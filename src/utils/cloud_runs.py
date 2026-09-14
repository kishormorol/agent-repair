"""Small, fail-closed helpers for the portable controlled-experiment notebook."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import yaml

from ..repair.controlled import fingerprint, plan_repairs
from ..repair.strategies import parse_strategy, uncertainty_strategy_name
from ..repair.position_matched import validate_position_profile
from .cohorts import read_ids
from .io import load_json, save_json, append_jsonl


QA_DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def write_once(path, value):
    """Allow identical resumes, never replace a frozen artifact with different data."""
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}. Use a new run ID.")
    else:
        save_json(value, path)


def primary_controls(strategy, *, include_diagnostics=True, include_position_control=False):
    if type(include_diagnostics) is not bool:
        raise ValueError("include_diagnostics must be a boolean")
    if type(include_position_control) is not bool:
        raise ValueError("include_position_control must be a boolean")
    p = parse_strategy(strategy)
    if (p["base"] != "uncertainty" or p["informed"] or p["backtrack"] not in (0, 2)
            or p["metric"] not in {"perplexity", "token_entropy_max", "max_token_prob_max"}
            or p["rule"] not in {"argmax", "topk", "cascade_weighted"}):
        raise ValueError("Choose one stored-token, generic policy from the bounded development grid")
    random = "random_step" + ("__bt2" if p["backtrack"] else "")
    controls = ["full_restart", random, "fixed_early", strategy]
    comparisons = ["full_restart", random]
    if include_position_control:
        if p["backtrack"] != 2:
            raise ValueError("The six-condition profile requires the selected backtrack-two policy")
        no_backtrack = uncertainty_strategy_name(p["metric"], p["rule"], 0)
        controls.extend(["position_matched_random", no_backtrack])
        comparisons.extend(["position_matched_random", no_backtrack])
    if include_diagnostics:
        controls.extend([uncertainty_strategy_name(p["metric"], p["rule"], 0),
                         uncertainty_strategy_name(p["metric"], p["rule"], 2), "random_step"])
    return list(dict.fromkeys(controls)), comparisons


def run_directory(storage, dataset, phase, run_id):
    if dataset not in QA_DATASETS or phase not in {"pilot", "development", "test"}:
        raise ValueError("Select a QA dataset and pilot, development, or test phase")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Run ID must be a simple directory name")
    return Path(storage).resolve() / "runs" / "qwen32b" / phase / run_id / dataset


def prepare_config(repo, run_dir, *, dataset, phase, run_id, model_pin,
                   strategy, pool_size, batch_size, origin_sweep, multipliers,
                   code_sha256, test_ids=None, explored_ids=None, policy_file=None,
                   include_diagnostics=True, development_ids=None, position_profile=None):
    repo, run_dir = Path(repo), Path(run_dir).resolve()
    if dataset not in QA_DATASETS or phase not in {"pilot", "development", "test"}:
        raise ValueError("Invalid dataset/phase")
    if type(batch_size) is not int or batch_size < 1 or type(pool_size) is not int or pool_size < 1:
        raise ValueError("Batch and pool sizes must be positive integers")
    if type(origin_sweep) is not bool:
        raise ValueError("origin_sweep must be a boolean")
    if not re.fullmatch(r"[0-9a-f]{40}", model_pin["revision"]):
        raise ValueError("A resolved Hugging Face commit SHA is required")
    if model_pin["repo_id"] != "Qwen/Qwen2.5-32B-Instruct-AWQ":
        raise ValueError("This notebook implements the recorded Qwen32B study, not model substitution")
    snapshot = Path(model_pin["snapshot_path"])
    if not snapshot.is_absolute() or not (snapshot / "config.json").is_file():
        raise ValueError("Download the pinned local model snapshot first")
    cfg = yaml.safe_load((repo / "config/config_iclr_pilot.yaml").read_text())
    catalog = yaml.safe_load((repo / "config/datasets.yaml").read_text())
    cfg["dataset"].update({k: catalog[dataset][k] for k in ("raw_filename", "url", "stratify_by")})
    cfg["dataset"].update(name=dataset, split="test" if phase == "test" else "development",
                          pool_size=pool_size)
    cfg["models"]["agent"]["name"] = str(snapshot)
    cfg["runtime"]["gen_batch_size"] = batch_size
    cfg["runtime"]["repair_batch_size"] = batch_size
    strategies, comparisons = primary_controls(strategy, include_diagnostics=include_diagnostics,
                                                include_position_control=position_profile is not None)
    cfg["repair"].update(run_id=run_id, strategies=strategies, origin_sweep=origin_sweep)
    profile = load_json(position_profile) if position_profile is not None else None
    if profile is not None:
        payload = validate_position_profile(profile, dataset=dataset, strategy=strategy)
        if payload["model"] != {"repo_id": model_pin["repo_id"], "revision": model_pin["revision"]}:
            raise ValueError("Position profile model differs from the frozen model")
        cfg["repair"]["position_matched_profile"] = profile
    cfg["repair"]["budget"]["multipliers"] = multipliers
    from ..repair.controlled import budget_multipliers
    budget_multipliers(cfg["repair"])
    cfg["notebook_provenance"] = {"code_sha256": code_sha256, "model": model_pin,
                                   "phase": phase, "primary_strategy": strategy}
    for key in cfg["paths"]:
        if key.endswith("_base"):
            cfg["paths"][key] = str(run_dir)
        elif key.startswith("data_"):
            cfg["paths"][key] = str(run_dir / "data" / key.removeprefix("data_"))
        else:
            cfg["paths"][key] = str(run_dir / "outputs" / key) if key != "outputs" else str(run_dir / "outputs")

    frozen_ids = {}
    if development_ids and phase != "development":
        raise ValueError("Development IDs require the development phase")
    if phase == "test":
        if not test_ids or not explored_ids or not policy_file:
            raise ValueError("Test runs require test IDs, all explored IDs, and a frozen study policy")
        ids, excluded = read_ids(test_ids), read_ids(explored_ids)
        if not ids or not excluded or set(ids) & set(excluded):
            raise ValueError("Test/explored IDs must be nonempty and disjoint")
        policy = load_json(policy_file)
        expected = {"strategy": strategy, "model_id": model_pin["repo_id"],
                    "model_revision": model_pin["revision"], "seeds": [0, 1, 2],
                    "multipliers": multipliers, "origin_sweep": origin_sweep,
                    "max_steps": cfg["agent"]["max_steps"],
                    "max_tokens_per_step": cfg["agent"]["max_tokens_per_step"],
                    "batch_size": batch_size,
                    "code_sha256": code_sha256}
        if any(policy.get(key) != value for key, value in expected.items()):
            raise ValueError("Run settings differ from the frozen study policy")
        # Earlier manifests always used the expanded diagnostics profile.
        if policy.get("include_diagnostics", True) is not include_diagnostics:
            raise ValueError("Diagnostic conditions differ from the frozen study policy")
        profile_digest = profile["sha256"] if profile is not None else None
        if policy.get("position_profile_sha256") != profile_digest:
            raise ValueError("Position profile differs from the frozen study policy")
        if profile is not None:
            validate_position_profile(profile, excluded_ids=excluded)
            if policy.get("strategies") != strategies:
                raise ValueError("Six-condition strategies differ from the frozen study policy")
            if policy.get("primary_comparisons") != comparisons:
                raise ValueError("Primary comparison family differs from the frozen study policy")
        if (not policy.get("selection_note") or not policy.get("precision_target")
                or policy.get("n_questions", {}).get(dataset) != len(ids)):
            raise ValueError("Freeze development selection, precision target, and per-dataset sample sizes")
        declared = policy.get("cohort_sha256", {}).get(dataset, {})
        if declared != {"test": fingerprint(ids), "explored": fingerprint(excluded)}:
            raise ValueError("ID lists differ from the frozen study policy")
        cfg["dataset"]["pool_size"] = len(ids)
        for name, values in (("test_ids", ids), ("explored_ids", excluded)):
            path = run_dir / (name + ".json")
            frozen_ids[path] = values
            key = "id_manifest" if name == "test_ids" else "exclude_ids_manifest"
            cfg["dataset"][key] = str(path)
        cfg["notebook_provenance"]["study_policy_sha256"] = fingerprint(policy)
        frozen_ids[run_dir / "study_policy.json"] = policy
    elif phase == "development" and development_ids and explored_ids:
        if test_ids or policy_file:
            raise ValueError("Test manifests are only accepted in the test phase")
        ids, excluded = read_ids(development_ids), read_ids(explored_ids)
        if len(ids) != pool_size:
            raise ValueError("Development cohort size must equal pool_size")
        if not excluded or set(ids) & set(excluded):
            raise ValueError("Development/explored IDs must be nonempty and disjoint")
        for name, values, key in (("development_ids", ids, "id_manifest"),
                                  ("explored_ids", excluded, "exclude_ids_manifest")):
            path = run_dir / (name + ".json")
            frozen_ids[path] = values
            cfg["dataset"][key] = str(path)
    elif development_ids:
        raise ValueError("Frozen development IDs require an explored-ID manifest")
    elif test_ids or explored_ids or policy_file:
        raise ValueError("Test manifests are only accepted in the test phase")

    config_path = run_dir / "config.yaml"
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
        raise ValueError("Run configuration changed; use a new run ID")
    for path, values in frozen_ids.items():
        write_once(path, values)
    if profile is not None:
        write_once(run_dir / "position_profile.json", profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_once(run_dir / "config_lock.json", {"sha256": fingerprint(cfg)})
    return config_path


def repair_plan(cfg):
    """Count actual deduplicated jobs before loading a repair model."""
    raw = cfg.raw
    ids = load_json(Path(cfg.path("data_processed")) / "failed_ids.json")
    records = load_json(Path(cfg.path("data_processed")) / "pool.json")
    pool = {record["_id"]: record for record in records}
    total = rows = cached = 0
    for qid in ids:
        original = load_json(Path(cfg.path("trajectories")) / (qid + ".json"))
        uncertainty = load_json(Path(cfg.path("uncertainty")) / (qid + ".json"))
        for seed in raw["repair"]["seeds"]:
            for multiplier in raw["repair"]["budget"]["multipliers"]:
                jobs = plan_repairs(raw, pool[qid], original, uncertainty, None, seed,
                                    multiplier, raw["repair"]["strategies"])
                total += len(jobs)
                rows += sum(len(job["strategies"]) for job in jobs)
                cached += sum((Path(cfg.path("repairs")) / "executions" /
                               (job["meta"]["execution_id"] + ".json")).exists() for job in jobs)
    origins = raw["agent"]["max_steps"]
    if not raw["repair"]["origin_sweep"]:
        origins = len(raw["repair"]["strategies"])
    upper_bound = origins * len(raw["repair"]["seeds"]) * len(raw["repair"]["budget"]["multipliers"])
    return {"failed_questions": len(ids), "planned_unique_executions": total,
            "strategy_rows": rows, "cached_executions": cached,
            "missing_executions": total - cached,
            "per_question_execution_upper_bound": upper_bound}


def run_stage(python, repo, run_dir, script, config, extra=(), env=None):
    """Stream subprocess errors and persist attempt timing, including failed attempts."""
    command = [str(python), str(Path(repo) / "scripts" / script), "--config", str(config), *extra]
    run_dir = Path(run_dir)
    started = time.time()
    status = "failed"
    before = len(list((run_dir / "outputs/repairs/executions").glob("*.json")))
    try:
        subprocess.run(command, cwd=repo, env=env, check=True)
        status = "complete"
    finally:
        after = len(list((run_dir / "outputs/repairs/executions").glob("*.json")))
        append_jsonl({"script": script, "extra": list(extra), "status": status,
                      "started_unix": started, "elapsed_seconds": time.time() - started,
                      "new_execution_files": after - before}, run_dir / "stage_attempts.jsonl")


def pin_model(run_dir, cache_dir, repo_id, revision=None):
    """Download model and tokenizer from one frozen snapshot; reuse the pin on resume."""
    from huggingface_hub import HfApi, snapshot_download
    path = Path(run_dir) / "model_pin.json"
    if path.exists():
        pin = load_json(path)
        if pin["repo_id"] != repo_id or (revision and revision != pin["revision"]):
            raise ValueError("Requested model differs from the saved snapshot pin")
        commit = pin["revision"]
    else:
        commit = HfApi().model_info(repo_id, revision=revision or "main").sha
        write_once(path, {"repo_id": repo_id, "revision": commit})
    snapshot = snapshot_download(repo_id, revision=commit, cache_dir=str(cache_dir))
    result = {"repo_id": repo_id, "revision": commit, "snapshot_path": str(Path(snapshot).resolve())}
    write_once(Path(run_dir) / "model_snapshot.json", result)
    return result
