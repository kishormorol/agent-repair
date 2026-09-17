"""Execute a frozen extension cell with independently auditable raw records."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.react_agent import ReActAgent, Step, build_messages
from src.env.extension import ExtensionEnv
from src.llm.vllm_client import VLLMClient
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import TREATMENT, DIAGNOSIS_STRATEGY, diagnose_original
from src.repair.position_matched import fit_position_profile
from src.repair.strategies import select_target_step, make_rng
from src.uncertainty.metrics import compute_math_metrics


def _restore_token_keys(value):
    """Recover TokenInfo's integer keys before checking its original hash.

    JSON object keys are strings, while TokenInfo.to_dict() returns integer
    token IDs. Sorting IDs such as 2 and 10 gives a different order after a
    plain JSON round trip. Restore only the token schema, leaving unrelated
    numeric string keys and the stored checksum unchanged.
    """
    if set(value) == {"token_id", "token_str", "logprob", "top_logprobs"}:
        probabilities = value["top_logprobs"]
        restored = {}
        for key, probability in probabilities.items():
            token_id = int(key)
            if str(token_id) != key:
                raise ValueError("Noncanonical token ID in top_logprobs")
            restored[token_id] = probability
        value["top_logprobs"] = restored
    return value


def save_envelope(path, payload, identity):
    path = Path(path)
    value = {"identity": identity, "sha256": fingerprint(payload), "payload": payload}
    if path.exists():
        if json.loads(path.read_text(), object_hook=_restore_token_keys) != value:
            raise ValueError(f"Preserve conflicting output: {path}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return payload


def load_envelope(path, identity):
    if not Path(path).is_file():
        raise ValueError(f"Missing required output: {path}")
    saved = json.loads(Path(path).read_text(), object_hook=_restore_token_keys)
    if saved.get("identity") != identity or saved.get("sha256") != fingerprint(saved.get("payload")):
        raise ValueError(f"Cached output identity/checksum mismatch: {path}")
    return saved["payload"]


def verify_package(package):
    package = Path(package)
    manifest = json.loads((package / "package-manifest.json").read_text())
    if manifest["sha256"] != fingerprint(manifest["files"]):
        raise ValueError("Package manifest checksum mismatch")
    for name, digest in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_fingerprint(package / path) != digest:
            raise ValueError(f"Package input checksum mismatch: {name}")
        if name.startswith("code/src/") or name.startswith("code/scripts/"):
            if file_fingerprint(ROOT / Path(*path.parts[1:])) != digest:
                raise ValueError("Run from the frozen package source, not changed working source")
    frozen = json.loads((package / "protocol.json").read_text())
    if frozen["sha256"] != fingerprint(frozen["payload"]):
        raise ValueError("Protocol checksum mismatch")
    return frozen["payload"], frozen["sha256"]


def uncertainty_record(original):
    started = time.monotonic()
    steps = []
    for step in original["steps"]:
        values = compute_math_metrics(step["generation"]["tokens"])
        if "perplexity" not in values or not math.isfinite(values["perplexity"]):
            raise ValueError("Missing/invalid original sampled-token log probabilities")
        steps.append({"index": step["index"], "uncertainty": values})
    return {"qid": original["qid"], "steps": steps,
            "acquisition_wall_latency_s": time.monotonic() - started}


def select_origins(protocol, original, uncertainty, diagnosis, seed, profile, strategies):
    choices = []
    for strategy in strategies:
        started = time.monotonic()
        if strategy == DIAGNOSIS_STRATEGY:
            origin = diagnosis["target_step"]
        elif strategy == "position_matched_random" and profile is None:
            origin = 0
        else:
            origin = select_target_step(strategy, len(original["steps"]), None, uncertainty,
                                        make_rng(original["qid"], strategy, seed), position_profile=profile)
        seconds = time.monotonic() - started
        cost = {"selection_gen_tokens": 0, "selection_prompt_tokens": 0,
                "selection_model_requests": 0, "selection_latency_s": seconds}
        if strategy.startswith("unc__"):
            cost["selection_latency_s"] += uncertainty["acquisition_wall_latency_s"]
        elif strategy == DIAGNOSIS_STRATEGY:
            cost.update({key: diagnosis[key] for key in cost if key != "selection_latency_s"})
            cost["selection_latency_s"] += diagnosis["selection_latency_s"]
        choices.append({"strategy": strategy, "origin": origin, "seed": seed, **cost})
    return choices


def repair_jobs(protocol, record, original, uncertainty, diagnosis, profile, *, runtime=False):
    strategies = protocol["runtime"]["strategies"] if runtime else protocol["strategies"]
    jobs = {}
    for seed in protocol["seeds"]:
        for choice in select_origins(protocol, original, uncertainty, diagnosis, seed, profile, strategies):
            origin = choice["origin"]
            budget = None if runtime else int(protocol["generated_token_multiplier"] * original["total_gen_tokens"])
            identity = {"qid": record["_id"], "seed": seed, "origin": origin, "budget": budget,
                        "runtime_policy": choice["strategy"] if runtime else None}
            execution_id = fingerprint(identity)
            if execution_id not in jobs:
                prefix = [Step.from_dict({**s, "generation": None}) for s in original["steps"][:origin]]
                jobs[execution_id] = {"execution_id": execution_id, **identity, "prefix": prefix,
                                      "prompt_sha256": fingerprint(build_messages(record["question"], prefix, protocol["retry_hint"])),
                                      "policies": []}
            jobs[execution_id]["policies"].append(choice)
    jobs = list(jobs.values())
    random.Random(fingerprint([protocol["run_id"], record["_id"], "runtime" if runtime else "token"])).shuffle(jobs)
    return jobs


def trial_rows(trace, job, *, model_key, dataset, runtime):
    rows = []
    for choice in job["policies"]:
        row = {"qid": trace["qid"], "model_key": model_key, "dataset": dataset,
               "mode": "runtime" if runtime else "token", "strategy": choice["strategy"], "seed": job["seed"],
               "origin": job["origin"], "execution_id": job["execution_id"], "budget": job["budget"],
               "prompt_sha256": job["prompt_sha256"],
               **{k: trace[k] for k in ["success", "em", "f1", "final_answer", "terminated_reason"]},
               **{k: v for k, v in choice.items() if k.startswith("selection_")},
               **{k: v for k, v in trace["meta"].items() if k.startswith("recovery_")}}
        for suffix in ["gen_tokens", "prompt_tokens", "model_requests"]:
            row["incremental_" + suffix] = row["recovery_" + suffix] + row["selection_" + suffix]
        row["incremental_wall_latency_s"] = row["recovery_wall_latency_s"] + row["selection_latency_s"]
        rows.append(row)
    return rows


def run_model(package, output, model_key, cache, deadline_utc, client=None):
    package, output = Path(package), Path(output)
    protocol, digest = verify_package(package)
    model = protocol["models"][model_key]
    deadline = dt.datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError("Use an explicit UTC execution deadline")

    def guard():
        if dt.datetime.now(dt.timezone.utc) >= deadline - dt.timedelta(minutes=8):
            raise TimeoutError("Stop for result archival before the independent EC2 deadline")

    guard()
    if client is None:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model["repo_id"], revision=model["revision"], cache_dir=str(cache),
                                       allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
                                       ignore_patterns=["consolidated.safetensors"])
        if Path(model_path).name != model["revision"]:
            raise ValueError("Downloaded snapshot differs from the model pin")
        client = VLLMClient(model_path, dtype=model["dtype"], max_model_len=protocol["max_model_len"],
                            logprobs_topk=protocol["logprobs_topk"], enable_prefix_caching=False,
                            trust_remote_code=False, seed=protocol["initial_seed"]).load()
    signature = {"name": client.model_name, "dtype": model["dtype"], "revision": model["revision"]}
    identity = {"protocol_sha256": digest, "model_key": model_key, "model": signature}
    save_envelope(output / model_key / "model.json", signature, identity)
    agent = ReActAgent(client, protocol["max_new_steps"], protocol["max_tokens_per_step"])
    warmup = []
    for index in range(3):
        guard()
        started = time.monotonic()
        result = client.chat([{"role": "system", "content": "Follow the instruction."},
                              {"role": "user", "content": "Write the numbers one through ten."}],
                             temperature=0., max_tokens=64, seed=42)[0]
        warmup.append({"seconds": time.monotonic()-started, "tokens": result.num_tokens})
    warmup_path = output / model_key / f"warmup-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    save_envelope(warmup_path, warmup, identity)
    for dataset, cohort in protocol["cohorts"].items():
        cell = output / model_key / dataset
        cell_identity = {**identity, "dataset": dataset}
        if (cell / "complete.json").exists():
            load_envelope(cell / "complete.json", cell_identity)
            continue
        records = {r["_id"]: r for r in json.loads((package / "data" / f"{dataset}.json").read_text())}
        originals, uncertainties = {}, {}
        for role in ["development", "main"]:
            for qid in cohort[role + "_ids"]:
                guard()
                path = cell / role / "originals" / f"{qid}.json"
                trace_identity = {**cell_identity, "qid": qid, "role": role, "record_sha256": fingerprint(records[qid])}
                if path.exists():
                    trace = load_envelope(path, trace_identity)
                else:
                    trace = agent.run(ExtensionEnv(records[qid]), temperature=0., seed=protocol["initial_seed"],
                                      record_timing=True, meta={"model_name": client.model_name}).to_dict()
                    save_envelope(path, trace, trace_identity)
                originals[qid] = trace
                unc_path = cell / role / "uncertainty" / f"{qid}.json"
                unc_identity = {**trace_identity, "original_sha256": fingerprint(trace)}
                uncertainties[qid] = (load_envelope(unc_path, unc_identity) if unc_path.exists() else
                                      save_envelope(unc_path, uncertainty_record(trace), unc_identity))
                print(json.dumps({"stage": "original", "model": model_key, "dataset": dataset, "role": role, "qid": qid}), flush=True)
            if role == "development":
                failed_dev = [originals[q] for q in cohort["development_ids"] if not originals[q]["success"]]
                profile = (fit_position_profile(failed_dev, uncertainties, source_ids=cohort["development_ids"],
                           dataset=dataset, strategy=TREATMENT, source_phase="development", source_run_id=protocol["run_id"],
                           model=signature, source_sha256=fingerprint(failed_dev)) if failed_dev else None)
                save_envelope(cell / "position-profile.json", profile, cell_identity)
        failed_main = [q for q in cohort["main_ids"] if not originals[q]["success"]]
        runtime_ids = (failed_main[:protocol["runtime"]["n_failures_max"]]
                       if model_key == protocol["runtime"]["model"] and dataset == protocol["runtime"]["dataset"] else [])
        for qid in failed_main:
            guard()
            original = originals[qid]
            diagnosis = diagnose_original(client, original, cell / "diagnoses" / f"{qid}.json", model_signature=signature)
            for runtime in ([False, True] if qid in runtime_ids else [False]):
                mode = "runtime" if runtime else "token"
                plan_path = cell / mode / "plans" / f"{qid}.json"
                plan_identity = {**cell_identity, "qid": qid, "original_sha256": fingerprint(original),
                                 "diagnosis_sha256": fingerprint(diagnosis), "runtime": runtime}
                if plan_path.exists():
                    serialized = load_envelope(plan_path, plan_identity)
                    jobs = [{**job, "prefix": [Step.from_dict(s) for s in job["prefix"]]} for job in serialized]
                else:
                    jobs = repair_jobs(protocol, records[qid], original, uncertainties[qid], diagnosis, profile, runtime=runtime)
                    serialized = [{**job, "prefix": [s.to_dict() for s in job["prefix"]]} for job in jobs]
                    save_envelope(plan_path, serialized, plan_identity)
                for job in jobs:
                    guard()
                    execution_path = cell / mode / "executions" / f"{job['execution_id']}.json"
                    execution_identity = {**plan_identity, "job_sha256": fingerprint({**job, "prefix": [s.to_dict() for s in job["prefix"]]})}
                    if execution_path.exists():
                        trace = load_envelope(execution_path, execution_identity)
                    else:
                        trace = agent.run(ExtensionEnv(records[qid]), temperature=protocol["repair_temperature"], seed=job["seed"],
                            prefix_steps=job["prefix"], nudge=protocol["retry_hint"], token_budget=job["budget"],
                            step_budget_mode="new", record_timing=True, meta={"model_name": client.model_name,
                            "execution_id": job["execution_id"], "prompt_sha256": job["prompt_sha256"]}).to_dict()
                        save_envelope(execution_path, trace, execution_identity)
                    rows = trial_rows(trace, job, model_key=model_key, dataset=dataset, runtime=runtime)
                    save_envelope(cell / mode / "trials" / f"{job['execution_id']}.json", rows, execution_identity)
                print(json.dumps({"stage": "repair", "model": model_key, "dataset": dataset, "mode": mode,
                                  "qid": qid, "unique_executions": len(jobs)}), flush=True)
        save_envelope(cell / "complete.json", {"main_questions": len(cohort["main_ids"]),
            "development_questions": len(cohort["development_ids"]), "main_failures": len(failed_main),
            "runtime_questions": len(runtime_ids), "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, cell_identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=["qwen32b", "mistral12b"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    run_model(args.package, args.output, args.model, args.cache, args.deadline_utc)
