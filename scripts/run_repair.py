"""Stage 5: controlled repair, optional all-origin sweep, and resumable raw executions."""
from __future__ import annotations

from pathlib import Path

from _common import parse_args, boot, load_agent, resolve_dataset  # type: ignore

from src.agent import run_repair_batch
from src.agent.react_agent import Trajectory
from src.repair import repair_record, parse_strategy
from src.repair.controlled import (
    configured_strategies, budget_multipliers, plan_repairs,
    freeze_manifest, file_fingerprint,
)
from src.utils import load_item, load_json, save_json, append_jsonl, read_jsonl


def _trial_key(row):
    return (row["qid"], row["strategy"], row["seed"], row["multiplier"])


def run(cfg, log, args):
    ds = resolve_dataset(cfg, args)
    if ds["name"] != cfg.dataset.name:
        raise ValueError("Use a dataset-specific config for repair; do not mix saved pools")
    repair = cfg.raw["repair"]
    strategies = configured_strategies(repair)
    multipliers = budget_multipliers(repair)
    seeds = repair["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
        raise ValueError("Repair seeds must be a nonempty list of unique integers")
    ids_path = Path(cfg.path("data_processed")) / "failed_ids.json"
    failed_ids = load_json(ids_path)
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError("Duplicate failed question IDs")
    if args.limit:
        failed_ids = failed_ids[:args.limit]
    if not failed_ids:
        raise ValueError("No failed trajectories to repair")
    pool_path = Path(cfg.path("data_processed")) / "pool.json"
    records = load_json(pool_path)
    pool = {record["_id"]: record for record in records}
    if len(pool) != len(records) or not set(failed_ids) <= pool.keys():
        raise ValueError("Pool contains duplicate IDs or is missing failed questions")

    needs_uncertainty = any(parse_strategy(s)["base"] == "uncertainty" for s in strategies)
    needs_annotations = any(parse_strategy(s)["base"] == "oracle_targeted" or
                            parse_strategy(s)["informed"] for s in strategies)
    inputs = [pool_path, ids_path]
    model_path = Path(cfg.path("data_processed")) / "agent_model.json"
    if not model_path.exists():
        raise ValueError("Missing original agent_model.json; verify model identity before repair")
    cached = load_json(model_path)
    if cached["name"] != cfg.models.agent.name or cached["dtype"] != cfg.models.agent.dtype:
        raise ValueError("Original agent model cache differs from repair configuration")
    inputs.append(model_path)
    for qid in failed_ids:
        inputs.append(Path(cfg.path("trajectories")) / f"{qid}.json")
        if needs_uncertainty:
            inputs.append(Path(cfg.path("uncertainty")) / f"{qid}.json")
        if needs_annotations:
            inputs.append(Path(cfg.path("annotations")) / f"{qid}.json")
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), *sorted((root / "src").rglob("*.py"))]
    payload = {
        "schema": 1, "configuration": cfg.raw, "question_ids": failed_ids,
        "input_sha256": {str(p): file_fingerprint(p) for p in inputs},
        "code_sha256": {str(p.relative_to(root)): file_fingerprint(p) for p in code},
    }
    result_path = Path(cfg.path("repairs")) / "results.jsonl"
    origin_path = Path(cfg.path("repairs")) / "origins.jsonl"
    execution_dir = Path(cfg.path("repairs")) / "executions"
    dry_run = getattr(args, "dry_run", False)
    digest = None if dry_run else freeze_manifest(
        Path(cfg.path("repairs")) / "manifest.json", payload,
        [result_path, origin_path, execution_dir])
    completed = set()
    origins_completed = set()
    if not dry_run:
        for path, seen in ((result_path, completed), (origin_path, origins_completed)):
            for row in read_jsonl(path):
                key = _trial_key(row)
                if key in seen or row.get("manifest_sha256") != digest:
                    raise ValueError("Duplicate or incompatible saved repair rows")
                seen.add(key)

    batch_size = cfg.raw["runtime"]["repair_batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Repair batch size must be positive")
    client = None
    total_jobs = new_jobs = total_rows = 0
    for multiplier in multipliers:
        for seed in seeds:
            for start in range(0, len(failed_ids), 64):
                jobs = []
                for qid in failed_ids[start:start + 64]:
                    original = load_item(cfg.path("trajectories"), qid)
                    uncertainty = load_item(cfg.path("uncertainty"), qid) if needs_uncertainty else None
                    annotation = load_item(cfg.path("annotations"), qid) if needs_annotations else None
                    jobs.extend(plan_repairs(cfg.raw, pool[qid], original, uncertainty,
                                              annotation, seed, multiplier, strategies))
                total_jobs += len(jobs)
                total_rows += sum(len(job["strategies"]) for job in jobs)
                if dry_run:
                    continue
                missing = [job for job in jobs if not
                           (execution_dir / (job["meta"]["execution_id"] + ".json")).exists()]
                if missing and client is None:
                    client = load_agent(cfg, log)
                    if client.model_name != cfg.models.agent.name:
                        raise ValueError("Loaded model differs from the frozen repair configuration")
                for offset in range(0, len(missing), batch_size):
                    batch = missing[offset:offset + batch_size]
                    trajectories = run_repair_batch(
                        client, batch, max_steps=cfg.agent.max_steps,
                        max_tokens_per_step=cfg.agent.max_tokens_per_step,
                        temperature=repair["nudge"].get("temperature", 0.7)
                                    if repair["nudge"].get("enabled") else 0.0,
                        seed=seed, env_cls=ds["env_cls"], score_fn=ds["score"],
                        step_budget_mode=repair.get("step_budget_mode", "total"))
                    for job, trajectory in zip(batch, trajectories):
                        if trajectory.meta["recovery_gen_tokens"] > job["token_budget"]:
                            raise ValueError("Inference exceeded the frozen generated-token budget")
                        raw = trajectory.to_dict()
                        for step in raw["steps"]:
                            step["generation"] = None
                        save_json({"manifest_sha256": digest, "trajectory": raw},
                                  execution_dir / (job["meta"]["execution_id"] + ".json"))
                    new_jobs += len(batch)
                for job in jobs:
                    saved = load_json(execution_dir / (job["meta"]["execution_id"] + ".json"))
                    if saved["manifest_sha256"] != digest:
                        raise ValueError("Execution cache has an incompatible manifest")
                    trajectory = Trajectory.from_dict(saved["trajectory"])
                    origin = job["meta"]["target_step"]
                    outputs = [(name, result_path, completed) for name in job["strategies"]]
                    if repair.get("origin_sweep"):
                        outputs.append((f"origin_{origin}", origin_path, origins_completed))
                    for name, path, seen in outputs:
                        row = repair_record(trajectory, name, seed, origin, job["oracle_step"],
                                            job["original_gen_tokens"], job["token_budget"])
                        row.update(multiplier=multiplier, manifest_sha256=digest)
                        row.update(job["policy_costs"].get(name, {}))
                        selection = row.get("selection_gen_tokens")
                        row["incremental_gen_tokens"] = (None if selection is None else
                                                           row["recovery_gen_tokens"] + selection)
                        key = _trial_key(row)
                        if key not in seen:
                            append_jsonl(row, path)
                            seen.add(key)
    log.info(f"{'DRY RUN' if dry_run else 'Completed'}: {total_jobs} unique planned repairs, "
             f"{total_rows} strategy rows, {new_jobs} new executions; "
             f"step mode={repair.get('step_budget_mode', 'total')}")
    return {"planned_executions": total_jobs, "strategy_rows": total_rows,
            "new_executions": new_jobs}


def main():
    args = parse_args("Repair experiments with controlled origin and budget settings",
                      extra=lambda p: p.add_argument("--dry-run", action="store_true"))
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    cfg, log = boot("stage5", args)
    run(cfg, log, args)


if __name__ == "__main__":
    main()
