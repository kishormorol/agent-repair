"""Execute a separately frozen, three-policy diagnosis/allowance follow-up."""
from __future__ import annotations

from pathlib import Path
import math

from _common import parse_args, boot, load_agent, resolve_dataset  # type: ignore
from src.agent import run_repair_batch
from src.agent.react_agent import Trajectory
from src.env.hotpot_env import score_answer
from src.repair import repair_record
from src.repair.controlled import fingerprint, file_fingerprint, freeze_manifest, budget_multipliers
from src.repair.diagnosis import (DIAGNOSIS_STRATEGY, TREATMENT, DIAGNOSIS_PROMPT,
                                 DIAGNOSIS_SETTINGS, diagnose_original, plan_diagnosis_repairs)
from src.utils import load_json, save_json, append_jsonl, read_jsonl


STRATEGIES = ["full_restart", TREATMENT, DIAGNOSIS_STRATEGY]


def trial_key(row):
    return row["qid"], row["strategy"], row["seed"], row["multiplier"]


def load_inputs(cfg, protocol_path):
    frozen = load_json(protocol_path)
    protocol = frozen["payload"]
    if frozen["sha256"] != fingerprint(protocol):
        raise ValueError("Follow-up protocol checksum mismatch")
    expected = {"strategies": STRATEGIES, "seeds": [0, 1, 2], "multipliers": [0.5, 1.0, 2.0],
                "max_new_steps": 8, "max_tokens_per_step": 512,
                "diagnosis_prompt_sha256": fingerprint(DIAGNOSIS_PROMPT),
                "diagnosis_settings": DIAGNOSIS_SETTINGS, "dataset": "hotpotqa"}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Follow-up differs from its declared policy/allowance design")
    all_ids = protocol["question_ids"]
    if (not all_ids or len(all_ids) != len(set(all_ids)) or not protocol["explored_ids"]
            or [q for batch in protocol["batches"] for q in batch["ids"]] != all_ids
            or len({b["run_id"] for b in protocol["batches"]}) != len(protocol["batches"])):
        raise ValueError("Follow-up protocol has incomplete or duplicate batch coverage")
    batches = [b for b in protocol["batches"] if b["run_id"] == cfg.raw["repair"]["run_id"]]
    if len(batches) != 1:
        raise ValueError("Run must identify exactly one frozen follow-up batch")
    if fingerprint(cfg.raw) != batches[0]["configuration_sha256"]:
        raise ValueError("Run configuration differs from the frozen follow-up batch")
    ids = batches[0]["ids"]
    if (not ids or len(ids) != len(set(ids)) or not set(ids) <= set(protocol["question_ids"])
            or set(protocol["question_ids"]) & set(protocol["explored_ids"])):
        raise ValueError("Invalid or overlapping follow-up cohort")
    repair = cfg.raw["repair"]
    model = cfg.raw.get("notebook_provenance", {}).get("model", {})
    if (cfg.dataset.name != "hotpotqa" or cfg.agent.max_steps != 8
            or cfg.agent.max_tokens_per_step != 512 or repair["seeds"] != protocol["seeds"]
            or budget_multipliers(repair) != protocol["multipliers"]
            or repair.get("step_budget_mode") != "new" or not repair.get("match_restart_hint")
            or not repair.get("nudge", {}).get("enabled")
            or repair["nudge"].get("temperature") != 0.7
            or repair["nudge"].get("retry_hint") != protocol["retry_hint"]
            or cfg.raw["runtime"]["repair_batch_size"] != protocol["batch_size"]
            or cfg.raw["runtime"]["gen_batch_size"] != protocol["batch_size"]
            or model.get("revision") != protocol["model_revision"]
            or model.get("repo_id") != protocol["model_id"]):
        raise ValueError("Run configuration differs from the frozen follow-up")
    processed = Path(cfg.path("data_processed"))
    paths = [Path(protocol_path), processed/"pool.json", processed/"failed_ids.json", processed/"agent_model.json"]
    records = load_json(paths[1])
    pool = {r["_id"]: r for r in records}
    if len(pool) != len(records) or set(pool) != set(ids):
        raise ValueError("Follow-up pool does not match the complete frozen batch")
    model_cache = load_json(paths[3])
    if model_cache["name"] != cfg.models.agent.name or model_cache["dtype"] != cfg.models.agent.dtype:
        raise ValueError("Original agent model differs from the follow-up model")
    originals, uncertainty = {}, {}
    for qid in ids:
        path = Path(cfg.path("trajectories"))/f"{qid}.json"
        original = load_json(path)
        score = score_answer(original["final_answer"], pool[qid]["answer"])
        if (original["qid"] != qid or original["question"] != pool[qid]["question"]
                or original["gold_answer"] != pool[qid]["answer"]
                or bool(original["success"]) != score["correct"] or original["em"] != score["em"]
                or not math.isclose(original["f1"], score["f1"], rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Initial follow-up trajectory identity/scoring mismatch")
        originals[qid] = original
        paths.append(path)
        if not original["success"]:
            path = Path(cfg.path("uncertainty"))/f"{qid}.json"
            uncertainty[qid] = load_json(path)
            paths.append(path)
    failed = load_json(processed/"failed_ids.json")
    if len(failed) != len(set(failed)) or set(failed) != set(uncertainty):
        raise ValueError("Incomplete or duplicate initial-failure cohort")
    return frozen, pool, originals, uncertainty, failed, paths


def run(cfg, log, args):
    if getattr(args, "limit", None) is not None:
        raise ValueError("Use a separately frozen smoke-test batch; do not truncate the follow-up cohort")
    ds = resolve_dataset(cfg, args)
    if ds["name"] != "hotpotqa":
        raise ValueError("This follow-up requires its frozen HotpotQA environment")
    frozen, pool, originals, uncertainty, failed, inputs = load_inputs(cfg, args.protocol)
    protocol = frozen["payload"]
    guard = protocol["max_repair_executions_per_batch"]
    if type(guard) is not int or guard < 1 or type(protocol["batch_size"]) is not int or protocol["batch_size"] < 1:
        raise ValueError("Follow-up execution guard and batch size must be positive integers")
    upper_bound = len(failed) * len(STRATEGIES) * len(protocol["seeds"]) * len(protocol["multipliers"])
    if upper_bound > guard:
        raise ValueError("Follow-up exceeds its frozen execution-count guard")
    if args.dry_run:
        return {"failed_questions": len(failed), "maximum_repair_executions": upper_bound,
                "diagnosis_requests": len(failed), "model_loaded": False}
    out = Path(cfg.path("repairs"))/"diagnosis-followup"
    rows_path, executions, diagnoses = out/"results.jsonl", out/"executions", out/"diagnoses"
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), *sorted((root/"src").rglob("*.py"))]
    payload = {"schema": 1, "study_sha256": frozen["sha256"], "configuration": cfg.raw,
               "question_ids": failed, "input_sha256": {str(p): file_fingerprint(p) for p in inputs},
               "code_sha256": {str(p.relative_to(root)): file_fingerprint(p) for p in code}}
    digest = freeze_manifest(out/"manifest.json", payload, [rows_path, executions, diagnoses])
    expected_keys = {(q, a, s, m) for q in failed for a in STRATEGIES
                     for s in protocol["seeds"] for m in protocol["multipliers"]}
    completed = {}
    for row in read_jsonl(rows_path):
        key = trial_key(row)
        if key not in expected_keys or key in completed or row.get("manifest_sha256") != digest:
            raise ValueError("Duplicate, unexpected or incompatible follow-up row")
        completed[key] = row
    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = load_agent(cfg, log)
            if client.model_name != cfg.models.agent.name:
                raise ValueError("Follow-up loaded a different model")
        return client

    diagnosis_rows = {}
    new_diagnoses = new_executions = planned = 0
    model_signature = {"name": cfg.models.agent.name, "dtype": cfg.models.agent.dtype,
                       "revision": protocol["model_revision"]}
    for qid in failed:
        path = diagnoses/f"{qid}.json"
        cached = path.exists()
        diagnosis_rows[qid] = diagnose_original(None if cached else get_client(), originals[qid], path,
                                                model_signature=model_signature)
        new_diagnoses += int(not cached)
    batch_size = cfg.raw["runtime"]["repair_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Follow-up batch size must be positive")
    for multiplier in protocol["multipliers"]:
        for seed in protocol["seeds"]:
            jobs = [job for qid in failed for job in plan_diagnosis_repairs(
                cfg.raw, pool[qid], originals[qid], uncertainty[qid], diagnosis_rows[qid], seed, multiplier)]
            planned += len(jobs)
            missing = [j for j in jobs if not (executions/(j["meta"]["execution_id"]+".json")).exists()]
            for offset in range(0, len(missing), batch_size):
                batch = missing[offset:offset+batch_size]
                trajectories = run_repair_batch(get_client(), batch, max_steps=8, max_tokens_per_step=512,
                    temperature=0.7, seed=seed, env_cls=ds["env_cls"], score_fn=ds["score"], step_budget_mode="new")
                if len(trajectories) != len(batch):
                    raise ValueError("Incomplete follow-up execution batch")
                for job, trajectory in zip(batch, trajectories):
                    if trajectory.meta["recovery_gen_tokens"] > job["token_budget"]:
                        raise ValueError("Follow-up exceeded its recovery token allowance")
                    raw = trajectory.to_dict()
                    for step in raw["steps"]:
                        step["generation"] = None
                    save_json({"manifest_sha256": digest, "trajectory_sha256": fingerprint(raw), "trajectory": raw},
                              executions/(job["meta"]["execution_id"]+".json"))
                new_executions += len(batch)
            for job in jobs:
                saved = load_json(executions/(job["meta"]["execution_id"]+".json"))
                if saved["manifest_sha256"] != digest or saved["trajectory_sha256"] != fingerprint(saved["trajectory"]):
                    raise ValueError("Follow-up execution checksum/manifest mismatch")
                trajectory = Trajectory.from_dict(saved["trajectory"])
                for name in job["strategies"]:
                    row = repair_record(trajectory, name, seed, job["meta"]["target_step"], None,
                                        job["original_gen_tokens"], job["token_budget"])
                    row.update(multiplier=multiplier, manifest_sha256=digest, **job["policy_costs"][name])
                    for cost in ["gen_tokens", "prompt_tokens", "model_requests"]:
                        selection, recovery = row["selection_"+cost], row["recovery_"+cost]
                        if selection is None or recovery is None:
                            raise ValueError("Missing measured follow-up cost")
                        row["incremental_"+cost] = selection + recovery
                    if name == DIAGNOSIS_STRATEGY:
                        diagnosis = diagnosis_rows[row["qid"]]
                        row.update(diagnosis_input_sha256=diagnosis["input_sha256"],
                                   diagnosis_fallback=diagnosis["fallback"],
                                   diagnosis_fallback_reason=diagnosis["fallback_reason"])
                    key = trial_key(row)
                    if key in completed:
                        if completed[key] != row:
                            raise ValueError("Saved follow-up row differs from its cached execution/diagnosis")
                    else:
                        append_jsonl(row, rows_path)
                        completed[key] = row
    if set(completed) != expected_keys:
        raise ValueError("Incomplete follow-up trial coverage")
    result = {"failed_questions": len(failed), "strategy_rows": len(completed),
              "planned_executions": planned, "new_executions": new_executions,
              "new_diagnoses": new_diagnoses, "study_sha256": frozen["sha256"]}
    save_json(result, out/"completion.json")
    log.info(str(result))
    return result


def main():
    def extra(parser):
        parser.add_argument("--protocol", required=True)
        parser.add_argument("--dry-run", action="store_true")
    args = parse_args(__doc__, extra=extra)
    cfg, log = boot("diagnosis-followup", args)
    print(run(cfg, log, args))


if __name__ == "__main__":
    main()
