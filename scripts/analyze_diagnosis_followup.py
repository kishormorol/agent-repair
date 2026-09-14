"""Audit a diagnosis follow-up and compute its separately frozen comparisons."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.env.hotpot_env import HotpotEnv
from src.eval.metrics import paired_mean_comparison, holm_correction
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import (DIAGNOSIS_STRATEGY, TREATMENT, diagnose_original,
                                 parse_diagnosis, plan_diagnosis_repairs)
from src.utils import load_json, read_jsonl


COSTS = ["gen_tokens", "prompt_tokens", "model_requests"]


def verify_bundle(bundle, expected):
    with zipfile.ZipFile(bundle) as stream:
        manifest = json.loads(stream.read("bundle_manifest.json"))
        if (manifest["code_sha256"] != expected
                or hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest() != expected):
            raise ValueError("Follow-up bundle identity mismatch")
        for name, digest in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid bundled source path")
            if hashlib.sha256(stream.read(name)).hexdigest() != digest:
                raise ValueError("Bundled source checksum mismatch")
            if name.startswith("src/") or name in ["scripts/analyze_diagnosis_followup.py", "scripts/run_diagnosis_followup.py"]:
                if file_fingerprint(ROOT/name) != digest:
                    raise ValueError("Audit must execute from the frozen source; extract the bundle into a separate directory")
    return manifest


def official_scorer(reference):
    names = {"normalize_answer", "exact_match_score", "f1_score"}
    tree = ast.parse(Path(reference).read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != names:
        raise ValueError("Incomplete official reference scorer")
    namespace = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), namespace)
    return namespace


def audit_batch(run, protocol_path, bundle, reference):
    run, protocol_path = Path(run), Path(protocol_path)
    frozen = load_json(protocol_path)
    protocol = frozen["payload"]
    if frozen["sha256"] != fingerprint(protocol):
        raise ValueError("Follow-up protocol checksum mismatch")
    source = verify_bundle(bundle, protocol["code_sha256"])
    if file_fingerprint(reference) != protocol["official_scorer_sha256"]:
        raise ValueError("Official scorer differs from frozen reference")
    scorer = official_scorer(reference)
    out = run/"outputs/repairs/diagnosis-followup"
    manifest = load_json(out/"manifest.json")
    payload = manifest["payload"]
    if manifest["sha256"] != fingerprint(payload) or payload["study_sha256"] != frozen["sha256"]:
        raise ValueError("Follow-up manifest checksum/identity mismatch")
    cfg = payload["configuration"]
    remote_root = Path(cfg["paths"]["local_base"])

    def local_path(remote):
        path = Path(remote)
        if not path.is_absolute():
            path = remote_root/path
        if path.is_relative_to(remote_root):
            return run/path.relative_to(remote_root)
        if path.name == protocol_path.name:
            return protocol_path
        raise ValueError("Unexpected external input in follow-up manifest")

    for name, digest in payload["input_sha256"].items():
        if file_fingerprint(local_path(name)) != digest:
            raise ValueError("Follow-up input hash mismatch")
    for name, digest in payload["code_sha256"].items():
        if source["files"].get(name) != digest:
            raise ValueError("Execution source differs from frozen bundle")
    batches = [b for b in protocol["batches"] if b["run_id"] == cfg["repair"]["run_id"]]
    if len(batches) != 1:
        raise ValueError("Unknown follow-up batch")
    if fingerprint(cfg) != batches[0]["configuration_sha256"]:
        raise ValueError("Execution configuration differs from frozen follow-up batch")
    ids = batches[0]["ids"]
    processed = local_path(cfg["paths"]["data_processed"])
    pool_rows = load_json(processed/"pool.json")
    pool = {r["_id"]: r for r in pool_rows}
    if len(pool) != len(pool_rows) or set(pool) != set(ids) or set(ids) & set(protocol["explored_ids"]):
        raise ValueError("Follow-up cohort/exclusion mismatch")
    originals = {p.stem: load_json(p) for p in local_path(cfg["paths"]["trajectories"]).glob("*.json")}
    if set(originals) != set(ids):
        raise ValueError("Incomplete original follow-up cohort")
    failed = load_json(processed/"failed_ids.json")
    if len(failed) != len(set(failed)) or set(failed) != {q for q,t in originals.items() if not t["success"]}:
        raise ValueError("Incorrect initial-failure cohort")
    diagnosis_dir = out/"diagnoses"
    if {p.stem for p in diagnosis_dir.glob("*.json")} != set(failed):
        raise ValueError("Incomplete diagnosis coverage")
    signature = {"name": cfg["models"]["agent"]["name"], "dtype": cfg["models"]["agent"]["dtype"],
                 "revision": protocol["model_revision"]}
    diagnoses = {q: diagnose_original(None, originals[q], diagnosis_dir/f"{q}.json", model_signature=signature)
                 for q in failed}
    planned = {}
    for q in failed:
        diagnosis = diagnoses[q]
        decoded = parse_diagnosis(diagnosis["response_text"], len(originals[q]["steps"]))
        if any(diagnosis[k] != v for k,v in decoded.items()):
            raise ValueError("Saved diagnosis differs from its recorded response")
        if (len(diagnosis["generated_token_ids"]) != diagnosis["selection_gen_tokens"]
                or not 0 <= diagnosis["selection_gen_tokens"] <= protocol["diagnosis_settings"]["max_tokens"]
                or diagnosis["selection_prompt_tokens"] < 0 or diagnosis["selection_model_requests"] != 1):
            raise ValueError("Invalid diagnosis token/request accounting")
        uncertainty = load_json(local_path(cfg["paths"]["uncertainty"])/f"{q}.json")
        for m in protocol["multipliers"]:
            for s in protocol["seeds"]:
                for job in plan_diagnosis_repairs(cfg, pool[q], originals[q], uncertainty, diagnosis, s, m):
                    for name in job["strategies"]:
                        planned[q, name, s, m] = job
    rows = list(read_jsonl(out/"results.jsonl"))
    keys = [(r["qid"],r["strategy"],r["seed"],r["multiplier"]) for r in rows]
    expected = {(q,a,s,m) for q in failed for a in protocol["strategies"]
                for s in protocol["seeds"] for m in protocol["multipliers"]}
    if len(keys) != len(set(keys)) or set(keys) != expected or set(planned) != expected:
        raise ValueError("Incomplete/duplicate follow-up trial coverage")
    executions = {}
    for path in (out/"executions").glob("*.json"):
        wrapper = load_json(path)
        if (wrapper["manifest_sha256"] != manifest["sha256"]
                or wrapper["trajectory_sha256"] != fingerprint(wrapper["trajectory"])):
            raise ValueError("Invalid raw follow-up execution checksum")
        executions[path.stem] = wrapper["trajectory"]
    if set(executions) != {job["meta"]["execution_id"] for job in planned.values()}:
        raise ValueError("Missing or unexpected physical executions")
    observations = prefixes = 0
    for trace in [*originals.values(), *executions.values()]:
        q = trace["qid"]
        env = HotpotEnv(pool[q])
        n_prefix = trace["meta"]["n_prefix_steps"]
        if (trace["question"] != pool[q]["question"] or type(n_prefix) is not int
                or not 0 <= n_prefix <= len(trace["steps"])):
            raise ValueError("Invalid trajectory question/prefix identity")
        finished = False
        for i, step in enumerate(trace["steps"]):
            if step["index"] != i:
                raise ValueError("Noncontiguous replay indices")
            if i < n_prefix:
                fields = ["index","thought","action","action_input","observation","is_tool_call","retrieved_title","n_gen_tokens"]
                if any(step[k] != originals[q]["steps"][i][k] for k in fields):
                    raise ValueError("Retained prefix changed")
                prefixes += 1
            if step["action"] in ["search", "lookup", "finish"]:
                replay = env.step(step["action"], step["action_input"])
                if any(step[k] != getattr(replay,k) for k in ["observation","is_tool_call","retrieved_title"]):
                    raise ValueError("Tool-observation replay mismatch")
                if replay.finished and (i != len(trace["steps"])-1 or trace["final_answer"] != replay.answer):
                    raise ValueError("Incorrect recorded final answer")
                finished = replay.finished
            elif (i != len(trace["steps"])-1 or step["observation"] != "Invalid action. Use search, lookup, or finish."
                  or trace["meta"].get("invalid_action_step") != i):
                raise ValueError("Incorrect invalid-action termination")
            observations += 1
        if ((not finished and trace["final_answer"] is not None)
                or (trace["terminated_reason"] == "finished") != finished):
            raise ValueError("Recorded final answer/termination disagrees with replay")
        answer, gold = trace["final_answer"], pool[q]["answer"]
        em = float(scorer["exact_match_score"](answer,gold)) if answer is not None else 0.0
        f1 = scorer["f1_score"](answer,gold)[0] if answer is not None else 0.0
        if (trace["gold_answer"] != gold or trace["em"] != em
                or not math.isclose(trace["f1"],f1,rel_tol=0,abs_tol=1e-12)
                or bool(trace["success"]) != bool(em or f1 >= 0.5)):
            raise ValueError("Official answer-scoring mismatch")
        if (trace["total_gen_tokens"] != sum(s["n_gen_tokens"] for s in trace["steps"])
                or trace["meta"]["recovery_gen_tokens"] != sum(s["n_gen_tokens"] for s in trace["steps"][n_prefix:])):
            raise ValueError("Raw token-accounting mismatch")
        # The agent counts retrieval calls (search/lookup), while the environment
        # also marks finish as a tool action. Finish still costs a model request.
        if (trace["meta"]["recovery_model_requests"] != len(trace["steps"])-n_prefix
                or trace["meta"]["recovery_tool_calls"] != sum(s["action"] in ("search", "lookup") for s in trace["steps"][n_prefix:])
                or trace["num_tool_calls"] != sum(s["action"] in ("search", "lookup") for s in trace["steps"])):
            raise ValueError("Raw request/tool accounting disagrees with replayed steps")
    for row, key in zip(rows, keys):
        job = planned[key]
        trace = executions[job["meta"]["execution_id"]]
        if any(trace["meta"].get(k) != v for k,v in job["meta"].items()):
            raise ValueError("Raw execution metadata differs from its frozen origin/prompt")
        if (row["manifest_sha256"] != manifest["sha256"] or row["execution_id"] != job["meta"]["execution_id"]
                or row["prompt_sha256"] != job["meta"]["prompt_sha256"]
                or row["target_step"] != job["meta"]["target_step"]
                or trace["qid"] != row["qid"] or trace["meta"]["n_prefix_steps"] != row["target_step"]
                or row["budget"] != job["token_budget"] or row["recovery_gen_tokens"] > row["budget"]
                or row["step_budget_mode"] != "new" or row["new_step_allowance"] != 8
                or len(trace["steps"])-row["target_step"] > 8):
            raise ValueError("Prespecified origin, prompt or allowance mismatch")
        for k in ["success","em","f1","final_answer","terminated_reason"]:
            if row[k] != trace[k]:
                raise ValueError("Trial outcome differs from raw execution")
        for suffix in COSTS:
            if (row["recovery_"+suffix] != trace["meta"]["recovery_"+suffix]
                    or row["selection_"+suffix] != job["policy_costs"][row["strategy"]]["selection_"+suffix]
                    or row["incremental_"+suffix] != row["recovery_"+suffix]+row["selection_"+suffix]):
                raise ValueError("Policy cost accounting mismatch")
        if row["recovery_tool_calls"] != trace["meta"]["recovery_tool_calls"]:
            raise ValueError("Policy request/tool accounting mismatch")
        if row["strategy"] == DIAGNOSIS_STRATEGY:
            diagnosis = diagnoses[row["qid"]]
            for stored, selected in [("diagnosis_input_sha256", "input_sha256"),
                                     ("diagnosis_fallback", "fallback"),
                                     ("diagnosis_fallback_reason", "fallback_reason")]:
                if row[stored] != diagnosis[selected]:
                    raise ValueError("Trial diagnosis metadata differs from the cached selection")
    audit = {"run_id": cfg["repair"]["run_id"], "initial_questions": len(ids), "failed_questions": len(failed),
             "strategy_seed_allowance_rows": len(rows), "unique_repairs": len(executions),
             "unique_diagnoses": len(diagnoses), "diagnosis_fallbacks": sum(d["fallback"] for d in diagnoses.values()),
             "diagnosis_generated_tokens": sum(d["selection_gen_tokens"] for d in diagnoses.values()),
             "diagnosis_prompt_tokens": sum(d["selection_prompt_tokens"] for d in diagnoses.values()),
             "trajectories_replayed_and_scored": len(originals)+len(executions),
             "observations_checked": observations, "prefix_steps_checked": prefixes,
             "source_files_hash_verified": len(source["files"]), "input_hashes_verified": len(payload["input_sha256"]),
             "official_scorer_sha256": file_fingerprint(reference), "mismatches": 0}
    return pd.DataFrame(rows), originals, audit


def validate_summary_trials(protocol, frame, originals):
    """Require every frozen question/seed/allowance before pooling or deduplicating."""
    keys = ["qid", "strategy", "seed", "multiplier"]
    costs = [prefix+suffix for prefix in ["recovery_", "selection_", "incremental_"] for suffix in COSTS]
    shared = ["qid", "seed", "multiplier", "target_step", "prompt_sha256", "success", "em", "f1",
              "final_answer", "terminated_reason", "budget", "recovery_tool_calls", *["recovery_"+s for s in COSTS]]
    diagnosis_fields = ["diagnosis_input_sha256", *["selection_"+s for s in COSTS]]
    required = {*keys, *costs, *shared, *diagnosis_fields, "execution_id", "run_id"}
    if not required <= set(frame):
        raise ValueError("Missing follow-up trial coverage or cost accounting fields")
    failed = {q for q,t in originals.items() if not t["success"]}
    expected = {(q,a,s,m) for q in failed for a in protocol["strategies"]
                for s in protocol["seeds"] for m in protocol["multipliers"]}
    actual = list(frame[keys].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete/duplicate follow-up trial coverage")
    batches = {q: b["run_id"] for b in protocol["batches"] for q in b["ids"]}
    if not frame.run_id.eq(frame.qid.map(batches)).all():
        raise ValueError("Follow-up trial coverage has an incorrect batch identity")
    numeric = frame[[*costs, "recovery_tool_calls"]].to_numpy(dtype=float)
    if any(not math.isfinite(v) or v < 0 or v != int(v) for v in numeric.flat):
        raise ValueError("Follow-up cost accounting requires finite nonnegative counts")
    if frame.groupby("execution_id")[shared].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Shared execution has inconsistent outcomes or cost accounting")
    for suffix in COSTS:
        if not frame["incremental_"+suffix].eq(frame["selection_"+suffix]+frame["recovery_"+suffix]).all():
            raise ValueError("Incremental cost accounting mismatch")
    diagnosis = frame[frame.strategy.eq(DIAGNOSIS_STRATEGY)]
    if (diagnosis.diagnosis_input_sha256.isna().any()
            or diagnosis.groupby("qid")[diagnosis_fields].nunique(dropna=False).gt(1).any().any()):
        raise ValueError("Repeated diagnosis has inconsistent identity or cost accounting")
    return diagnosis.drop_duplicates("qid")


def summarize(protocol, frame, originals):
    if set(originals) != set(protocol["question_ids"]):
        raise ValueError("Analyze the complete frozen follow-up cohort")
    declared = [{"treatment": TREATMENT, "control": c, "multiplier": m}
                for m in [0.5, 1.0, 2.0] for c in ["full_restart", DIAGNOSIS_STRATEGY]]
    if protocol.get("primary_comparisons", declared) != declared:
        raise ValueError("Follow-up primary family differs from the six frozen comparisons")
    if not any(not t["success"] for t in originals.values()):
        if len(frame):
            raise ValueError("Repair rows exist without any initial failures")
        return {"analysis_scope": protocol.get("cohort_role", "confirmatory_followup"),
                "n_initial_questions": len(originals), "n_failed_questions": 0,
                "primary_family_size": 6, "primary_contrasts": {}, "exploratory_em_f1": {},
                "policy_accounting": {}, "status": "No initial failures; repair comparisons are not estimable.",
                "initial_mean": {k: sum(t[k] for t in originals.values())/len(originals) for k in ["success","em","f1"]},
                "unique_study_usage": {"initial_generated_tokens": sum(t["total_gen_tokens"] for t in originals.values()),
                    "unique_repairs": 0, "unique_diagnoses": 0, "repair_tool_calls": 0,
                    **{prefix+suffix: 0 for prefix in ["repair_", "diagnosis_"] for suffix in COSTS}}}
    diagnoses = validate_summary_trials(protocol, frame, originals)
    primary, secondary, policies = {}, {}, {}
    for m in protocol["multipliers"]:
        subset = frame[frame.multiplier.eq(m)]
        # Batch run IDs differ within one frozen study; the question is the paired unit.
        comparison = subset.drop(columns="run_id")
        for control in ["full_restart", DIAGNOSIS_STRATEGY]:
            name = f"allowance_{m:g}_versus_{control}"
            primary[name] = paired_mean_comparison(comparison, TREATMENT, control, expected_seeds=protocol["seeds"], iters=10000)
            secondary[name] = {metric: paired_mean_comparison(comparison, TREATMENT, control,
                expected_seeds=protocol["seeds"], iters=10000, outcome=metric) for metric in ["em","f1"]}
        for name, group in subset.groupby("strategy"):
            costs = [prefix+suffix for prefix in ["recovery_","selection_","incremental_"] for suffix in COSTS]
            costs += ["recovery_tool_calls"]
            means = group.groupby("qid")[["success","em","f1", *costs]].mean()
            accepted = [t for t in originals.values() if t["success"]]
            policies[f"{name}@{m:g}"] = {"n_failed_questions": len(means), "seed_rows": len(group),
                "mean_per_attempt": {k: float(means[k].mean()) for k in means},
                "reference_gated_full_cohort": {k: (sum(t[k] for t in accepted)+means[k].sum())/len(originals)
                                                for k in ["success","em","f1"]},
                "termination_counts": group.terminated_reason.value_counts().to_dict()}
    for name, adjusted in zip(primary, holm_correction([p["p_value"] for p in primary.values()])):
        primary[name]["p_value_holm"] = adjusted
    unique = frame.drop_duplicates("execution_id")
    return {"analysis_scope": protocol.get("cohort_role", "confirmatory_followup"),
            "n_initial_questions": len(originals), "n_failed_questions": sum(not t["success"] for t in originals.values()),
            "primary_family_size": 6, "primary_contrasts": primary, "exploratory_em_f1": secondary,
            "policy_accounting": policies,
            "unique_study_usage": {"initial_generated_tokens": sum(t["total_gen_tokens"] for t in originals.values()),
                "unique_repairs": len(unique), "unique_diagnoses": len(diagnoses),
                "repair_tool_calls": int(unique.recovery_tool_calls.sum()),
                **{"repair_"+suffix: int(unique["recovery_"+suffix].sum()) for suffix in COSTS},
                **{"diagnosis_"+suffix: int(diagnoses["selection_"+suffix].sum()) for suffix in COSTS}},
            "cost_note": "Each diagnosis policy attempt pays for one complete diagnosis. Shared study executions are counted separately. Token counts are not latency or FLOPs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "bundle", "reference", "output"]:
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    frozen = load_json(args.protocol)
    frames, originals, audits = [], {}, []
    for run in args.runs:
        frame, traces, audit = audit_batch(run, args.protocol, args.bundle, args.reference)
        if set(originals) & set(traces):
            raise ValueError("Overlapping follow-up batches")
        frames.append(frame)
        originals.update(traces)
        audits.append(audit)
    frame = pd.concat(frames, ignore_index=True)
    result = {"study_sha256": frozen["sha256"], "audits": audits,
              **summarize(frozen["payload"], frame, originals)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".trials.csv"), index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"initial_questions": result["n_initial_questions"], "failed_questions": result["n_failed_questions"],
                      "primary_contrasts": len(result["primary_contrasts"]), "audited_batches": len(audits)}, indent=2))


if __name__ == "__main__":
    main()
