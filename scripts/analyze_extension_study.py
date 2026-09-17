"""Independently replay, rescore, and analyze the complete frozen extension."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.prepare_extension_study import official_functions, official_score
from scripts.run_extension_study import (verify_package, load_envelope, uncertainty_record,
                                         repair_jobs, trial_rows)
from src.agent.react_agent import parse_action
from src.env.extension import ExtensionEnv
from src.eval.metrics import paired_mean_comparison, holm_correction
from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT, DIAGNOSIS_STRATEGY, diagnose_original, parse_diagnosis
from src.repair.position_matched import fit_position_profile

UNCERTAINTY_ABS_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_trace(trace, record, protocol, official, *, original=None, job=None):
    require(trace["qid"] == record["_id"] and trace["question"] == record["question"]
            and trace["gold_answer"] == record["answer"], "Trajectory question/reference mismatch")
    env = ExtensionEnv(record)
    steps = trace["steps"]
    prefix_n = job["origin"] if job else 0
    require(trace["meta"]["n_prefix_steps"] == prefix_n, "Prefix length mismatch")
    require(len(steps)-prefix_n <= protocol["max_new_steps"], "New-step allowance overrun")
    finished = False
    for index, step in enumerate(steps):
        require(step["index"] == index, "Noncontiguous trajectory indices")
        if index < prefix_n:
            expected = {**original["steps"][index], "generation": None}
            require(step == expected, "Retained prefix differs from the original")
        else:
            generation = step["generation"]
            require(generation is not None and len(generation["tokens"]) == len(generation["token_ids"]) == step["n_gen_tokens"],
                    "Generated-token accounting mismatch")
            require(type(generation["prompt_tokens"]) is int and generation["prompt_tokens"] >= 0,
                    "Missing prompt-token accounting")
            require(step["n_gen_tokens"] <= protocol["max_tokens_per_step"], "Per-request token cap exceeded")
            for token_id, token in zip(generation["token_ids"], generation["tokens"]):
                require(token_id == token["token_id"] and math.isfinite(token["logprob"]) and token["logprob"] <= 0,
                        "Invalid sampled-token log probabilities")
            thought, action, argument = parse_action(generation["text"])
            require((thought, action or "invalid", argument) == (step["thought"], step["action"], step["action_input"]),
                    "Recorded action differs from model output")
        if step["action"] in {"search", "lookup", "finish"}:
            result = env.step(step["action"], step["action_input"])
            require(all(step[k] == getattr(result, k) for k in ["observation", "is_tool_call", "retrieved_title"]),
                    "Tool-observation replay mismatch")
            if result.finished:
                require(index == len(steps)-1 and trace["final_answer"] == result.answer, "Final answer/finish mismatch")
                finished = True
        else:
            require(index == len(steps)-1 and step["observation"] == "Invalid action. Use search, lookup, or finish."
                    and not step["is_tool_call"] and trace["meta"].get("invalid_action_step") == index,
                    "Invalid-action termination mismatch")
    require(finished == (trace["terminated_reason"] == "finished") and (finished or trace["final_answer"] is None),
            "Trajectory termination mismatch")
    score = official_score(trace["final_answer"], record, official)
    require(trace["em"] == score["em"] and math.isclose(trace["f1"], score["f1"], abs_tol=1e-12)
            and trace["success"] == score["correct"], "Official answer-scoring mismatch")
    new_steps = steps[prefix_n:]
    totals = {"recovery_gen_tokens": sum(s["n_gen_tokens"] for s in new_steps),
              "recovery_model_requests": len(new_steps),
              "recovery_tool_calls": sum(s["action"] in {"search", "lookup"} for s in new_steps),
              "recovery_prompt_tokens": sum(s["generation"]["prompt_tokens"] for s in new_steps)}
    require(all(trace["meta"][k] == v for k, v in totals.items()), "Recovery accounting mismatch")
    require(trace["total_gen_tokens"] == sum(s["n_gen_tokens"] for s in steps)
            and trace["num_tool_calls"] == sum(s["action"] in {"search", "lookup"} for s in steps), "Total token/tool accounting mismatch")
    for key in ["recovery_latency_s", "recovery_wall_latency_s"]:
        require(math.isfinite(trace["meta"][key]) and trace["meta"][key] >= 0, "Invalid measured latency")
    require(math.isclose(trace["meta"]["recovery_latency_s"], sum(s["latency_s"] for s in new_steps), abs_tol=1e-9),
            "Request latency accounting mismatch")
    if job:
        require(job["budget"] is None or totals["recovery_gen_tokens"] <= job["budget"], "Recovery generated-token allowance exceeded")
        require(trace["meta"]["execution_id"] == job["execution_id"] and trace["meta"]["prompt_sha256"] == job["prompt_sha256"],
                "Execution prompt/identity mismatch")
    return {"observations": len(steps), "prefix_steps": prefix_n, "new_steps": len(new_steps)}


def stable_jobs(jobs):
    result = []
    for job in jobs:
        value = {**job, "prefix": [s.to_dict() if hasattr(s, "to_dict") else s for s in job["prefix"]],
                 "policies": [{k: v for k, v in policy.items() if k != "selection_latency_s"} for policy in job["policies"]]}
        result.append(value)
    return result


def compare_uncertainty(stored, recomputed):
    """Allow numerical roundoff while requiring the same steps and signals."""
    message = "Stored uncertainty differs from sampled log probabilities"
    require(isinstance(stored, list) and len(stored) == len(recomputed), message)
    differing, maximum = 0, 0.
    for actual, expected in zip(stored, recomputed):
        require(isinstance(actual, dict) and actual.keys() == expected.keys()
                and type(actual["index"]) is int and actual["index"] == expected["index"], message)
        values = actual["uncertainty"]
        require(isinstance(values, dict) and values.keys() == expected["uncertainty"].keys(), message)
        for key, value in values.items():
            calculated = expected["uncertainty"][key]
            require(type(value) in (int, float) and math.isfinite(value) and math.isfinite(calculated)
                    and math.isclose(value, calculated, rel_tol=0, abs_tol=UNCERTAINTY_ABS_TOLERANCE), message)
            error = abs(value - calculated)
            differing += error != 0
            maximum = max(maximum, error)
    return differing, maximum


def audit_cell(package, output, model_key, dataset):
    package, output = Path(package), Path(output)
    protocol, digest = verify_package(package)
    model, cohort = protocol["models"][model_key], protocol["cohorts"][dataset]
    saved_model = json.loads((output / model_key / "model.json").read_text())
    signature = saved_model["payload"]
    require(signature["revision"] == model["revision"] and signature["dtype"] == model["dtype"], "Model pin mismatch")
    require(Path(signature["name"]).name == model["revision"], "Model snapshot path is not pinned")
    identity = {"protocol_sha256": digest, "model_key": model_key, "model": signature}
    load_envelope(output / model_key / "model.json", identity)
    cell_identity = {**identity, "dataset": dataset}
    cell = output / model_key / dataset
    complete = load_envelope(cell / "complete.json", cell_identity)
    record_list = json.loads((package / "data" / f"{dataset}.json").read_text())
    records = {r["_id"]: r for r in record_list}
    ids = cohort["main_ids"] + cohort["development_ids"]
    require(len(ids) == len(set(ids)) == len(records) == len(record_list) and set(ids) == set(records)
            and not set(ids) & set(cohort["excluded_ids"]), "Frozen cohort mismatch")
    official = official_functions(package / "references" / f"{dataset}.py", dataset)
    originals, uncertainties, raw_repairs, diagnoses, rows = {}, {}, {}, {}, []
    counts = {"trajectories": 0, "observations": 0, "prefix_steps": 0, "new_steps": 0,
              "uncertainty_values_with_roundoff": 0, "uncertainty_max_abs_error": 0.}
    for role in ["development", "main"]:
        require({p.stem for p in (cell / role / "originals").glob("*.json")} == set(cohort[role+"_ids"]), "Incomplete original cohort")
        require({p.stem for p in (cell / role / "uncertainty").glob("*.json")} == set(cohort[role+"_ids"]), "Incomplete uncertainty cohort")
        for qid in cohort[role+"_ids"]:
            trace_identity = {**cell_identity, "qid": qid, "role": role, "record_sha256": fingerprint(records[qid])}
            original = load_envelope(cell / role / "originals" / f"{qid}.json", trace_identity)
            originals[qid] = original
            detail = audit_trace(original, records[qid], protocol, official)
            counts["trajectories"] += 1
            for key, value in detail.items(): counts[key] += value
            uncertainty = load_envelope(cell / role / "uncertainty" / f"{qid}.json",
                                         {**trace_identity, "original_sha256": fingerprint(original)})
            recomputed = uncertainty_record(original)
            differing, maximum = compare_uncertainty(uncertainty["steps"], recomputed["steps"])
            counts["uncertainty_values_with_roundoff"] += differing
            counts["uncertainty_max_abs_error"] = max(counts["uncertainty_max_abs_error"], maximum)
            require(math.isfinite(uncertainty["acquisition_wall_latency_s"]) and uncertainty["acquisition_wall_latency_s"] >= 0,
                    "Invalid uncertainty acquisition latency")
            # Rebuild profiles and policy origins from the independently
            # recomputed values; accepting roundoff must not change a decision.
            uncertainties[qid] = {**uncertainty, "steps": recomputed["steps"]}
    failed_dev = [originals[q] for q in cohort["development_ids"] if not originals[q]["success"]]
    expected_profile = (fit_position_profile(failed_dev, uncertainties, source_ids=cohort["development_ids"], dataset=dataset,
                         strategy=TREATMENT, source_phase="development", source_run_id=protocol["run_id"], model=signature,
                         source_sha256=fingerprint(failed_dev)) if failed_dev else None)
    profile = load_envelope(cell / "position-profile.json", cell_identity)
    require(profile == expected_profile, "Position profile differs from frozen development-only fitting")
    failures = [q for q in cohort["main_ids"] if not originals[q]["success"]]
    runtime_ids = (failures[:protocol["runtime"]["n_failures_max"]]
                   if model_key == protocol["runtime"]["model"] and dataset == protocol["runtime"]["dataset"] else [])
    require({p.stem for p in (cell / "diagnoses").glob("*.json")} == set(failures), "Diagnosis coverage mismatch")
    for qid in failures:
        original = originals[qid]
        diagnosis = diagnose_original(None, original, cell / "diagnoses" / f"{qid}.json", model_signature=signature)
        require(all(diagnosis[k] == v for k, v in parse_diagnosis(diagnosis["response_text"], len(original["steps"])).items()),
                "Diagnosis parse mismatch")
        require(diagnosis["selection_gen_tokens"] == len(diagnosis["generated_token_ids"])
                and diagnosis["selection_gen_tokens"] <= protocol["diagnosis_settings"]["max_tokens"]
                and diagnosis["selection_model_requests"] == 1 and diagnosis["selection_prompt_tokens"] >= 0
                and math.isfinite(diagnosis["selection_latency_s"]) and diagnosis["selection_latency_s"] >= 0, "Diagnosis cost mismatch")
        diagnoses[qid] = diagnosis
    for runtime in [False, True]:
        mode = "runtime" if runtime else "token"
        selected = runtime_ids if runtime else failures
        require({p.stem for p in (cell / mode / "plans").glob("*.json")} == set(selected), "Repair plan coverage mismatch")
        expected_execution_ids = set()
        for qid in selected:
            original, diagnosis = originals[qid], diagnoses[qid]
            plan_identity = {**cell_identity, "qid": qid, "original_sha256": fingerprint(original),
                             "diagnosis_sha256": fingerprint(diagnosis), "runtime": runtime}
            jobs = load_envelope(cell / mode / "plans" / f"{qid}.json", plan_identity)
            planned = repair_jobs(protocol, records[qid], original, uncertainties[qid], diagnosis, profile, runtime=runtime)
            require(stable_jobs(jobs) == stable_jobs(planned), "Policy origins, prompt, cost, seed, or allowance differ from protocol")
            for job in jobs:
                expected_execution_ids.add(job["execution_id"])
                for policy in job["policies"]:
                    minimum = (diagnosis["selection_latency_s"] if policy["strategy"] == DIAGNOSIS_STRATEGY else
                               uncertainties[qid]["acquisition_wall_latency_s"] if policy["strategy"].startswith("unc__") else 0)
                    require(math.isfinite(policy["selection_latency_s"]) and policy["selection_latency_s"] >= minimum,
                            "Acquisition latency omitted from policy accounting")
                execution_identity = {**plan_identity, "job_sha256": fingerprint(job)}
                trace = load_envelope(cell / mode / "executions" / f"{job['execution_id']}.json", execution_identity)
                detail = audit_trace(trace, records[qid], protocol, official, original=original, job=job)
                counts["trajectories"] += 1
                for key, value in detail.items(): counts[key] += value
                raw_repairs[mode+":"+job["execution_id"]] = trace
                saved_rows = load_envelope(cell / mode / "trials" / f"{job['execution_id']}.json", execution_identity)
                require(saved_rows == trial_rows(trace, job, model_key=model_key, dataset=dataset, runtime=runtime),
                        "Trial outcome/accounting differs from audited raw execution")
                rows.extend(saved_rows)
        for folder in ["executions", "trials"]:
            require({p.stem for p in (cell / mode / folder).glob("*.json")} == expected_execution_ids,
                    "Missing or extra execution/trial artifacts")
    expected = {(q, s, seed, "token") for q in failures for s in protocol["strategies"] for seed in protocol["seeds"]}
    expected.update((q, s, seed, "runtime") for q in runtime_ids for s in protocol["runtime"]["strategies"] for seed in protocol["seeds"])
    keys = [(r["qid"], r["strategy"], r["seed"], r["mode"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete/duplicate condition-seed coverage")
    require(complete["main_questions"] == len(cohort["main_ids"]) and complete["main_failures"] == len(failures)
            and complete["development_questions"] == len(cohort["development_ids"])
            and complete["runtime_questions"] == len(runtime_ids), "Completion marker differs from audited records")
    initial = [originals[q] for q in cohort["main_ids"]]
    audit = {"model_key": model_key, "dataset": dataset, "main_questions": len(initial), "main_failures": len(failures),
             "development_questions": len(cohort["development_ids"]), "position_profile_fallback": profile is None,
             "initial_success": sum(t["success"] for t in initial)/len(initial),
             "initial_em": sum(t["em"] for t in initial)/len(initial), "initial_f1": sum(t["f1"] for t in initial)/len(initial),
             "unique_repairs": len(raw_repairs), "diagnoses": len(diagnoses), "diagnosis_fallbacks": sum(d["fallback"] for d in diagnoses.values()),
             "trial_rows": len(rows), "runtime_questions": len(runtime_ids), "mismatches": 0,
             "uncertainty_abs_tolerance": UNCERTAINTY_ABS_TOLERANCE, **counts,
             "physical_repair_generated_tokens": sum(t["meta"]["recovery_gen_tokens"] for t in raw_repairs.values()),
             "physical_diagnosis_generated_tokens": sum(d["selection_gen_tokens"] for d in diagnoses.values())}
    return audit, pd.DataFrame(rows)


def comparisons(frame, controls, protocol, outcome="success"):
    results = []
    for control in controls:
        result = paired_mean_comparison(frame, TREATMENT, control, iters=protocol["bootstrap_iters"],
                                        seed=protocol["analysis_seed"], expected_seeds=protocol["seeds"], outcome=outcome)
        results.append(result)
    return results


def analyze(package, output, destination, model_keys=None):
    package, output, destination = Path(package), Path(output), Path(destination)
    protocol, digest = verify_package(package)
    audits, frames, primary, runtime_results, summaries = [], [], [], [], []
    selected_models = model_keys or list(protocol["models"])
    for model_key in selected_models:
        for dataset in protocol["cohorts"]:
            audit, frame = audit_cell(package, output, model_key, dataset)
            audits.append(audit)
            frames.append(frame)
            if frame.empty:
                continue
            token = frame[frame["mode"] == "token"]
            for result in comparisons(token, protocol["primary_controls"], protocol):
                primary.append({"model_key": model_key, "dataset": dataset, **result})
            for (mode, strategy), sub in frame.groupby(["mode", "strategy"]):
                summaries.append({"model_key": model_key, "dataset": dataset, "mode": mode, "strategy": strategy,
                    "questions": sub.qid.nunique(), "trials": len(sub), "successful_trials": int(sub.success.sum()),
                    **{c: float(sub[c].mean()) for c in ["success", "em", "f1", "incremental_gen_tokens",
                       "incremental_prompt_tokens", "incremental_model_requests", "incremental_wall_latency_s"]}})
            runtime = frame[frame["mode"] == "runtime"].copy()
            if not runtime.empty:
                for seconds in protocol["runtime"]["deadlines_s"]:
                    runtime["deadline_success"] = runtime.success & (runtime.incremental_wall_latency_s <= seconds)
                    results = comparisons(runtime, ["full_restart", DIAGNOSIS_STRATEGY], protocol, outcome="deadline_success")
                    adjusted = holm_correction([r["p_value"] for r in results]) if seconds == protocol["runtime"]["primary_deadline_s"] else [None]*len(results)
                    for result, holm in zip(results, adjusted):
                        runtime_results.append({"model_key": model_key, "dataset": dataset, "deadline_s": seconds,
                                                "p_value_holm": float(holm) if holm is not None else None, **result})
                    for strategy, sub in runtime.groupby("strategy"):
                        summaries.append({"model_key": model_key, "dataset": dataset, "mode": "deadline", "strategy": strategy,
                            "deadline_s": seconds, "questions": sub.qid.nunique(), "trials": len(sub),
                            "successful_trials": int(sub.deadline_success.sum()), "success": float(sub.deadline_success.mean())})
    # Missing zero-failure cells contribute p=1 to the declared fixed family.
    family_n = len(protocol["models"])*len(protocol["cohorts"])*len(protocol["primary_controls"])
    family = [r["p_value"] for r in primary] + [1.] * (family_n-len(primary))
    for result, holm in zip(primary, holm_correction(family)):
        result["p_value_holm"] = float(holm)
    payload = {"protocol_sha256": digest, "complete": set(selected_models) == set(protocol["models"]),
               "audits": audits, "summaries": summaries, "primary_comparisons": primary, "runtime_comparisons": runtime_results,
               "primary_family_size": family_n, "runtime_note": protocol["runtime"]["measurement"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(destination.with_suffix(".trials.csv"), index=False)
    print(json.dumps({"complete": payload["complete"], "audited_cells": len(audits),
                      "initial_questions": sum(a["main_questions"] for a in audits),
                      "unique_repairs": sum(a["unique_repairs"] for a in audits), "mismatches": 0}), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model", action="append")
    args = parser.parse_args()
    analyze(args.package, args.output, args.destination, args.model)
