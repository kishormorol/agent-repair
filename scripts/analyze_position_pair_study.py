"""Independently replay, rescore, and analyze the frozen position-pair study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.analyze_extension_study import (audit_trace, compare_uncertainty, require,
                                             stable_jobs, UNCERTAINTY_ABS_TOLERANCE)
from scripts.prepare_extension_study import official_functions
from scripts.run_extension_study import load_envelope, uncertainty_record, verify_package
from scripts.run_position_pair_study import (build_schedule, load_profile, pair_jobs,
                                             plan_identity, trial_rows)
from src.eval.metrics import holm_correction
from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_pairs import (DEV_CONTROL, POLICIES, SWAP, block_comparison,
                                       pair_by_length, validate_exact_balance)


def audit_cell(package, output, dataset):
    package, output = Path(package), Path(output)
    protocol, digest = verify_package(package)
    model_key = protocol["model_key"]
    model, cohort = protocol["models"][model_key], protocol["cohorts"][dataset]
    signature = json.loads((output / model_key / "model.json").read_text())["payload"]
    require(signature["revision"] == model["revision"] and signature["dtype"] == model["dtype"], "Model pin mismatch")
    require(Path(signature["name"]).name == model["revision"], "Model snapshot path is not pinned")
    identity = {"protocol_sha256": digest, "model_key": model_key, "model": signature}
    load_envelope(output / model_key / "model.json", identity)
    cell_identity = {**identity, "dataset": dataset}
    cell = output / model_key / dataset
    complete = load_envelope(cell / "complete.json", cell_identity)

    record_list = json.loads((package / "data" / f"{dataset}.json").read_text())
    records = {r["_id"]: r for r in record_list}
    ids = cohort["main_ids"]
    require(len(ids) == len(set(ids)) == len(records) == len(record_list) and set(ids) == set(records)
            and not set(ids) & set(cohort["excluded_ids"]), "Frozen cohort mismatch")
    require(len(ids) == protocol["main_n_per_dataset"], "Cohort size differs from the frozen protocol")
    official = official_functions(package / "references" / f"{dataset}.py", dataset)
    profile = load_profile(package, dataset, cohort)

    originals, uncertainties, rows, raw_repairs = {}, {}, [], {}
    counts = {"trajectories": 0, "observations": 0, "prefix_steps": 0, "new_steps": 0,
              "uncertainty_values_with_roundoff": 0, "uncertainty_max_abs_error": 0.}
    for folder in ["originals", "uncertainty"]:
        require({p.stem for p in (cell / "main" / folder).glob("*.json")} == set(ids), f"Incomplete {folder} cohort")
    for qid in ids:
        trace_identity = {**cell_identity, "qid": qid, "role": "main", "record_sha256": fingerprint(records[qid])}
        original = load_envelope(cell / "main" / "originals" / f"{qid}.json", trace_identity)
        originals[qid] = original
        detail = audit_trace(original, records[qid], protocol, official)
        counts["trajectories"] += 1
        for key, value in detail.items(): counts[key] += value
        uncertainty = load_envelope(cell / "main" / "uncertainty" / f"{qid}.json",
                                    {**trace_identity, "original_sha256": fingerprint(original)})
        recomputed = uncertainty_record(original)
        differing, maximum = compare_uncertainty(uncertainty["steps"], recomputed["steps"])
        counts["uncertainty_values_with_roundoff"] += differing
        counts["uncertainty_max_abs_error"] = max(counts["uncertainty_max_abs_error"], maximum)
        require(math.isfinite(uncertainty["acquisition_wall_latency_s"])
                and uncertainty["acquisition_wall_latency_s"] >= 0, "Invalid uncertainty acquisition latency")
        uncertainties[qid] = {**uncertainty, "steps": recomputed["steps"]}

    failures = [q for q in ids if not originals[q]["success"]]
    expected_pairing = pair_by_length({q: len(originals[q]["steps"]) for q in failures}, seed=protocol["pairing_seed"])
    pairing = load_envelope(cell / "pairing.json", cell_identity)
    require(pairing == expected_pairing, "Pairing differs from length-only reproduction of the audited failures")
    matched = [q for pair in pairing["pairs"] for q in pair["qids"]]
    require(set(matched) | set(pairing["unmatched_ids"]) == set(failures)
            and len(matched) == len(set(matched)) == pairing["n_matched_questions"], "Pair membership mismatch")

    pair_ids = {pair["pair_id"] for pair in pairing["pairs"]}
    require({p.stem for p in (cell / "plans").glob("*.json")} == pair_ids, "Repair plan coverage mismatch")
    expected_execution_ids = set()
    for pair in pairing["pairs"]:
        identity_for_plan = plan_identity(cell_identity, pair, originals, profile)
        jobs = load_envelope(cell / "plans" / f"{pair['pair_id']}.json", identity_for_plan)
        planned = pair_jobs(protocol, pair, records, originals, uncertainties, profile)
        require(stable_jobs(jobs) == stable_jobs(planned),
                "Policy origins, prompt, seed, or allowance differ from the frozen protocol")
        for job in jobs:
            expected_execution_ids.add(job["execution_id"])
            for policy in job["policies"]:
                minimum = 0. if policy["strategy"] == DEV_CONTROL else min(
                    uncertainties[q]["acquisition_wall_latency_s"] for q in pair["qids"])
                require(math.isfinite(policy["selection_latency_s"]) and policy["selection_latency_s"] >= minimum,
                        "Acquisition latency omitted from policy accounting")
            execution_identity = {**identity_for_plan, "job_sha256": fingerprint(job)}
            trace = load_envelope(cell / "executions" / f"{job['execution_id']}.json", execution_identity)
            detail = audit_trace(trace, records[job["qid"]], protocol, official,
                                 original=originals[job["qid"]], job=job)
            counts["trajectories"] += 1
            for key, value in detail.items(): counts[key] += value
            raw_repairs[job["execution_id"]] = trace
            saved_rows = load_envelope(cell / "trials" / f"{job['execution_id']}.json", execution_identity)
            require(saved_rows == trial_rows(trace, job, model_key=model_key, dataset=dataset),
                    "Trial outcome/accounting differs from the audited raw execution")
            rows.extend(saved_rows)
    for folder in ["executions", "trials"]:
        require({p.stem for p in (cell / folder).glob("*.json")} == expected_execution_ids,
                "Missing or extra execution/trial artifacts")

    keys = [(r["qid"], r["strategy"], r["seed"]) for r in rows]
    expected = {(q, s, seed) for q in matched for s in POLICIES for seed in protocol["seeds"]}
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete/duplicate policy-seed coverage")
    require(complete["main_questions"] == len(ids) and complete["main_failures"] == len(failures)
            and complete["pairs"] == len(pairing["pairs"])
            and complete["matched_questions"] == pairing["n_matched_questions"]
            and complete["unmatched_ids"] == pairing["unmatched_ids"], "Completion marker differs from audited records")

    frame = pd.DataFrame(rows)
    balance = validate_exact_balance(frame, pairing, protocol["seeds"])
    initial = [originals[q] for q in ids]
    audit = {"model_key": model_key, "dataset": dataset, "main_questions": len(initial),
             "main_failures": len(failures), "matched_questions": pairing["n_matched_questions"],
             "pairs": len(pairing["pairs"]), "unmatched_failures": len(pairing["unmatched_ids"]),
             "initial_success": sum(t["success"] for t in initial) / len(initial),
             "initial_em": sum(t["em"] for t in initial) / len(initial),
             "initial_f1": sum(t["f1"] for t in initial) / len(initial),
             "unique_repairs": len(raw_repairs), "trial_rows": len(rows), "mismatches": 0,
             "uncertainty_abs_tolerance": UNCERTAINTY_ABS_TOLERANCE, **counts, "balance": balance,
             "profile_sha256": profile["sha256"],
             "physical_repair_generated_tokens": sum(t["meta"]["recovery_gen_tokens"] for t in raw_repairs.values())}
    return audit, frame, pairing


def analyze(package, output, destination, datasets=None):
    package, output, destination = Path(package), Path(output), Path(destination)
    protocol, digest = verify_package(package)
    selected = datasets or list(protocol["cohorts"])
    require(protocol["primary_control"] == SWAP and protocol["strategies"] == POLICIES,
            "Analysis does not match the frozen policy set or declared primary control")
    audits, frames, primary, secondary, summaries = [], [], [], [], []
    for dataset in selected:
        audit, frame, pairing = audit_cell(package, output, dataset)
        audits.append(audit)
        frames.append(frame)
        for control, bucket in [(SWAP, primary), (DEV_CONTROL, secondary)]:
            result = block_comparison(frame, pairing, protocol["seeds"], control=control,
                                      iters=protocol["bootstrap_iters"], seed=protocol["analysis_seed"])
            bucket.append({"model_key": protocol["model_key"], "dataset": dataset, **result})
        for strategy, sub in frame.groupby("strategy"):
            summaries.append({"model_key": protocol["model_key"], "dataset": dataset, "strategy": strategy,
                "questions": sub.qid.nunique(), "trials": len(sub), "successful_trials": int(sub.success.sum()),
                "mean_origin": float(sub.origin.mean()), "mean_original_n_steps": float(sub.original_n_steps.mean()),
                **{c: float(sub[c].mean()) for c in ["success", "em", "f1", "incremental_gen_tokens",
                   "incremental_prompt_tokens", "incremental_model_requests", "incremental_wall_latency_s"]}})
    # Zero-pair datasets contribute p=1 to the declared fixed two-test family.
    family_n = protocol["primary_family_size"]
    require(len(primary) <= family_n, "More primary tests than the frozen family declares")
    family = [r["p_value"] for r in primary] + [1.] * (family_n - len(primary))
    for result, holm in zip(primary, holm_correction(family)):
        result["p_value_holm"] = float(holm)
    for result in secondary:
        result["p_value_holm"] = None
        result["inference_note"] = protocol["secondary_control"]
    payload = {"protocol_sha256": digest, "complete": set(selected) == set(protocol["cohorts"]),
               "audits": audits, "summaries": summaries, "primary_comparisons": primary,
               "secondary_comparisons": secondary, "primary_family_size": family_n,
               "primary_family": protocol["primary_family"], "resampling_unit": protocol["resampling_unit"],
               "precision_note": protocol["precision_note"], "scope": protocol["scope"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(destination.with_suffix(".trials.csv"), index=False)
    print(json.dumps({"complete": payload["complete"], "audited_cells": len(audits),
                      "initial_questions": sum(a["main_questions"] for a in audits),
                      "pairs": sum(a["pairs"] for a in audits),
                      "unique_repairs": sum(a["unique_repairs"] for a in audits), "mismatches": 0}), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    analyze(args.package, args.output, args.destination, args.dataset)
