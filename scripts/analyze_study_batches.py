"""Pool every prespecified batch of one study without treating batches as datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paired_analysis import load_verified_results
from src.eval.metrics import paired_mean_comparison, holm_correction
from src.repair.controlled import fingerprint
from src.utils.io import load_json, save_json


def load_study_batches(paths, study_path):
    frozen = load_json(study_path)
    study = frozen["payload"]
    if frozen["sha256"] != fingerprint(study):
        raise ValueError("Invalid frozen study checksum")
    batches = {b["run_id"]: b for b in study["batches"]}
    ids = [q for b in study["batches"] for q in b["ids"]]
    if (len(batches) != len(study["batches"]) or len(ids) != len(set(ids))
            or ids != study["question_ids"] or len(ids) != study["n_initial_questions"]
            or set(ids).intersection(study["explored_ids"])):
        raise ValueError("Frozen study batches must be complete, unique and disjoint")
    frames, seen = [], set()
    for name in paths:
        path = Path(name)
        run = path.parents[2]
        manifest = load_json(path.with_name("manifest.json"))
        cfg = manifest["payload"]["configuration"]
        run_id = cfg["repair"]["run_id"]
        if run_id not in batches or run_id in seen:
            raise ValueError("Unexpected or duplicated study batch")
        batch = batches[run_id]
        seen.add(run_id)
        provenance = cfg.get("notebook_provenance", {})
        policy = load_json(run / "study_policy.json")
        if (fingerprint(policy) != batch["policy_sha256"]
                or provenance.get("study_policy_sha256") != batch["policy_sha256"]
                or provenance.get("code_sha256") != study["code_sha256"]
                or provenance.get("phase") != "test"
                or cfg["dataset"].get("split") != "test"
                or cfg["dataset"]["name"] != study["dataset"]
                or cfg["repair"]["strategies"] != study["strategies"]
                or cfg["repair"]["seeds"] != study["seeds"]):
            raise ValueError("Batch configuration differs from the frozen study")
        pool = load_json(run / "data/processed/pool.json")
        if ([r["_id"] for r in pool] != batch["ids"]
                or load_json(run / "test_ids.json") != batch["ids"]):
            raise ValueError("Batch initial-question coverage differs from the study")
        originals = [load_json(run / "outputs/trajectories" / (q + ".json")) for q in batch["ids"]]
        failed = {t["qid"] for t in originals if not t["success"]}
        if (set(manifest["payload"]["question_ids"]) != failed
                or set(load_json(run / "data/processed/failed_ids.json")) != failed):
            raise ValueError("Batch failure gate differs from the completed initial cohort")
        other_conditions = [s for s in study["strategies"] if s != study["strategy"]]
        frame = load_verified_results([path], study["strategy"], other_conditions,
                                       study["seeds"], study["multiplier"])
        if (set(frame.strategy) != set(study["strategies"])
                or frame.duplicated(["qid", "strategy", "seed", "multiplier"]).any()
                or not frame.run_id.eq(run_id).all()):
            raise ValueError("Unexpected or duplicate study trials")
        frames.append(frame)
    if seen != set(batches):
        raise ValueError("All frozen study batches must complete before pooled analysis")
    pooled = pd.concat(frames, ignore_index=True)
    pooled["batch_run_id"] = pooled["run_id"]
    # The validated batch manifest defines one study. Preserve each original
    # run ID separately while supplying that study identity to the estimator.
    pooled["run_id"] = "study:" + frozen["sha256"]
    return study, pooled


def analyze(paths, study_path, output, iters=10000):
    study, frame = load_study_batches(paths, study_path)
    primary, secondary = {}, {}
    for baseline in study["primary_comparisons"]:
        primary[baseline] = paired_mean_comparison(frame, study["strategy"], baseline,
            iters=iters, expected_seeds=study["seeds"])
        secondary[baseline] = {metric: paired_mean_comparison(frame, study["strategy"], baseline,
            iters=iters, expected_seeds=study["seeds"], outcome=metric) for metric in ("em", "f1")}
    for comparison, adjusted in zip(primary.values(), holm_correction([r["p_value"] for r in primary.values()])):
        comparison["p_value_holm"] = adjusted
    result = {"study_sha256": load_json(study_path)["sha256"], "dataset": study["dataset"],
              "n_initial_questions": study["n_initial_questions"],
              "n_failed_questions": int(frame.qid.nunique()), "n_complete_batches": len(study["batches"]),
              "n_strategy_seed_rows": len(frame), "primary_contrasts": primary,
              "exploratory_sensitivities": secondary,
              "note": "All frozen batches pooled; question-level seed means, paired question bootstrap, "
                      "and Holm correction across the prespecified comparison family."}
    save_json(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.results, args.study, args.output)
    print(f"Verified {result['n_initial_questions']} initial questions across {result['n_complete_batches']} batches")


if __name__ == "__main__":
    main()
