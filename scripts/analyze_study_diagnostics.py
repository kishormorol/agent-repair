"""Add review-stage diagnostics from verified archives without changing the primary analysis."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_study_batches import load_study_batches
from src.analysis.review_diagnostics import summarize_diagnostics
from src.repair.controlled import file_fingerprint
from src.utils.io import load_json


def verify_archive_inputs(archive, expected_sha256, members):
    """Compare selected local inputs with a checksummed archive; never extract it."""
    if file_fingerprint(archive) != expected_sha256:
        raise ValueError("Archive checksum mismatch")
    checked = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if member.name not in members:
                continue
            if member.name in checked or not member.isfile():
                raise ValueError("Duplicate or non-file diagnostic archive member")
            digest = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            local = members[member.name]
            if file_fingerprint(local) != digest:
                raise ValueError(f"Local diagnostic input differs from archive: {member.name}")
            checked[member.name] = digest
    if set(checked) != set(members):
        raise ValueError("Archive omits a required diagnostic input")
    return checked


def analyze(study_root, output, iters=10000):
    study_root, output = Path(study_root).resolve(), Path(output).resolve()
    freeze_path = study_root / "study-freeze.json"
    frozen = load_json(freeze_path)
    archives = load_json(study_root / "archive-verification.json")
    expected_batches = [f"batch{i:02}" for i in range(1, len(frozen["payload"]["batches"]) + 1)]
    if sorted(a["batch"] for a in archives) != expected_batches:
        raise ValueError("Missing, duplicate or unexpected archive batch")
    by_batch = {a["batch"]: a for a in archives}
    paths = [study_root / "results" / b / "run/outputs/repairs/results.jsonl" for b in expected_batches]
    study, frame = load_study_batches(paths, freeze_path)
    originals, uncertainty, sources, counts, context_sizes = {}, {}, {}, {}, []
    for batch, path in zip(expected_batches, paths):
        run = path.parents[2]
        pool = load_json(run / "data/processed/pool.json")
        context_sizes.extend(len(record["context"]) for record in pool)
        failures = load_json(run / "data/processed/failed_ids.json")
        relative = ["outputs/repairs/results.jsonl", "outputs/repairs/manifest.json",
                    "data/processed/pool.json", "data/processed/failed_ids.json",
                    "study_policy.json", "test_ids.json"]
        relative += [f"outputs/trajectories/{q['_id']}.json" for q in pool]
        relative += [f"outputs/uncertainty/{q}.json" for q in failures]
        if any(not (run / rel).resolve().is_relative_to(run.resolve()) for rel in relative):
            raise ValueError("Diagnostic input escapes its run directory")
        members = {"run/" + rel: run / rel for rel in relative}
        members["session/study-freeze.json"] = freeze_path
        record = by_batch[batch]
        archive = ROOT / record["archive"]
        checked = verify_archive_inputs(archive, record["sha256"], members)
        sources[str(archive.relative_to(ROOT))] = record["sha256"]
        sources.update({str(members[name].relative_to(ROOT)): digest for name, digest in checked.items()})
        counts[batch] = len(checked)
        originals.update({q["_id"]: load_json(run / f"outputs/trajectories/{q['_id']}.json") for q in pool})
        uncertainty.update({q: load_json(run / f"outputs/uncertainty/{q}.json") for q in failures})
    result = summarize_diagnostics(study, frame, originals, uncertainty, iters=iters)
    result["cohort"]["context_paragraph_counts"] = dict(sorted(Counter(context_sizes).items()))
    result.update(study_sha256=frozen["sha256"], frozen_code_sha256=study["code_sha256"],
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  bootstrap_resamples=iters, analysis_seed=0,
                  archived_input_checks_by_batch=counts, source_sha256=sources)
    code_paths = [Path(__file__), ROOT / "src/analysis/review_diagnostics.py",
                  ROOT / "scripts/analyze_study_batches.py", ROOT / "scripts/run_paired_analysis.py",
                  ROOT / "src/eval/metrics.py", ROOT / "src/env/hotpot_env.py",
                  ROOT / "src/localize/rules.py", ROOT / "src/repair/controlled.py"]
    result["analysis_code_sha256"] = {str(p.relative_to(ROOT)): file_fingerprint(p) for p in code_paths}
    plan = ROOT / "docs/reviewer_diagnostics_plan_2026-09-12.md"
    result["review_analysis_plan"] = {"path": str(plan.relative_to(ROOT)), "sha256": file_fingerprint(plan)}
    primary = load_json(study_root / "pooled-analysis-local.json")
    for name, contrast in result["execution_overlap"].items():
        for key in ["mean_a", "mean_b", "delta"]:
            if abs(contrast["paired_success"][key] - primary["primary_contrasts"][name][key]) > 1e-12:
                raise ValueError("Diagnostic analysis differs from the frozen primary estimates")
    output.parent.mkdir(parents=True, exist_ok=True)
    trial_export = output.with_suffix(".trials.csv")
    columns = ["qid", "strategy", "seed", "success", "em", "f1", "target_step",
               "execution_id", "prompt_sha256", "terminated_reason", "final_answer", "budget",
               "n_prefix_steps", "recovery_gen_tokens", "recovery_prompt_tokens",
               "recovery_model_requests", "recovery_tool_calls"]
    frame[columns].sort_values(["qid", "strategy", "seed"]).to_csv(trial_export, index=False)
    original_export = output.with_suffix(".initial.csv")
    pd.DataFrame([{"qid": q, "success": t["success"], "em": t["em"], "f1": t["f1"],
                   "n_steps": len(t["steps"]), "generated_tokens": t["total_gen_tokens"],
                   "terminated_reason": t["terminated_reason"], "final_answer": t["final_answer"]}
                  for q, t in sorted(originals.items())]).to_csv(original_export, index=False)
    result["export_sha256"] = {p.name: file_fingerprint(p) for p in [trial_export, original_export]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(args.study_root, args.output, args.iters)
    print(f"Verified {sum(result['archived_input_checks_by_batch'].values())} archive inputs; "
          f"diagnosed {result['n_failed_questions']} failed questions.")


if __name__ == "__main__":
    main()
