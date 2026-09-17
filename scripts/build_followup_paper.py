"""Build paper tables from the independently audited diagnosis follow-up."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.build_iclr_draft import write_table
from src.repair.controlled import fingerprint

BASE = ROOT / "output/aws-experiment/2026-09-14-diagnosis-v2"
OUT = ROOT / "paper/generated"
TREATMENT = "unc__perplexity__argmax__bt2"
LABELS = {"full_restart": "Restart", TREATMENT: "Uncertainty, backtrack 2", "diagnosis_replay": "Diagnosis/replay"}
INPUTS = ["pooled-analysis-local.json", "pooled-analysis-local.trials.csv",
          "analysis-reproduction-check.json", "prepared/main/protocol.json",
          "retrieved/main/pooled-analysis.json", "retrieved/main/pooled-analysis.trials.csv"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_values(actual, expected, path="analysis"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Changed fields: {path}")
        for key in expected:
            compare_values(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Changed coverage: {path}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare_values(a, b, f"{path}[{index}]")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
                f"Changed numeric evidence: {path}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Changed evidence: {path}")


def load_followup(base=BASE):
    base = Path(base)
    analysis = json.loads((base / "pooled-analysis-local.json").read_text())
    reproduced = json.loads((base / "analysis-reproduction-check.json").read_text())
    frozen = json.loads((base / "prepared/main/protocol.json").read_text())
    require(frozen["sha256"] == fingerprint(frozen["payload"]) == analysis["study_sha256"],
            "Follow-up protocol identity mismatch")
    require(reproduced["matched"] and reproduced["trial_rows_compared"] == 3186,
            "Follow-up reproduction is incomplete")
    require(analysis["n_initial_questions"] == 250 and analysis["n_failed_questions"] == 118,
            "Follow-up cohort coverage changed")
    require(len(analysis["audits"]) == 5 and all(a["mismatches"] == 0 for a in analysis["audits"]),
            "Follow-up audit is incomplete or contains a mismatch")
    expected_family = {f"allowance_{m:g}_versus_{c}" for m in [.5, 1., 2.]
                       for c in ["full_restart", "diagnosis_replay"]}
    require(analysis["primary_family_size"] == 6 and set(analysis["primary_contrasts"]) == expected_family,
            "Follow-up primary comparison family changed")
    compare_values(analysis, json.loads((base / "retrieved/main/pooled-analysis.json").read_text()))

    keys = ["qid", "strategy", "seed", "multiplier"]
    frames = [pd.read_csv(base / name, keep_default_na=False).sort_values(keys).reset_index(drop=True)
              for name in ["pooled-analysis-local.trials.csv", "retrieved/main/pooled-analysis.trials.csv"]]
    try:
        pd.testing.assert_frame_equal(*frames, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as error:
        raise ValueError("Local and remote follow-up trials differ") from error
    trials = frames[0]
    require(len(trials) == 3186 and not trials.duplicated(keys).any(), "Incomplete/duplicate follow-up trials")
    require(trials.execution_id.nunique() == analysis["unique_study_usage"]["unique_repairs"] == 2079,
            "Follow-up unique execution coverage changed")
    failed = set(trials.qid)
    require(len(failed) == 118 and failed <= set(frozen["payload"]["question_ids"]),
            "Follow-up failed-question coverage changed")
    expected = {(q, s, seed, m) for q in failed for s in LABELS for seed in [0, 1, 2] for m in [.5, 1., 2.]}
    require(set(trials[keys].itertuples(index=False, name=None)) == expected, "Follow-up policy/seed coverage changed")
    require(len(analysis["policy_accounting"]) == 9, "Follow-up policy/allowance coverage changed")
    for (strategy, multiplier), frame in trials.groupby(["strategy", "multiplier"]):
        policy = analysis["policy_accounting"][f"{strategy}@{multiplier:g}"]
        require(policy["seed_rows"] == len(frame), "Follow-up policy row count changed")
        for name, value in policy["mean_per_attempt"].items():
            require(math.isclose(float(frame[name].mean()), value, rel_tol=0, abs_tol=1e-12),
                    f"Follow-up policy mean differs from trials: {strategy}@{multiplier:g}/{name}")
    return analysis


def build_assets(out, analysis):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results, costs, contrasts = [], [], []
    for allowance in [.5, 1., 2.]:
        for strategy, label in LABELS.items():
            entry = analysis["policy_accounting"][f"{strategy}@{allowance:g}"]
            row = entry["mean_per_attempt"]
            n = entry["seed_rows"]
            results.append([f"${allowance:g}\\times$", label, f"{round(row['success']*n)}/{n}",
                            f"{100*row['success']:.2f}", f"{100*row['em']:.2f}", f"{row['f1']:.4f}"])
            costs.append([f"${allowance:g}\\times$", label, f"{row['incremental_gen_tokens']:.1f}",
                          f"{row['incremental_prompt_tokens']:.1f}", f"{row['incremental_model_requests']:.2f}"])
        for control in ["full_restart", "diagnosis_replay"]:
            row = analysis["primary_contrasts"][f"allowance_{allowance:g}_versus_{control}"]
            contrasts.append([f"${allowance:g}\\times$", LABELS[control], f"{100*row['delta']:+.2f}",
                              f"[{100*row['delta_lo']:+.2f}, {100*row['delta_hi']:+.2f}]", f"{row['p_value']:.3f}",
                              f"{row['p_value_holm']:.3f}"])
    write_table(out/"tables"/"iclr2027_diagnosis_results.tex", ["Allowance", "Policy", "Successes", "Success (\\%)", "EM (\\%)", "F1"], results)
    write_table(out/"tables"/"iclr2027_diagnosis_costs.tex", ["Allowance", "Policy", "Generated", "Prompt", "Requests"], costs)
    write_table(out/"tables"/"iclr2027_diagnosis_contrasts.tex", ["Allowance", "Control", "$\\Delta$ (pp)", "95\\% interval (pp)", "$p$", "$p_{\\rm Holm}$"], contrasts)


def main():
    analysis = load_followup()
    build_assets(OUT, analysis)
    provenance = {"study_sha256": analysis["study_sha256"], "initial_questions": 250,
                  "failed_questions": 118, "unique_repairs": 2079, "trial_rows": 3186,
                  "checks": "Protocol, five audits, local/remote JSON and all trial columns, complete policy/seed coverage and measured policy means",
                  "source_sha256": {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in INPUTS},
                  "asset_sha256": {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((OUT/"tables").glob("iclr2027_diagnosis_*.tex"))}}
    (OUT/"iclr2027_diagnosis_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Built audited diagnosis results, paired contrasts, and measured policy-cost tables.")


if __name__ == "__main__":
    main()
