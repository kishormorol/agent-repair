"""Package the completed study's statistical reproduction without cloud or author credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "output/aws-experiment/2026-09-11-main"
REVIEW = ROOT / "output/reviewer-2026-09-12"

README = """# Statistical reproduction supplement

This local supplement reproduces statistics from the completed, six-condition
HotpotQA study. It contains 250 initial-question records, 2,034 repair rows,
the frozen study manifest, the analysis implementation and expected results.

Tested with Python 3.10, NumPy 1.23.5 and pandas 1.5.3. In an environment with
these dependencies, run `python reproduce.py`. It verifies the file hashes,
full cohort/seed coverage, four primary comparisons including Holm correction,
eight exploratory EM/F1 comparisons, policy token accounting, unique study
usage and reference-gated full-cohort rates. It writes `verification.json`.
No cloud account, GPU, model download, repository checkout or network call
is needed for this statistical reproduction once dependencies are installed.

The primary study was frozen before its outcomes. Review-stage diagnostics
were added afterwards and remain exploratory. A nonsignificant result is
not evidence of equivalence. The 40 initially accepted non-EM answers remain
unrepaired; the population rates are not an EM-only rerun or a deployed
failure detector. Historical exclusions are reconstructed, and the original
full historical pool file was not compared.

This is a statistical supplement. It does not reproduce model generations,
verify all tool observations, supply human labels, or package the complete
GPU runtime and raw archives. Those remain separate retained artifacts.
The CSVs preserve measured outcomes; the frozen manifest and reference JSON
identify the study. `manifest.json` gives checksums for every packaged input.
"""

REPRODUCE = '''"""Reproduce the packaged statistics locally; no cloud or network access."""
from pathlib import Path
import hashlib
import json
import math

import pandas as pd
from metrics import paired_mean_comparison, holm_correction

BASE = Path(__file__).resolve().parent

def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Reproduction mismatch: {a} != {b}")

def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    frozen = json.loads((BASE / "study-freeze.json").read_text())
    study = frozen["payload"]
    digest = hashlib.sha256(json.dumps(study, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != frozen["sha256"]:
        raise ValueError("Frozen study identity mismatch")
    reference = json.loads((BASE / "reference-analysis.json").read_text())
    diagnostics = json.loads((BASE / "reference-diagnostics.json").read_text())
    if reference["study_sha256"] != digest or diagnostics["study_sha256"] != digest:
        raise ValueError("Reference study identity mismatch")
    rows = pd.read_csv(BASE / "repair-trials.csv", dtype={"qid": str, "strategy": str})
    initial = pd.read_csv(BASE / "initial-questions.csv", dtype={"qid": str})
    initial["success"] = initial.success.map(lambda value: {"true": True, "false": False}[str(value).lower()])
    if initial.qid.duplicated().any() or set(initial.qid) != set(study["question_ids"]):
        raise ValueError("Initial cohort coverage mismatch")
    failed = set(initial.loc[~initial.success, "qid"])
    expected = {(q, a, r) for q in failed for a in study["strategies"] for r in study["seeds"]}
    keys = list(rows[["qid", "strategy", "seed"]].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Repair coverage mismatch")
    primary = {}
    for control in study["primary_comparisons"]:
        primary[control] = paired_mean_comparison(rows, study["strategy"], control,
            expected_seeds=study["seeds"], iters=10000)
        for metric in ["em", "f1"]:
            found = paired_mean_comparison(rows, study["strategy"], control,
                expected_seeds=study["seeds"], iters=10000, outcome=metric)
            wanted = reference["exploratory_sensitivities"][control][metric]
            for key in ["mean_a", "mean_b", "delta", "delta_lo", "delta_hi", "p_value"]:
                close(found[key], wanted[key])
    for control, adjusted in zip(primary, holm_correction([r["p_value"] for r in primary.values()])):
        primary[control]["p_value_holm"] = adjusted
        for key in ["mean_a", "mean_b", "delta", "delta_lo", "delta_hi", "p_value", "p_value_holm"]:
            close(primary[control][key], reference["primary_contrasts"][control][key])
    costs = ["recovery_gen_tokens", "recovery_prompt_tokens", "recovery_model_requests", "recovery_tool_calls"]
    for name, group in rows.groupby("strategy"):
        means = group.groupby("qid")[["success", "em", "f1", *costs]].mean()
        wanted = diagnostics["policy_accounting"][name]
        for metric in costs:
            close(means[metric].mean(), wanted["mean_cost_per_attempt"][metric])
        for metric in ["success", "em", "f1"]:
            value = (initial.loc[initial.success, metric].sum() + means[metric].sum()) / len(initial)
            close(value, wanted["reference_gated_full_cohort"][metric]["mean"])
    unique = rows.drop_duplicates("execution_id")
    close(len(unique), diagnostics["unique_study_usage"]["executions"])
    for metric in costs:
        close(unique[metric].sum(), diagnostics["unique_study_usage"][metric])
    result = {"status": "passed", "study_sha256": digest, "initial_questions": len(initial),
        "failed_questions": len(failed), "trial_rows": len(rows), "primary_contrasts_reproduced": 4,
        "secondary_contrasts_reproduced": 8, "policy_costs_and_population_rates_reproduced": 6,
        "tolerance": 1e-12, "scope": "statistical reproduction from packaged CSVs"}
    (BASE / "verification.json").write_text(json.dumps(result, indent=2) + "\\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
'''


def build(output_dir):
    output_dir = Path(output_dir)
    bundle = output_dir / "statistical-supplement"
    bundle.mkdir(parents=True, exist_ok=True)
    copies = {
        "repair-trials.csv": REVIEW / "diagnostics.trials.csv",
        "initial-questions.csv": REVIEW / "diagnostics.initial.csv",
        "study-freeze.json": STUDY / "study-freeze.json",
        "reference-analysis.json": STUDY / "pooled-analysis-local.json",
        "reference-diagnostics.json": REVIEW / "diagnostics.json",
        "metrics.py": ROOT / "src/eval/metrics.py",
    }
    for name, source in copies.items():
        shutil.copy2(source, bundle / name)
    (bundle / "README.md").write_text(README)
    (bundle / "reproduce.py").write_text(REPRODUCE)
    (bundle / "requirements.txt").write_text("numpy==1.23.5\npandas==1.5.3\n")
    names = sorted([*copies, "README.md", "reproduce.py", "requirements.txt"])
    forbidden = ["kishormorol", "/Users/", "692430448570", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY"]
    for name in names:
        if any(value in (bundle / name).read_text() for value in forbidden):
            raise ValueError(f"Direct author/account/private-key identifier in packaged input: {name}")
    manifest = {"scope": "statistical reproduction; full GPU replay is separate",
                "sha256": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in names}}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output_dir / "agent-repair-statistical-supplement.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in [*names, "manifest.json"]:
            stream.write(bundle / name, "statistical-supplement/" + name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REVIEW))
    args = parser.parse_args()
    print(build(args.output_dir))


if __name__ == "__main__":
    main()
