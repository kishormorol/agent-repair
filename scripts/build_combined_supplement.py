"""Package every completed study for independent statistical reproduction.

Scope is deliberately narrow and stated in the bundle: this reproduces the
packaged statistics from packaged trial exports. Raw replay and reference
scoring is a separate audit, and regenerating trajectories on a GPU is a third
environment; neither is claimed here. Nothing in the bundle reads a private
workspace path or needs cloud credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "output/reviewer-2026-09-12"
AWS = ROOT / "output/aws-experiment"

# Each study contributes its trial export, its reference analysis, and the
# primary family size the paper declares for it.
STUDIES = {
    "main": {"trials": REVIEW / "diagnostics.trials.csv",
             "analysis": AWS / "2026-09-11-main/pooled-analysis-local.json",
             "contrasts": "primary_contrasts", "family": 4, "rows": 2034,
             "title": "Main HotpotQA study"},
    # Its six tests partition by token allowance, which lives in the contrast
    # key rather than the contrast body, so the filter is declared explicitly.
    "diagnosis": {"trials": AWS / "2026-09-14-diagnosis-v2/pooled-analysis-local.trials.csv",
                  "analysis": AWS / "2026-09-14-diagnosis-v2/pooled-analysis-local.json",
                  "contrasts": "primary_contrasts", "family": 6, "rows": 3186,
                  "title": "Diagnosis and allowance follow-up",
                  "key_filter": {"column": "multiplier", "pattern": r"^allowance_([0-9.]+)_versus_"}},
    "replication": {"trials": AWS / "2026-09-15-extension-completion/pooled-analysis-local.trials.csv",
                    "analysis": AWS / "2026-09-15-extension-completion/pooled-analysis-local.json",
                    "contrasts": "primary_comparisons", "family": 24, "rows": 8016,
                    "title": "Two-model, three-dataset replication"},
    "position_pairs": {"trials": AWS / "2026-09-16-position-pairs/analysis-local.trials.csv",
                       "analysis": AWS / "2026-09-16-position-pairs/analysis-local.json",
                       "contrasts": "primary_comparisons", "family": 2, "rows": 738,
                       "title": "Exactly balanced length-matched swap"},
}
# Identifiers that must never leave the working tree.
FORBIDDEN = ["kishormorol", "/Users/", "692430448570", "i-03b33e00c47b11be6",
             "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY", "agent-repair-aws119"]

README = """# Combined statistical reproduction supplement

Anonymous bundle covering every completed study in the paper.

## Scope

This bundle reproduces **packaged statistics from packaged trial exports**:
per-condition means, primary contrast point estimates, and the Holm adjustment
within each declared family. Three environments are deliberately separate:

| Environment | Covered here | What it needs |
| --- | --- | --- |
| Statistical reproduction | **yes** | Python, numpy, pandas |
| Raw replay and reference scoring | no | frozen raw trajectories and dataset files |
| GPU regeneration of trajectories | no | the pinned model weights and a GPU |

Reproducing the statistics does not re-derive the trial outcomes; it checks
that the reported statistics follow from the exported trials.

## Contents

One directory per study, each with `trials.csv` (the exported per-trial rows)
and `analysis.json` (the reference analysis). `metrics.py` holds the bootstrap
and Holm routines. `manifest.json` lists a SHA-256 for every file.

## Run

    pip install -r requirements.txt
    python reproduce.py

`reproduce.py` verifies every file against `manifest.json`, recomputes each
study's condition means and contrast deltas from its `trials.csv`, recomputes
the Holm adjustment over each declared family, and writes `verification.json`.
It exits non-zero if any packaged file was altered or any value disagrees.

## Families

| Study | Primary tests | Trial rows |
| --- | --- | --- |
{family_table}

No file here reads an absolute workspace path, and no step contacts a network
or cloud account.
"""

REPRODUCE = '''"""Reproduce the packaged statistics locally; no cloud or network access."""
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from metrics import holm_correction

BASE = Path(__file__).resolve().parent
TOLERANCE = 1e-9


def close(a, b, tolerance=TOLERANCE):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def check_manifest():
    manifest = json.loads((BASE / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"Checksum mismatch: {name}")
    return manifest


def contrasts_of(analysis, key):
    """Return (key, contrast) pairs; a dict key can carry the partition value."""
    container = analysis[key]
    if isinstance(container, dict):
        return list(container.items())
    return [(None, contrast) for contrast in container]


def study_means(trials, contrast, name=None, key_filter=None):
    """Recompute each arm's equal-weight mean from the packaged trial rows."""
    frame = trials
    for column in ["model_key", "dataset"]:
        if column in contrast and column in frame.columns:
            frame = frame[frame[column] == contrast[column]]
    if key_filter and name is not None:
        match = re.search(key_filter["pattern"], name)
        if match is None:
            raise SystemExit(f"Contrast key {name!r} does not match its declared filter")
        frame = frame[frame[key_filter["column"]] == float(match.group(1))]
    if "mode" in frame.columns and "deadline_s" not in contrast:
        frame = frame[frame["mode"] == "token"]
    outcome = contrast.get("outcome", "success")
    if outcome not in frame.columns:
        return None, None
    per = frame.groupby(["strategy", "qid"])[outcome].mean().unstack(0)
    a, b = contrast["strategy_a"], contrast["strategy_b"]
    if a not in per.columns or b not in per.columns:
        return None, None
    paired = per[[a, b]].dropna()
    return float(paired[a].mean()), float(paired[b].mean())


def main():
    manifest = check_manifest()
    report = {"status": "passed", "tolerance": TOLERANCE, "studies": {},
              "scope": "statistical reproduction from packaged trial exports"}
    for study, spec in sorted(manifest["studies"].items()):
        trials = pd.read_csv(BASE / study / "trials.csv")
        analysis = json.loads((BASE / study / "analysis.json").read_text())
        contrasts = contrasts_of(analysis, spec["contrasts"])
        if len(contrasts) != spec["family"]:
            raise SystemExit(f"{study}: expected {spec['family']} primary tests, found {len(contrasts)}")
        if len(trials) != spec["rows"]:
            raise SystemExit(f"{study}: expected {spec['rows']} trial rows, found {len(trials)}")

        means_checked = 0
        for name, contrast in contrasts:
            mean_a, mean_b = study_means(trials, contrast, name, spec.get("key_filter"))
            if mean_a is None:
                continue
            if not (close(mean_a, contrast["mean_a"]) and close(mean_b, contrast["mean_b"])):
                raise SystemExit(f"{study}: condition means differ for "
                                 f"{contrast['strategy_a']} vs {contrast['strategy_b']}")
            if not close(mean_a - mean_b, contrast["delta"]):
                raise SystemExit(f"{study}: delta differs for "
                                 f"{contrast['strategy_a']} vs {contrast['strategy_b']}")
            means_checked += 1

        adjusted = holm_correction([c["p_value"] for _, c in contrasts])
        for (_, contrast), value in zip(contrasts, adjusted):
            reported = contrast.get("p_value_holm")
            if reported is not None and not close(float(reported), float(value)):
                raise SystemExit(f"{study}: Holm adjustment differs for "
                                 f"{contrast['strategy_a']} vs {contrast['strategy_b']}")
        report["studies"][study] = {"primary_tests": len(contrasts), "trial_rows": len(trials),
                                    "contrast_means_recomputed": means_checked,
                                    "holm_family_recomputed": True}
    (BASE / "verification.json").write_text(json.dumps(report, indent=2) + "\\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
'''


def build(output_dir):
    output_dir = Path(output_dir)
    bundle = output_dir / "combined-supplement"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    names, studies = [], {}
    for study, spec in STUDIES.items():
        (bundle / study).mkdir()
        shutil.copy2(spec["trials"], bundle / study / "trials.csv")
        shutil.copy2(spec["analysis"], bundle / study / "analysis.json")
        names += [f"{study}/trials.csv", f"{study}/analysis.json"]
        studies[study] = {k: spec[k] for k in ["contrasts", "family", "rows", "title"]}
        if "key_filter" in spec:
            studies[study]["key_filter"] = spec["key_filter"]

    shutil.copy2(ROOT / "src/eval/metrics.py", bundle / "metrics.py")
    family_table = "\n".join(
        f"| {spec['title']} | {spec['family']} | {spec['rows']:,} |" for spec in STUDIES.values())
    (bundle / "README.md").write_text(README.format(family_table=family_table))
    (bundle / "reproduce.py").write_text(REPRODUCE)
    (bundle / "requirements.txt").write_text("numpy==1.23.5\npandas==1.5.3\n")
    names += ["metrics.py", "README.md", "reproduce.py", "requirements.txt"]

    for name in sorted(names):
        text = (bundle / name).read_text(errors="ignore")
        for value in FORBIDDEN:
            if value in text:
                raise ValueError(f"Identifying value {value!r} in packaged input: {name}")
    manifest = {"scope": "statistical reproduction; raw replay and GPU regeneration are separate",
                "studies": studies,
                "sha256": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                           for name in sorted(names)}}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    archive = output_dir / "agent-repair-combined-supplement.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in [*sorted(names), "manifest.json"]:
            stream.write(bundle / name, "combined-supplement/" + name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "output/supplement"))
    args = parser.parse_args()
    print(build(args.output_dir))


if __name__ == "__main__":
    main()
