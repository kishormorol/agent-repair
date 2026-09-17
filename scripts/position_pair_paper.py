"""Turn the completed position-pair cells into paper assets, never hand-typed.

The swap control gives each paired question its partner's uncertainty-chosen
origin, so the two arms carry identical origin multisets by construction. This
loader refuses any analysis that does not carry that exact balance, that is
incomplete, or whose audits report a mismatch.

Two model cells share one question cohort. That makes them a direct contrast
and, for the same reason, forbids a pooled confirmatory statistic: the
identifiers are not independent between cells.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_extension_study import require
from scripts.paper_stats import sign_flip_resolution_floor
from src.repair.controlled import file_fingerprint, fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_pairs import DEV_CONTROL, SWAP

BASE = ROOT / "output/aws-experiment/2026-09-16-position-pairs"
DATASETS = {"hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki"}
# Retrieval directories are scoped by model: both cells emit an identically
# named analysis and logs, so one shared directory would collide.
CELLS = {
    "qwen32b": {"label": "Qwen", "prefix": "PairQwen", "package": "prepared-v2",
                "analysis": "analysis-local.json",
                "remote": "retrieved-qwen32b/results/analysis.json",
                "trials": "retrieved-qwen32b/results/analysis.trials.csv",
                "run_id": "position-pairs-20260916-v2"},
    "mistral12b": {"label": "Mistral", "prefix": "PairMistral", "package": "prepared-mistral12b",
                   "analysis": "analysis-local-mistral.json",
                   "remote": "retrieved/results/analysis.json",
                   "trials": "retrieved/results/analysis.trials.csv",
                   "run_id": "position-pairs-mistral-20260917-v2"},
}


def load_cell(model_key, base=BASE):
    """Read one cell's independent reproduction and verify it against its package."""
    base, spec = Path(base), CELLS[model_key]
    protocol = json.loads((base / spec["package"] / "protocol.json").read_text())
    require(protocol["sha256"] == fingerprint(protocol["payload"]), "Protocol checksum mismatch")
    require(protocol["payload"]["run_id"] == spec["run_id"], f"Protocol is not {spec['run_id']}")
    require(protocol["payload"]["model_key"] == model_key, "Protocol names a different model")
    analysis = json.loads((base / spec["analysis"]).read_text())
    require(analysis["protocol_sha256"] == protocol["sha256"],
            "Analysis does not describe the frozen protocol")
    require(analysis["complete"], "Analysis is incomplete; partial runs are not reportable")
    require(analysis["primary_family_size"] == protocol["payload"]["primary_family_size"],
            "Primary family size differs from the frozen protocol")
    remote = json.loads((base / spec["remote"]).read_text())
    require(remote["protocol_sha256"] == analysis["protocol_sha256"],
            "On-instance analysis describes a different protocol")

    datasets = [a["dataset"] for a in analysis["audits"]]
    require(len(datasets) == len(set(datasets)) == len(protocol["payload"]["cohorts"]),
            "Audited cells do not cover the frozen cohorts")
    for audit in analysis["audits"]:
        require(audit["mismatches"] == 0, f"Audited mismatches in {audit['dataset']}")
        balance = audit["balance"]
        require(balance["raw_origin_distribution_equal"]
                and balance["normalized_origin_distribution_equal"],
                f"Origin distributions are not exactly balanced in {audit['dataset']}")
        require(audit["matched_questions"] == 2 * audit["pairs"], "Pair membership mismatch")
        require(audit["main_failures"] == audit["matched_questions"] + audit["unmatched_failures"],
                "Failure accounting does not close")

    primary = {r["dataset"]: r for r in analysis["primary_comparisons"]}
    require(set(primary) == set(datasets), "Primary comparisons do not cover every cell")
    for result in primary.values():
        require(result["strategy_a"] == TREATMENT and result["strategy_b"] == SWAP,
                "Primary comparison is not treatment versus the length-matched swap")
        blocks = result["positive_blocks"] + result["negative_blocks"] + result["zero_blocks"]
        require(blocks == result["n_pairs"], "Block counts do not sum to the pair count")
    secondary = {r["dataset"]: r for r in analysis["secondary_comparisons"]}
    for result in secondary.values():
        require(result["strategy_b"] == DEV_CONTROL and result["p_value_holm"] is None,
                "Secondary comparison must stay descriptive")
    return {**spec, "model_key": model_key, "protocol": protocol["payload"],
            "analysis": analysis, "primary": primary, "secondary": secondary,
            "audits": {a["dataset"]: a for a in analysis["audits"]},
            "paths": {k: spec[k] for k in ["analysis", "remote", "trials", "package"]}}


def load_position_pairs(base=BASE):
    """Load every cell and confirm they share one question cohort."""
    cells = {key: load_cell(key, base) for key in CELLS}
    cohorts = [tuple(sorted(c["protocol"]["cohorts"][d]["main_ids"]))
               for c in cells.values() for d in sorted(DATASETS)]
    require(len(set(cohorts)) == len(DATASETS),
            "Cells must face identical question cohorts, one per dataset")
    return cells


def resolution_floor(result):
    """Smallest attainable two-sided sign-flip p for this block configuration."""
    return sign_flip_resolution_floor(result["positive_blocks"] + result["negative_blocks"])


def ordered_rows(cells):
    for key in CELLS:
        cell = cells[key]
        for dataset in DATASETS:
            if dataset in cell["primary"]:
                yield cell, dataset, cell["primary"][dataset], cell["audits"][dataset]


def results_table(cells):
    rows = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
            r"Model & Dataset & $N_f$ & Pairs & $U$ & $S$ & $\Delta$ & 95\% CI & $p$ & $p_H$ \\",
            r"\midrule"]
    for cell, dataset, result, audit in ordered_rows(cells):
        rows.append(
            f"{cell['label']} & {DATASETS[dataset]} & {audit['main_failures']} & "
            f"{result['n_pairs']} & {100 * result['mean_a']:.2f} & {100 * result['mean_b']:.2f} & "
            f"{100 * result['delta']:+.2f} & "
            f"[{100 * result['delta_lo']:+.2f}, {100 * result['delta_hi']:+.2f}] & "
            f"{result['p_value']:.3f} & {result['p_value_holm']:.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def blocks_table(cells):
    rows = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
            r"Model & Dataset & Pairs & $+$ & $-$ & $0$ & Floor \\", r"\midrule"]
    for cell, dataset, result, _ in ordered_rows(cells):
        rows.append(
            f"{cell['label']} & {DATASETS[dataset]} & {result['n_pairs']} & "
            f"{result['positive_blocks']} & {result['negative_blocks']} & "
            f"{result['zero_blocks']} & {resolution_floor(result):.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def results_figure(out, cells):
    """Paired effect with its interval, beside the block composition that bounds it."""
    rows = list(ordered_rows(cells))
    labels = [f"{c['label']}/{DATASETS[d]}" for c, d, _, _ in rows]
    deltas = [100 * r["delta"] for _, _, r, _ in rows]
    low = [100 * (r["delta"] - r["delta_lo"]) for _, _, r, _ in rows]
    high = [100 * (r["delta_hi"] - r["delta"]) for _, _, r, _ in rows]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), layout="constrained")
        y = np.arange(len(rows))
        axes[0].errorbar(deltas, y, xerr=[low, high], fmt="o", color="#2b6cb0",
                         ecolor="#8aa9c9", elinewidth=2, capsize=3, markersize=5)
        axes[0].axvline(0, color="#8c8c8c", linewidth=.8, zorder=0)
        axes[0].set_yticks(y, labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Uncertainty minus swapped origin (pp)")
        axes[0].grid(axis="x", color="#e4e7ea", linewidth=.6)

        left = np.zeros(len(rows))
        for key, colour, label in [("positive", "#2b6cb0", "Higher"),
                                   ("negative", "#c05621", "Lower"), ("zero", "#cbd5e0", "Tied")]:
            values = np.array([r[f"{key}_blocks"] for _, _, r, _ in rows], dtype=float)
            axes[1].barh(y, values, left=left, color=colour, label=label, height=.6)
            left += values
        for index, (_, _, result, _) in enumerate(rows):
            axes[1].text(left[index] + .6, index, f"floor {resolution_floor(result):.3f}",
                         va="center", fontsize=7, color="#666666")
        axes[1].set_yticks(y, labels, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(left) * 1.38)
        axes[1].set_ylim(len(rows) - 0.4, -1.1)
        axes[1].set_xlabel("Pairs by sign of difference")
        axes[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper center", columnspacing=1.1)
        for ax in axes:
            ax.set_axisbelow(True)
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out) / "figures" / f"iclr2027_position_pairs.{extension}", dpi=300)
        plt.close(fig)


def macros(cells):
    audits = [a for c in cells.values() for a in c["analysis"]["audits"]]
    values = {
        "PairCells": len(cells),
        "PairQuestions": sum(a["main_questions"] for a in audits),
        "PairCohort": sum(a["main_questions"] for a in cells["qwen32b"]["analysis"]["audits"]),
        "PairFailures": sum(a["main_failures"] for a in audits),
        "PairPairs": sum(a["pairs"] for a in audits),
        "PairMatched": sum(a["matched_questions"] for a in audits),
        "PairUnmatched": sum(a["unmatched_failures"] for a in audits),
        "PairRepairs": sum(a["unique_repairs"] for a in audits),
        "PairRows": sum(a["trial_rows"] for a in audits),
        "PairFamily": cells["qwen32b"]["analysis"]["primary_family_size"],
        # Cells whose exact test could not have reached a conventional level,
        # whatever the data showed, because too few blocks are non-zero.
        "PairBlockedCells": sum(resolution_floor(r) > 0.05 for _, _, r, _ in ordered_rows(cells)),
        "PairTotalCells": sum(1 for _ in ordered_rows(cells)),
        "PairTiedPairs": sum(r["zero_blocks"] for _, _, r, _ in ordered_rows(cells)),
    }
    lines = [r"\newcommand{\%s}{%s}" % item for item in values.items()]
    for cell, dataset, result, audit in ordered_rows(cells):
        prefix = cell["prefix"] + ("Hotpot" if dataset == "hotpotqa" else "TwoWiki")
        emit = {
            "Failures": f"{audit['main_failures']}", "Pairs": f"{result['n_pairs']}",
            "Treatment": f"{100 * result['mean_a']:.2f}", "Swap": f"{100 * result['mean_b']:.2f}",
            "Delta": f"{100 * result['delta']:+.2f}",
            "Low": f"{100 * result['delta_lo']:+.2f}", "High": f"{100 * result['delta_hi']:+.2f}",
            "P": f"{result['p_value']:.3f}", "Holm": f"{result['p_value_holm']:.3f}",
            "Positive": f"{result['positive_blocks']}", "Negative": f"{result['negative_blocks']}",
            "Zero": f"{result['zero_blocks']}", "Floor": f"{resolution_floor(result):.3f}",
            "Shared": f"{audit['balance']['shared_execution_pairs']}",
            "Attempts": f"{audit['balance']['paired_attempts']}",
            "SharedPct": f"{100 * audit['balance']['shared_execution_fraction']:.1f}",
            "SecondaryDelta": f"{100 * cell['secondary'][dataset]['delta']:+.2f}",
        }
        lines += [r"\newcommand{\%s%s}{%s}" % (prefix, name, value) for name, value in emit.items()]
    return lines


def provenance(base, cells):
    base = Path(base)
    record = {"cells": {}, "pooling_limitation":
              "The two cells share question identifiers, so they support a descriptive "
              "contrast only and no pooled confirmatory statistic.",
              "code_sha256": {name: file_fingerprint(ROOT / name) for name in
                              ["scripts/position_pair_paper.py",
                               "scripts/analyze_position_pair_study.py",
                               "src/repair/position_pairs.py"]}}
    for key, cell in cells.items():
        paths = cell["paths"]
        sources = [paths["analysis"], paths["remote"], paths["trials"],
                   f"{paths['package']}/protocol.json"]
        record["cells"][key] = {
            "run_id": cell["run_id"], "protocol_sha256": cell["analysis"]["protocol_sha256"],
            "scope": cell["protocol"]["scope"], "precision_note": cell["protocol"]["precision_note"],
            "primary_family": cell["protocol"]["primary_family"],
            "source_sha256": {name: file_fingerprint(base / name) for name in sources
                              if (base / name).is_file()}}
    return record


def build(base=BASE, out=ROOT / "paper/generated"):
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cells = load_position_pairs(base)
    (out / "tables/iclr2027_position_pairs.tex").write_text(results_table(cells))
    (out / "tables/iclr2027_position_pair_blocks.tex").write_text(blocks_table(cells))
    results_figure(out, cells)
    (out / "iclr2027_position_pairs.tex").write_text(
        "% Generated by scripts/position_pair_paper.py from the audited position-pair cells.\n"
        + "\n".join(macros(cells)) + "\n")
    record = provenance(base, cells)
    (out / "iclr2027_position_pairs_provenance.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    return cells, record


if __name__ == "__main__":
    cells, _ = build()
    print(json.dumps({"cells": list(cells),
                      "pairs": sum(a["pairs"] for c in cells.values()
                                   for a in c["analysis"]["audits"]),
                      "holm": {f"{c['label']}/{d}": r["p_value_holm"]
                               for c, d, r, _ in ordered_rows(cells)}}, indent=2))
