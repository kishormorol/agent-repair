"""Turn the completed position-pair study into paper assets, never hand-typed.

The swap control gives each paired question its partner's uncertainty-chosen
origin, so the two arms carry identical origin multisets by construction. This
loader refuses any analysis that does not carry that exact balance, that is
incomplete, or whose audits report a mismatch.
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
RUN_ID = "position-pairs-20260916-v2"
LABELS = {"hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki"}


def load_position_pairs(base=BASE, run_id=RUN_ID):
    """Read the local independent reproduction and verify it against the package."""
    base = Path(base)
    protocol = json.loads((base / "prepared-v2/protocol.json").read_text())
    require(protocol["sha256"] == fingerprint(protocol["payload"]), "Protocol checksum mismatch")
    require(protocol["payload"]["run_id"] == run_id, "Protocol is not this run")
    analysis = json.loads((base / "analysis-local.json").read_text())
    require(analysis["protocol_sha256"] == protocol["sha256"],
            "Analysis does not describe the frozen protocol")
    require(analysis["complete"], "Analysis is incomplete; partial runs are not reportable")
    require(analysis["primary_family_size"] == protocol["payload"]["primary_family_size"],
            "Primary family size differs from the frozen protocol")

    remote = json.loads((base / "retrieved/results/analysis.json").read_text())
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
    return protocol["payload"], analysis, primary, secondary


def resolution_floor(result):
    """Smallest attainable two-sided sign-flip p for this block configuration."""
    return sign_flip_resolution_floor(result["positive_blocks"] + result["negative_blocks"])


def results_table(analysis, primary, secondary):
    audits = {a["dataset"]: a for a in analysis["audits"]}
    rows = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
            r"Dataset & $N_f$ & Pairs & $U$ & $S$ & $\Delta$ & 95\% CI & $p$ & $p_H$ \\",
            r"\midrule"]
    for dataset in sorted(audits, key=lambda d: LABELS.get(d, d)):
        audit, result = audits[dataset], primary[dataset]
        rows.append(
            f"{LABELS.get(dataset, dataset)} & {audit['main_failures']} & {result['n_pairs']} & "
            f"{100 * result['mean_a']:.2f} & {100 * result['mean_b']:.2f} & "
            f"{100 * result['delta']:+.2f} & "
            f"[{100 * result['delta_lo']:+.2f}, {100 * result['delta_hi']:+.2f}] & "
            f"{result['p_value']:.3f} & {result['p_value_holm']:.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def blocks_table(primary):
    rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
            r"Dataset & Pairs & $+$ & $-$ & $0$ & Floor \\", r"\midrule"]
    for dataset in sorted(primary, key=lambda d: LABELS.get(d, d)):
        result = primary[dataset]
        rows.append(
            f"{LABELS.get(dataset, dataset)} & {result['n_pairs']} & {result['positive_blocks']} & "
            f"{result['negative_blocks']} & {result['zero_blocks']} & "
            f"{resolution_floor(result):.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def macros(protocol, analysis, primary, secondary):
    audits = {a["dataset"]: a for a in analysis["audits"]}
    values = {
        "PairRunID": protocol["run_id"].replace("_", r"\_"),
        "PairQuestions": sum(a["main_questions"] for a in analysis["audits"]),
        "PairFailures": sum(a["main_failures"] for a in analysis["audits"]),
        "PairPairs": sum(a["pairs"] for a in analysis["audits"]),
        "PairMatched": sum(a["matched_questions"] for a in analysis["audits"]),
        "PairUnmatched": sum(a["unmatched_failures"] for a in analysis["audits"]),
        "PairRepairs": sum(a["unique_repairs"] for a in analysis["audits"]),
        "PairRows": sum(a["trial_rows"] for a in analysis["audits"]),
        "PairSeeds": len(protocol["seeds"]),
        "PairFamily": analysis["primary_family_size"],
    }
    lines = [r"\newcommand{\%s}{%s}" % (name, value) for name, value in values.items()]
    for dataset, prefix in [("hotpotqa", "PairHotpot"), ("2wikimultihopqa", "PairTwoWiki")]:
        if dataset not in primary:
            continue
        audit, result, other = audits[dataset], primary[dataset], secondary[dataset]
        emit = {
            "Failures": f"{audit['main_failures']}",
            "Pairs": f"{result['n_pairs']}",
            "Treatment": f"{100 * result['mean_a']:.2f}",
            "Swap": f"{100 * result['mean_b']:.2f}",
            "Delta": f"{100 * result['delta']:+.2f}",
            "Low": f"{100 * result['delta_lo']:+.2f}",
            "High": f"{100 * result['delta_hi']:+.2f}",
            "P": f"{result['p_value']:.3f}",
            "Holm": f"{result['p_value_holm']:.3f}",
            "Positive": f"{result['positive_blocks']}",
            "Negative": f"{result['negative_blocks']}",
            "Zero": f"{result['zero_blocks']}",
            "Floor": f"{resolution_floor(result):.3f}",
            "Shared": f"{audit['balance']['shared_execution_pairs']}",
            "Attempts": f"{audit['balance']['paired_attempts']}",
            "SharedPct": f"{100 * audit['balance']['shared_execution_fraction']:.1f}",
            "SecondaryDelta": f"{100 * other['delta']:+.2f}",
            "SecondaryP": f"{other['p_value']:.3f}",
        }
        lines += [r"\newcommand{\%s%s}{%s}" % (prefix, name, value) for name, value in emit.items()]
    return lines


def results_figure(out, analysis, primary):
    """Paired effect with its interval, beside the block composition that bounds it."""
    order = sorted(primary, key=lambda d: LABELS.get(d, d))
    labels = [LABELS.get(d, d) for d in order]
    deltas = [100 * primary[d]["delta"] for d in order]
    low = [100 * (primary[d]["delta"] - primary[d]["delta_lo"]) for d in order]
    high = [100 * (primary[d]["delta_hi"] - primary[d]["delta"]) for d in order]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), layout="constrained")
        y = np.arange(len(order))
        axes[0].errorbar(deltas, y, xerr=[low, high], fmt="o", color="#2b6cb0",
                         ecolor="#8aa9c9", elinewidth=2, capsize=3, markersize=5)
        axes[0].axvline(0, color="#8c8c8c", linewidth=.8, zorder=0)
        axes[0].set_yticks(y, labels)
        axes[0].set_xlabel("Uncertainty minus swapped origin (pp)")
        axes[0].grid(axis="x", color="#e4e7ea", linewidth=.6)

        widths = {key: [primary[d][f"{key}_blocks"] for d in order]
                  for key in ["positive", "negative", "zero"]}
        left = np.zeros(len(order))
        for key, colour, label in [("positive", "#2b6cb0", "Higher"),
                                   ("negative", "#c05621", "Lower"), ("zero", "#cbd5e0", "Tied")]:
            axes[1].barh(y, widths[key], left=left, color=colour, label=label, height=.6)
            left += np.asarray(widths[key], dtype=float)
        for index, dataset in enumerate(order):
            axes[1].text(left[index] + .4, index, f"floor {resolution_floor(primary[dataset]):.3f}",
                         va="center", fontsize=7, color="#666666")
        axes[1].set_yticks(y, labels)
        axes[1].set_xlim(0, max(left) * 1.42)
        # Keep headroom above the top bar so the legend cannot overlap it.
        axes[1].set_ylim(-0.7, len(order) - 1 + 1.1)
        axes[1].set_xlabel("Pairs by sign of the paired difference")
        axes[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper center", columnspacing=1.1)
        for ax in axes:
            ax.set_axisbelow(True)
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out) / "figures" / f"iclr2027_position_pairs.{extension}", dpi=300)
        plt.close(fig)


def provenance(base, protocol, analysis):
    base = Path(base)
    sources = ["analysis-local.json", "retrieved/results/analysis.json",
               "retrieved/results/analysis.trials.csv", "prepared-v2/protocol.json",
               "retrieval-verification.json"]
    return {"run_id": protocol["run_id"], "protocol_sha256": analysis["protocol_sha256"],
            "scope": protocol["scope"], "precision_note": protocol["precision_note"],
            "primary_family": protocol["primary_family"],
            "resampling_unit": protocol["resampling_unit"],
            "source_sha256": {name: file_fingerprint(base / name) for name in sources
                              if (base / name).is_file()},
            "code_sha256": {name: file_fingerprint(ROOT / name) for name in
                            ["scripts/position_pair_paper.py", "scripts/analyze_position_pair_study.py",
                             "src/repair/position_pairs.py"]}}


def build(base=BASE, out=ROOT / "paper/generated", run_id=RUN_ID):
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    protocol, analysis, primary, secondary = load_position_pairs(base, run_id)
    results_figure(out, analysis, primary)
    (out / "tables/iclr2027_position_pairs.tex").write_text(results_table(analysis, primary, secondary))
    (out / "tables/iclr2027_position_pair_blocks.tex").write_text(blocks_table(primary))
    (out / "iclr2027_position_pairs.tex").write_text(
        "% Generated by scripts/position_pair_paper.py from the audited position-pair study.\n"
        + "\n".join(macros(protocol, analysis, primary, secondary)) + "\n")
    record = provenance(base, protocol, analysis)
    (out / "iclr2027_position_pairs_provenance.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    return protocol, analysis, primary, secondary, record


if __name__ == "__main__":
    _, analysis, primary, _, _ = build()
    print(json.dumps({"run_id": RUN_ID, "cells": len(analysis["audits"]),
                      "pairs": sum(a["pairs"] for a in analysis["audits"]),
                      "holm": {d: r["p_value_holm"] for d, r in primary.items()}}, indent=2))
