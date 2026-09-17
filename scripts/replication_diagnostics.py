"""Per-cell replication diagnostics: paired outcomes, overlap, termination, bounds.

These are descriptive extensions of the frozen replication, not new tests. The
primary Holm-adjusted outputs are read from the completed analysis and reported
unchanged; the exact sign-flip values computed here are labelled sensitivity
checks. No pooled confirmatory statistic is produced, because question
identifiers are shared across model cells.
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
import pandas as pd
from scripts.analyze_extension_study import require
from scripts.paper_stats import sign_flip_resolution_floor, wins_losses_ties
from src.repair.controlled import file_fingerprint
from src.repair.diagnosis import TREATMENT

BASE = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
RESTART = "full_restart"
LIMIT_REASONS = ("budget", "max_steps")
MODELS = {"qwen32b": "Qwen", "mistral12b": "Mistral"}
DATASETS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "2wikimultihopqa": "2Wiki"}


def load_replication(base=BASE):
    """Read the completed replication and its independently reproduced analysis."""
    base = Path(base)
    analysis = json.loads((base / "pooled-analysis-local.json").read_text())
    require(analysis["complete"], "Replication analysis is incomplete")
    require(sum(a["mismatches"] for a in analysis["audits"]) == 0,
            "Replication analysis reports audit mismatches")
    reproduction = json.loads((base / "analysis-reproduction-check.json").read_text())
    require(reproduction["matched"] and reproduction["mismatches"] == 0,
            "Local and remote replication analyses do not agree")
    trials = pd.read_csv(base / "pooled-analysis-local.trials.csv")
    token = trials[trials["mode"] == "token"].copy()
    require(not token.empty, "No token-mode replication rows")
    return analysis, token


def paired_outcomes(cell, control, treatment=TREATMENT):
    """Wins, losses and ties over question-level mean success differences."""
    per = cell.groupby(["strategy", "qid"]).success.mean().unstack(0)
    require(treatment in per.columns and control in per.columns,
            f"Missing {treatment} or {control} in this cell")
    differences = (per[treatment] - per[control]).dropna()
    return wins_losses_ties(differences.tolist())


def execution_overlap(cell, control, treatment=TREATMENT):
    """How often the two policies ran the identical physical execution."""
    pair = cell[cell.strategy.isin([treatment, control])]
    counts = pair.groupby(["qid", "seed"]).execution_id.nunique()
    differing = counts[counts > 1].reset_index().qid.nunique()
    shared = int((counts == 1).sum())
    return {"attempts": int(len(counts)), "shared_attempts": shared,
            "shared_fraction": shared / len(counts) if len(counts) else None,
            "questions_with_any_differing_execution": int(differing),
            "questions": int(pair.qid.nunique())}


def termination(cell, strategy):
    rows = cell[cell.strategy == strategy]
    counts = rows.terminated_reason.value_counts().to_dict()
    limited = sum(counts.get(reason, 0) for reason in LIMIT_REASONS)
    return {"rows": int(len(rows)), "limit_stopped": int(limited),
            "limit_fraction": limited / len(rows) if len(rows) else None,
            "by_reason": {k: int(v) for k, v in sorted(counts.items())}}


def build_diagnostics(base=BASE):
    analysis, token = load_replication(base)
    primary = {(r["model_key"], r["dataset"], r["strategy_b"]): r
               for r in analysis["primary_comparisons"]}
    controls = sorted({key[2] for key in primary})
    cells, totals = [], {"rows": 0, "limit_stopped": 0}
    for (model_key, dataset), cell in token.groupby(["model_key", "dataset"]):
        record = {"model_key": model_key, "dataset": dataset,
                  "termination": termination(cell, TREATMENT), "contrasts": []}
        totals["rows"] += record["termination"]["rows"]
        totals["limit_stopped"] += record["termination"]["limit_stopped"]
        for control in controls:
            result = primary.get((model_key, dataset, control))
            if result is None:
                continue
            outcomes = paired_outcomes(cell, control)
            record["contrasts"].append({
                "control": control, **outcomes,
                "overlap": execution_overlap(cell, control),
                # Frozen primary outputs, reported unchanged.
                "delta": result["delta"], "delta_lo": result["delta_lo"],
                "delta_hi": result["delta_hi"], "p_value": result["p_value"],
                "p_value_holm": result["p_value_holm"],
                "inference_note": "frozen primary output; floor is a design property"})
        cells.append(record)
    require(totals["rows"] > 0, "No treatment rows found")
    return {"source": "completed replication, token mode",
            "status": "descriptive extension; primary Holm outputs unchanged",
            "pooled_confirmatory_claim": False,
            "pooled_note": "Question identifiers are shared across model cells, "
                           "so no pooled confirmatory statistic is reported.",
            "treatment": TREATMENT, "cells": cells,
            "treatment_rows": totals["rows"], "treatment_limit_stopped": totals["limit_stopped"],
            "nonzero_question_range": [min(c["nonzero"] for cell in cells for c in cell["contrasts"]),
                                       max(c["nonzero"] for cell in cells for c in cell["contrasts"])]}


def restart_contrast(diagnostics, model_key, dataset):
    cell = next(c for c in diagnostics["cells"]
                if c["model_key"] == model_key and c["dataset"] == dataset)
    return next(c for c in cell["contrasts"] if c["control"] == RESTART)


def outcomes_table(diagnostics, control=RESTART):
    rows = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
            r"Model & Dataset & $N_q$ & W & L & T & Floor & $\Delta$ & 95\% CI \\",
            r"\midrule"]
    for cell in diagnostics["cells"]:
        contrast = next((c for c in cell["contrasts"] if c["control"] == control), None)
        if contrast is None:
            continue
        rows.append(
            f"{MODELS.get(cell['model_key'], cell['model_key'])} & "
            f"{DATASETS.get(cell['dataset'], cell['dataset'])} & "
            f"{contrast['questions']} & {contrast['wins']} & {contrast['losses']} & "
            f"{contrast['ties']} & {contrast['resolution_floor']:.3f} & "
            f"{100 * contrast['delta']:+.2f} & "
            f"[{100 * contrast['delta_lo']:+.2f}, {100 * contrast['delta_hi']:+.2f}] " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def overlap_table(diagnostics, control=RESTART):
    rows = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
            r"Model & Dataset & Attempts & Shared & Shared \% & $Q_{\neq}$ \\", r"\midrule"]
    for cell in diagnostics["cells"]:
        contrast = next((c for c in cell["contrasts"] if c["control"] == control), None)
        if contrast is None:
            continue
        overlap = contrast["overlap"]
        rows.append(
            f"{MODELS.get(cell['model_key'], cell['model_key'])} & "
            f"{DATASETS.get(cell['dataset'], cell['dataset'])} & "
            f"{overlap['attempts']} & {overlap['shared_attempts']} & "
            f"{100 * overlap['shared_fraction']:.1f} & "
            f"{overlap['questions_with_any_differing_execution']} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def macros(diagnostics):
    low, high = diagnostics["nonzero_question_range"]
    shared = [c["overlap"]["shared_fraction"] for cell in diagnostics["cells"]
              for c in cell["contrasts"] if c["control"] == RESTART]
    example = restart_contrast(diagnostics, "qwen32b", "2wikimultihopqa")
    values = {
        "RepNonzeroLow": low, "RepNonzeroHigh": high,
        "RepTreatmentRows": diagnostics["treatment_rows"],
        "RepLimitStopped": diagnostics["treatment_limit_stopped"],
        "RepLimitPct": f"{100 * diagnostics['treatment_limit_stopped'] / diagnostics['treatment_rows']:.1f}",
        "RepSharedLow": f"{100 * min(shared):.1f}", "RepSharedHigh": f"{100 * max(shared):.1f}",
        "RepExampleWins": example["wins"], "RepExampleLosses": example["losses"],
        "RepExampleTies": example["ties"],
        "RepExampleFloor": f"{example['resolution_floor']:.3f}",
        "RepExampleDelta": f"{100 * example['delta']:+.2f}",
        "RepExampleLow": f"{100 * example['delta_lo']:+.2f}",
        "RepExampleHigh": f"{100 * example['delta_hi']:+.2f}",
    }
    return [r"\newcommand{\%s}{%s}" % item for item in values.items()]


def diagnostics_figure(out, diagnostics, control=RESTART, level=0.05):
    """Show how few questions differ, and which cells their test cannot resolve."""
    cells = [(c, next(x for x in c["contrasts"] if x["control"] == control))
             for c in diagnostics["cells"]]
    labels = [f"{MODELS.get(c['model_key'], c['model_key'])}/{DATASETS.get(c['dataset'], c['dataset'])}"
              for c, _ in cells]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
        x = np.arange(len(cells))
        bottom = np.zeros(len(cells))
        for key, colour, label in [("wins", "#2b6cb0", "Higher"), ("losses", "#c05621", "Lower"),
                                   ("ties", "#cbd5e0", "Tied")]:
            values = np.array([contrast[key] for _, contrast in cells], dtype=float)
            axes[0].bar(x, values, bottom=bottom, color=colour, label=label, width=.62)
            bottom += values
        axes[0].set_xticks(x, labels, fontsize=7, rotation=30, ha="right")
        axes[0].set_ylabel("Paired questions")
        axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper center", columnspacing=1.1)
        axes[0].set_ylim(0, bottom.max() * 1.28)
        axes[0].grid(axis="y", color="#e4e7ea", linewidth=.6)

        floors = [contrast["resolution_floor"] for _, contrast in cells]
        colours = ["#c05621" if f > level else "#2b6cb0" for f in floors]
        axes[1].bar(x, floors, color=colours, width=.62)
        axes[1].axhline(level, color="#8c8c8c", linewidth=.9, linestyle="--")
        axes[1].text(len(cells) - .45, level * 1.25, f"{level:g}", fontsize=7,
                     color="#666666", ha="right")
        axes[1].set_yscale("log")
        axes[1].set_xticks(x, labels, fontsize=7, rotation=30, ha="right")
        axes[1].set_ylabel("Smallest attainable $p$")
        axes[1].grid(axis="y", color="#e4e7ea", linewidth=.6)
        for ax in axes:
            ax.set_axisbelow(True)
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out) / "figures" / f"iclr2027_replication_diagnostics.{extension}", dpi=300)
        plt.close(fig)


def build(base=BASE, out=ROOT / "paper/generated"):
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    diagnostics = build_diagnostics(base)
    diagnostics_figure(out, diagnostics)
    (out / "tables/iclr2027_replication_outcomes.tex").write_text(outcomes_table(diagnostics))
    (out / "tables/iclr2027_replication_overlap.tex").write_text(overlap_table(diagnostics))
    (out / "iclr2027_replication_diagnostics.tex").write_text(
        "% Generated by scripts/replication_diagnostics.py; descriptive extensions only.\n"
        + "\n".join(macros(diagnostics)) + "\n")
    record = {**{k: v for k, v in diagnostics.items() if k != "cells"},
              "source_sha256": {name: file_fingerprint(Path(base) / name) for name in
                                ["pooled-analysis-local.json", "pooled-analysis-local.trials.csv",
                                 "analysis-reproduction-check.json"]},
              "code_sha256": {name: file_fingerprint(ROOT / name) for name in
                              ["scripts/replication_diagnostics.py", "scripts/paper_stats.py"]},
              "cells": diagnostics["cells"]}
    (out / "iclr2027_replication_diagnostics.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    return diagnostics, record


if __name__ == "__main__":
    diagnostics, _ = build()
    print(json.dumps({"cells": len(diagnostics["cells"]),
                      "nonzero_question_range": diagnostics["nonzero_question_range"],
                      "treatment_limit_stopped": diagnostics["treatment_limit_stopped"],
                      "treatment_rows": diagnostics["treatment_rows"]}, indent=2))
