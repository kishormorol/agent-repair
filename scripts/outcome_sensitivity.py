"""How far the replication conclusion depends on the success definition.

The study gate accepts an answer on exact match **or** token F1 at least 0.5.
Rescoring completed records under strict EM is a sensitivity check, not a new
failure cohort: questions accepted under the gate were never offered for
repair, so their repair outcomes do not exist and cannot be recovered here.
Throughout, "accepted" means accepted under the study gate, never strict EM.
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
from scripts.replication_diagnostics import DATASETS, MODELS, RESTART, load_replication
from src.repair.controlled import file_fingerprint
from src.repair.diagnosis import TREATMENT

BASE = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
POLICIES = (TREATMENT, RESTART)


def initial_cohort(analysis):
    """Split each cell's initial evaluations into accepted, strict EM and F1-only."""
    cells = []
    for audit in analysis["audits"]:
        n = audit["main_questions"]
        accepted = round(audit["initial_success"] * n)
        exact = round(audit["initial_em"] * n)
        require(0 <= exact <= accepted <= n, "Initial EM cannot exceed the accepted count")
        cells.append({"model_key": audit["model_key"], "dataset": audit["dataset"],
                      "questions": n, "accepted": accepted, "exact_match": exact,
                      "accepted_not_exact": accepted - exact,
                      "failures_offered_repair": audit["main_failures"]})
    return cells


def repair_outcomes(token, model_key, dataset, policy):
    """Accepted, strict-EM and F1-only counts among one policy's repair attempts."""
    rows = token[(token.model_key == model_key) & (token.dataset == dataset)
                 & (token.strategy == policy)]
    accepted, exact = int(rows.success.sum()), int(rows.em.sum())
    require(exact <= accepted, "Repair EM cannot exceed accepted repairs")
    return {"attempts": int(len(rows)), "accepted": accepted, "exact_match": exact,
            "accepted_not_exact": accepted - exact,
            "mean_success": float(rows.success.mean()) if len(rows) else None,
            "mean_em": float(rows.em.mean()) if len(rows) else None,
            "mean_f1": float(rows.f1.mean()) if len(rows) else None}


def full_cohort(token, cell, policy):
    """Outcome over all questions, preserving answers already accepted.

    A question accepted initially keeps that answer; only failures contribute a
    repair outcome. This is the quantity a deployment would see, unlike the
    conditional repair rate over failures alone.
    """
    rows = token[(token.model_key == cell["model_key"]) & (token.dataset == cell["dataset"])
                 & (token.strategy == policy)]
    per_question = rows.groupby("qid").success.mean()
    require(len(per_question) == cell["failures_offered_repair"],
            "Repaired questions must equal the failures offered repair")
    return (cell["accepted"] + float(per_question.sum())) / cell["questions"]


def build_sensitivity(base=BASE):
    analysis, token = load_replication(base)
    cells = initial_cohort(analysis)
    for cell in cells:
        cell["policies"] = {}
        for policy in POLICIES:
            cell["policies"][policy] = {
                **repair_outcomes(token, cell["model_key"], cell["dataset"], policy),
                "full_cohort": full_cohort(token, cell, policy)}
        cell["full_cohort_delta"] = (cell["policies"][TREATMENT]["full_cohort"]
                                     - cell["policies"][RESTART]["full_cohort"])
    totals = {
        "questions": sum(c["questions"] for c in cells),
        "accepted": sum(c["accepted"] for c in cells),
        "exact_match": sum(c["exact_match"] for c in cells),
        "accepted_not_exact": sum(c["accepted_not_exact"] for c in cells),
        "repair_accepted": sum(c["policies"][p]["accepted"] for c in cells for p in POLICIES),
        "repair_accepted_not_exact": sum(c["policies"][p]["accepted_not_exact"]
                                         for c in cells for p in POLICIES),
    }
    return {"gate": "exact match or token F1 at least 0.5",
            "status": "exploratory rescoring of completed records; not a new failure cohort",
            "terminology": "accepted means accepted under the study gate, never strict EM",
            "unrepaired_note": "Questions accepted under the gate were never offered repair, "
                               "so an EM-gated conclusion needs a separately frozen study.",
            "cells": cells, "totals": totals}


def initial_table(sensitivity):
    rows = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
            r"Model & Dataset & $N$ & Accepted & EM & Gate only \\", r"\midrule"]
    for cell in sensitivity["cells"]:
        rows.append(f"{MODELS.get(cell['model_key'], cell['model_key'])} & "
                    f"{DATASETS.get(cell['dataset'], cell['dataset'])} & {cell['questions']} & "
                    f"{cell['accepted']} & {cell['exact_match']} & {cell['accepted_not_exact']} " + r"\\")
    totals = sensitivity["totals"]
    rows += [r"\midrule",
             f"\\textbf{{All}} & & {totals['questions']} & {totals['accepted']} & "
             f"{totals['exact_match']} & {totals['accepted_not_exact']} " + r"\\",
             r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def full_cohort_table(sensitivity):
    rows = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
            r"Model & Dataset & Initial & Uncertainty & Restart & $\Delta$ \\", r"\midrule"]
    for cell in sensitivity["cells"]:
        initial = cell["accepted"] / cell["questions"]
        rows.append(f"{MODELS.get(cell['model_key'], cell['model_key'])} & "
                    f"{DATASETS.get(cell['dataset'], cell['dataset'])} & "
                    f"{100 * initial:.1f} & "
                    f"{100 * cell['policies'][TREATMENT]['full_cohort']:.1f} & "
                    f"{100 * cell['policies'][RESTART]['full_cohort']:.1f} & "
                    f"{100 * cell['full_cohort_delta']:+.2f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def sensitivity_figure(out, sensitivity):
    """Contrast the two success definitions on initial answers and on repairs."""
    cells = sensitivity["cells"]
    labels = [f"{MODELS.get(c['model_key'], c['model_key'])}/{DATASETS.get(c['dataset'], c['dataset'])}"
              for c in cells]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
        x = np.arange(len(cells))
        exact = np.array([c["exact_match"] for c in cells], dtype=float)
        gate_only = np.array([c["accepted_not_exact"] for c in cells], dtype=float)
        axes[0].bar(x, exact, color="#2b6cb0", label="Exact match", width=.62)
        axes[0].bar(x, gate_only, bottom=exact, color="#dd9a4a", label="Gate only", width=.62)
        axes[0].set_xticks(x, labels, fontsize=7, rotation=30, ha="right")
        axes[0].set_ylabel("Initial answers accepted")
        axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
        axes[0].set_ylim(0, max(exact + gate_only) * 1.3)
        axes[0].grid(axis="y", color="#e4e7ea", linewidth=.6)

        width = .38
        for offset, policy, colour, label in [(-width / 2, TREATMENT, "#2b6cb0", "Uncertainty"),
                                              (width / 2, RESTART, "#c05621", "Restart")]:
            values = [100 * c["policies"][policy]["full_cohort"] for c in cells]
            axes[1].bar(x + offset, values, width=width, color=colour, label=label)
        initial = [100 * c["accepted"] / c["questions"] for c in cells]
        axes[1].scatter(x, initial, color="#2d3748", marker="_", s=180, linewidths=1.6,
                        zorder=4, label="Initial")
        axes[1].set_xticks(x, labels, fontsize=7, rotation=30, ha="right")
        axes[1].set_ylabel("Full-cohort accepted (%)")
        axes[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
        axes[1].set_ylim(0, max(initial) * 1.5)
        axes[1].grid(axis="y", color="#e4e7ea", linewidth=.6)
        for ax in axes:
            ax.set_axisbelow(True)
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out) / "figures" / f"iclr2027_outcome_sensitivity.{extension}", dpi=300)
        plt.close(fig)


def macros(sensitivity):
    totals = sensitivity["totals"]
    values = {
        "GateQuestions": totals["questions"], "GateAccepted": totals["accepted"],
        "GateExact": totals["exact_match"], "GateOnly": totals["accepted_not_exact"],
        "GateOnlyPct": f"{100 * totals['accepted_not_exact'] / totals['questions']:.2f}",
        "GateRepairAccepted": totals["repair_accepted"],
        "GateRepairOnly": totals["repair_accepted_not_exact"],
        "GateRepairOnlyPct": f"{100 * totals['repair_accepted_not_exact'] / totals['repair_accepted']:.1f}",
        "GateFullLow": f"{100 * min(c['full_cohort_delta'] for c in sensitivity['cells']):+.2f}",
        "GateFullHigh": f"{100 * max(c['full_cohort_delta'] for c in sensitivity['cells']):+.2f}",
    }
    return [r"\newcommand{\%s}{%s}" % item for item in values.items()]


def build(base=BASE, out=ROOT / "paper/generated"):
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sensitivity = build_sensitivity(base)
    (out / "tables/iclr2027_outcome_gate.tex").write_text(initial_table(sensitivity))
    (out / "tables/iclr2027_outcome_full_cohort.tex").write_text(full_cohort_table(sensitivity))
    sensitivity_figure(out, sensitivity)
    (out / "iclr2027_outcome_sensitivity.tex").write_text(
        "% Generated by scripts/outcome_sensitivity.py; exploratory rescoring only.\n"
        + "\n".join(macros(sensitivity)) + "\n")
    record = {**sensitivity,
              "source_sha256": {name: file_fingerprint(Path(base) / name) for name in
                                ["pooled-analysis-local.json", "pooled-analysis-local.trials.csv"]},
              "code_sha256": {"scripts/outcome_sensitivity.py":
                              file_fingerprint(ROOT / "scripts/outcome_sensitivity.py")}}
    (out / "iclr2027_outcome_sensitivity.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    return sensitivity, record


if __name__ == "__main__":
    sensitivity, _ = build()
    print(json.dumps(sensitivity["totals"], indent=2))
