"""Descriptive runtime evidence, kept inside the measured setup.

The runtime component answers a completion-time question for one execution
configuration. It cannot speak to deployed caching, concurrency, repeated
acquisition or a matched total-resource allowance, so this module reports the
measured latency distribution and termination counts and nothing further.
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
from scripts.replication_diagnostics import RESTART, load_replication
from src.repair.controlled import file_fingerprint
from src.repair.diagnosis import DIAGNOSIS_STRATEGY, TREATMENT

BASE = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
LABELS = {TREATMENT: "Uncertainty", RESTART: "Restart", DIAGNOSIS_STRATEGY: "Diagnosis"}
QUANTILES = (0.25, 0.5, 0.75, 0.9)


def runtime_rows(base=BASE):
    analysis, _ = load_replication(base)
    trials = pd.read_csv(Path(base) / "pooled-analysis-local.trials.csv")
    rows = trials[trials["mode"] == "runtime"].copy()
    require(not rows.empty, "No runtime rows in the completed replication")
    require(set(rows.strategy) == set(LABELS), "Runtime policies differ from the frozen design")
    return analysis, rows


def latency_summary(rows):
    """Measured wall time per attempt, including selection and prefix replay."""
    summary = []
    for strategy, sub in rows.groupby("strategy"):
        latency = sub.incremental_wall_latency_s
        counts = sub.terminated_reason.value_counts().to_dict()
        require(latency.notna().all() and (latency >= 0).all(), "Invalid measured latency")
        summary.append({
            "strategy": strategy, "label": LABELS[strategy],
            "attempts": int(len(sub)), "questions": int(sub.qid.nunique()),
            "finished": int(counts.get("finished", 0)),
            "step_limited": int(counts.get("max_steps", 0)),
            "token_limited": int(counts.get("budget", 0)),
            "errored": int(counts.get("error", 0)),
            "mean_s": float(latency.mean()), "max_s": float(latency.max()),
            **{f"q{int(100 * q)}_s": float(latency.quantile(q)) for q in QUANTILES},
            "selection_mean_s": float(sub.selection_latency_s.mean()),
            "recovery_mean_s": float(sub.recovery_wall_latency_s.mean()),
        })
    return sorted(summary, key=lambda row: list(LABELS).index(row["strategy"]))


def build_scope(base=BASE):
    analysis, rows = runtime_rows(base)
    summary = latency_summary(rows)
    totals = {"attempts": sum(s["attempts"] for s in summary),
              "questions": summary[0]["questions"],
              "step_limited": sum(s["step_limited"] for s in summary),
              "token_limited": sum(s["token_limited"] for s in summary)}
    require(all(s["questions"] == totals["questions"] for s in summary),
            "Runtime policies must share one question cohort")
    return {
        "scope": "completion-time success for one measured execution configuration",
        "measured_time": "selection, prefix replay and recovery generation for the attempt; "
                         "diagnosis acquisition is measured once per question and charged to "
                         "each of its attempts",
        "deadline_role": "attempts execute fully and a threshold is applied afterwards, so a "
                         "deadline classifies completions and never cancels work or shortens an attempt",
        "excluded": "deployed prefix caching, concurrency, repeated acquisition and matched "
                    "total-resource comparisons are outside this configuration",
        "policies": summary, "totals": totals}


def latency_table(scope):
    rows = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
            r"Policy & $n$ & Median & $q_{75}$ & $q_{90}$ & Max & Finished & Step-limited \\",
            r"\midrule"]
    for policy in scope["policies"]:
        rows.append(f"{policy['label']} & {policy['attempts']} & {policy['q50_s']:.2f} & "
                    f"{policy['q75_s']:.2f} & {policy['q90_s']:.2f} & {policy['max_s']:.2f} & "
                    f"{policy['finished']} & {policy['step_limited']} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def latency_figure(out, scope, rows):
    """Full measured latency distribution, not only the threshold crossings."""
    order = [p["strategy"] for p in scope["policies"]]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.4, 2.5), layout="constrained")
        for strategy, colour, style in zip(order, ["#2b6cb0", "#c05621", "#6b7280"],
                                           ["-", "--", ":"]):
            latency = np.sort(rows[rows.strategy == strategy].incremental_wall_latency_s.to_numpy())
            fraction = np.arange(1, len(latency) + 1) / len(latency)
            ax.step(latency, 100 * fraction, where="post", color=colour, linestyle=style,
                    linewidth=1.8, label=LABELS[strategy])
        ax.axvline(10, color="#8c8c8c", linewidth=.8, linestyle="--", zorder=0)
        ax.text(10.15, 8, "10 s primary", fontsize=8, color="#666666")
        ax.set(xlabel="Measured time per attempt (seconds)",
               ylabel="Attempts at or below (%)", xlim=(0, None), ylim=(0, 104))
        ax.grid(axis="y", color="#e4e7ea", linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=2.4)
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out) / "figures" / f"iclr2027_runtime_latency.{extension}", dpi=300)
        plt.close(fig)


def macros(scope):
    by_strategy = {p["strategy"]: p for p in scope["policies"]}
    values = {"RuntimeAttempts": scope["totals"]["attempts"],
              "RuntimeQuestions": scope["totals"]["questions"],
              "RuntimeStepLimited": scope["totals"]["step_limited"],
              "RuntimeTokenLimited": scope["totals"]["token_limited"]}
    for strategy, prefix in [(TREATMENT, "RuntimeUnc"), (RESTART, "RuntimeRestart"),
                             (DIAGNOSIS_STRATEGY, "RuntimeDiag")]:
        policy = by_strategy[strategy]
        values[prefix + "Median"] = f"{policy['q50_s']:.2f}"
        values[prefix + "Ninetieth"] = f"{policy['q90_s']:.2f}"
        values[prefix + "Finished"] = policy["finished"]
    return [r"\newcommand{\%s}{%s}" % item for item in values.items()]


def build(base=BASE, out=ROOT / "paper/generated"):
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _, rows = runtime_rows(base)
    scope = build_scope(base)
    (out / "tables/iclr2027_runtime_latency.tex").write_text(latency_table(scope))
    latency_figure(out, scope, rows)
    (out / "iclr2027_runtime_scope.tex").write_text(
        "% Generated by scripts/runtime_scope.py; descriptive within the measured setup.\n"
        + "\n".join(macros(scope)) + "\n")
    record = {**scope,
              "source_sha256": {"pooled-analysis-local.trials.csv":
                                file_fingerprint(Path(base) / "pooled-analysis-local.trials.csv")},
              "code_sha256": {"scripts/runtime_scope.py":
                              file_fingerprint(ROOT / "scripts/runtime_scope.py")}}
    (out / "iclr2027_runtime_scope.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    return scope, record


if __name__ == "__main__":
    scope, _ = build()
    print(json.dumps({"totals": scope["totals"],
                      "median_s": {p["label"]: round(p["q50_s"], 2) for p in scope["policies"]}},
                     indent=2))
