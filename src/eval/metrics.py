"""Evaluation & statistics (Stage 6).

Aggregates the Stage-5 repair rows into the main results table, computes
bootstrap CIs and cost-normalized success, and runs the paired significance
tests behind the three research questions.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: List[float], iters: int = 10000, ci: float = 0.95,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                     dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return (float(arr.mean()), lo, hi)


# --------------------------------------------------------------------------- #
# Per-strategy summary
# --------------------------------------------------------------------------- #
def summarize_strategies(df: pd.DataFrame, iters: int = 10000,
                         ci: float = 0.95) -> pd.DataFrame:
    """One row per strategy: success (+CI), recovery cost, cost-normalized success."""
    rows = []
    for strat, g in df.groupby("strategy"):
        mean, lo, hi = bootstrap_ci(g["success"].tolist(), iters, ci)
        tok = g["recovery_gen_tokens"].mean()
        rows.append({
            "strategy": strat,
            "n": len(g),
            "success": round(mean, 4),
            "success_lo": round(lo, 4),
            "success_hi": round(hi, 4),
            "recovery_gen_tokens": round(tok, 1),
            "recovery_tool_calls": round(g["recovery_tool_calls"].mean(), 3),
            "recovery_latency_s": round(g["recovery_latency_s"].mean(), 3),
            "success_per_1k_tokens": round(1000 * mean / tok, 4) if tok > 0 else float("nan"),
            "targeted_oracle_match": round(g["targeted_oracle_match"].mean(), 4),
        })
    # order: random, full_restart, then uncertainty strategies (sorted), oracle last
    def rank(s: str) -> tuple:
        if s == "random_step": return (0, s)
        if s == "full_restart": return (1, s)
        if s == "oracle_targeted": return (3, s)
        if s.startswith("uncertainty_ensemble"): return (2.5, s)
        return (2, s)   # unc__* strategies
    out = pd.DataFrame(rows)
    out["__o"] = out["strategy"].apply(rank)
    return out.sort_values("__o").drop(columns="__o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-qid success (aggregate seeds -> binary) for paired tests
# --------------------------------------------------------------------------- #
def per_qid_success(df: pd.DataFrame, strategy: str) -> Dict[str, int]:
    """success for a strategy per qid = majority vote over seeds (>=0.5 -> 1)."""
    g = df[df["strategy"] == strategy].groupby("qid")["success"].mean()
    return {qid: int(v >= 0.5) for qid, v in g.items()}


# --------------------------------------------------------------------------- #
# McNemar paired test
# --------------------------------------------------------------------------- #
def mcnemar(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, Any]:
    """Paired McNemar test between two strategies' per-qid success dicts.

    b_only = a wrong & b right ; a_only = a right & b wrong.
    Returns discordant counts, delta (mean_a - mean_b), and p-value.
    """
    from scipy import stats
    qids = sorted(set(a) & set(b))
    a_only = b_only = 0
    for q in qids:
        if a[q] == 1 and b[q] == 0:
            a_only += 1
        elif a[q] == 0 and b[q] == 1:
            b_only += 1
    n = a_only + b_only
    mean_a = np.mean([a[q] for q in qids]) if qids else float("nan")
    mean_b = np.mean([b[q] for q in qids]) if qids else float("nan")
    if n == 0:
        p = 1.0
    elif n < 25:  # exact binomial
        k = min(a_only, b_only)
        p = float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))
    else:         # chi-square with continuity correction
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n
        p = float(stats.chi2.sf(chi2, df=1))
    return {"n_pairs": len(qids), "a_only": a_only, "b_only": b_only,
            "mean_a": round(float(mean_a), 4), "mean_b": round(float(mean_b), 4),
            "delta": round(float(mean_a - mean_b), 4), "p_value": p}


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def rq_comparisons(df: pd.DataFrame) -> Dict[str, Any]:
    """Paired RQ comparisons with Holm-corrected p-values.

    RQ1: oracle vs full_restart.
    RQ2: EACH uncertainty rule vs oracle (cost of imperfect localization).
    RQ3: EACH uncertainty rule vs random (is the rule better than luck).
    """
    strategies = set(df["strategy"].unique())
    succ = {s: per_qid_success(df, s) for s in strategies}
    unc_rules = sorted(s for s in strategies if s.startswith("unc"))

    pairs: Dict[str, tuple] = {}
    if "oracle_targeted" in strategies and "full_restart" in strategies:
        pairs["RQ1_oracle_vs_restart"] = ("oracle_targeted", "full_restart")
    for u in unc_rules:
        if "oracle_targeted" in strategies:
            pairs[f"RQ2_{u}_vs_oracle"] = (u, "oracle_targeted")
        if "random_step" in strategies:
            pairs[f"RQ3_{u}_vs_random"] = (u, "random_step")

    results, pvals, keys = {}, [], []
    for name, (a, b) in pairs.items():
        r = mcnemar(succ[a], succ[b])
        results[name] = r
        pvals.append(r["p_value"]); keys.append(name)
    adj = holm_correction(pvals)
    for name, p in zip(keys, adj):
        results[name]["p_value_holm"] = round(p, 5)
        results[name]["significant_0.05"] = bool(p < 0.05)
    return results


def ensemble_rows(df: pd.DataFrame, member_strategies: List[str],
                  name: str = "uncertainty_ensemble_any") -> pd.DataFrame:
    """Build pseudo-strategy rows: per (qid, seed), success = ANY member
    succeeded; cost = SUM of members' recovery cost (running all of them).

    Lets the ensemble be summarized/compared like any other strategy.
    """
    sub = df[df["strategy"].isin(member_strategies)]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby(["qid", "seed"]).agg(
        success=("success", "max"),
        recovery_gen_tokens=("recovery_gen_tokens", "sum"),
        recovery_tool_calls=("recovery_tool_calls", "sum"),
        recovery_latency_s=("recovery_latency_s", "sum"),
        targeted_oracle_match=("targeted_oracle_match", "max"),
    ).reset_index()
    agg["strategy"] = name
    agg["multiplier"] = 1.0
    return agg


def pareto_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Cost (recovery tokens) vs success per strategy, for the Pareto plot."""
    return summary[["strategy", "recovery_gen_tokens", "success",
                    "success_lo", "success_hi"]].copy()
