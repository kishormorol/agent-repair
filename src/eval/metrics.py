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
    """Mean seed success per question, with a question-level bootstrap CI.

    Seeds on the same failed trajectory are repeated measurements, not
    independent questions. Keep them together when estimating uncertainty.
    Callers must select one dataset, model, and budget before summarizing.
    """
    for column in ("dataset", "model_key", "multiplier"):
        if column in df and df[column].nunique(dropna=False) > 1:
            raise ValueError("Select one dataset, model, and budget before summarizing")
    if df.duplicated(["qid", "strategy", "seed"]).any():
        raise ValueError("Found duplicate (qid, strategy, seed) repair trials")
    rows = []
    for strat, g in df.groupby("strategy"):
        per_question = g.groupby("qid")[[
            "success", "recovery_gen_tokens", "recovery_tool_calls",
            "recovery_latency_s", "targeted_oracle_match",
        ]].mean()
        mean, lo, hi = bootstrap_ci(per_question["success"].tolist(), iters, ci)
        tok = per_question["recovery_gen_tokens"].mean()
        rows.append({
            "strategy": strat,
            "n": len(g),
            "n_questions": len(per_question),
            "success": round(mean, 4),
            "success_lo": round(lo, 4),
            "success_hi": round(hi, 4),
            "recovery_gen_tokens": round(tok, 1),
            "recovery_tool_calls": round(per_question["recovery_tool_calls"].mean(), 3),
            "recovery_latency_s": round(per_question["recovery_latency_s"].mean(), 3),
            "success_per_1k_tokens": round(1000 * mean / tok, 4) if tok > 0 else float("nan"),
            "targeted_oracle_match": round(per_question["targeted_oracle_match"].mean(), 4),
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


def _paired_question_values(df, a, b, outcome, expected_seeds=None):
    required = {"qid", "strategy", "seed", outcome}
    if not required <= set(df.columns) or a == b:
        raise ValueError("Paired analysis requires two strategies and complete trial columns")
    sub = df[df.strategy.isin([a, b])].copy()
    if sub.empty or sub[list(required)].isna().any().any():
        raise ValueError("Paired trials contain missing values")
    for column in ("dataset", "model_key", "model_name", "multiplier", "run_id"):
        if column in sub and sub[column].nunique(dropna=False) > 1:
            raise ValueError("Select one dataset, model, run, and budget for paired analysis")
    if sub.duplicated(["qid", "strategy", "seed"]).any():
        raise ValueError("Duplicate paired trial")
    values = pd.to_numeric(sub[outcome], errors="raise")
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError("Outcomes must be finite values in [0, 1]")
    if outcome in ("success", "em") and not values.isin([0, 1]).all():
        raise ValueError("Success and exact match must be binary")
    sub[outcome] = values
    seeds = set(sub.seed) if expected_seeds is None else set(expected_seeds)
    if not seeds or set(sub.seed) != seeds:
        raise ValueError("Observed seeds differ from the prespecified seeds")
    coverage = sub.groupby(["qid", "strategy"]).seed.agg(set)
    if len(coverage) != 2 * sub.qid.nunique() or any(s != seeds for s in coverage):
        raise ValueError("Incomplete paired question/strategy/seed coverage")
    means = sub.groupby(["qid", "strategy"])[outcome].mean().unstack("strategy")
    if not {a, b} <= set(means.columns) or means[[a, b]].isna().any().any():
        raise ValueError("Incomplete strategy coverage")
    return means[a].to_numpy(), means[b].to_numpy()


def _paired_inference(groups, iters, ci, seed):
    if not isinstance(iters, int) or iters < 1 or not 0 < ci < 1:
        raise ValueError("Require positive bootstrap iterations and ci in (0, 1)")
    deltas = [a - b for a, b in groups]
    estimate = float(np.mean([d.mean() for d in deltas]))
    rng = np.random.default_rng(seed)
    boot, permuted = np.zeros(iters), np.zeros(iters)
    # Resample paired questions within datasets, with equal dataset weight.
    for d in deltas:
        for start in range(0, iters, 256):
            count = min(256, iters - start)
            boot[start:start + count] += d[rng.integers(0, len(d), (count, len(d)))].mean(axis=1)
            signs = rng.choice([-1, 1], size=(count, len(d)))
            permuted[start:start + count] += (signs * d).mean(axis=1)
    boot /= len(deltas)
    permuted /= len(deltas)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, (1 + ci) / 2])
    return {
        "estimand": "equal_question_mean_seed_success" if len(groups) == 1 else
                    "equal_dataset_macro_mean_of_question_seed_means",
        "n_questions": sum(len(a) for a, _ in groups),
        "mean_a": float(np.mean([a.mean() for a, _ in groups])),
        "mean_b": float(np.mean([b.mean() for _, b in groups])),
        "delta": estimate, "delta_lo": float(lo), "delta_hi": float(hi),
        "ci": ci, "resamples": iters, "analysis_seed": seed,
        "p_value": float((1 + (np.abs(permuted) >= abs(estimate) - 1e-12).sum()) / (iters + 1)),
        "test": "paired_sign_flip_assuming_within_question_exchangeability",
    }


def paired_mean_comparison(df: pd.DataFrame, a: str, b: str, iters=10000,
                           ci=0.95, seed=0, expected_seeds=None,
                           outcome="success") -> Dict[str, Any]:
    """Paired question bootstrap and sign-flip test, without seed-majority voting."""
    groups = [_paired_question_values(df, a, b, outcome, expected_seeds)]
    result = _paired_inference(groups, iters, ci, seed)
    result.update(strategy_a=a, strategy_b=b, outcome=outcome)
    return result


def paired_macro_comparison(df: pd.DataFrame, a: str, b: str, iters=10000,
                            ci=0.95, seed=0, expected_seeds=None,
                            outcome="success") -> Dict[str, Any]:
    """Equal-dataset macro comparison; preserve dataset and question clusters."""
    if "dataset" not in df or df.dataset.isna().any() or df.empty:
        raise ValueError("Macro comparison requires dataset labels")
    for column in ("model_key", "model_name", "multiplier", "run_id"):
        if column in df and df[column].nunique(dropna=False) > 1:
            raise ValueError("Select one model, run, and budget for macro analysis")
    groups = [_paired_question_values(g, a, b, outcome, expected_seeds)
              for _, g in df.groupby("dataset", sort=True)]
    result = _paired_inference(groups, iters, ci, seed)
    result.update(strategy_a=a, strategy_b=b, outcome=outcome,
                  datasets=sorted(df.dataset.unique()))
    return result


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
        r["estimand"] = "majority_over_seeds_repeatability_secondary"
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
