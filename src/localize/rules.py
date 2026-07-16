"""Error localization (Stage 4).

Given a trajectory's per-step uncertainty for one metric key, predict the broken
step via three rules and score against the oracle label.

Convention: every uncertainty metric is 'higher = more uncertain', so the most
suspicious step is always the argmax.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def get_step_scores(traj_unc: Dict[str, Any], metric_key: str) -> List[Tuple[int, float]]:
    """Return [(step_index, score)] for steps that have a finite value of
    `metric_key`. Steps missing the metric (e.g. sampling metrics not computed)
    are skipped."""
    out = []
    for s in traj_unc["steps"]:
        v = s.get("uncertainty", {}).get(metric_key)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out.append((s["index"], float(v)))
    return out


def _ranked(scores: List[Tuple[int, float]]) -> List[int]:
    """Step indices sorted by score desc (ties broken by earlier index)."""
    return [idx for idx, _ in sorted(scores, key=lambda x: (-x[1], x[0]))]


def rule_argmax(scores: List[Tuple[int, float]]) -> Optional[int]:
    if not scores:
        return None
    return _ranked(scores)[0]


def rule_topk(scores: List[Tuple[int, float]], k: int) -> List[int]:
    return _ranked(scores)[:k]


def rule_earliest_above_threshold(scores: List[Tuple[int, float]],
                                  percentile: float) -> Optional[int]:
    """Earliest step whose score is >= the given within-trajectory percentile."""
    if not scores:
        return None
    vals = sorted(v for _, v in scores)
    # percentile threshold (linear interpolation)
    rank = (percentile / 100.0) * (len(vals) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    thr = vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)
    for idx, v in sorted(scores, key=lambda x: x[0]):   # earliest first
        if v >= thr:
            return idx
    return None


def mrr(scores: List[Tuple[int, float]], oracle_step: int) -> float:
    """Reciprocal rank of the oracle step in the descending uncertainty order."""
    ranking = _ranked(scores)
    if oracle_step not in ranking:
        return 0.0
    return 1.0 / (ranking.index(oracle_step) + 1)


def rule_cascade_upstream(scores: List[Tuple[int, float]],
                         lookback: int = 2) -> Optional[int]:
    """Cascade-aware: find the uncertainty peak, then go `lookback` steps
    upstream.  Rationale: errors cascade forward, so by the time uncertainty
    peaks the damage started earlier.  Clamped to step 0."""
    if not scores:
        return None
    peak = rule_argmax(scores)
    if peak is None:
        return None
    step_indices = sorted(idx for idx, _ in scores)
    peak_pos = step_indices.index(peak) if peak in step_indices else 0
    target_pos = max(0, peak_pos - lookback)
    return step_indices[target_pos]


def rule_cascade_gradient(scores: List[Tuple[int, float]]) -> Optional[int]:
    """Cascade-aware: find the step with the largest uncertainty *increase*
    from the previous step.  A sudden jump in uncertainty suggests that step
    is where the reasoning first went wrong, before the error cascades into
    higher uncertainty downstream."""
    if not scores or len(scores) < 2:
        return rule_argmax(scores)
    ordered = sorted(scores, key=lambda x: x[0])
    best_step, best_delta = ordered[0][0], -float("inf")
    for i in range(1, len(ordered)):
        delta = ordered[i][1] - ordered[i - 1][1]
        if delta > best_delta:
            best_delta = delta
            best_step = ordered[i][0]
    return best_step


def rule_cascade_weighted(scores: List[Tuple[int, float]],
                          position_weight: float = 0.5) -> Optional[int]:
    """Cascade-aware: score each step as a blend of its uncertainty and how
    early it is.  Earlier steps get a bonus because errors cascade forward.

    composite = (1 - w) * norm_uncertainty + w * (1 - norm_position)

    where norm_uncertainty and norm_position are min-max normalized to [0, 1].
    Returns the step with the highest composite score.
    """
    if not scores:
        return None
    if len(scores) == 1:
        return scores[0][0]
    ordered = sorted(scores, key=lambda x: x[0])
    indices = [idx for idx, _ in ordered]
    vals = [v for _, v in ordered]
    v_min, v_max = min(vals), max(vals)
    v_range = v_max - v_min if v_max > v_min else 1.0
    i_min, i_max = min(indices), max(indices)
    i_range = i_max - i_min if i_max > i_min else 1.0

    best_step, best_score = ordered[0][0], -float("inf")
    w = position_weight
    for idx, v in ordered:
        norm_unc = (v - v_min) / v_range
        norm_pos = (idx - i_min) / i_range
        composite = (1.0 - w) * norm_unc + w * (1.0 - norm_pos)
        if composite > best_score:
            best_score = composite
            best_step = idx
    return best_step


def localize_step(scores: List[Tuple[int, float]], rule: str,
                  k: int = 3, percentile: float = 75.0) -> Optional[int]:
    """Collapse a rule to a SINGLE step to repair from (used by Stage 5).

    Standard rules:
    - 'argmax'                   -> highest-uncertainty step
    - 'topk'                     -> EARLIEST step among the top-k
    - 'earliest_above_threshold' -> earliest step over the percentile bar

    Cascade-aware rules (address the upstream error problem):
    - 'cascade_upstream'         -> go 2 steps before the uncertainty peak
    - 'cascade_gradient'         -> step with largest uncertainty jump
    - 'cascade_weighted'         -> blend of uncertainty + early-position bonus
    """
    if not scores:
        return None
    if rule == "argmax":
        return rule_argmax(scores)
    if rule == "topk":
        topset = rule_topk(scores, k)
        return min(topset) if topset else None      # earliest index in the set
    if rule == "earliest_above_threshold":
        return rule_earliest_above_threshold(scores, percentile)
    if rule == "cascade_upstream":
        return rule_cascade_upstream(scores, lookback=2)
    if rule == "cascade_gradient":
        return rule_cascade_gradient(scores)
    if rule == "cascade_weighted":
        return rule_cascade_weighted(scores, position_weight=0.5)
    raise ValueError(f"unknown localization rule '{rule}'")


def evaluate_localization(traj_unc: Dict[str, Any], oracle_step: int,
                          metric_key: str, topk_list: List[int],
                          threshold_percentile: float) -> Dict[str, Any]:
    """Score all rules for one trajectory + metric against the oracle step."""
    scores = get_step_scores(traj_unc, metric_key)
    if not scores:
        return {"has_scores": False}

    pred_argmax = rule_argmax(scores)
    pred_thr = rule_earliest_above_threshold(scores, threshold_percentile)
    pred_upstream = rule_cascade_upstream(scores, lookback=2)
    pred_gradient = rule_cascade_gradient(scores)
    pred_weighted = rule_cascade_weighted(scores, position_weight=0.5)

    res: Dict[str, Any] = {
        "has_scores": True,
        "n_steps": len(scores),
        "oracle_step": oracle_step,
        "pred_argmax": pred_argmax,
        "argmax_top1": int(pred_argmax == oracle_step),
        "argmax_within1": int(abs(pred_argmax - oracle_step) <= 1),
        "mrr": mrr(scores, oracle_step),
        "pred_threshold": pred_thr,
        "threshold_hit": int(pred_thr == oracle_step),
        "threshold_within1": int(pred_thr is not None and abs(pred_thr - oracle_step) <= 1),
        # cascade-aware rules
        "pred_cascade_upstream": pred_upstream,
        "cascade_upstream_hit": int(pred_upstream == oracle_step),
        "cascade_upstream_within1": int(pred_upstream is not None and abs(pred_upstream - oracle_step) <= 1),
        "pred_cascade_gradient": pred_gradient,
        "cascade_gradient_hit": int(pred_gradient == oracle_step),
        "cascade_gradient_within1": int(pred_gradient is not None and abs(pred_gradient - oracle_step) <= 1),
        "pred_cascade_weighted": pred_weighted,
        "cascade_weighted_hit": int(pred_weighted == oracle_step),
        "cascade_weighted_within1": int(pred_weighted is not None and abs(pred_weighted - oracle_step) <= 1),
    }
    for k in topk_list:
        res[f"top{k}_hit"] = int(oracle_step in rule_topk(scores, k))
    return res
