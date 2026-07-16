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


def localize_step(scores: List[Tuple[int, float]], rule: str,
                  k: int = 3, percentile: float = 75.0) -> Optional[int]:
    """Collapse a rule to a SINGLE step to repair from (used by Stage 5).

    - 'argmax'                  -> highest-uncertainty step
    - 'topk'                    -> EARLIEST step among the top-k (per user design:
                                   act on the earliest suspicious step)
    - 'earliest_above_threshold'-> earliest step over the percentile bar
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
    }
    for k in topk_list:
        res[f"top{k}_hit"] = int(oracle_step in rule_topk(scores, k))
    return res
