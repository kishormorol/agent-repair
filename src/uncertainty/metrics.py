"""Step-level uncertainty metrics.

Given a step's generated-token logprobs (stored during Stage 1), compute:

MATHEMATICAL (free, from logprobs):
  * token_entropy   — entropy over the top-k next-token distribution per token,
                      aggregated (mean/max) across the step's tokens.
  * perplexity      — exp(mean surprisal) of the step's sampled tokens.
  * max_token_prob  — uncertainty = 1 - p(sampled token), aggregated (mean/max).

SAMPLING-BASED (need extra model calls — driven by the Stage-2 notebook):
  * self_consistency      — disagreement among n resampled actions for the step.
  * verbalized_confidence — model rates its own confidence 0..100.

This module provides the math computations + scoring helpers for the sampling
metrics (prompt builder / parsers). The notebook performs the resampling.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Low-level helpers over one token's top-k logprobs
# --------------------------------------------------------------------------- #
def _entropy_from_logprobs(top_logprobs: Dict[int, float], topk: Optional[int] = None) -> float:
    """Entropy (nats) over the top-k next-token distribution, renormalized to
    sum to 1 over the available top-k entries."""
    if not top_logprobs:
        return 0.0
    items = sorted(top_logprobs.values(), reverse=True)
    if topk is not None:
        items = items[:topk]
    probs = [math.exp(lp) for lp in items]
    z = sum(probs)
    if z <= 0:
        return 0.0
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _agg(values: List[float], how: str) -> float:
    if not values:
        return float("nan")
    if how == "mean":
        return sum(values) / len(values)
    if how == "max":
        return max(values)
    raise ValueError(f"unknown agg '{how}'")


# --------------------------------------------------------------------------- #
# Mathematical metrics for one step
# --------------------------------------------------------------------------- #
def compute_math_metrics(tokens: List[Dict[str, Any]],
                         entropy_topk: Optional[int] = None) -> Dict[str, float]:
    """Compute the three mathematical uncertainty metrics for a step.

    Args:
        tokens: list of token dicts (from Step.generation.tokens[*].to_dict()),
                each with 'logprob' (float) and 'top_logprobs' (dict).
        entropy_topk: cap the top-k used for entropy (for the top-k ablation).

    Returns dict with *_mean and *_max variants plus scalar perplexity.
    """
    if not tokens:
        return {}

    # per-token arrays
    entropies, surprisals, one_minus_p = [], [], []
    for t in tokens:
        top = {int(k): float(v) for k, v in t.get("top_logprobs", {}).items()}
        entropies.append(_entropy_from_logprobs(top, entropy_topk))
        lp = t.get("logprob")
        if lp is None or (isinstance(lp, float) and math.isnan(lp)):
            # sampled token missing from top-k: approximate with the smallest
            # returned logprob (a lower bound on its probability).
            lp = min(top.values()) if top else -20.0
        surprisals.append(-lp)
        one_minus_p.append(1.0 - math.exp(lp))

    mean_surprisal = _agg(surprisals, "mean")
    return {
        "token_entropy_mean": _agg(entropies, "mean"),
        "token_entropy_max": _agg(entropies, "max"),
        "max_token_prob_mean": _agg(one_minus_p, "mean"),   # 1 - p(sampled)
        "max_token_prob_max": _agg(one_minus_p, "max"),
        "perplexity": math.exp(min(mean_surprisal, 20.0)),   # cap to avoid overflow
        "surprisal_max": _agg(surprisals, "max"),
    }


# --------------------------------------------------------------------------- #
# Self-consistency
# --------------------------------------------------------------------------- #
def _norm_action(action: str, action_input: str) -> str:
    a = (action or "").strip().lower()
    ai = re.sub(r"\s+", " ", (action_input or "").strip().lower())
    return f"{a}::{ai}"


def self_consistency_score(samples: List[tuple[str, str]]) -> float:
    """Uncertainty in [0,1] = 1 - (fraction of samples matching the mode action).

    `samples` is a list of (action, action_input) from n resamples of one step.
    All identical -> 0 (certain). All different -> ->1 (uncertain).
    """
    if not samples:
        return float("nan")
    keys = [_norm_action(a, ai) for a, ai in samples]
    mode_count = Counter(keys).most_common(1)[0][1]
    return 1.0 - mode_count / len(keys)


# --------------------------------------------------------------------------- #
# Verbalized confidence
# --------------------------------------------------------------------------- #
def build_confidence_prompt(question: str, scratchpad: str,
                            thought: str, action: str, action_input: str) -> List[Dict[str, str]]:
    """Chat messages asking the model to rate confidence 0..100 in this step."""
    sys = ("You assess your own confidence. Given a question, the reasoning so "
           "far, and a proposed next step, output ONLY an integer 0-100 where 100 "
           "means fully confident the step is correct and useful.")
    user = (f"Question: {question}\n\nReasoning so far:\n{scratchpad}\n\n"
            f"Proposed next step:\nThought: {thought}\nAction: {action}\n"
            f"Action Input: {action_input}\n\nConfidence (0-100):")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_confidence(text: str, scale: int = 100) -> float:
    """Extract the confidence integer; return uncertainty = 1 - conf/scale."""
    m = re.search(r"\d{1,3}", text or "")
    if not m:
        return float("nan")
    conf = max(0, min(scale, int(m.group())))
    return 1.0 - conf / scale


# --------------------------------------------------------------------------- #
# Trajectory-level driver for the mathematical metrics
# --------------------------------------------------------------------------- #
def annotate_trajectory_math(traj: Dict[str, Any],
                             entropy_topk: Optional[int] = None) -> Dict[str, Any]:
    """Attach an 'uncertainty' dict (math metrics only) to each step in a
    trajectory dict (in place) and return it."""
    for step in traj.get("steps", []):
        gen = step.get("generation") or {}
        tokens = gen.get("tokens", [])
        step.setdefault("uncertainty", {})
        step["uncertainty"].update(compute_math_metrics(tokens, entropy_topk))
    return traj
