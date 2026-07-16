"""Repair strategies (Stage 5).

Four ways to recover from a failed trajectory, all implemented as "resume the
ReAct loop from step k":

  full_restart          k = 0                      (re-execute everything)
  random_step           k = seeded random index    (lower-bound targeting)
  uncertainty_targeted  k = argmax uncertainty      (the practical method)
  oracle_targeted       k = annotated broken step   (upper bound)

Fairness controls (see design):
  * identical `nudge` (retry hint + temperature) applied to EVERY strategy;
  * matched compute budget: token cap = multiplier * original episode cost.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..agent.react_agent import ReActAgent, Step, Trajectory
from ..env.hotpot_env import HotpotEnv
from ..localize.rules import get_step_scores, localize_step
from ..utils.seed import derive_seed


BASELINES = ["full_restart", "random_step", "oracle_targeted"]

# An uncertainty strategy name encodes its metric + rule as:
#   "unc__<metric_key>__<rule>"   e.g. "unc__max_token_prob_max__argmax"
# (double underscore separates the three parts; metric/rule use single ones.)
_SEP = "__"


def uncertainty_strategy_name(metric: str, rule: str) -> str:
    return f"unc{_SEP}{metric}{_SEP}{rule}"


def parse_uncertainty_strategy(name: str) -> tuple[str, str]:
    """Return (metric_key, rule) from an 'unc__metric__rule' strategy name."""
    _, metric, rule = name.split(_SEP)
    return metric, rule


def build_strategies(metric_keys: List[str], rule_keys: List[str]) -> List[str]:
    """3 baselines + (metrics x rules) uncertainty strategies."""
    strats = list(BASELINES)
    for m in metric_keys:
        for r in rule_keys:
            strats.append(uncertainty_strategy_name(m, r))
    return strats


def select_target_step(strategy: str, n_steps: int, oracle_step: int,
                       uncertainty_traj: Optional[Dict[str, Any]],
                       rng: random.Random,
                       topk: int = 3, percentile: float = 75.0) -> int:
    """Return the step index k to resume FROM (keep steps[:k]).

    For an uncertainty strategy, the metric and rule are read from its name; a
    'miss' vs the oracle is fine — we repair from the rule's chosen step
    regardless. Clamped to [0, n_steps-1].
    """
    if n_steps <= 0:
        return 0
    if strategy == "full_restart":
        k = 0
    elif strategy == "random_step":
        k = rng.randint(0, n_steps - 1)
    elif strategy == "oracle_targeted":
        k = oracle_step
    elif strategy.startswith("unc" + _SEP):
        metric, rule = parse_uncertainty_strategy(strategy)
        scores = get_step_scores(uncertainty_traj, metric) if uncertainty_traj else []
        pred = localize_step(scores, rule, k=topk, percentile=percentile)
        k = pred if pred is not None else 0
    else:
        raise ValueError(f"unknown strategy '{strategy}'")
    return max(0, min(n_steps - 1, k))


def run_repair(agent: ReActAgent, record: Dict[str, Any],
               original_steps: List[Dict[str, Any]], target_step: int,
               strategy: str, seed: int, nudge_cfg: Dict[str, Any],
               token_budget: Optional[int]) -> Trajectory:
    """Execute one repair episode and return the resulting Trajectory."""
    prefix = [Step.from_dict(s) for s in original_steps[:target_step]]
    env = HotpotEnv(record=record)
    nudge = nudge_cfg.get("retry_hint") if nudge_cfg.get("enabled") else None
    temperature = nudge_cfg.get("temperature", 0.7) if nudge_cfg.get("enabled") else 0.0
    meta = {"strategy": strategy, "seed": seed, "target_step": target_step}
    return agent.run(env, temperature=temperature, seed=seed,
                     prefix_steps=prefix, nudge=nudge,
                     token_budget=token_budget, meta=meta)


def repair_record(traj: Trajectory, strategy: str, seed: int, target_step: int,
                  oracle_step: int, original_gen_tokens: int,
                  budget: Optional[int]) -> Dict[str, Any]:
    """Flatten one repair outcome into an analysis row."""
    m = traj.meta
    metric, rule = (parse_uncertainty_strategy(strategy)
                    if strategy.startswith("unc" + _SEP) else (None, None))
    return {
        "qid": traj.qid,
        "strategy": strategy,
        "metric": metric,
        "rule": rule,
        "seed": seed,
        "target_step": target_step,
        "oracle_step": oracle_step,
        "targeted_oracle_match": int(target_step == oracle_step),
        "success": int(traj.success),
        "em": traj.em,
        "f1": traj.f1,
        "final_answer": traj.final_answer,
        "terminated_reason": traj.terminated_reason,
        # recovery-only cost (excludes reused prefix) — the fair cost of recovery
        "recovery_gen_tokens": m.get("recovery_gen_tokens", traj.total_gen_tokens),
        "recovery_tool_calls": m.get("recovery_tool_calls", traj.num_tool_calls),
        "recovery_latency_s": m.get("recovery_latency_s", traj.total_latency_s),
        "n_prefix_steps": m.get("n_prefix_steps", target_step),
        "total_gen_tokens": traj.total_gen_tokens,
        "original_gen_tokens": original_gen_tokens,
        "budget": budget,
    }


def make_rng(qid: str, strategy: str, seed: int) -> random.Random:
    """Deterministic RNG for target selection (reproducible random_step)."""
    return random.Random(derive_seed(qid, strategy, seed))
