"""Repair strategies (Stage 5).

Four ways to recover from a failed trajectory, all implemented as "resume the
ReAct loop from step k":

  full_restart          k = 0                      (re-execute everything)
  random_step           k = seeded random index    (lower-bound targeting)
  uncertainty_targeted  k = argmax uncertainty      (the practical method)
  oracle_targeted       k = annotated broken step   (upper bound)

Backtrack offsets (btN): after selecting step k, back up N additional steps to
k-N.  This tests the cascade hypothesis — if errors propagate forward, repairing
from further upstream should help.

Nudge types:
  generic   — "Your previous attempt may have been wrong. Reconsider carefully."
  informed  — error-type-specific hint from the judge annotation, e.g.
              "Your search query was likely wrong. Try different search terms."

Fairness controls (see design):
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

# Strategy name encoding:
#   "unc__<metric>__<rule>"                         base case
#   "unc__<metric>__<rule>__bt2"                    + backtrack
#   "unc__<metric>__<rule>__bt2__informed"           + backtrack + informed nudge
#   "oracle_targeted__bt1__informed"                oracle + backtrack + informed
# double underscore separates parts; metric/rule use single underscores.
_SEP = "__"
_BT_PREFIX = "bt"
_NUDGE_INFORMED = "informed"

# Error-type-specific nudge hints (keyed by annotation error_type)
INFORMED_NUDGES = {
    "wrong_search_query": (
        "Your previous search query was likely wrong or too vague. "
        "Try searching for a different entity or using more specific terms."
    ),
    "wrong_fact_extraction": (
        "You may have extracted the wrong fact from the search results. "
        "Re-read the passage carefully and look for the specific detail "
        "the question asks about."
    ),
    "faulty_reasoning": (
        "Your reasoning in the previous attempt had a logical error. "
        "Reconsider the relationships between the facts you have gathered "
        "and check your conclusion step by step."
    ),
    "premature_or_wrong_answer": (
        "Your previous answer was given too early or was incorrect. "
        "You may need more information before answering. Consider doing "
        "another search to verify your answer."
    ),
    "formatting_tool_error": (
        "Your previous attempt had a formatting or tool-use error. "
        "Make sure to use the exact format: Action: <tool>\\nAction Input: <argument>."
    ),
}
GENERIC_NUDGE_DEFAULT = "Your previous attempt at this step may have been wrong. Reconsider carefully."


def uncertainty_strategy_name(metric: str, rule: str,
                              backtrack: int = 0,
                              informed: bool = False) -> str:
    base = f"unc{_SEP}{metric}{_SEP}{rule}"
    if backtrack > 0:
        base += f"{_SEP}{_BT_PREFIX}{backtrack}"
    if informed:
        base += f"{_SEP}{_NUDGE_INFORMED}"
    return base


def parse_strategy(name: str) -> dict:
    """Parse any strategy name into its components.

    Returns dict with keys: base, metric, rule, backtrack, informed.
    """
    backtrack = 0
    informed = False

    # Check for informed suffix
    if name.endswith(_SEP + _NUDGE_INFORMED):
        informed = True
        name = name[:-(len(_SEP) + len(_NUDGE_INFORMED))]

    # Check for backtrack suffix
    if _SEP + _BT_PREFIX in name:
        parts = name.rsplit(_SEP, 1)
        bt_str = parts[1]
        if bt_str.startswith(_BT_PREFIX) and bt_str[len(_BT_PREFIX):].isdigit():
            backtrack = int(bt_str[len(_BT_PREFIX):])
            name = parts[0]

    if name.startswith("unc" + _SEP):
        _, metric, rule = name.split(_SEP)
        return {"base": "uncertainty", "metric": metric, "rule": rule,
                "backtrack": backtrack, "informed": informed}
    return {"base": name, "metric": None, "rule": None,
            "backtrack": backtrack, "informed": informed}


def parse_uncertainty_strategy(name: str) -> tuple[str, str]:
    """Return (metric_key, rule) from an 'unc__metric__rule[__btN][__informed]' name."""
    p = parse_strategy(name)
    return p["metric"], p["rule"]


def get_nudge_text(strategy: str, error_type: Optional[str],
                   generic_hint: str = GENERIC_NUDGE_DEFAULT) -> Optional[str]:
    """Return the appropriate nudge text for a strategy.

    - full_restart: no nudge (fresh start)
    - informed strategies: error-type-specific hint
    - all others: generic hint
    """
    p = parse_strategy(strategy)
    if p["base"] == "full_restart":
        return None
    if p["informed"] and error_type:
        return INFORMED_NUDGES.get(error_type, generic_hint)
    return generic_hint


def build_strategies(metric_keys: List[str], rule_keys: List[str],
                     backtrack_offsets: Optional[List[int]] = None,
                     nudge_types: Optional[List[str]] = None) -> List[str]:
    """Baselines + (metrics x rules) uncertainty strategies, optionally with
    backtrack variants and nudge type variants.

    backtrack_offsets=[0, 1, 2, 3] adds bt1/bt2/bt3 variants.
    nudge_types=["generic", "informed"] adds informed variants.
    """
    offsets = backtrack_offsets or [0]
    ntypes = nudge_types or ["generic"]
    strats = ["full_restart", "random_step"]
    # oracle + backtrack × nudge variants
    for bt in offsets:
        for nt in ntypes:
            suffix = ""
            if bt > 0:
                suffix += f"{_SEP}{_BT_PREFIX}{bt}"
            if nt == "informed":
                suffix += f"{_SEP}{_NUDGE_INFORMED}"
            strats.append(f"oracle_targeted{suffix}")
    # uncertainty strategies + backtrack × nudge variants
    for m in metric_keys:
        for r in rule_keys:
            for bt in offsets:
                for nt in ntypes:
                    strats.append(uncertainty_strategy_name(
                        m, r, bt, informed=(nt == "informed")))
    return strats


def select_target_step(strategy: str, n_steps: int, oracle_step: int,
                       uncertainty_traj: Optional[Dict[str, Any]],
                       rng: random.Random,
                       topk: int = 3, percentile: float = 75.0) -> int:
    """Return the step index k to resume FROM (keep steps[:k]).

    For an uncertainty strategy, the metric and rule are read from its name; a
    'miss' vs the oracle is fine — we repair from the rule's chosen step
    regardless. Clamped to [0, n_steps-1].

    Backtrack: if the strategy name ends with '__btN', the selected step is
    shifted upstream by N additional positions.
    """
    if n_steps <= 0:
        return 0
    p = parse_strategy(strategy)
    base = p["base"]
    backtrack = p["backtrack"]

    if base == "full_restart":
        k = 0
    elif base == "random_step":
        k = rng.randint(0, n_steps - 1)
    elif base == "oracle_targeted":
        k = oracle_step
    elif base == "uncertainty":
        scores = get_step_scores(uncertainty_traj, p["metric"]) if uncertainty_traj else []
        pred = localize_step(scores, p["rule"], k=topk, percentile=percentile)
        k = pred if pred is not None else 0
    else:
        raise ValueError(f"unknown strategy '{strategy}'")

    k = max(0, k - backtrack)
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
    p = parse_strategy(strategy)
    metric, rule = p["metric"], p["rule"]
    return {
        "qid": traj.qid,
        "strategy": strategy,
        "metric": metric,
        "rule": rule,
        "backtrack": p["backtrack"],
        "informed": p["informed"],
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
