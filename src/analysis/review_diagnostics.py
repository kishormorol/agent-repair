"""Exploratory diagnostics for a completed, frozen repair study."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from src.env.hotpot_env import score_answer
from src.eval.metrics import bootstrap_ci, paired_mean_comparison
from src.localize.rules import get_step_scores


COST_FIELDS = ("recovery_gen_tokens", "recovery_prompt_tokens",
               "recovery_model_requests", "recovery_tool_calls")
SHARED_FIELDS = ("qid", "seed", "target_step", "prompt_sha256", "success", "em", "f1",
                 "final_answer", "terminated_reason", "budget", *COST_FIELDS)


def distribution(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "min": None, "q25": None, "median": None,
                "q75": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Distribution contains nonfinite values")
    return dict(n=len(values), mean=float(values.mean()),
                **dict(zip(("min", "q25", "median", "q75", "max"),
                           map(float, np.quantile(values, [0, .25, .5, .75, 1])))))


def mean_interval(values, iters):
    values = list(values)
    if not values:
        return {"n_questions": 0, "mean": None, "lo": None, "hi": None}
    mean, lo, hi = bootstrap_ci(values, iters=iters)
    return {"n_questions": len(values), "mean": mean, "lo": lo, "hi": hi}


def validate_trials(study, frame, originals, uncertainty):
    """Check fields needed by diagnostics in addition to the frozen-study loader."""
    required = {"strategy", "execution_id", "n_prefix_steps", "original_gen_tokens",
                "new_step_allowance", *SHARED_FIELDS}
    if not required <= set(frame):
        raise ValueError("Missing diagnostic trial fields")
    if set(originals) != set(study["question_ids"]):
        raise ValueError("Incomplete original-question coverage")
    failed = {q for q, t in originals.items() if not t["success"]}
    if not failed:
        raise ValueError("Diagnostics require at least one failed question")
    expected = {(q, a, r) for q in failed for a in study["strategies"] for r in study["seeds"]}
    keys = [tuple(row) for row in frame[["qid", "strategy", "seed"]].to_numpy()]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate diagnostic trial coverage")
    if set(uncertainty) != failed:
        raise ValueError("Incomplete uncertainty coverage")
    numeric = [*COST_FIELDS, "budget", "target_step", "original_gen_tokens", "n_prefix_steps", "new_step_allowance"]
    values = frame[numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Diagnostic costs and indices must be finite nonnegative integers")
    if frame[["execution_id", "prompt_sha256", "terminated_reason"]].isna().any().any():
        raise ValueError("Missing execution identity or termination reason")
    if not frame.terminated_reason.isin(["finished", "budget", "max_steps"]).all():
        raise ValueError("Unexpected termination reason")
    if frame.groupby("execution_id")[list(SHARED_FIELDS)].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Shared execution has inconsistent outcomes, origins or costs")
    if frame.groupby(["qid", "seed", "target_step"]).execution_id.nunique().gt(1).any():
        raise ValueError("Identical origin/seed has multiple executions in the matched-hint study")
    if not frame.loc[frame.strategy == "full_restart", "target_step"].eq(0).all():
        raise ValueError("Full restart must use origin zero")
    for q, original in originals.items():
        if original["qid"] != q or not original["steps"]:
            raise ValueError("Invalid original trajectory identity or length")
        score = score_answer(original["final_answer"], original["gold_answer"])
        if (bool(original["success"]) != score["correct"] or original["em"] != score["em"]
                or not math.isclose(original["f1"], score["f1"], abs_tol=1e-12)):
            raise ValueError("Original reference-gate scoring mismatch")
        if q in failed:
            profile = uncertainty[q]
            if (profile["qid"] != q or [s["index"] for s in profile["steps"]]
                    != list(range(len(original["steps"])))):
                raise ValueError("Uncertainty trajectory coverage mismatch")
    for row in frame.itertuples(index=False):
        original = originals[row.qid]
        score = score_answer(row.final_answer, original["gold_answer"])
        if (row.success != int(score["correct"]) or row.em != score["em"]
                or not math.isclose(row.f1, score["f1"], abs_tol=1e-12)):
            raise ValueError("Repair scoring mismatch")
        if (row.original_gen_tokens != original["total_gen_tokens"]
                or row.budget != int(study["multiplier"] * original["total_gen_tokens"])
                or row.target_step >= len(original["steps"])
                or row.n_prefix_steps != row.target_step
                or row.recovery_gen_tokens > row.budget
                or row.recovery_model_requests > row.new_step_allowance):
            raise ValueError("Repair origin or allowance mismatch")
    return failed


def summarize_diagnostics(study, frame, originals, uncertainty, iters=10000):
    failed = validate_trials(study, frame, originals, uncertainty)
    treatment = study["strategy"]
    n_questions = len(failed)
    policy_summaries = {}
    for name in study["strategies"]:
        rows = frame[frame.strategy == name].copy()
        means = rows.groupby("qid")[["success", "em", "f1", *COST_FIELDS]].mean()
        accepted = [t for t in originals.values() if t["success"]]
        overall = {}
        for metric in ["success", "em", "f1"]:
            values = [float(t[metric]) for t in accepted] + means[metric].tolist()
            overall[metric] = mean_interval(values, iters)
        policy_summaries[name] = {
            "n_questions": n_questions, "n_seed_rows": len(rows),
            "success": mean_interval(means.success, iters),
            "mean_cost_per_attempt": {k: float(means[k].mean()) for k in COST_FIELDS},
            "policy_trial_totals": {k: int(rows[k].sum()) for k in COST_FIELDS},
            "termination_counts": {k: int(rows.terminated_reason.eq(k).sum())
                                   for k in ["finished", "budget", "max_steps"]},
            "missing_final_answers": int(rows.final_answer.isna().sum()),
            "budget_stop_fraction": float(rows.terminated_reason.eq("budget").mean()),
            "token_cap_reached_fraction": float(rows.recovery_gen_tokens.eq(rows.budget).mean()),
            "zero_token_caps": int(rows.budget.eq(0).sum()),
            "token_cap_utilization": distribution(rows.loc[rows.budget.gt(0), "recovery_gen_tokens"]
                                                 / rows.loc[rows.budget.gt(0), "budget"]),
            "reference_gated_full_cohort": overall,
        }
    contrasts = {}
    a = frame[frame.strategy == treatment].set_index(["qid", "seed"])
    for name in study["primary_comparisons"]:
        b = frame[frame.strategy == name].set_index(["qid", "seed"]).reindex(a.index)
        same = a.execution_id.eq(b.execution_id)
        difference = a.success - b.success
        per_question = difference.groupby(level="qid").mean()
        paired = paired_mean_comparison(frame, treatment, name, iters=iters,
                                        expected_seeds=study["seeds"])
        variance = float(per_question.var(ddof=1)) if n_questions > 1 else None
        contrasts[name] = {
            "same_execution_seed_rows": int(same.sum()),
            "different_execution_seed_rows": int((~same).sum()),
            "same_execution_fraction": float(same.mean()),
            "questions_with_any_different_execution": int((~same).groupby(level="qid").any().sum()),
            "question_mean_wins": int((per_question > 1e-12).sum()),
            "question_mean_losses": int((per_question < -1e-12).sum()),
            "question_mean_ties": int((per_question.abs() <= 1e-12).sum()),
            "same_execution_contribution": float(difference.where(same, 0).groupby(level="qid").mean().mean()),
            "different_execution_contribution": float(difference.where(~same, 0).groupby(level="qid").mean().mean()),
            "paired_success": paired,
            "paired_cost_differences": {metric: mean_interval(
                (a[metric] - b[metric]).groupby(level="qid").mean(), iters) for metric in COST_FIELDS},
            "secondary_outcomes": {metric: paired_mean_comparison(frame, treatment, name,
                iters=iters, expected_seeds=study["seeds"], outcome=metric) for metric in ["em", "f1"]},
            "prospective_precision": {
                "observed_question_difference_variance": variance,
                "approximate_n_for_95pct_half_width_pp": {
                    str(width): max(2, math.ceil(1.96 ** 2 * variance / (width / 100) ** 2))
                    if variance is not None else None for width in [2, 5]},
                "note": "Normal-approximation planning from this cohort's variance; not observed power, simultaneous coverage or a guaranteed future interval width.",
            },
        }
    origins = a.target_step.groupby(level="qid")
    if origins.nunique().gt(1).any():
        raise ValueError("Treatment subgroup requires a deterministic origin across seeds")
    active = set(origins.first()[lambda s: s.gt(0)].index)
    subset = frame[frame.qid.isin(active)]
    active_effect = (paired_mean_comparison(subset, treatment, "full_restart", iters=iters,
                     expected_seeds=study["seeds"]) if active else None)
    unique = frame.drop_duplicates("execution_id")
    failure_traces = [originals[q] for q in sorted(failed)]
    signals = [get_step_scores(uncertainty[q], "perplexity") for q in sorted(failed)]
    question_means = frame.groupby(["qid", "strategy"]).success.mean().unstack("strategy")
    differences = question_means[treatment] - question_means.full_restart
    cases = []
    for label, mask in [("treatment_higher", differences.gt(1e-12)),
                        ("restart_higher", differences.lt(-1e-12)),
                        ("both_unsuccessful", question_means[treatment].eq(0) & question_means.full_restart.eq(0))]:
        ids = sorted(question_means.index[mask])
        if not ids:
            continue
        q = ids[0]
        original = originals[q]
        cases.append({"selection_stratum": label, "qid": q, "question": original["question"],
            "gold_answer": original["gold_answer"], "original_answer": original["final_answer"],
            "original_termination": original["terminated_reason"], "original_steps": len(original["steps"]),
            "token_allowance": original["total_gen_tokens"],
            "rows": frame[frame.qid.eq(q) & frame.strategy.isin([treatment, "full_restart"])]
                .sort_values(["strategy", "seed"])[["strategy", "seed", "target_step", "success",
                    "em", "f1", "final_answer", "terminated_reason", "recovery_gen_tokens"]].to_dict("records")})
    return {
        "analysis_status": "exploratory_review_stage", "n_initial_questions": len(originals),
        "n_failed_questions": n_questions, "n_strategy_seed_rows": len(frame),
        "policy_accounting": policy_summaries, "execution_overlap": contrasts,
        "descriptive_fixed_early": {metric: paired_mean_comparison(frame, treatment, "fixed_early",
            iters=iters, expected_seeds=study["seeds"], outcome=metric) for metric in ["success", "em", "f1"]},
        "nonzero_treatment_origin": {"n_questions": len(active), "question_fraction": len(active) / n_questions,
            "paired_success_versus_restart": active_effect,
            "note": "Exploratory subgroup defined by the frozen treatment origin on the original trace; does not estimate an origin-wide optimum."},
        "cohort": {"initial_successes": sum(bool(t["success"]) for t in originals.values()),
            "initial_exact_matches": sum(t["em"] for t in originals.values()),
            "accepted_without_exact_match": sum(t["success"] and not t["em"] for t in originals.values()),
            "original_failure_steps": distribution(len(t["steps"]) for t in failure_traces),
            "original_failure_token_allowance": distribution(t["total_gen_tokens"] for t in failure_traces),
            "original_failure_termination": dict(Counter(t["terminated_reason"] for t in failure_traces))},
        "signal_quality": {"profiles": len(signals), "empty_usable_profiles": sum(not s for s in signals),
            "total_original_failure_steps": sum(len(t["steps"]) for t in failure_traces),
            "usable_perplexity_steps": sum(map(len, signals))},
        "unique_study_usage": {"executions": len(unique), **{k: int(unique[k].sum()) for k in COST_FIELDS}},
        "illustrative_cases": cases,
        "case_selection": "First sorted question ID per outcome stratum, all three seeds shown; outcome-selected illustrations, not failure-type prevalence or human validation.",
    }
