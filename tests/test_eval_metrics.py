import pandas as pd
import pytest

from src.eval.metrics import bootstrap_ci, summarize_strategies


def repair_rows(outcomes):
    return pd.DataFrame([
        {
            "qid": qid, "seed": seed, "strategy": "full_restart",
            "success": success, "recovery_gen_tokens": 10,
            "recovery_tool_calls": 1, "recovery_latency_s": 0.1,
            "targeted_oracle_match": 0,
        }
        for qid, seeds in outcomes.items()
        for seed, success in enumerate(seeds)
    ])


def test_summary_resamples_questions_not_seed_rows():
    df = repair_rows({"q1": [0, 0, 0], "q2": [1, 1, 1]})
    result = summarize_strategies(df, iters=2000).iloc[0]
    expected = bootstrap_ci([0, 1], iters=2000)
    assert result.success == expected[0]
    assert result.success_lo == expected[1]
    assert result.success_hi == expected[2]
    assert result.n_questions == 2
    assert result.n == 6


def test_summary_reports_mean_seed_success_not_majority_success():
    df = repair_rows({"q1": [1, 0, 0], "q2": [1, 1, 0]})
    result = summarize_strategies(df, iters=100).iloc[0]
    assert result.success == 0.5
    assert result.success_lo == pytest.approx(1 / 3, abs=0.0001)
    assert result.success_hi == pytest.approx(2 / 3, abs=0.0001)


def test_summary_weights_each_question_equally_with_missing_seed():
    df = repair_rows({"q1": [0], "q2": [1, 1, 1]})
    result = summarize_strategies(df, iters=100).iloc[0]
    assert result.success == 0.5


def test_summary_rejects_duplicate_trials():
    df = repair_rows({"q1": [1, 0, 0]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_strategies(df, iters=100)


def test_summary_rejects_pooled_budgets():
    df = repair_rows({"q1": [1, 0, 0]})
    df["multiplier"] = [0.5, 1.0, 2.0]
    with pytest.raises(ValueError, match="budget"):
        summarize_strategies(df, iters=100)
