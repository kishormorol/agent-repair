import itertools

import pandas as pd
import pytest

from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_pairs import (
    DEV_CONTROL, POLICIES, SWAP, block_comparison, exact_block_sign_flip,
    pair_by_length, validate_exact_balance,
)

SEEDS = [0, 1]


def _row(qid, seed, strategy, origin, pair, successes):
    budget = 100 * pair["n_steps"]
    return {"qid": qid, "seed": seed, "strategy": strategy, "origin": origin,
            "pair_id": pair["pair_id"], "original_n_steps": pair["n_steps"], "budget": budget,
            "execution_id": fingerprint([qid, seed, origin, budget]),
            "prompt_sha256": fingerprint([qid, origin]),
            "success": successes[(qid, seed, strategy)]}


def build_frame(pairing, origins, successes, seeds=SEEDS):
    """Lay out one exactly swapped frame; origins maps qid to its own origin."""
    rows = []
    for pair in pairing["pairs"]:
        left, right = pair["qids"]
        for qid, partner in [(left, right), (right, left)]:
            for seed in seeds:
                rows.append(_row(qid, seed, TREATMENT, origins[qid], pair, successes))
                rows.append(_row(qid, seed, SWAP, origins[partner], pair, successes))
                rows.append(_row(qid, seed, DEV_CONTROL, 0, pair, successes))
    return pd.DataFrame(rows)


@pytest.fixture
def pairing():
    return pair_by_length({"a": 4, "b": 4, "c": 4, "d": 6, "e": 6, "f": 3}, seed=20260916)


@pytest.fixture
def fixed_successes():
    """Give the treatment one extra success per pair on the first seed."""
    values = {}
    for qid, seed, strategy in itertools.product("abcdef", SEEDS, POLICIES):
        values[(qid, seed, strategy)] = strategy == TREATMENT and seed == 0
    return values


def test_pairing_matches_exact_lengths_and_reports_the_odd_remainder(pairing):
    assert pairing["n_failures"] == 6 and pairing["n_matched_questions"] == 4
    assert pairing["unmatched_ids"] in (["c", "f"], ["a", "f"], ["b", "f"])
    assert {p["n_steps"] for p in pairing["pairs"]} == {4, 6}
    for pair in pairing["pairs"]:
        assert len(pair["qids"]) == 2 and len(set(pair["qids"])) == 2
    assert pairing == pair_by_length({"f": 3, "e": 6, "d": 6, "c": 4, "b": 4, "a": 4}, seed=20260916)


def test_pairing_ignores_outcomes_and_rejects_malformed_lengths():
    assert sorted(pair_by_length({"a": 4, "b": 4}, seed=1)["pairs"][0]["qids"]) == ["a", "b"]
    for lengths in [{"a": 0}, {"a": -1}, {"": 3}, {"a": True}, {"a": 3.0}, {3: 3}]:
        with pytest.raises(ValueError, match="positive integer lengths"):
            pair_by_length(lengths, seed=1)


def test_balance_audit_accepts_an_exact_swap_and_counts_shared_executions(pairing, fixed_successes):
    origins = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 0, "f": 0}
    audit = validate_exact_balance(build_frame(pairing, origins, fixed_successes), pairing, SEEDS)
    assert audit["paired_attempts"] == 8 and audit["matched_questions"] == 4
    assert audit["raw_origin_distribution_equal"] and audit["normalized_origin_distribution_equal"]
    # Only the equal-length pair whose members share an origin reuses executions.
    shared = sum(2 * len(SEEDS) for p in pairing["pairs"] if origins[p["qids"][0]] == origins[p["qids"][1]])
    assert audit["shared_execution_pairs"] == shared
    assert audit["shared_execution_fraction"] == shared / 8


def test_balance_audit_rejects_incomplete_coverage_and_inexact_swaps(pairing, fixed_successes):
    origins = {"a": 1, "b": 2, "c": 2, "d": 3, "e": 0, "f": 0}
    frame = build_frame(pairing, origins, fixed_successes)
    with pytest.raises(ValueError, match="Incomplete paired-origin"):
        validate_exact_balance(frame.iloc[1:], pairing, SEEDS)
    with pytest.raises(ValueError, match="Incomplete paired-origin"):
        validate_exact_balance(pd.concat([frame, frame.iloc[:1]]), pairing, SEEDS)
    broken = frame.copy()
    swap = (broken.strategy == SWAP) & (broken.qid == pairing["pairs"][0]["qids"][0])
    broken.loc[swap, "origin"] = 7
    with pytest.raises(ValueError, match="exchange the paired questions' origins"):
        validate_exact_balance(broken, pairing, SEEDS)


def test_balance_audit_requires_execution_sharing_to_track_the_origin(pairing, fixed_successes):
    origins = {"a": 1, "b": 2, "c": 2, "d": 3, "e": 0, "f": 0}
    frame = build_frame(pairing, origins, fixed_successes)
    pair = next(p for p in pairing["pairs"] if origins[p["qids"][0]] != origins[p["qids"][1]])
    qid = pair["qids"][0]
    swapped = (frame.strategy == SWAP) & (frame.qid == qid)
    # Differing origins must not reuse the question's own execution.
    borrowed = frame.copy()
    own = frame[(frame.strategy == TREATMENT) & (frame.qid == qid)].execution_id.tolist()
    borrowed.loc[swapped, "execution_id"] = own
    with pytest.raises(ValueError, match="Execution sharing differs"):
        validate_exact_balance(borrowed, pairing, SEEDS)
    # Match each column's own dtype; pandas 3 rejects incompatible setitem.
    for column, value in [("budget", 999), ("original_n_steps", 999), ("pair_id", "changed")]:
        broken = frame.copy()
        broken.loc[swapped, column] = value
        with pytest.raises(ValueError, match="per-question allowance differs"):
            validate_exact_balance(broken, pairing, SEEDS)


def test_exact_sign_flip_enumerates_the_same_distribution_as_brute_force():
    for differences in [[1, -1, 2], [3, 0, 0, -1], [2, 2, 2], [-4, 1, 1, 1], []]:
        nonzero = [d for d in differences if d]
        observed = abs(sum(differences))
        total = 0.
        for signs in itertools.product([1, -1], repeat=len(nonzero)):
            total += abs(sum(s * abs(d) for s, d in zip(signs, nonzero))) >= observed
        expected = total / 2 ** len(nonzero)
        assert exact_block_sign_flip(differences) == pytest.approx(expected)
    assert exact_block_sign_flip([0, 0]) == 1.
    with pytest.raises(ValueError, match="integer success-count differences"):
        exact_block_sign_flip([1.0])


def test_block_comparison_resamples_pairs_and_reports_every_block(pairing, fixed_successes):
    origins = {"a": 1, "b": 2, "c": 2, "d": 3, "e": 0, "f": 0}
    frame = build_frame(pairing, origins, fixed_successes)
    result = block_comparison(frame, pairing, SEEDS, iters=200)
    assert result["n_pairs"] == 2 and result["n_questions"] == 4
    assert (result["strategy_a"], result["strategy_b"]) == (TREATMENT, SWAP)
    # Each pair has two questions and two seeds; only seed 0 succeeds under treatment.
    assert result["mean_a"] == 0.5 and result["mean_b"] == 0.
    assert result["delta"] == pytest.approx(0.5)
    assert [b["successes_a"] - b["successes_b"] for b in result["blocks"]] == [2, 2]
    assert result["positive_blocks"] == 2 and result["negative_blocks"] == result["zero_blocks"] == 0
    assert result["p_value"] == pytest.approx(0.5)
    assert result["resampling_unit"].startswith("two-question length-matched pair")


def test_block_comparison_rejects_undeclared_comparators_and_nonbinary_outcomes(pairing, fixed_successes):
    frame = build_frame(pairing, {q: 0 for q in "abcdef"}, fixed_successes)
    with pytest.raises(ValueError, match="Undeclared paired-origin comparator"):
        block_comparison(frame, pairing, SEEDS, control="full_restart")
    broken = frame.copy()
    # Widen the boolean column first; pandas 3 refuses to store 2 in a bool.
    broken["success"] = broken.success.astype(object)
    broken.loc[broken.strategy == TREATMENT, "success"] = 2
    with pytest.raises(ValueError, match="success must be binary"):
        block_comparison(broken, pairing, SEEDS, iters=200)


def test_dev_control_is_comparable_without_the_swap_balance_claim(pairing, fixed_successes):
    frame = build_frame(pairing, {q: 0 for q in "abcdef"}, fixed_successes)
    result = block_comparison(frame, pairing, SEEDS, control=DEV_CONTROL, iters=200)
    assert result["strategy_b"] == DEV_CONTROL and result["n_pairs"] == 2
