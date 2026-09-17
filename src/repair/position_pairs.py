"""Length-matched origin swaps and inference that keeps each question pair intact."""
from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
import pandas as pd

from .controlled import fingerprint
from .diagnosis import TREATMENT
from ..eval.metrics import bootstrap_ci

SWAP = "length_matched_swap"
DEV_CONTROL = "position_matched_random"
POLICIES = [TREATMENT, SWAP, DEV_CONTROL]


def pair_by_length(lengths, seed):
    """Pair using IDs and lengths alone; leave at most one unmatched per length.

    Input must contain only failed originals. No uncertainty or repair outcomes
    enter the pairing. Each pair is an independent analysis block, not two
    independent questions, because the swap control borrows its partner's origin.
    """
    groups = defaultdict(list)
    for qid, length in lengths.items():
        if not isinstance(qid, str) or not qid or type(length) is not int or length < 1:
            raise ValueError("Pairing requires question IDs and positive integer lengths")
        groups[length].append(qid)
    pairs, unmatched = [], []
    for length, ids in sorted(groups.items()):
        ids.sort()
        random.Random(fingerprint(["length-pairing", seed, length])).shuffle(ids)
        for start in range(0, len(ids)-1, 2):
            members = ids[start:start+2]
            pairs.append({"pair_id": fingerprint([seed, length, sorted(members)]),
                          "qids": members, "n_steps": length})
        if len(ids) % 2:
            unmatched.append(ids[-1])
    random.Random(fingerprint(["pair-order", seed])).shuffle(pairs)
    return {"pairs": pairs, "unmatched_ids": sorted(unmatched), "n_failures": len(lengths),
            "n_matched_questions": 2*len(pairs), "pairing_seed": seed}


def validate_exact_balance(frame, pairing, seeds):
    """Check coverage and exact within-pair origin distributions for every seed."""
    expected = {(q, s, k) for pair in pairing["pairs"] for q in pair["qids"]
                for s in seeds for k in POLICIES}
    keys = ["qid", "seed", "strategy"]
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete paired-origin policy/seed coverage")
    indexed = frame.set_index(keys)
    shared = 0
    for pair in pairing["pairs"]:
        left, right = pair["qids"]
        for seed in seeds:
            a, b = [indexed.loc[(q, seed, TREATMENT)] for q in [left, right]]
            ca, cb = [indexed.loc[(q, seed, SWAP)] for q in [left, right]]
            if (ca.origin, cb.origin) != (b.origin, a.origin):
                raise ValueError("Swapped origins must exchange the paired questions' origins exactly")
            for own, control in [(a, ca), (b, cb)]:
                if (own.pair_id != pair["pair_id"] or control.pair_id != pair["pair_id"]
                        or own.original_n_steps != pair["n_steps"] or control.original_n_steps != pair["n_steps"]
                        or own.budget != control.budget):
                    raise ValueError("Pair identity, original length, or per-question allowance differs")
                same = own.origin == control.origin
                if (own.execution_id == control.execution_id) != same:
                    raise ValueError("Execution sharing differs from the selected origin")
                if same and own.prompt_sha256 != control.prompt_sha256:
                    raise ValueError("Shared execution has different prompts")
                shared += same
    n = pairing["n_matched_questions"]*len(seeds)
    return {"paired_attempts": n, "shared_execution_pairs": int(shared),
            "shared_execution_fraction": shared/n if n else None,
            "raw_origin_distribution_equal": True, "normalized_origin_distribution_equal": True,
            "matched_questions": pairing["n_matched_questions"], "unmatched_ids": pairing["unmatched_ids"]}


def exact_block_sign_flip(integer_differences):
    """Exact two-sided sign-flip distribution via integer dynamic programming.

    This tests the declared within-block exchangeability assumption. It is
    not an inference that policy labels were randomly assigned in deployment.
    Zero-difference blocks remain in the point estimate/CI; their signs do
    not change the exact null distribution.
    """
    if any(type(x) is not int for x in integer_differences):
        raise ValueError("Exact sign flips require integer success-count differences")
    distribution = Counter({0: 1})
    for value in integer_differences:
        if not value:
            continue
        updated = Counter()
        for total, count in distribution.items():
            updated[total+abs(value)] += count
            updated[total-abs(value)] += count
        distribution = updated
    observed = abs(sum(integer_differences))
    return sum(count for total, count in distribution.items() if abs(total) >= observed) / sum(distribution.values())


def block_comparison(frame, pairing, seeds, *, control=SWAP, iters=10000, seed=20260916):
    validate_exact_balance(frame, pairing, seeds)
    if control not in [SWAP, DEV_CONTROL]:
        raise ValueError("Undeclared paired-origin comparator")
    block_size = 2*len(seeds)
    rows = []
    for pair in pairing["pairs"]:
        sub = frame[frame.qid.isin(pair["qids"])]
        own = sub[sub.strategy == TREATMENT]
        other = sub[sub.strategy == control]
        if not own.success.isin([True, False, 0, 1]).all() or not other.success.isin([True, False, 0, 1]).all():
            raise ValueError("Repair success must be binary")
        rows.append({"pair_id": pair["pair_id"], "qids": pair["qids"],
                     "successes_a": int(own.success.sum()), "successes_b": int(other.success.sum())})
    if not rows:
        return {"n_pairs": 0, "n_questions": 0, "delta": None, "delta_lo": None, "delta_hi": None,
                "p_value": 1., "mean_a": None, "mean_b": None, "blocks": []}
    differences = [r["successes_a"]-r["successes_b"] for r in rows]
    mean, lo, hi = bootstrap_ci([d/block_size for d in differences], iters=iters, seed=seed)
    return {"strategy_a": TREATMENT, "strategy_b": control,
            "estimand": "equal_question_mean_seed_success_on_length_matched_failure_pairs",
            "resampling_unit": "two-question length-matched pair; all seeds retained",
            "n_pairs": len(rows), "n_questions": 2*len(rows),
            "mean_a": sum(r["successes_a"] for r in rows)/(block_size*len(rows)),
            "mean_b": sum(r["successes_b"] for r in rows)/(block_size*len(rows)),
            "delta": mean, "delta_lo": lo, "delta_hi": hi,
            "p_value": exact_block_sign_flip(differences),
            "test": "exact_pair_block_sign_flip_assuming_within_block_exchangeability",
            "bootstrap_iters": iters, "analysis_seed": seed,
            "positive_blocks": sum(d > 0 for d in differences), "negative_blocks": sum(d < 0 for d in differences),
            "zero_blocks": sum(d == 0 for d in differences), "blocks": rows}
