"""Post-results position diagnostics; never refit or change frozen trial exports."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.analyze_extension_study import compare_uncertainty, require
from scripts.run_extension_study import load_envelope, select_origins, uncertainty_record
from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_matched import fit_position_profile

CONTROL = "position_matched_random"


def position_distribution(entries):
    """Keep every support point; zero is the convention for one-step traces."""
    if not entries:
        return {"n": 0, "origin_zero_fraction": None, "mean_retained_steps": None,
                "mean_normalized_origin": None, "retained_steps": [], "normalized_origin": []}
    raw, normalized = Counter(), Counter()
    for entry in entries:
        k, length = entry["origin"], entry["n_steps"]
        require(type(k) is int and type(length) is int and length > 0 and 0 <= k < length,
                "Invalid origin or original trajectory length")
        raw[k] += 1
        normalized[Fraction(k, max(1, length - 1))] += 1
    n = len(entries)
    return {"n": n, "origin_zero_fraction": raw[0] / n,
            "mean_retained_steps": sum(k * count for k, count in raw.items()) / n,
            "mean_normalized_origin": float(sum(k * count for k, count in normalized.items()) / n),
            "retained_steps": [{"position": k, "count": count, "proportion": count / n}
                               for k, count in sorted(raw.items())],
            "normalized_origin": [{"position": float(k), "numerator": k.numerator,
                                   "denominator": k.denominator, "count": count, "proportion": count / n}
                                  for k, count in sorted(normalized.items())]}


def summarize_positions(trials, lengths, seeds, development_entries):
    """Equal weight per question and then seed; runtime attempts are excluded."""
    selected = trials[(trials["mode"] == "token") & trials.strategy.isin([TREATMENT, CONTROL])].copy()
    keys = ["qid", "seed", "strategy"]
    expected = {(q, seed, policy) for q in lengths for seed in seeds for policy in [TREATMENT, CONTROL]}
    require(len(seeds) == len(set(seeds)) and bool(seeds), "Invalid position diagnostic seeds")
    require(not selected.duplicated(keys).any()
            and set(selected[keys].itertuples(index=False, name=None)) == expected,
            "Incomplete position diagnostic question/policy/seed coverage")
    require(all(type(n) is int and n > 0 for n in lengths.values()), "Invalid original trajectory length")
    require(all(value == int(value) for value in selected.origin), "Noninteger repair origin")
    selected["origin"] = selected.origin.astype(int)
    # Runtime has blank budgets, so the combined CSV reads this column as text.
    # Convert only token rows; preserve the frozen numeric allowance exactly.
    selected["budget"] = pd.to_numeric(selected.budget, errors="raise")
    require(all(value >= 0 and value == int(value) for value in selected.budget), "Invalid token allowance")
    selected["budget"] = selected.budget.astype(int)
    selected["n_steps"] = selected.qid.map(lengths).astype(int)
    selected["normalized_origin"] = selected.origin / (selected.n_steps - 1).clip(lower=1)
    policies = {policy: position_distribution(selected.loc[selected.strategy == policy,
                                ["origin", "n_steps"]].to_dict("records")) for policy in [TREATMENT, CONTROL]}
    paired = selected[selected.strategy == TREATMENT].merge(
        selected[selected.strategy == CONTROL], on=["qid", "seed"], suffixes=("_unc", "_control"),
        validate="one_to_one")
    same_origin = paired.origin_unc == paired.origin_control
    shared = paired.execution_id_unc == paired.execution_id_control
    require((shared == same_origin).all(), "Shared execution differs from shared token-mode origin")
    require((paired.budget_unc == paired.budget_control).all()
            and (paired.loc[shared, "prompt_sha256_unc"] == paired.loc[shared, "prompt_sha256_control"]).all(),
            "Paired token budgets or shared-execution prompts differ")
    n = len(paired)
    support = sorted({r["position"] for p in policies.values() for r in p["normalized_origin"]})
    gap = (max(abs(sum(r["proportion"] for r in policies[TREATMENT]["normalized_origin"] if r["position"] <= x)
                   - sum(r["proportion"] for r in policies[CONTROL]["normalized_origin"] if r["position"] <= x))
               for x in support) if n else None)
    by_question = paired.assign(shared=shared).groupby("qid").shared.all()
    return {"main_failures": len(lengths), "seeds": list(seeds), "paired_seed_rows": n,
            "development": position_distribution(development_entries), "policies": policies,
            "max_normalized_ecdf_gap": gap,
            "shared_execution_pairs": int(shared.sum()), "shared_execution_fraction": float(shared.mean()) if n else None,
            "questions_all_seeds_shared": int(by_question.sum()),
            "questions_any_seed_differs": int((~by_question).sum()),
            "failed_trace_lengths": [{"n_steps": k, "questions": count} for k, count in sorted(Counter(lengths.values()).items())]}, selected


def load_position_balance(base, analysis, protocol):
    """Join reproduced trials to checksum-checked originals and rebuilt dev profiles.

    The caller first validates the frozen exports with load_extension. This
    additional pass verifies the raw inputs needed only for position reporting.
    It recomputes original uncertainty but does not rerun or rescore repairs.
    """
    base = Path(base)
    digest = fingerprint(protocol)
    require(digest == analysis["protocol_sha256"], "Position diagnostic protocol mismatch")
    source_hashes = {}

    def remember(path):
        source_hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def envelope(path, identity):
        payload = load_envelope(path, identity)
        remember(path)
        return payload

    package, results = base / "retrieved/package", base / "retrieved/results"
    manifest = json.loads(remember(package / "package-manifest.json").read_text())
    require(manifest["sha256"] == fingerprint(manifest["files"]), "Package manifest checksum mismatch")
    trials = pd.read_csv(remember(base / "pooled-analysis-local.trials.csv"), keep_default_na=False)
    cells, frames = [], []
    for audit in analysis["audits"]:
        model, dataset = audit["model_key"], audit["dataset"]
        cohort, pin = protocol["cohorts"][dataset], protocol["models"][model]
        data_path = remember(package / "data" / f"{dataset}.json")
        require(source_hashes[str(data_path.relative_to(base))] == manifest["files"][f"data/{dataset}.json"],
                "Frozen cohort data checksum mismatch")
        record_list = json.loads(data_path.read_text())
        records = {r["_id"]: r for r in record_list}
        ids = cohort["main_ids"] + cohort["development_ids"]
        require(len(ids) == len(set(ids)) == len(records) == len(record_list) and set(ids) == set(records)
                and not set(ids) & set(cohort["excluded_ids"]), "Frozen position cohort mismatch")
        model_path = results / model / "model.json"
        signature = json.loads(model_path.read_text())["payload"]
        require(signature["revision"] == pin["revision"] and signature["dtype"] == pin["dtype"]
                and Path(signature["name"]).name == pin["revision"], "Position diagnostic model pin mismatch")
        identity = {"protocol_sha256": digest, "model_key": model, "model": signature}
        envelope(model_path, identity)
        identity = {**identity, "dataset": dataset}
        cell = results / model / dataset
        originals, uncertainties = {}, {}
        for role in ["development", "main"]:
            for qid in cohort[role + "_ids"]:
                trace_identity = {**identity, "qid": qid, "role": role, "record_sha256": fingerprint(records[qid])}
                original = envelope(cell / role / "originals" / f"{qid}.json", trace_identity)
                require(original["qid"] == qid and type(original["success"]) is bool
                        and bool(original["steps"])
                        and [s["index"] for s in original["steps"]] == list(range(len(original["steps"]))),
                        "Position diagnostic original trace mismatch")
                originals[qid] = original
                saved = envelope(cell / role / "uncertainty" / f"{qid}.json",
                                 {**trace_identity, "original_sha256": fingerprint(original)})
                recomputed = uncertainty_record(original)
                compare_uncertainty(saved["steps"], recomputed["steps"])
                uncertainties[qid] = {**saved, "steps": recomputed["steps"]}
        failed_dev = [originals[q] for q in cohort["development_ids"] if not originals[q]["success"]]
        expected_profile = (fit_position_profile(failed_dev, uncertainties, source_ids=cohort["development_ids"],
                            dataset=dataset, strategy=TREATMENT, source_phase="development", source_run_id=protocol["run_id"],
                            model=signature, source_sha256=fingerprint(failed_dev)) if failed_dev else None)
        profile = envelope(cell / "position-profile.json", identity)
        require(profile == expected_profile, "Position profile differs from frozen development-only fitting")
        require(audit["position_profile_fallback"] == (profile is None), "Position profile fallback differs from audit")
        failures = [q for q in cohort["main_ids"] if not originals[q]["success"]]
        require(len(failures) == audit["main_failures"], "Raw initial failure count differs from audit")
        sub = trials[(trials.model_key == model) & (trials.dataset == dataset)]
        entries = profile["payload"]["origins"] if profile is not None else []
        summary, selected = summarize_positions(sub, {q: len(originals[q]["steps"]) for q in failures},
                                                protocol["seeds"], entries)
        # Reconstruct choices to tie every displayed position to the frozen policy.
        indexed = selected.set_index(["qid", "seed", "strategy"])
        for qid in failures:
            for seed in protocol["seeds"]:
                choices = select_origins(protocol, originals[qid], uncertainties[qid], None, seed, profile, [TREATMENT, CONTROL])
                budget = int(protocol["generated_token_multiplier"] * originals[qid]["total_gen_tokens"])
                for choice in choices:
                    row = indexed.loc[(qid, seed, choice["strategy"])]
                    execution = fingerprint({"qid": qid, "seed": seed, "origin": choice["origin"],
                                             "budget": budget, "runtime_policy": None})
                    require(row.origin == choice["origin"] and row.budget == budget and row.execution_id == execution,
                            "Position trial differs from frozen policy or execution")
        cells.append({"model_key": model, "dataset": dataset, "development_questions": len(cohort["development_ids"]),
                      "profile_fallback": profile is None, **summary})
        frames.append(selected)
    return {"schema": 1, "analysis_status": "exploratory_post_results", "protocol_sha256": digest,
            "normalization": "origin / max(1, original_n_steps - 1)",
            "weighting": "Equal weight per failed main question, then per frozen seed; development has one row per failure.",
            "interpretation": "Descriptive diagnostics, not independent seed observations, balance tests, or a refitted control.",
            "source_sha256": source_hashes, "cells": cells}, pd.concat(frames, ignore_index=True)
