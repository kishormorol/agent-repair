"""Deterministic repair planning and provenance guards for controlled reruns."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..agent.react_agent import Step, build_messages, resolve_step_limit
from ..utils.io import load_json, save_json
from .strategies import (build_strategies, select_target_step, get_nudge_text,
                         make_rng, parse_strategy)
from .position_matched import validate_position_profile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_manifest(path, payload, existing_outputs=()):
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    path = Path(path)
    if path.exists():
        if load_json(path) != manifest:
            raise ValueError("Run inputs or code changed; preserve this run and use a new output directory")
    else:
        if any(Path(p).exists() for p in existing_outputs):
            raise ValueError("Existing outputs have no matching manifest; use a fresh output directory")
        save_json(manifest, path)
    return manifest["sha256"]


def configured_strategies(repair):
    strategies = repair.get("strategies")
    if strategies is None:
        strategies = build_strategies(
            repair.get("uncertainty_metrics", []), repair.get("uncertainty_rules", []),
            backtrack_offsets=repair.get("backtrack_offsets", [0]),
            nudge_types=repair.get("nudge_types", ["generic"]),
            baselines=repair.get("baselines"))
    if not isinstance(strategies, list) or not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("repair.strategies must be a nonempty list of unique strategy names")
    for strategy in strategies:
        parsed = parse_strategy(strategy)
        if parsed["base"] not in {"full_restart", "random_step", "fixed_early", "position_matched_random",
                                   "oracle_targeted", "uncertainty"}:
            raise ValueError(f"Unknown repair strategy: {strategy}")
    return strategies


def budget_multipliers(repair):
    values = repair.get("budget", {}).get("multipliers", [1.0])
    if not isinstance(values, list) or not values:
        raise ValueError("Specify a nonempty budget multiplier list")
    values = [float(v) for v in values]
    if len(values) != len(set(values)) or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Budget multipliers must be unique positive finite numbers")
    return values


def acquisition_cost(strategy, uncertainty, annotation):
    parsed = parse_strategy(strategy)
    sources = []
    if parsed["base"] == "uncertainty" and parsed["metric"] in {
        "self_consistency", "verbalized_confidence"}:
        sources.extend(step.get("acquisition_cost", {}).get(parsed["metric"])
                       for step in uncertainty["steps"])
    if parsed["base"] == "oracle_targeted" or parsed["informed"]:
        sources.append((annotation or {}).get("acquisition_cost"))
    result = {}
    for source, target in (("generated_tokens", "selection_gen_tokens"),
                           ("prompt_tokens", "selection_prompt_tokens"),
                           ("requests", "selection_model_requests")):
        result[target] = (None if any(cost is None or cost.get(source) is None for cost in sources)
                          else sum(cost[source] for cost in sources))
    return result


def plan_repairs(raw, record, original, uncertainty, annotation, seed, multiplier, strategies):
    repair = raw["repair"]
    qid = record["_id"]
    if original["qid"] != qid or original.get("success"):
        raise ValueError("Repair requires a matching failed original trajectory")
    if original["question"] != record["question"] or original["gold_answer"] != record["answer"]:
        raise ValueError("Original trajectory does not match the frozen question/answer")
    steps = original["steps"]
    for artifact in (uncertainty, annotation):
        if artifact and artifact.get("qid", qid) != qid:
            raise ValueError("Question ID mismatch in uncertainty or annotation artifact")
    if not steps or [s["index"] for s in steps] != list(range(len(steps))):
        raise ValueError("Original trajectory must have contiguous zero-based step indices")
    oracle = annotation.get("broken_step") if annotation else None
    mode = repair.get("step_budget_mode", "total")
    sweep = repair.get("origin_sweep", False)
    matched = repair.get("match_restart_hint", False)
    nudge_cfg = repair.get("nudge", {})
    hint = nudge_cfg.get("retry_hint") if nudge_cfg.get("enabled") else None
    if sweep and (mode != "new" or not matched or any(parse_strategy(s)["informed"] for s in strategies)):
        raise ValueError("Origin sweeps require new-step budgets, matched hints, and no informed variants")
    base = original["total_gen_tokens"]
    budget = int(multiplier * base)
    choices = {}
    position_profile = repair.get("position_matched_profile")
    if any(parse_strategy(s)["base"] == "position_matched_random" for s in strategies):
        profile = validate_position_profile(position_profile, dataset=raw["dataset"]["name"])
        if raw["dataset"].get("split") == "test" and qid in profile["source_question_ids"]:
            raise ValueError("Test question overlaps position-profile development data")
    for strategy in strategies:
        parsed = parse_strategy(strategy)
        if (parsed["base"] == "oracle_targeted" or parsed["informed"]) and oracle is None:
            raise ValueError("Reference-targeted or informed repair requires an annotation")
        if parsed["base"] == "uncertainty" and uncertainty is None:
            raise ValueError("Uncertainty repair requires saved uncertainty measurements")
        k = select_target_step(strategy, len(steps), oracle, uncertainty,
                               make_rng(qid, strategy, seed),
                               topk=repair.get("topk_for_repair", 3),
                               percentile=repair.get("threshold_percentile", 75),
                               position_profile=position_profile)
        nudge = get_nudge_text(strategy, annotation.get("error_type") if annotation else None,
                               hint, match_restart=matched)
        choices.setdefault((k, nudge), []).append(strategy)
    if sweep:
        for k in range(len(steps)):
            choices.setdefault((k, hint), [])
    jobs = []
    for (origin, nudge), names in sorted(choices.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        prefix = [Step.from_dict({**s, "generation": None}) for s in steps[:origin]]
        messages = build_messages(record["question"], prefix, nudge)
        effective = {"qid": qid, "seed": seed, "origin": origin, "messages": messages,
                     "record": record, "token_budget": budget, "multiplier": multiplier,
                     "max_steps": raw["agent"]["max_steps"], "step_budget_mode": mode,
                     "max_tokens_per_step": raw["agent"]["max_tokens_per_step"],
                     "temperature": nudge_cfg.get("temperature", 0.7) if nudge_cfg.get("enabled") else 0.0,
                     "model": raw["models"]["agent"]}
        resolve_step_limit(raw["agent"]["max_steps"], origin, mode)
        meta = {"execution_id": fingerprint(effective), "prompt_sha256": fingerprint(messages),
                "target_step": origin, "nudge_text": nudge, "model_name": raw["models"]["agent"]["name"],
                "dataset": raw["dataset"]["name"], "run_id": repair.get("run_id", "unspecified")}
        jobs.append({"record": record, "prefix_steps": prefix, "nudge": nudge,
                     "token_budget": budget, "meta": meta, "strategies": names,
                     "policy_costs": {s: acquisition_cost(s, uncertainty, annotation) for s in names},
                     "oracle_step": oracle, "original_gen_tokens": base})
    return jobs
