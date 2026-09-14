"""A development-fitted origin distribution with no question-specific scores."""
from __future__ import annotations

import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_position_profile(profile, *, dataset=None, strategy=None, excluded_ids=None):
    if not isinstance(profile, dict) or set(profile) != {"sha256", "payload"}:
        raise ValueError("A frozen position profile with a payload and SHA256 is required")
    payload = profile["payload"]
    if not isinstance(payload, dict):
        raise ValueError("Position profile payload must be an object")
    if profile["sha256"] != _digest(payload):
        raise ValueError("Position profile checksum does not match its payload")
    if (payload.get("schema") != 1 or payload.get("source_phase") != "development"
            or payload.get("mapping") != "nearest_normalized_origin_half_up"):
        raise ValueError("Position profiles must be fitted on development data")
    if dataset is not None and payload.get("dataset") != dataset:
        raise ValueError("Position profile dataset does not match the run")
    if strategy is not None and payload.get("source_strategy") != strategy:
        raise ValueError("Position profile policy does not match the selected strategy")
    source_ids = payload.get("source_question_ids")
    if (not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(q, str) or not q for q in source_ids)
            or len(source_ids) != len(set(source_ids))):
        raise ValueError("Position profile requires unique development question IDs")
    if excluded_ids is not None and not set(source_ids) <= set(excluded_ids):
        raise ValueError("Exclude every profile development question from test data")
    entries = payload.get("origins")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Position profile has no development origins")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Position profile origin entries must be objects")
        q, origin, length = entry.get("qid"), entry.get("origin"), entry.get("n_steps")
        if (q not in source_ids or q in seen or type(origin) is not int
                or type(length) is not int or length < 1 or not 0 <= origin < length):
            raise ValueError("Position profile requires one valid origin per failed question")
        seen.add(q)
    return payload


def sample_position(profile, n_steps, rng):
    """Draw one development question uniformly; map its normalized origin."""
    payload = validate_position_profile(profile)
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("A position draw requires a nonempty trajectory")
    entry = rng.choice(payload["origins"])
    denominator = max(1, entry["n_steps"] - 1)
    # Exact rational arithmetic, with half-step ties rounded upward.
    return (2 * entry["origin"] * (n_steps - 1) + denominator) // (2 * denominator)


def fit_position_profile(originals, uncertainties, *, source_ids, dataset, strategy,
                         source_phase, source_run_id, model, source_sha256):
    from .strategies import parse_strategy, select_target_step, make_rng

    parsed = parse_strategy(strategy)
    if source_phase != "development" or parsed["base"] != "uncertainty" or parsed["informed"]:
        raise ValueError("Fit a generic uncertainty policy on development data only")
    if not originals:
        raise ValueError("Position fitting requires failed development trajectories")
    origins = []
    for original in sorted(originals, key=lambda t: t["qid"]):
        qid = original["qid"]
        if original.get("success") is not False or qid not in uncertainties:
            raise ValueError("Position fitting needs failed originals and their uncertainty")
        steps = original["steps"]
        if not steps or [s["index"] for s in steps] != list(range(len(steps))):
            raise ValueError("Position fitting requires contiguous original steps")
        uncertainty = uncertainties[qid]
        if uncertainty.get("qid") != qid:
            raise ValueError("Position fitting uncertainty belongs to another question")
        origin = select_target_step(strategy, len(steps), None, uncertainty,
                                    make_rng(qid, strategy, 0))
        origins.append({"qid": qid, "origin": origin, "n_steps": len(steps)})
    payload = {"schema": 1, "source_phase": source_phase, "dataset": dataset,
               "source_run_id": source_run_id, "source_strategy": strategy,
               "source_question_ids": sorted(source_ids), "origins": origins,
               "model": model, "source_sha256": source_sha256,
               "mapping": "nearest_normalized_origin_half_up"}
    profile = {"sha256": _digest(payload), "payload": payload}
    validate_position_profile(profile, dataset=dataset, strategy=strategy)
    return profile
