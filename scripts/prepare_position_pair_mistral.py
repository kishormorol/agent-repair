"""Freeze the Mistral cell of the length-matched swap design.

Kept separate from `prepare_position_pair_study.py` on purpose: that file is
frozen by checksum inside the already-executed Qwen package, so editing it
would invalidate a completed study's provenance.

The question cohort is copied from the Qwen cell so the two face identical
questions, as the replication does. Shared identifiers are exactly why no
pooled confirmatory statistic is reported across model cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_extension_study import audit_trace, compare_uncertainty, require
from scripts.prepare_extension_study import freeze_file, official_functions, MODELS, HINT
from scripts.prepare_position_pair_study import BASE, DATASETS, PREVIOUS
from scripts.run_extension_study import load_envelope, uncertainty_record
from src.agent.react_agent import SYSTEM_PROMPT
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_matched import fit_position_profile
from src.repair.position_pairs import POLICIES, SWAP
from src.utils.cloud_runs import write_once

MODEL = "mistral12b"
RUN_ID = "position-pairs-mistral-20260917-v2"
QWEN_PACKAGE = BASE / "prepared-v2"
# The authorized allocation is frozen directly into the protocol rather than
# amended downstream. validate_live_checks is called from three places, and an
# amendment applied to only some of them let a controller refuse mid-launch
# while the GPU ran. Freezing it once means every gate agrees by construction.
PRIOR_SPENT_USD = 99.13
AUTHORIZED_RESERVE_USD = 11.0
FROZEN_SCRIPTS = ["run_position_pair_study.py", "analyze_position_pair_study.py",
                  "run_position_pair_cloud.py", "prepare_position_pair_study.py",
                  "prepare_position_pair_mistral.py", "run_extension_study.py",
                  "analyze_extension_study.py", "prepare_extension_study.py",
                  "configure_aws_stop.py", "run_aws_extension_study.py"]


def calibration(previous, prior_protocol, dataset, destination):
    """Reclassify this model's earlier originals as development for the fresh cell.

    Mirrors the Qwen cell's calibration, bound to this model. The function is
    copied rather than imported because the original is frozen inside a
    completed package and pins its own model key.
    """
    results = previous / "retrieved/results" / MODEL
    signature = json.loads((results / "model.json").read_text())["payload"]
    identity = {"protocol_sha256": fingerprint(prior_protocol), "model_key": MODEL, "model": signature}
    load_envelope(results / "model.json", identity)
    identity = {**identity, "dataset": dataset}
    records = {r["_id"]: r for r in json.loads((previous / "retrieved/package/data" / f"{dataset}.json").read_text())}
    official = official_functions(previous / "retrieved/package/references" / f"{dataset}.py", dataset)
    originals, uncertainties, source_hashes = [], {}, {}
    cohort = prior_protocol["cohorts"][dataset]
    ids = cohort["development_ids"] + cohort["main_ids"]
    for role in ["development", "main"]:
        for qid in cohort[role + "_ids"]:
            trace_identity = {**identity, "qid": qid, "role": role, "record_sha256": fingerprint(records[qid])}
            original_path = results / dataset / role / "originals" / f"{qid}.json"
            original = load_envelope(original_path, trace_identity)
            audit_trace(original, records[qid], prior_protocol, official)
            unc_path = results / dataset / role / "uncertainty" / f"{qid}.json"
            saved = load_envelope(unc_path, {**trace_identity, "original_sha256": fingerprint(original)})
            recomputed = uncertainty_record(original)
            compare_uncertainty(saved["steps"], recomputed["steps"])
            for path in [original_path, unc_path]:
                source_hashes[str(path.relative_to(previous))] = file_fingerprint(path)
            if not original["success"]:
                originals.append(original)
                uncertainties[qid] = {**saved, "steps": recomputed["steps"]}
    profile = fit_position_profile(originals, uncertainties, source_ids=ids, dataset=dataset,
                                   strategy=TREATMENT, source_phase="development", source_run_id=RUN_ID,
                                   model=MODELS[MODEL], source_sha256=fingerprint(originals))
    payload = {"source_protocol_sha256": fingerprint(prior_protocol), "source_question_ids": ids,
               "reused_for": "Development only in this fresh study; previous study unchanged",
               "originals": originals, "uncertainties": uncertainties, "source_sha256": source_hashes,
               "records": {t["qid"]: records[t["qid"]] for t in originals}, "profile": profile}
    write_once(destination, payload)
    return {"questions": len(ids), "failures": len(originals), "profile_sha256": profile["sha256"],
            "calibration_sha256": file_fingerprint(destination)}


def prepare(out=None, source=QWEN_PACKAGE, previous=PREVIOUS):
    out = Path(out) if out else BASE / f"prepared-{MODEL}"
    source, previous = Path(source), Path(previous)
    frozen = json.loads((source / "protocol.json").read_text())
    require(frozen["sha256"] == fingerprint(frozen["payload"]), "Source protocol checksum mismatch")
    donor = frozen["payload"]
    require(donor["model_key"] != MODEL, "Cohort source must be the other model's cell")

    old = json.loads((previous / "retrieved/package/protocol.json").read_text())["payload"]
    cohorts, calibrated = {}, {}
    for dataset in DATASETS:
        cohorts[dataset] = donor["cohorts"][dataset]
        freeze_file(out / "data" / f"{dataset}.json",
                    (source / "data" / f"{dataset}.json").read_bytes())
        freeze_file(out / "references" / f"{dataset}.py",
                    (source / "references" / f"{dataset}.py").read_bytes())
        calibrated[dataset] = calibration(previous, old, dataset,
                                          out / "calibration" / f"{dataset}.json")

    protocol = {"schema": 1, "run_id": RUN_ID, "frozen_date_utc": "2026-09-17",
        "model_key": MODEL, "models": {MODEL: MODELS[MODEL]}, "cohorts": cohorts,
        "calibration": calibrated, "strategies": POLICIES, "seeds": donor["seeds"],
        "main_n_per_dataset": donor["main_n_per_dataset"], "initial_seed": donor["initial_seed"],
        "initial_temperature": donor["initial_temperature"],
        "repair_temperature": donor["repair_temperature"],
        "max_new_steps": donor["max_new_steps"], "max_tokens_per_step": donor["max_tokens_per_step"],
        "max_model_len": donor["max_model_len"], "logprobs_topk": donor["logprobs_topk"],
        "top_p": donor["top_p"], "retry_hint": HINT,
        "generated_token_multiplier": donor["generated_token_multiplier"],
        "system_prompt_sha256": fingerprint(SYSTEM_PROMPT), "prefix_caching": False,
        "pairing_seed": donor["pairing_seed"], "analysis_seed": donor["analysis_seed"],
        "bootstrap_iters": donor["bootstrap_iters"], "pairing": donor["pairing"],
        "swap": donor["swap"], "scope": donor["scope"], "primary_control": SWAP,
        "primary_family_size": donor["primary_family_size"],
        "primary_family": donor["primary_family"], "resampling_unit": donor["resampling_unit"],
        "secondary_control": donor["secondary_control"], "precision_note": donor["precision_note"],
        "execution_order": donor["execution_order"], "completion": donor["completion"],
        "cohort_source_run_id": donor["run_id"],
        "cohort_source_protocol_sha256": frozen["sha256"],
        "pooling_limitation": "Question identifiers are shared with the Qwen cell, so the two "
                              "cells support a descriptive contrast only and no pooled "
                              "confirmatory statistic.",
        "allocation": {"total_usd": 119., "reserve_usd": AUTHORIZED_RESERVE_USD,
                       "prior_gross_conservative_bound_usd": PRIOR_SPENT_USD,
                       "session_gross_ceiling_usd": 8.50, "maximum_minutes": 70,
                       "maximum_hourly_usd": 5.84531, "overhead_allowance_usd": 1.50},
        "history_limitation": old["history_limitation"],
        "prior_protocol_sha256": fingerprint(old),
        "authorization": "User authorized on September 17. The reserve is reduced from 20 to "
                         "11 USD for this cell only, against a conservative prior of 99.13. This "
                         "is the last session the 119 USD allocation supports."}
    write_once(out / "protocol.json", {"payload": protocol, "sha256": fingerprint(protocol)})
    for folder in ["src", "tests"]:
        for path in sorted((ROOT / folder).rglob("*.py")):
            freeze_file(out / "code" / path.relative_to(ROOT), path.read_bytes())
    for name in FROZEN_SCRIPTS:
        freeze_file(out / "code/scripts" / name, (ROOT / "scripts" / name).read_bytes())
    hashes = {str(p.relative_to(out)): file_fingerprint(p) for p in sorted(out.rglob("*"))
              if p.is_file() and p.name != "package-manifest.json" and "__pycache__" not in p.parts}
    write_once(out / "package-manifest.json", {"sha256": fingerprint(hashes), "files": hashes})
    print(json.dumps({"protocol_sha256": fingerprint(protocol), "model": MODEL,
                      "cohort_source": donor["run_id"],
                      "main_questions": protocol["main_n_per_dataset"] * len(DATASETS),
                      "calibration": calibrated, "allocation": protocol["allocation"]}, indent=2))
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path, default=QWEN_PACKAGE)
    args = parser.parse_args()
    prepare(args.out, args.source)
