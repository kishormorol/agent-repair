"""Freeze a fresh, capped Qwen experiment with exactly balanced origin swaps."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_extension_study import audit_trace, compare_uncertainty, require
from scripts.prepare_extension_study import freeze_file, official_functions, MODELS, HINT
from scripts.run_extension_study import load_envelope, uncertainty_record
from src.agent.react_agent import SYSTEM_PROMPT
from src.env.extension import convert_2wiki
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_matched import fit_position_profile
from src.repair.position_pairs import POLICIES, SWAP
from src.utils.cloud_runs import write_once

RUN_ID = "position-pairs-20260916-v2"
BASE = ROOT / "output/aws-experiment/2026-09-16-position-pairs"
PREVIOUS = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
DATASETS = ["hotpotqa", "2wikimultihopqa"]


def fresh_ids(records, previous_cohort, n, seed):
    excluded = sorted(set(previous_cohort["excluded_ids"] + previous_cohort["main_ids"]
                          + previous_cohort["development_ids"]))
    ids = [r["_id"] for r in records]
    require(len(ids) == len(set(ids)) and set(excluded) <= set(ids), "Input IDs or historical exclusions differ")
    eligible = sorted(set(ids)-set(excluded))
    require(type(n) is int and 0 < n <= len(eligible), "Invalid fresh cohort size")
    return {"main_ids": random.Random(seed).sample(eligible, n), "excluded_ids": excluded,
            "sampling_seed": seed, "eligible_count": len(eligible)}


def calibration(previous, prior_protocol, dataset, destination):
    """Reclassify old originals as development for this separate fresh study."""
    model = "qwen32b"
    results = previous / "retrieved/results" / model
    signature = json.loads((results / "model.json").read_text())["payload"]
    identity = {"protocol_sha256": fingerprint(prior_protocol), "model_key": model, "model": signature}
    load_envelope(results / "model.json", identity)
    identity = {**identity, "dataset": dataset}
    records = {r["_id"]: r for r in json.loads((previous / "retrieved/package/data" / f"{dataset}.json").read_text())}
    official = official_functions(previous / "retrieved/package/references" / f"{dataset}.py", dataset)
    originals, uncertainties, source_hashes = [], {}, {}
    cohort = prior_protocol["cohorts"][dataset]
    ids = cohort["development_ids"] + cohort["main_ids"]
    for role in ["development", "main"]:
        for qid in cohort[role+"_ids"]:
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
    profile = fit_position_profile(originals, uncertainties, source_ids=ids, dataset=dataset, strategy=TREATMENT,
                                   source_phase="development", source_run_id=RUN_ID,
                                   model=MODELS[model], source_sha256=fingerprint(originals))
    payload = {"source_protocol_sha256": fingerprint(prior_protocol), "source_question_ids": ids,
               "reused_for": "Development only in this fresh study; previous study unchanged",
               "originals": originals, "uncertainties": uncertainties, "source_sha256": source_hashes,
               "records": {t["qid"]: records[t["qid"]] for t in originals}, "profile": profile}
    write_once(destination, payload)
    return {"questions": len(ids), "failures": len(originals), "profile_sha256": profile["sha256"],
            "calibration_sha256": file_fingerprint(destination)}


def prepare(out=BASE / "prepared-v2", n=100, previous=PREVIOUS):
    out, previous = Path(out), Path(previous)
    reproduction = json.loads((previous / "analysis-reproduction-check.json").read_text())
    require(reproduction["matched"] and reproduction["mismatches"] == 0, "Prior reproduction did not pass")
    for name, digest in reproduction["sha256"].items():
        require(file_fingerprint(previous/name) == digest, f"Prior input changed: {name}")
    old = json.loads((previous / "retrieved/package/protocol.json").read_text())["payload"]
    inputs = ROOT / "output/aws-experiment/2026-09-14-extension/source-inputs"
    hotpot = ROOT / "output/aws-experiment/2026-09-14-diagnosis-v2/prepared/raw/hotpot_dev_distractor_v1.json"
    for path in [hotpot, inputs/"dev.json", inputs/"id_aliases.json"]:
        require(file_fingerprint(path) == old["source_input_sha256"][path.name], "Official source bytes changed")
    aliases = {r["Q_id"]: r for r in map(json.loads, (inputs/"id_aliases.json").read_text().splitlines())}
    data = {"hotpotqa": json.loads(hotpot.read_text()),
            "2wikimultihopqa": [convert_2wiki(r, aliases) for r in json.loads((inputs/"dev.json").read_text())]}
    cohorts, calibrated = {}, {}
    for i, dataset in enumerate(DATASETS):
        cohorts[dataset] = fresh_ids(data[dataset], old["cohorts"][dataset], n, 20260916+i)
        by_id = {r["_id"]: r for r in data[dataset]}
        write_once(out/"data"/f"{dataset}.json", [by_id[q] for q in cohorts[dataset]["main_ids"]])
        freeze_file(out/"references"/f"{dataset}.py", (previous/"retrieved/package/references"/f"{dataset}.py").read_bytes())
        calibrated[dataset] = calibration(previous, old, dataset, out/"calibration"/f"{dataset}.json")
    protocol = {"schema": 1, "run_id": RUN_ID, "frozen_date_utc": "2026-09-16",
        "model_key": "qwen32b", "models": {"qwen32b": MODELS["qwen32b"]}, "cohorts": cohorts,
        "calibration": calibrated, "strategies": POLICIES, "seeds": [0, 1, 2],
        "main_n_per_dataset": n, "initial_seed": 42, "initial_temperature": 0., "repair_temperature": .7,
        "max_new_steps": 8, "max_tokens_per_step": 512, "max_model_len": 16384,
        "logprobs_topk": 20, "top_p": .95, "retry_hint": HINT, "generated_token_multiplier": 1.,
        "system_prompt_sha256": fingerprint(SYSTEM_PROMPT), "prefix_caching": False,
        "pairing_seed": 20260916, "analysis_seed": 20260916, "bootstrap_iters": 10000,
        "pairing": "Seeded pairing within exact original length, using failed-question IDs and lengths only; odd remainder excluded and reported",
        "swap": "Exchange uncertainty+BT2 origins within each pair; same per-question seeds, hints and allowances; no test recovery outcomes enter pairing or choice",
        "scope": "Prospective origin-assignment diagnostic on a fresh length-matched failure cohort; the swap control uses its partner's uncertainty and is not a standalone deployable selector",
        "primary_control": SWAP, "primary_family_size": 2,
        "primary_family": "Two exact two-sided pair-block sign-flip tests (one per dataset), Holm adjusted; within-block exchangeability assumption explicit",
        "resampling_unit": "Resample whole two-question pairs and retain all three seeds; individual 95% percentile intervals are not simultaneous",
        "secondary_control": "Expanded-development random; descriptive contrasts only, no new primary test",
        "precision_note": "Resource-bounded diagnostic, no equivalence or power claim. Report all equal-origin and zero-difference pairs.",
        "execution_order": "All fresh originals before repairs; seeded job order within a pair; alternate dataset pairs; share identical origin/prompt/seed/budget executions",
        "completion": "All frozen originals and matched policy/seed rows must pass raw replay, scoring and exact-balance audit. Partial runs remain incomplete.",
        "allocation": {"total_usd": 119., "reserve_usd": 20., "prior_gross_conservative_bound_usd": 90.12,
                       "session_gross_ceiling_usd": 8.50, "maximum_minutes": 70,
                       "maximum_hourly_usd": 5.84531, "overhead_allowance_usd": 1.50},
        "history_limitation": old["history_limitation"], "prior_protocol_sha256": fingerprint(old),
        "authorization": "User requested on September 16: we have credits right, so do the experiment"}
    write_once(out/"protocol.json", {"payload": protocol, "sha256": fingerprint(protocol)})
    for folder in ["src", "tests"]:
        for path in sorted((ROOT/folder).rglob("*.py")):
            freeze_file(out/"code"/path.relative_to(ROOT), path.read_bytes())
    for name in ["run_position_pair_study.py", "analyze_position_pair_study.py", "run_position_pair_cloud.py",
                 "prepare_position_pair_study.py", "run_extension_study.py", "analyze_extension_study.py",
                 "prepare_extension_study.py", "configure_aws_stop.py", "run_aws_extension_study.py"]:
        freeze_file(out/"code/scripts"/name, (ROOT/"scripts"/name).read_bytes())
    hashes = {str(p.relative_to(out)): file_fingerprint(p) for p in sorted(out.rglob("*"))
              if p.is_file() and p.name != "package-manifest.json" and "__pycache__" not in p.parts}
    write_once(out/"package-manifest.json", {"sha256": fingerprint(hashes), "files": hashes})
    print(json.dumps({"protocol_sha256": fingerprint(protocol), "main_questions": n*len(DATASETS),
                      "calibration": calibrated, "allocation": protocol["allocation"]}, indent=2))
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE/"prepared-v2")
    args = parser.parse_args()
    prepare(args.out)
