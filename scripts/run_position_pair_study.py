"""Execute the frozen position-pair cell with exactly swapped, length-matched origins."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_extension_study import (load_envelope, save_envelope, uncertainty_record,
                                         verify_package)
from src.agent.react_agent import ReActAgent, Step, build_messages
from src.env.extension import ExtensionEnv
from src.llm.vllm_client import VLLMClient
from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_matched import validate_position_profile
from src.repair.position_pairs import DEV_CONTROL, SWAP, pair_by_length


def load_profile(package, dataset, cohort):
    """Read the frozen development profile fitted on the earlier study's questions."""
    calibration = json.loads((Path(package) / "calibration" / f"{dataset}.json").read_text())
    profile = calibration["profile"]
    validate_position_profile(profile, dataset=dataset, strategy=TREATMENT,
                              excluded_ids=cohort["excluded_ids"])
    return profile


def select_pair_origins(protocol, pair, originals, uncertainties, profile, seed):
    """Choose each policy's origin; the swap takes its partner's own choice.

    Exact length matching is what makes the swap admissible: the partner's
    origin is always a valid index for this question, so the two policies
    differ only in which trajectory supplied the number.
    """
    from src.repair.strategies import make_rng, select_target_step

    left, right = pair["qids"]
    own, cost = {}, {}
    for qid in pair["qids"]:
        if len(originals[qid]["steps"]) != pair["n_steps"]:
            raise ValueError("Paired questions must share their exact original length")
        started = time.monotonic()
        own[qid] = select_target_step(TREATMENT, pair["n_steps"], None, uncertainties[qid],
                                      make_rng(qid, TREATMENT, seed))
        cost[qid] = time.monotonic() - started + uncertainties[qid]["acquisition_wall_latency_s"]
    choices = {}
    for qid, partner in [(left, right), (right, left)]:
        started = time.monotonic()
        drawn = select_target_step(DEV_CONTROL, pair["n_steps"], None, None,
                                   make_rng(qid, DEV_CONTROL, seed), position_profile=profile)
        policies = [{"strategy": TREATMENT, "origin": own[qid], "selection_latency_s": cost[qid]},
                    {"strategy": SWAP, "origin": own[partner], "selection_latency_s": cost[partner]},
                    {"strategy": DEV_CONTROL, "origin": drawn,
                     "selection_latency_s": time.monotonic() - started}]
        for policy in policies:
            policy.update(seed=seed, selection_gen_tokens=0, selection_prompt_tokens=0,
                          selection_model_requests=0)
        choices[qid] = policies
    return choices


def pair_jobs(protocol, pair, records, originals, uncertainties, profile):
    """Plan one pair's executions, sharing any execution the policies agree on."""
    jobs = {}
    for seed in protocol["seeds"]:
        for qid, policies in select_pair_origins(protocol, pair, originals, uncertainties, profile, seed).items():
            budget = int(protocol["generated_token_multiplier"] * originals[qid]["total_gen_tokens"])
            for choice in policies:
                identity = {"qid": qid, "seed": seed, "origin": choice["origin"], "budget": budget}
                execution_id = fingerprint(identity)
                if execution_id not in jobs:
                    prefix = [Step.from_dict({**s, "generation": None})
                              for s in originals[qid]["steps"][:choice["origin"]]]
                    jobs[execution_id] = {"execution_id": execution_id, **identity,
                        "pair_id": pair["pair_id"], "original_n_steps": pair["n_steps"], "prefix": prefix,
                        "prompt_sha256": fingerprint(build_messages(records[qid]["question"], prefix, protocol["retry_hint"])),
                        "policies": []}
                jobs[execution_id]["policies"].append(choice)
    jobs = list(jobs.values())
    random.Random(fingerprint([protocol["run_id"], pair["pair_id"]])).shuffle(jobs)
    return jobs


def trial_rows(trace, job, *, model_key, dataset):
    rows = []
    for choice in job["policies"]:
        row = {"qid": trace["qid"], "model_key": model_key, "dataset": dataset,
               "strategy": choice["strategy"], "seed": job["seed"], "origin": job["origin"],
               "pair_id": job["pair_id"], "original_n_steps": job["original_n_steps"],
               "execution_id": job["execution_id"], "budget": job["budget"],
               "prompt_sha256": job["prompt_sha256"],
               **{k: trace[k] for k in ["success", "em", "f1", "final_answer", "terminated_reason"]},
               **{k: v for k, v in choice.items() if k.startswith("selection_")},
               **{k: v for k, v in trace["meta"].items() if k.startswith("recovery_")}}
        for suffix in ["gen_tokens", "prompt_tokens", "model_requests"]:
            row["incremental_" + suffix] = row["recovery_" + suffix] + row["selection_" + suffix]
        row["incremental_wall_latency_s"] = row["recovery_wall_latency_s"] + row["selection_latency_s"]
        rows.append(row)
    return rows


def plan_identity(cell_identity, pair, originals, profile):
    return {**cell_identity, "pair_id": pair["pair_id"], "qids": pair["qids"],
            "n_steps": pair["n_steps"], "profile_sha256": profile["sha256"],
            "originals_sha256": fingerprint([originals[q] for q in pair["qids"]])}


def build_schedule(protocol, cells):
    """Alternate datasets pair by pair so a truncated run is not one-sided."""
    schedule, datasets = [], [d for d in protocol["cohorts"] if d in cells]
    depth = max((len(cells[d]["pairing"]["pairs"]) for d in datasets), default=0)
    for index in range(depth):
        for dataset in datasets:
            pairs = cells[dataset]["pairing"]["pairs"]
            if index < len(pairs):
                schedule.append((dataset, pairs[index]))
    return schedule


def run(package, output, cache, deadline_utc, client=None):
    package, output = Path(package), Path(output)
    protocol, digest = verify_package(package)
    model_key = protocol["model_key"]
    model = protocol["models"][model_key]
    deadline = dt.datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError("Use an explicit UTC execution deadline")

    def guard():
        if dt.datetime.now(dt.timezone.utc) >= deadline - dt.timedelta(minutes=8):
            raise TimeoutError("Stop for result archival before the independent EC2 deadline")

    guard()
    if client is None:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model["repo_id"], revision=model["revision"], cache_dir=str(cache),
                                       allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
                                       ignore_patterns=["consolidated.safetensors"])
        if Path(model_path).name != model["revision"]:
            raise ValueError("Downloaded snapshot differs from the model pin")
        client = VLLMClient(model_path, dtype=model["dtype"], max_model_len=protocol["max_model_len"],
                            logprobs_topk=protocol["logprobs_topk"], enable_prefix_caching=protocol["prefix_caching"],
                            trust_remote_code=False, seed=protocol["initial_seed"]).load()
    signature = {"name": client.model_name, "dtype": model["dtype"], "revision": model["revision"]}
    identity = {"protocol_sha256": digest, "model_key": model_key, "model": signature}
    save_envelope(output / model_key / "model.json", signature, identity)
    agent = ReActAgent(client, protocol["max_new_steps"], protocol["max_tokens_per_step"])
    warmup = []
    for _ in range(3):
        guard()
        started = time.monotonic()
        result = client.chat([{"role": "system", "content": "Follow the instruction."},
                              {"role": "user", "content": "Write the numbers one through ten."}],
                             temperature=0., max_tokens=64, seed=protocol["initial_seed"])[0]
        warmup.append({"seconds": time.monotonic() - started, "tokens": result.num_tokens})
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save_envelope(output / model_key / f"warmup-{stamp}.json", warmup, identity)

    # Every fresh original is produced before any repair, so pairing never
    # sees a recovery outcome and a truncated run cannot select its cohort.
    cells = {}
    for dataset, cohort in protocol["cohorts"].items():
        cell = output / model_key / dataset
        cell_identity = {**identity, "dataset": dataset}
        if (cell / "complete.json").exists():
            load_envelope(cell / "complete.json", cell_identity)
            continue
        records = {r["_id"]: r for r in json.loads((package / "data" / f"{dataset}.json").read_text())}
        profile = load_profile(package, dataset, cohort)
        originals, uncertainties = {}, {}
        for qid in cohort["main_ids"]:
            guard()
            trace_identity = {**cell_identity, "qid": qid, "role": "main",
                              "record_sha256": fingerprint(records[qid])}
            path = cell / "main" / "originals" / f"{qid}.json"
            if path.exists():
                trace = load_envelope(path, trace_identity)
            else:
                trace = agent.run(ExtensionEnv(records[qid]), temperature=protocol["initial_temperature"],
                                  seed=protocol["initial_seed"], record_timing=True,
                                  meta={"model_name": client.model_name}).to_dict()
                save_envelope(path, trace, trace_identity)
            originals[qid] = trace
            unc_path = cell / "main" / "uncertainty" / f"{qid}.json"
            unc_identity = {**trace_identity, "original_sha256": fingerprint(trace)}
            uncertainties[qid] = (load_envelope(unc_path, unc_identity) if unc_path.exists() else
                                  save_envelope(unc_path, uncertainty_record(trace), unc_identity))
            print(json.dumps({"stage": "original", "dataset": dataset, "qid": qid}), flush=True)
        failures = [q for q in cohort["main_ids"] if not originals[q]["success"]]
        pairing = pair_by_length({q: len(originals[q]["steps"]) for q in failures},
                                 seed=protocol["pairing_seed"])
        save_envelope(cell / "pairing.json", pairing, cell_identity)
        cells[dataset] = {"cell": cell, "cell_identity": cell_identity, "records": records,
                          "profile": profile, "originals": originals, "uncertainties": uncertainties,
                          "failures": failures, "pairing": pairing}
        print(json.dumps({"stage": "pairing", "dataset": dataset, "failures": len(failures),
                          "pairs": len(pairing["pairs"]), "unmatched": len(pairing["unmatched_ids"])}), flush=True)

    for dataset, pair in build_schedule(protocol, cells):
        state = cells[dataset]
        cell, records, profile = state["cell"], state["records"], state["profile"]
        originals, uncertainties = state["originals"], state["uncertainties"]
        guard()
        identity_for_plan = plan_identity(state["cell_identity"], pair, originals, profile)
        path = cell / "plans" / f"{pair['pair_id']}.json"
        if path.exists():
            jobs = [{**job, "prefix": [Step.from_dict(s) for s in job["prefix"]]}
                    for job in load_envelope(path, identity_for_plan)]
        else:
            jobs = pair_jobs(protocol, pair, records, originals, uncertainties, profile)
            save_envelope(path, [{**job, "prefix": [s.to_dict() for s in job["prefix"]]} for job in jobs],
                          identity_for_plan)
        for job in jobs:
            guard()
            execution_identity = {**identity_for_plan,
                                  "job_sha256": fingerprint({**job, "prefix": [s.to_dict() for s in job["prefix"]]})}
            execution_path = cell / "executions" / f"{job['execution_id']}.json"
            if execution_path.exists():
                trace = load_envelope(execution_path, execution_identity)
            else:
                trace = agent.run(ExtensionEnv(records[job["qid"]]), temperature=protocol["repair_temperature"],
                    seed=job["seed"], prefix_steps=job["prefix"], nudge=protocol["retry_hint"],
                    token_budget=job["budget"], step_budget_mode="new", record_timing=True,
                    meta={"model_name": client.model_name, "execution_id": job["execution_id"],
                          "prompt_sha256": job["prompt_sha256"]}).to_dict()
                save_envelope(execution_path, trace, execution_identity)
            save_envelope(cell / "trials" / f"{job['execution_id']}.json",
                          trial_rows(trace, job, model_key=model_key, dataset=dataset), execution_identity)
        print(json.dumps({"stage": "pair", "dataset": dataset, "pair_id": pair["pair_id"],
                          "unique_executions": len(jobs)}), flush=True)

    for dataset, state in cells.items():
        pairing = state["pairing"]
        save_envelope(state["cell"] / "complete.json", {
            "main_questions": len(protocol["cohorts"][dataset]["main_ids"]),
            "main_failures": len(state["failures"]), "pairs": len(pairing["pairs"]),
            "matched_questions": pairing["n_matched_questions"],
            "unmatched_ids": pairing["unmatched_ids"],
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, state["cell_identity"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    run(args.package, args.output, args.cache, args.deadline_utc)
