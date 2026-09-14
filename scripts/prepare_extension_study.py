"""Freeze official inputs and disjoint cohorts before extension outcomes exist."""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import json
from pathlib import Path
import random
import re
import shutil
import string
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.extension import convert_musique, convert_2wiki, score_record
from src.env.musique_env import sample_pool
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import TREATMENT, DIAGNOSIS_STRATEGY, DIAGNOSIS_PROMPT, DIAGNOSIS_SETTINGS
from src.agent.react_agent import SYSTEM_PROMPT
from src.utils.cloud_runs import write_once

MODELS = {
    "qwen32b": {"repo_id": "Qwen/Qwen2.5-32B-Instruct-AWQ", "revision": "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c", "dtype": "auto"},
    "mistral12b": {"repo_id": "mistralai/Mistral-Nemo-Instruct-2407", "revision": "04d8a90549d23fc6bd7f642064003592df51e9b3", "dtype": "bfloat16"},
}
STRATEGIES = ["full_restart", "random_step__bt2", "fixed_early", TREATMENT,
              "position_matched_random", "unc__perplexity__argmax", DIAGNOSIS_STRATEGY]
HINT = "Your previous attempt at this step may have been wrong. Reconsider carefully."


def official_functions(path, dataset):
    names = ({"normalize_answer", "get_tokens", "compute_exact", "compute_f1"}
             if dataset == "musique" else {"normalize_answer", "exact_match_score", "f1_score"})
    tree = ast.parse(Path(path).read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != names:
        raise ValueError("Incomplete official scorer")
    namespace = {"re": re, "string": string, "collections": collections, "Counter": collections.Counter}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def official_score(prediction, record, functions):
    if prediction is None:
        return {"em": 0., "f1": 0., "correct": False}
    golds = [record["answer"], *record.get("answer_aliases", [])]
    if record.get("scoring") == "musique":
        em = max(functions["compute_exact"](g, prediction) for g in golds)
        f1 = max(functions["compute_f1"](g, prediction) for g in golds)
    else:
        em = max(functions["exact_match_score"](prediction, g) for g in golds)
        f1 = max(functions["f1_score"](prediction, g)[0] for g in golds)
    return {"em": float(em), "f1": float(f1), "correct": bool(em or f1 >= .5)}


def freeze_file(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen input changed: {path}; use a new package")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def prepare(source, output, n=100, development_n=20):
    source, output = Path(source), Path(output)
    previous = ROOT / "output/aws-experiment/2026-09-14-diagnosis-v2/prepared"
    hotpot_path = previous / "raw/hotpot_dev_distractor_v1.json"
    hotpot = json.loads(hotpot_path.read_text())
    with zipfile.ZipFile(source / "musique-official.zip") as archive:
        musique_bytes = archive.read("data/musique_ans_v1.0_dev.jsonl")
    if musique_bytes != (source / "musique-mirror-dev.jsonl").read_bytes():
        raise ValueError("MuSiQue mirror differs from the official development split")
    musique_raw = [json.loads(line) for line in musique_bytes.splitlines()]
    musique = [convert_musique(record) for record in musique_raw]
    with zipfile.ZipFile(source / "2wiki-official.zip") as archive:
        wiki_bytes = archive.read("dev.json")
        aliases_bytes = archive.read("id_aliases.json")
    if wiki_bytes != (source / "dev.json").read_bytes() or aliases_bytes != (source / "id_aliases.json").read_bytes():
        raise ValueError("2Wiki extracted inputs differ from the official archive")
    aliases = {entry["Q_id"]: entry for entry in map(json.loads, aliases_bytes.splitlines())}
    wiki_raw = json.loads(wiki_bytes)
    wiki = [convert_2wiki(record, aliases) for record in wiki_raw]
    data = {"hotpotqa": hotpot, "musique": musique, "2wikimultihopqa": wiki}
    if [len(data[d]) for d in data] != [7405, 2417, 12576]:
        raise ValueError("Official dataset counts changed")
    references = {"hotpotqa": previous / "hotpot_evaluate_v1.reference.py",
                  "musique": source / "musique-answer.py", "2wikimultihopqa": source / "2wiki-evaluate.py"}
    prior = json.loads((previous / "main/protocol.json").read_text())["payload"]
    exclusions = {"hotpotqa": set(prior["explored_ids"]) | set(prior["question_ids"])}
    for dataset in ["musique", "2wikimultihopqa"]:
        exclusions[dataset] = {r["_id"] for r in sample_pool(data[dataset], 500, ["type"], 42)}
    # The historical 2Wiki loader tried this mirror first. Exclude both
    # reconstructed sample orders, even if the mirror reordered its records.
    import pandas as pd
    mirror = pd.read_parquet(source / "2wiki-mirror-dev.parquet").to_dict("records")
    mirror_ids = [{"_id": r.get("_id", r.get("id")), "type": r.get("type", "bridge")} for r in mirror]
    exclusions["2wikimultihopqa"].update(r["_id"] for r in sample_pool(mirror_ids, 500, ["type"], 42))
    cohorts, audit = {}, {}
    for index, (dataset, records) in enumerate(data.items()):
        by_id = {r["_id"]: r for r in records}
        if len(by_id) != len(records) or not exclusions[dataset] <= set(by_id):
            raise ValueError("Dataset identity or historical exclusions changed")
        seed = 20260914 + index
        eligible = sorted(set(by_id) - exclusions[dataset])
        ids = random.Random(seed).sample(eligible, n + development_n)
        cohorts[dataset] = {"main_ids": ids[development_n:], "development_ids": ids[:development_n],
                            "excluded_ids": sorted(exclusions[dataset]), "sampling_seed": seed,
                            "eligible_count": len(eligible), "record_count": len(records)}
        write_once(output / "data" / f"{dataset}.json", [by_id[q] for q in ids])
        frozen_reference = output / "references" / f"{dataset}.py"
        freeze_file(frozen_reference, references[dataset].read_bytes())
        official = official_functions(frozen_reference, dataset)
        checks = 0
        for record in records:
            predictions = [None, "", "yes", "no", "noanswer", record["answer"],
                           "wrong " + record["answer"], *record.get("answer_aliases", [])[:2]]
            for prediction in predictions:
                if score_record(prediction, record) != official_score(prediction, record, official):
                    raise ValueError(f"Official scorer mismatch: {dataset}/{record['_id']}")
                checks += 1
        audit[dataset] = {"official_scorer_checks": checks, "mismatches": 0,
                          "dataset_records": len(records), "excluded_ids": len(exclusions[dataset]),
                          "selected_records": len(ids)}
    source_inputs = {p.name: file_fingerprint(p) for p in source.iterdir() if p.is_file()}
    source_inputs["hotpot_dev_distractor_v1.json"] = file_fingerprint(hotpot_path)
    payload = {
        "schema": 1, "run_id": "aws119-extension-20260914-v1", "frozen_date_utc": "2026-09-14",
        "models": MODELS, "cohorts": cohorts, "strategies": STRATEGIES, "seeds": [0, 1, 2],
        "main_n_per_dataset_model": n, "development_n_per_dataset_model": development_n,
        "max_new_steps": 8, "max_tokens_per_step": 512, "max_model_len": 16384,
        "initial_seed": 42, "initial_temperature": 0., "repair_temperature": .7,
        "top_p": .95, "logprobs_topk": 20, "retry_hint": HINT,
        "generated_token_multiplier": 1., "prefix_caching": False, "batch_size": 1,
        "system_prompt_sha256": fingerprint(SYSTEM_PROMPT),
        "diagnosis_prompt_sha256": fingerprint(DIAGNOSIS_PROMPT), "diagnosis_settings": DIAGNOSIS_SETTINGS,
        "primary_controls": ["full_restart", "position_matched_random", "unc__perplexity__argmax", DIAGNOSIS_STRATEGY],
        "primary_family": "24 two-sided paired question sign-flip tests; Holm across six model/dataset cells and four controls",
        "bootstrap_iters": 10000, "analysis_seed": 20260914,
        "runtime": {"model": "qwen32b", "dataset": "hotpotqa", "n_failures_max": 25,
                    "selection": "First 25 main initial failures in frozen question order; use all if fewer",
                    "strategies": ["full_restart", TREATMENT, DIAGNOSIS_STRATEGY],
                    "deadlines_s": [5., 10., 20.], "primary_deadline_s": 10.,
                    "primary_family": "Two comparisons at 10 seconds, separate Holm family; 5/20 seconds exploratory",
                    "token_budget": None,
                    "measurement": "Sequential warmed-model completion latency, including origin acquisition and prefix replay; no token cap, eight new steps. Outcomes count only if completed within the declared deadline. Full attempts are measured; no claim of compute saved by hard cancellation."},
        "history_limitation": "Excludes documented/reconstructed prior cohorts. Original complete historical Drive pools remain unavailable for byte-level comparison.",
        "precision_note": "Resource-bounded replication; no promised power or equivalence conclusion. The same questions are paired across models, but their initial failure cohorts may differ.",
        "retrieval_note": "MuSiQue retains full paragraphs as lookup units, concatenating repeated titles in source order; HotpotQA and 2Wiki retain source sentences. No support labels enter agent prompts.",
        "position_profile": "Fit separately per model/dataset on the 20 disjoint development originals. If none fail, use origin zero and report this fallback.",
        "execution_order": "Seeded shuffle of unique origin/seed jobs within each question; no cross-request prefix caching. Runtime trials execute every policy separately in seeded shuffled order.",
        "completion": "All frozen questions and applicable policy/seed rows, independent official rescoring and tool/prefix replay. Partial cohorts remain incomplete.",
        "allocation": {"total_usd": 119., "reserve_usd": 20., "session_gross_ceiling_usd": 35.,
                       "prior_gross_conservative_bound_usd": 45., "maximum_minutes": 330,
                       "maximum_hourly_usd": 5.84531, "overhead_allowance_usd": 2.5},
        "source_input_sha256": source_inputs,
    }
    write_once(output / "protocol.json", {"sha256": fingerprint(payload), "payload": payload})
    write_once(output / "data-audit.json", audit)
    source_files = [p for folder in ["src", "tests"] for p in (ROOT / folder).rglob("*.py")]
    source_files += list((ROOT / "scripts").glob("*.py"))
    source_files += list((ROOT / "config").glob("*.yaml"))
    for path in source_files:
        freeze_file(output / "code" / path.relative_to(ROOT), path.read_bytes())
    input_hashes = {str(p.relative_to(output)): file_fingerprint(p) for p in sorted(output.rglob("*"))
                   if p.is_file() and p.name != "package-manifest.json" and "__pycache__" not in p.parts}
    write_once(output / "package-manifest.json", {"sha256": fingerprint(input_hashes), "files": input_hashes})
    print(json.dumps({"protocol_sha256": fingerprint(payload), "cohorts": {d: {k: v for k, v in a.items() if k != "excluded_ids"} for d, a in cohorts.items()}, "data_audit": audit}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--development-n", type=int, default=20)
    args = parser.parse_args()
    prepare(args.source, args.output, args.n, args.development_n)
