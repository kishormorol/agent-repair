"""Freeze a position-only control from a completed development run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair.controlled import file_fingerprint
from src.repair.position_matched import fit_position_profile
from src.utils.cloud_runs import write_once
from src.utils.io import load_json


def fit_from_run(run, strategy, output):
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg.get("notebook_provenance", {}).get("phase") != "development":
        raise ValueError("Use a completed development run, never test outcomes")
    pool = load_json(run / "data/processed/pool.json")
    ids = [r["_id"] for r in pool]
    failed = load_json(run / "data/processed/failed_ids.json")
    if len(failed) != len(set(failed)) or not set(failed) <= set(ids):
        raise ValueError("Development failures do not match the saved pool")
    initial_paths = [run / "outputs/trajectories" / (q + ".json") for q in ids]
    all_originals = [load_json(p) for p in initial_paths]
    if {t["qid"] for t in all_originals if not t["success"]} != set(failed):
        raise ValueError("Development failure gate differs from the saved failed IDs")
    originals = [t for t in all_originals if t["qid"] in failed]
    uncertainty_paths = [run / "outputs/uncertainty" / (q + ".json") for q in failed]
    uncertainties = {q: load_json(p) for q, p in zip(failed, uncertainty_paths)}
    model = load_json(run / "model_snapshot.json")
    inputs = [run / "config.yaml", run / "data/processed/pool.json",
              run / "data/processed/failed_ids.json", *initial_paths, *uncertainty_paths]
    profile = fit_position_profile(originals, uncertainties, source_ids=ids,
        dataset=cfg["dataset"]["name"], strategy=strategy, source_phase="development",
        source_run_id=cfg["repair"]["run_id"],
        model={"repo_id": model["repo_id"], "revision": model["revision"]},
        source_sha256={str(p.relative_to(run)): file_fingerprint(p) for p in inputs})
    write_once(output, profile)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit_from_run(args.development_run, args.strategy, args.output)
    print(json.dumps({"sha256": result["sha256"],
                      "development_questions": len(result["payload"]["source_question_ids"]),
                      "fitted_origins": len(result["payload"]["origins"])}, indent=2))


if __name__ == "__main__":
    main()
