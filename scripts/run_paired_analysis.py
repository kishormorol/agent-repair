"""Analyze a frozen policy against prespecified baselines from complete raw trials."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.eval.metrics import paired_mean_comparison, paired_macro_comparison, holm_correction
from src.utils import read_jsonl, save_json
from src.utils import load_json
from src.repair.controlled import fingerprint


def load_verified_results(paths, strategy, baselines, seeds, multiplier):
    frames = []
    datasets = set()
    for path in paths:
        manifest = load_json(Path(path).with_name("manifest.json"))
        payload = manifest["payload"]
        if manifest["sha256"] != fingerprint(payload):
            raise ValueError("Invalid repair manifest digest")
        dataset = payload["configuration"]["dataset"]["name"]
        if dataset in datasets:
            raise ValueError("Provide exactly one frozen result file per dataset")
        datasets.add(dataset)
        frame = pd.DataFrame(list(read_jsonl(path)))
        if frame.empty or not {"multiplier", "manifest_sha256", "dataset"} <= set(frame):
            raise ValueError("Missing result provenance")
        if not frame.manifest_sha256.eq(manifest["sha256"]).all() or not frame.dataset.eq(dataset).all():
            raise ValueError("Results do not match their frozen manifest")
        frame = frame[frame.multiplier == multiplier].copy()
        expected_ids = set(payload["question_ids"])
        for name in [strategy, *baselines]:
            observed = set(zip(frame.loc[frame.strategy == name, "qid"],
                               frame.loc[frame.strategy == name, "seed"]))
            if observed != {(qid, seed) for qid in expected_ids for seed in seeds}:
                raise ValueError("Results omit or add prespecified question/seed pairs")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--strategy", required=True, help="Frozen before inspecting these outcomes")
    parser.add_argument("--baselines", nargs="+", default=["full_restart", "random_step"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = load_verified_results(args.results, args.strategy, args.baselines,
                                   args.seeds, args.multiplier)
    if frame.empty or "dataset" not in frame or frame.dataset.isna().any():
        parser.error("Raw results must be nonempty and include dataset labels")
    primary, secondary = {}, {}
    if len(set(args.baselines)) != len(args.baselines):
        parser.error("Baseline contrasts must be unique")
    for baseline in args.baselines:
        primary[baseline] = paired_macro_comparison(
            frame, args.strategy, baseline, iters=args.iters, expected_seeds=args.seeds)
        secondary[baseline] = {
            dataset: {metric: paired_mean_comparison(
                group, args.strategy, baseline, iters=args.iters,
                expected_seeds=args.seeds, outcome=metric)
                for metric in ("success", "em", "f1")}
            for dataset, group in frame.groupby("dataset", sort=True)
        }
    adjusted = holm_correction([value["p_value"] for value in primary.values()])
    for value, p in zip(primary.values(), adjusted):
        value["p_value_holm"] = p
    save_json({"primary_macro_contrasts": primary, "exploratory_dataset_sensitivities": secondary,
               "note": "Expected question/seed coverage verified against repair manifests. "
                       "Policy selection must still be frozen externally before test outcomes are inspected."}, args.output)


if __name__ == "__main__":
    main()
