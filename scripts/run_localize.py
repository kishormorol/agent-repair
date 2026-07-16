"""Stage 4 — error localization (CPU, no model).

    python scripts/run_localize.py --config config/config_local.yaml
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _common import parse_args, boot  # type: ignore

from src.localize import evaluate_localization
from src.utils import load_item, load_json, save_json


def main() -> None:
    args = parse_args("Error localization scoring")
    cfg, log = boot("stage4", args)

    topks_abl = cfg.raw["uncertainty"]["entropy_topk_ablation"]
    metric_keys = ["token_entropy_mean", "token_entropy_max", "perplexity",
                   "max_token_prob_mean", "max_token_prob_max",
                   "self_consistency", "verbalized_confidence"]
    metric_keys += [f"token_entropy_mean_k{k}" for k in topks_abl]

    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    if args.limit:
        failed_ids = failed_ids[:args.limit]
    topk_list = cfg.raw["localization"]["topk"]
    thr = cfg.raw["localization"]["threshold_percentile"]

    per_traj = []
    for q in failed_ids:
        unc = load_item(cfg.path("uncertainty"), q)
        ann = load_item(cfg.path("annotations"), q)
        for mk in metric_keys:
            r = evaluate_localization(unc, ann["broken_step"], mk, topk_list, thr)
            if not r.get("has_scores"):
                continue
            r.update({"qid": q, "metric": mk, "error_type": ann["error_type"]})
            per_traj.append(r)

    save_json(per_traj, os.path.join(cfg.path("localization"), "per_traj.json"))
    df = pd.DataFrame(per_traj)
    log.info(f"records: {len(df)} ({len(failed_ids)} trajectories x {len(metric_keys)} metrics)")

    agg_cols = ["argmax_top1", "argmax_within1", "mrr", "threshold_hit",
                "threshold_within1"] + [f"top{k}_hit" for k in topk_list]
    summary = df.groupby("metric")[agg_cols].mean().sort_values(
        "argmax_top1", ascending=False)
    summary["n"] = df.groupby("metric").size()
    summary = summary.round(3)
    out_csv = os.path.join(cfg.path("tables"), "localization_summary.csv")
    summary.to_csv(out_csv)
    print(summary.to_string())

    rand_top1 = float(np.mean([r["n_steps"] and 1.0 / r["n_steps"]
                               for r in per_traj if r["metric"] == metric_keys[0]]))
    best = summary["argmax_top1"].idxmax()
    save_json({"random_top1": rand_top1, "best_metric": best,
               "best_top1": float(summary.loc[best, "argmax_top1"])},
              os.path.join(cfg.path("tables"), "localization_headline.json"))
    log.info(f"Random top-1 baseline: {rand_top1:.3f} | best metric: {best} "
             f"({summary.loc[best, 'argmax_top1']:.3f})")
    log.info(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
