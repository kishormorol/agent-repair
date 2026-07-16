"""Master experiment runner — iterates over (model, dataset) pairs.

Usage:
    # Run ALL models x ALL datasets:
    python scripts/run_experiment.py --config config/config_experiment.yaml

    # Run specific model + dataset:
    python scripts/run_experiment.py --config config/config_experiment.yaml \
        --model qwen2.5-7b --dataset hotpotqa

    # Smoke test (20 questions):
    python scripts/run_experiment.py --config config/config_experiment.yaml \
        --model qwen2.5-7b --dataset hotpotqa --limit 20

    # Run all models on one dataset:
    python scripts/run_experiment.py --config config/config_experiment.yaml \
        --dataset hotpotqa

    # Run one model on all datasets:
    python scripts/run_experiment.py --config config/config_experiment.yaml \
        --model qwen2.5-7b

Each (model, dataset) run creates outputs under:
    outputs/{dataset}/{model_key}/{trajectories,uncertainty,...}

The pipeline is fully resumable — re-run to pick up where it left off.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.utils import load_config, get_logger


def load_models_catalog(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_datasets_catalog(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_experiment_matrix(cfg, args) -> list:
    """Build list of (model_key, model_info, dataset_name, dataset_info) tuples."""
    exp = cfg.raw.get("experiment", {})
    models_file = exp.get("models_file", "config/models.yaml")
    datasets_file = exp.get("datasets_file", "config/datasets.yaml")

    models_cat = load_models_catalog(models_file)
    datasets_cat = load_datasets_catalog(datasets_file)

    # Resolve which models to run
    tiers = exp.get("tiers", ["small", "medium", "large"])
    use_awq = exp.get("use_awq", True)
    target_model = getattr(args, "model", None)
    target_dataset = getattr(args, "dataset", None)

    models = []
    for tier in tiers:
        if tier not in models_cat:
            continue
        for key, info in models_cat[tier].items():
            if target_model and key != target_model:
                continue
            name = info.get("awq", info["name"]) if use_awq else info["name"]
            models.append((key, {
                "name": name,
                "dtype": info.get("dtype", "auto"),
                "gpu_memory_utilization": info.get("gpu_memory_utilization", 0.90),
                "family": info.get("family", "unknown"),
                "tier": tier,
            }))

    datasets = []
    for ds_name in exp.get("datasets", ["hotpotqa"]):
        if target_dataset and ds_name != target_dataset:
            continue
        if ds_name in datasets_cat:
            datasets.append((ds_name, datasets_cat[ds_name]))

    matrix = []
    for model_key, model_info in models:
        for ds_name, ds_info in datasets:
            matrix.append((model_key, model_info, ds_name, ds_info))

    return matrix


def setup_experiment_dir(base: str, dataset: str, model_key: str) -> dict:
    """Create output directory structure for one experiment."""
    exp_dir = os.path.join(base, "outputs", dataset, model_key)
    subdirs = {
        "trajectories": os.path.join(exp_dir, "trajectories"),
        "uncertainty": os.path.join(exp_dir, "uncertainty"),
        "annotations": os.path.join(exp_dir, "annotations"),
        "localization": os.path.join(exp_dir, "localization"),
        "repairs": os.path.join(exp_dir, "repairs"),
        "tables": os.path.join(exp_dir, "tables"),
        "figures": os.path.join(exp_dir, "figures"),
        "logs": os.path.join(exp_dir, "logs"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    return subdirs


def write_experiment_config(base_cfg: dict, model_info: dict,
                            ds_info: dict, ds_name: str,
                            exp_dirs: dict, out_path: str) -> None:
    """Write a temporary per-experiment config YAML."""
    cfg = dict(base_cfg)  # shallow copy

    # Override model
    cfg["models"] = dict(cfg["models"])
    cfg["models"]["agent"] = {
        "name": model_info["name"],
        "dtype": model_info["dtype"],
        "max_model_len": cfg["models"]["agent"]["max_model_len"],
        "gpu_memory_utilization": model_info["gpu_memory_utilization"],
    }

    # Override dataset
    cfg["dataset"] = dict(cfg["dataset"])
    cfg["dataset"]["name"] = ds_name
    cfg["dataset"]["raw_filename"] = ds_info.get("raw_filename", "")
    cfg["dataset"]["pool_size"] = ds_info.get("pool_size", 500)
    cfg["dataset"]["stratify_by"] = ds_info.get("stratify_by", ["type"])
    if ds_info.get("url"):
        cfg["dataset"]["url"] = ds_info["url"]

    # Override paths to point to experiment-specific dirs
    cfg["paths"] = dict(cfg["paths"])
    for key, path in exp_dirs.items():
        cfg["paths"][key] = os.path.relpath(path, cfg["paths"].get("local_base", "./"))

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_stage(script: str, config: str, limit: int = None,
              dataset: str = None, log=None) -> bool:
    """Run a pipeline stage script as a subprocess."""
    cmd = [sys.executable, script, "--config", config]
    if limit:
        cmd += ["--limit", str(limit)]
    if dataset:
        cmd += ["--dataset", dataset]

    if log:
        log.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Master experiment runner")
    p.add_argument("--config", default="config/config_experiment.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=None, help="Run only this model key")
    p.add_argument("--dataset", default=None, help="Run only this dataset")
    p.add_argument("--stages", default="0,1,2,3,4,5,6",
                   help="Comma-separated stages to run (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the experiment matrix without running")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = get_logger("experiment", cfg.path("logs"))

    matrix = build_experiment_matrix(cfg, args)
    stages = [int(s) for s in args.stages.split(",")]

    stage_scripts = {
        0: "scripts/run_setup.py",
        1: "scripts/run_generate.py",
        2: "scripts/run_uncertainty.py",
        3: "scripts/run_annotate.py",
        4: "scripts/run_localize.py",
        5: "scripts/run_repair.py",
        6: "scripts/run_eval.py",
    }

    log.info(f"Experiment matrix: {len(matrix)} runs "
             f"({len(set(m[0] for m in matrix))} models x "
             f"{len(set(m[2] for m in matrix))} datasets)")
    log.info(f"Stages: {stages}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("EXPERIMENT MATRIX (dry run)")
        print("=" * 70)
        for i, (mk, mi, dn, di) in enumerate(matrix, 1):
            print(f"  {i:3d}. {mk:20s} x {dn:20s}  ({mi['tier']}, {mi['family']})")
        print(f"\nTotal: {len(matrix)} runs")
        return

    # Track results
    results = []
    t_start = time.time()

    for run_idx, (model_key, model_info, ds_name, ds_info) in enumerate(matrix, 1):
        print(f"\n{'=' * 70}")
        print(f"  RUN {run_idx}/{len(matrix)}: {model_key} x {ds_name}")
        print(f"  Model: {model_info['name']}")
        print(f"  Tier: {model_info['tier']}, Family: {model_info['family']}")
        print(f"{'=' * 70}")

        # Setup experiment directory
        exp_dirs = setup_experiment_dir(cfg.base, ds_name, model_key)

        # Write temporary config for this experiment
        tmp_config = os.path.join(exp_dirs["logs"], "experiment_config.yaml")
        write_experiment_config(cfg.raw, model_info, ds_info, ds_name,
                                exp_dirs, tmp_config)

        run_result = {"model": model_key, "dataset": ds_name,
                      "tier": model_info["tier"], "family": model_info["family"],
                      "stages_ok": [], "stages_fail": []}

        for stage in stages:
            if stage not in stage_scripts:
                continue
            script = stage_scripts[stage]
            log.info(f"[{model_key} x {ds_name}] Stage {stage}")
            ok = run_stage(script, tmp_config, args.limit, ds_name, log)
            if ok:
                run_result["stages_ok"].append(stage)
            else:
                run_result["stages_fail"].append(stage)
                log.error(f"[{model_key} x {ds_name}] Stage {stage} FAILED")
                break  # don't continue if a stage fails

        results.append(run_result)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE ({elapsed / 3600:.1f} hours)")
    print(f"{'=' * 70}")
    n_ok = sum(1 for r in results if not r["stages_fail"])
    print(f"  Successful: {n_ok}/{len(results)}")
    for r in results:
        status = "OK" if not r["stages_fail"] else f"FAILED at stage {r['stages_fail'][0]}"
        print(f"  {r['model']:20s} x {r['dataset']:20s} — {status}")

    # Save results log
    results_path = os.path.join(cfg.base, "outputs", "experiment_log.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "elapsed_hours": elapsed / 3600}, f, indent=2)
    print(f"\nResults log: {results_path}")


if __name__ == "__main__":
    main()
