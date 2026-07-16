"""Shared bootstrap for the CLI stage scripts."""
from __future__ import annotations

import argparse
import os
import sys

# make `src` importable when run as `python scripts/xxx.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_logger  # noqa: E402

DEFAULT_CONFIG = "config/config_local.yaml"


def parse_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to config yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N items (smoke test)")
    p.add_argument("--dataset", default=None,
                   help="dataset name (hotpotqa, fever, 2wikimultihopqa, musique)")
    p.add_argument("--model", default=None,
                   help="model key from models.yaml (e.g. qwen2.5-7b)")
    p.add_argument("--model-name", default=None,
                   help="explicit HuggingFace model name (overrides --model)")
    if extra:
        extra(p)
    return p.parse_args()


def boot(stage_name: str, args: argparse.Namespace):
    cfg = load_config(args.config)
    log = get_logger(stage_name, cfg.path("logs"))
    log.info(f"config={args.config}  base={cfg.base}")
    return cfg, log


def resolve_dataset(cfg, args):
    """Resolve dataset env class, loader, sampler, scorer from args or config.

    Returns: (dataset_name, env_cls, load_fn, sample_fn, score_fn, raw_filename,
              pool_size, stratify_by)
    """
    from src.env import get_dataset, DATASET_REGISTRY
    import yaml

    dataset_name = getattr(args, "dataset", None) or cfg.raw.get("dataset", {}).get("name", "hotpotqa")

    # Default to hotpotqa if the name isn't in registry
    if dataset_name not in DATASET_REGISTRY:
        # Try to match partial names
        for k in DATASET_REGISTRY:
            if dataset_name.lower().replace("_", "") in k.lower().replace("_", ""):
                dataset_name = k
                break

    ds_info = get_dataset(dataset_name)

    # Load dataset catalog for metadata
    datasets_file = cfg.raw.get("experiment", {}).get("datasets_file", "config/datasets.yaml")
    ds_meta = {}
    if os.path.exists(datasets_file):
        with open(datasets_file) as f:
            all_ds = yaml.safe_load(f) or {}
        ds_meta = all_ds.get(dataset_name, {})

    raw_filename = ds_meta.get("raw_filename", cfg.raw["dataset"].get("raw_filename", ""))
    pool_size = ds_meta.get("pool_size", cfg.raw["dataset"].get("pool_size", 500))
    stratify_by = ds_meta.get("stratify_by", cfg.raw["dataset"].get("stratify_by", ["type"]))

    return {
        "name": dataset_name,
        "env_cls": ds_info["env_cls"],
        "load": ds_info["load"],
        "sample": ds_info["sample"],
        "download": ds_info["download"],
        "score": ds_info["score"],
        "raw_filename": raw_filename,
        "pool_size": pool_size,
        "stratify_by": stratify_by,
    }


def resolve_model(cfg, args, log=None):
    """Resolve model name from --model or --model-name args, or fall back to config."""
    import yaml

    model_name = getattr(args, "model_name", None)
    model_key = getattr(args, "model", None)

    if model_name:
        # Explicit model name given
        return {"name": model_name, "dtype": "auto",
                "gpu_memory_utilization": 0.90,
                "reason": f"explicit: {model_name}"}

    if model_key:
        # Look up in models.yaml
        models_file = cfg.raw.get("experiment", {}).get("models_file", "config/models.yaml")
        if os.path.exists(models_file):
            with open(models_file) as f:
                all_models = yaml.safe_load(f) or {}
            # Search across tiers
            for tier in ["small", "medium", "large"]:
                if tier in all_models and model_key in all_models[tier]:
                    m = all_models[tier][model_key]
                    use_awq = cfg.raw.get("experiment", {}).get("use_awq", True)
                    name = m.get("awq", m["name"]) if use_awq else m["name"]
                    return {"name": name,
                            "dtype": m.get("dtype", "auto"),
                            "gpu_memory_utilization": m.get("gpu_memory_utilization", 0.90),
                            "reason": f"models.yaml: {model_key} ({'AWQ' if use_awq else 'fp16'})"}
        if log:
            log.warning(f"Model key '{model_key}' not found in models.yaml")

    # Fall back to config's default or auto-resolution
    return None  # let the caller use resolve_agent_model()


def load_agent(cfg, log, args=None):
    """Load the agent model (auto fp16/AWQ by VRAM, or from --model flag)."""
    from src.llm import VLLMClient, resolve_agent_model, gpu_vram_gb

    model_override = resolve_model(cfg, args, log) if args else None
    m = model_override or resolve_agent_model(cfg)
    log.info(f"GPU VRAM: {gpu_vram_gb()} GB | agent model: {m['name']} ({m['reason']})")
    client = VLLMClient(m["name"], dtype=m["dtype"],
                        max_model_len=cfg.models.agent.max_model_len,
                        gpu_memory_utilization=m["gpu_memory_utilization"],
                        logprobs_topk=cfg.agent.logprobs_topk,
                        seed=cfg.project.seed).load()
    return client


def experiment_output_dir(cfg, dataset_name: str, model_key: str = None) -> str:
    """Get the output directory for a specific (dataset, model) experiment.

    Structure: {base}/outputs/{dataset}/{model_key}/
    If model_key is None, uses the flat structure (backwards compatible).
    """
    if model_key:
        return os.path.join(cfg.base, "outputs", dataset_name, model_key)
    return os.path.join(cfg.base, "outputs")
