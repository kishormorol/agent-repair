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
    if extra:
        extra(p)
    return p.parse_args()


def boot(stage_name: str, args: argparse.Namespace):
    cfg = load_config(args.config)
    log = get_logger(stage_name, cfg.path("logs"))
    log.info(f"config={args.config}  base={cfg.base}")
    return cfg, log


def load_agent(cfg, log):
    """Load the agent model (auto fp16/AWQ by VRAM)."""
    from src.llm import VLLMClient, resolve_agent_model, gpu_vram_gb
    m = resolve_agent_model(cfg)
    log.info(f"GPU VRAM: {gpu_vram_gb()} GB | agent model: {m['name']} ({m['reason']})")
    client = VLLMClient(m["name"], dtype=m["dtype"],
                        max_model_len=cfg.models.agent.max_model_len,
                        gpu_memory_utilization=m["gpu_memory_utilization"],
                        logprobs_topk=cfg.agent.logprobs_topk,
                        seed=cfg.project.seed).load()
    return client
