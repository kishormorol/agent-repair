"""Stage 1 — batched ReAct trajectory generation.

    python scripts/run_generate.py --config config/config_local.yaml
    python scripts/run_generate.py --config config/config_colab.yaml --dataset fever

Resumable: re-run after an interruption and it skips finished questions.
"""
from __future__ import annotations

import os
import time

from _common import parse_args, boot, load_agent, resolve_dataset  # type: ignore

from src.agent import run_generation_batch
from src.utils import (Checkpoint, save_item, save_json, load_json,
                       load_all_items)


def main() -> None:
    args = parse_args("Generate ReAct trajectories (batched)")
    cfg, log = boot("stage1", args)

    ds_info = resolve_dataset(cfg, args)
    load_dataset = ds_info["load"]
    sample_pool = ds_info["sample"]
    env_cls = ds_info["env_cls"]
    log.info(f"Dataset: {ds_info['name']} (env: {env_cls.__name__})")

    # ---- pool (sampled once, then fixed) ---------------------------------- #
    pool_path = os.path.join(cfg.path("data_processed"), "pool.json")
    if os.path.exists(pool_path):
        pool = load_json(pool_path)
    else:
        raw_filename = ds_info["raw_filename"]
        raw = load_dataset(os.path.join(cfg.path("data_raw"), raw_filename))
        pool = sample_pool(raw, ds_info["pool_size"], ds_info["stratify_by"],
                           cfg.project.seed)
        save_json(pool, pool_path)
    log.info(f"Pool: {len(pool)} questions")

    limit = args.limit or cfg.raw["dataset"].get("process_limit")
    run_pool = pool[:limit] if limit else pool

    ckpt = Checkpoint(os.path.join(cfg.path("logs"), "stage1_generate.jsonl"))
    todo = [r for r in run_pool if not ckpt.is_done(r["_id"])]
    log.info(f"To process: {len(todo)} of {len(run_pool)} "
             f"({len(run_pool) - len(todo)} already done)")
    if not todo:
        log.info("Nothing to do.")
    else:
        client = load_agent(cfg, log)
        B = cfg.raw["runtime"]["gen_batch_size"]
        traj_dir = cfg.path("trajectories")
        t_start = time.time()

        for i in range(0, len(todo), B):
            chunk = todo[i:i + B]
            trajs = run_generation_batch(
                client, chunk,
                max_steps=cfg.agent.max_steps,
                max_tokens_per_step=cfg.agent.max_tokens_per_step,
                temperature=cfg.agent.temperature,
                seed=cfg.project.seed,
                env_cls=env_cls,
                score_fn=ds_info["score"],
            )
            for t in trajs:
                save_item(traj_dir, t.qid, t.to_dict())
                ckpt.mark_done(t.qid, {"success": t.success, "steps": len(t.steps)})
            done = min(i + B, len(todo))
            rate = done / max(1e-9, time.time() - t_start)
            eta = (len(todo) - done) / max(1e-9, rate) / 60
            log.info(f"{done}/{len(todo)}  ({rate:.2f} q/s, ETA {eta:.0f} min)")

    # ---- summary + failed set --------------------------------------------- #
    trajs = load_all_items(cfg.path("trajectories"))
    n = len(trajs)
    n_succ = sum(t["success"] for t in trajs)
    failed_ids = [t["qid"] for t in trajs if not t["success"]]
    save_json(failed_ids, os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    log.info(f"TOTAL {n} | success {n_succ} ({100*n_succ/max(1,n):.1f}%) "
             f"| FAILED {len(failed_ids)}")
    log.info("Wrote failed_ids.json — input for Stages 2-5.")


if __name__ == "__main__":
    main()
