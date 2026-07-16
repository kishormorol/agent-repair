"""Stage 3 — ground-truth error annotation (batched 72B judge, unattended).

    python scripts/run_annotate.py --config config/config_local.yaml

Signal A (programmatic gold-retrieval check) + Signal B (LLM judge).
Human validation is a separate interactive script: scripts/label_human.py
"""
from __future__ import annotations

import os
import time
from collections import Counter

from _common import parse_args, boot  # type: ignore

from src.llm import VLLMClient, resolve_judge_model, gpu_vram_gb
from src.annotate import (programmatic_retrieval_check, build_judge_prompt,
                          parse_judge_output, combine_annotation)
from src.utils import Checkpoint, save_item, load_item, load_json


def main() -> None:
    args = parse_args("Ground-truth error annotation (batched judge)")
    cfg, log = boot("stage3", args)

    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    if args.limit:
        failed_ids = failed_ids[:args.limit]
    pool = {r["_id"]: r for r in
            load_json(os.path.join(cfg.path("data_processed"), "pool.json"))}

    def gold_facts_str(rec):
        ctx = {t: sents for t, sents in rec["context"]}
        out = []
        for title, sid in rec["supporting_facts"]:
            if title in ctx and 0 <= sid < len(ctx[title]):
                out.append(f"{title}: {ctx[title][sid]}")
        return " | ".join(out)

    def load_failed(qid):
        t = load_item(cfg.path("trajectories"), qid)
        t["gold_supporting_facts_str"] = gold_facts_str(pool[qid])
        return t

    ann_dir = cfg.path("annotations")
    ckpt = Checkpoint(os.path.join(cfg.path("logs"), "stage3_annotate.jsonl"))
    todo = [q for q in failed_ids if not ckpt.is_done(q)]
    log.info(f"To annotate: {len(todo)} of {len(failed_ids)} failed trajectories")

    if todo:
        jm = resolve_judge_model(cfg)
        log.info(f"GPU VRAM: {gpu_vram_gb()} GB | judge: {jm['name']} ({jm['reason']})")
        judge = VLLMClient(jm["name"], dtype=jm["dtype"],
                           max_model_len=cfg.models.judge.max_model_len,
                           gpu_memory_utilization=jm["gpu_memory_utilization"],
                           logprobs_topk=1, seed=cfg.project.seed).load()

        etypes = cfg.raw["annotation"]["error_types"]
        B = cfg.raw["runtime"]["judge_batch_size"]
        t_start = time.time()

        for i in range(0, len(todo), B):
            batch_ids = todo[i:i + B]
            trajs = [load_failed(q) for q in batch_ids]
            rcs = [programmatic_retrieval_check(t) for t in trajs]
            prompts = [build_judge_prompt(t, etypes, rc) for t, rc in zip(trajs, rcs)]
            outs = judge.chat_batch(
                prompts, temperature=cfg.raw["annotation"]["judge_temperature"],
                max_tokens=128)
            for q, t, rc, o in zip(batch_ids, trajs, rcs, outs):
                parsed = parse_judge_output(o.text, len(t["steps"]), etypes)
                save_item(ann_dir, q, combine_annotation(q, parsed, rc))
                ckpt.mark_done(q)
            done = min(i + B, len(todo))
            rate = done / max(1e-9, time.time() - t_start)
            log.info(f"{done}/{len(todo)}  ({rate:.2f} traj/s, "
                     f"ETA {(len(todo)-done)/max(1e-9,rate)/60:.0f} min)")

    labs = [load_item(ann_dir, q) for q in failed_ids if
            os.path.exists(os.path.join(ann_dir, f"{q}.json"))]
    log.info(f"Annotated: {len(labs)}")
    log.info(f"error types: {Counter(l['error_type'] for l in labs)}")
    log.info(f"sources    : {Counter(l['source'] for l in labs)}")
    log.info("Next: run `python scripts/label_human.py` to hand-label 50 and check agreement.")


if __name__ == "__main__":
    main()
