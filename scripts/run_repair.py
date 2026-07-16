"""Stage 5 — repair experiments (batched + deduplicated).

    python scripts/run_repair.py --config config/config_local.yaml

18 strategies (3 baselines + 5 metrics x 3 rules) x seeds, on every failed
trajectory. Two efficiencies:
  * DEDUP  — strategies that pick the SAME step share one repair (same step +
             same seed => same result), so we run at most `n_steps` repairs per
             trajectory instead of 18.
  * BATCH  — the unique repairs are executed many-at-once on the GPU.
Resumable: checkpointed per (qid, strategy, seed).
"""
from __future__ import annotations

import os
import time

from _common import parse_args, boot, load_agent  # type: ignore

from src.agent import Step, run_repair_batch
from src.repair import (build_strategies, select_target_step, repair_record,
                        make_rng)
from src.utils import Checkpoint, load_item, load_json, append_jsonl


def main() -> None:
    args = parse_args("Repair experiments (batched + dedup)")
    cfg, log = boot("stage5", args)

    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    if args.limit:
        failed_ids = failed_ids[:args.limit]
    pool = {r["_id"]: r for r in
            load_json(os.path.join(cfg.path("data_processed"), "pool.json"))}

    STRATS = build_strategies(cfg.raw["repair"]["uncertainty_metrics"],
                              cfg.raw["repair"]["uncertainty_rules"])
    seeds = cfg.raw["repair"]["seeds"]
    nudge_cfg = cfg.raw["repair"]["nudge"]
    topk = cfg.raw["repair"]["topk_for_repair"]
    pctl = cfg.raw["repair"]["threshold_percentile"]
    multiplier = 1.0

    nudge = nudge_cfg.get("retry_hint") if nudge_cfg.get("enabled") else None
    temperature = nudge_cfg.get("temperature", 0.7) if nudge_cfg.get("enabled") else 0.0

    results_path = os.path.join(cfg.path("repairs"), "results.jsonl")
    ckpt = Checkpoint(os.path.join(cfg.path("logs"), "stage5_repair.jsonl"))

    total_rows = len(failed_ids) * len(STRATS) * len(seeds)
    log.info(f"{len(failed_ids)} failed x {len(STRATS)} strategies x {len(seeds)} seeds "
             f"= {total_rows} result rows (actual GPU runs are far fewer: dedup)")

    client = load_agent(cfg, log)
    B = cfg.raw["runtime"]["repair_batch_size"]
    QB = 64                       # questions staged per planning chunk
    n_gen = 0
    t_start = time.time()

    for seed in seeds:
        log.info(f"--- seed {seed} ---")
        for ci in range(0, len(failed_ids), QB):
            chunk = failed_ids[ci:ci + QB]

            job_map = {}          # (qid, step) -> [strategies needing it]
            info = {}             # qid -> (orig_steps, oracle, base, budget, record)
            for qid in chunk:
                pending = [s for s in STRATS
                           if not ckpt.is_done(f"{qid}|{s}|{seed}|{multiplier}")]
                if not pending:
                    continue
                orig = load_item(cfg.path("trajectories"), qid)
                unc = load_item(cfg.path("uncertainty"), qid)
                ann = load_item(cfg.path("annotations"), qid)
                oracle = ann["broken_step"]
                n = len(orig["steps"])
                base = max(1, orig["total_gen_tokens"])
                info[qid] = (orig["steps"], oracle, base, int(multiplier * base),
                             pool[qid])
                for s in pending:
                    k = select_target_step(s, n, oracle, unc,
                                           make_rng(qid, s, seed),
                                           topk=topk, percentile=pctl)
                    job_map.setdefault((qid, k), []).append(s)

            if not job_map:
                continue

            # unique repair jobs (this is the dedup)
            jobs = []
            for (qid, k) in job_map:
                steps_d, oracle, base, budget, record = info[qid]
                prefix = [Step.from_dict(s) for s in steps_d[:k]]
                for s in prefix:
                    s.generation = None          # prefix logprobs not needed -> save RAM
                jobs.append({"record": record, "prefix_steps": prefix,
                             "nudge": nudge, "token_budget": budget,
                             "meta": {"qid": qid, "target_step": k}})

            # run the unique jobs, batched
            done_jobs = {}
            for i in range(0, len(jobs), B):
                sub = jobs[i:i + B]
                trajs = run_repair_batch(
                    client, sub,
                    max_steps=cfg.agent.max_steps,
                    max_tokens_per_step=cfg.agent.max_tokens_per_step,
                    temperature=temperature, seed=seed)
                for j, t in zip(sub, trajs):
                    done_jobs[(j["meta"]["qid"], j["meta"]["target_step"])] = t
                n_gen += len(sub)

            # fan the shared result back out to every strategy that picked that step
            for (qid, k), strats in job_map.items():
                traj = done_jobs[(qid, k)]
                _, oracle, base, budget, _ = info[qid]
                for s in strats:
                    row = repair_record(traj, s, seed, k, oracle, base, budget)
                    row["multiplier"] = multiplier
                    append_jsonl(row, results_path)
                    ckpt.mark_done(f"{qid}|{s}|{seed}|{multiplier}")

            done_q = min(ci + QB, len(failed_ids))
            el = time.time() - t_start
            log.info(f"seed {seed}: {done_q}/{len(failed_ids)} questions | "
                     f"{n_gen} GPU repairs so far | {el/60:.1f} min elapsed")

    log.info(f"Repair complete. Actual GPU repair runs: {n_gen} "
             f"(vs {total_rows} result rows — dedup saved "
             f"{100*(1-n_gen/max(1,total_rows)):.0f}%)")
    log.info(f"Results -> {results_path}")


if __name__ == "__main__":
    main()
