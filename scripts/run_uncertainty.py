"""Stage 2 — step-level uncertainty (batched).

    python scripts/run_uncertainty.py --config config/config_local.yaml

Math metrics for every trajectory (from stored logprobs), then self-consistency
and verbalized confidence for the FAILED set — batched across steps and questions.
"""
from __future__ import annotations

import os
import time

from _common import parse_args, boot, load_agent  # type: ignore

from src.agent import Step, build_messages, build_scratchpad, parse_action
from src.uncertainty import (compute_math_metrics, self_consistency_score,
                             build_confidence_prompt, parse_confidence)
from src.utils import (Checkpoint, save_item, load_item, load_json,
                       list_item_ids)


def _lightweight(step: dict) -> dict:
    keys = ["index", "thought", "action", "action_input", "observation",
            "is_tool_call", "retrieved_title", "n_gen_tokens"]
    return {k: step[k] for k in keys}


def main() -> None:
    args = parse_args("Step-level uncertainty (batched)")
    cfg, log = boot("stage2", args)

    traj_dir = cfg.path("trajectories")
    unc_dir = cfg.path("uncertainty")
    topks = cfg.raw["uncertainty"]["entropy_topk_ablation"]

    # ---- 1. math metrics for every trajectory (no model needed) ------------ #
    math_ckpt = Checkpoint(os.path.join(cfg.path("logs"), "stage2_math.jsonl"))
    ids = list_item_ids(traj_dir)
    todo = [q for q in ids if not math_ckpt.is_done(q)]
    log.info(f"Math metrics: {len(todo)} of {len(ids)} trajectories to do")
    for qid in todo:
        t = load_item(traj_dir, qid)
        out = {"qid": qid, "question": t["question"], "gold_answer": t["gold_answer"],
               "gold_titles": t["gold_titles"], "success": t["success"], "steps": []}
        for s in t["steps"]:
            tokens = (s.get("generation") or {}).get("tokens", [])
            u = compute_math_metrics(tokens)
            for k in topks:
                u[f"token_entropy_mean_k{k}"] = compute_math_metrics(
                    tokens, entropy_topk=k)["token_entropy_mean"]
            row = _lightweight(s)
            row["uncertainty"] = u
            out["steps"].append(row)
        save_item(unc_dir, qid, out)
        math_ckpt.mark_done(qid)
    log.info("Math metrics done.")

    # ---- 2. sampling-based metrics on the FAILED set (batched) ------------- #
    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    if args.limit:
        failed_ids = failed_ids[:args.limit]
    ckpt = Checkpoint(os.path.join(cfg.path("logs"), "stage2_sampling.jsonl"))
    todo = [q for q in failed_ids if not ckpt.is_done(q)]
    log.info(f"Sampling metrics: {len(todo)} of {len(failed_ids)} failed trajectories")
    if not todo:
        log.info("Nothing to do.")
        return

    client = load_agent(cfg, log)
    sc_cfg = cfg.raw["uncertainty"]["self_consistency"]
    vc_scale = cfg.raw["uncertainty"]["verbalized_confidence"]["scale"]
    QB = cfg.raw["runtime"]["uncertainty_batch_size"]   # questions per chunk
    t_start = time.time()

    for i in range(0, len(todo), QB):
        chunk = todo[i:i + QB]

        # build every (qid, step) prompt in this chunk -> one big batch
        sc_msgs, vc_msgs, index = [], [], []
        cache = {}
        for qid in chunk:
            full = load_item(traj_dir, qid)
            steps = [Step.from_dict(s) for s in full["steps"]]
            cache[qid] = steps
            q = full["question"]
            for si, s in enumerate(steps):
                prefix = steps[:si]
                sc_msgs.append(build_messages(q, prefix, None))
                vc_msgs.append(build_confidence_prompt(
                    q, build_scratchpad(q, prefix, None),
                    s.thought, s.action, s.action_input))
                index.append((qid, si))

        # self-consistency: n samples for every step, in ONE call
        sc_out = client.chat_batch_n(
            sc_msgs, n=sc_cfg["n_samples"], temperature=sc_cfg["temperature"],
            max_tokens=cfg.agent.max_tokens_per_step, stop=["Observation:"])
        # verbalized confidence: one short call per step, in ONE call
        vc_out = client.chat_batch(vc_msgs, temperature=0.0, max_tokens=8)

        scores = {}
        for (qid, si), samples, conf in zip(index, sc_out, vc_out):
            acts = [parse_action(g.text)[1:] for g in samples]   # (action, input)
            scores[(qid, si)] = (self_consistency_score(acts),
                                 parse_confidence(conf.text, scale=vc_scale))

        for qid in chunk:
            enriched = load_item(unc_dir, qid)
            for si in range(len(cache[qid])):
                sc, vc = scores[(qid, si)]
                enriched["steps"][si]["uncertainty"]["self_consistency"] = sc
                enriched["steps"][si]["uncertainty"]["verbalized_confidence"] = vc
            save_item(unc_dir, qid, enriched)
            ckpt.mark_done(qid)

        done = min(i + QB, len(todo))
        rate = done / max(1e-9, time.time() - t_start)
        log.info(f"{done}/{len(todo)}  ({rate:.2f} q/s, "
                 f"ETA {(len(todo)-done)/max(1e-9,rate)/60:.0f} min)")

    log.info("Sampling-based uncertainty done.")


if __name__ == "__main__":
    main()
