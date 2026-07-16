"""Batched ReAct execution.

The sequential ReActAgent runs one question at a time, which leaves a big GPU
almost idle. This module keeps MANY episodes in flight: at each step it builds
the next-step prompt for every still-active episode and sends them to vLLM in a
SINGLE batched call. Episodes terminate at different times and drop out.

The same driver serves:
  * generation  -> episodes start empty (greedy)
  * repair      -> episodes start with a kept prefix + a nudge (sampled)

A batch is homogeneous in (temperature, seed) — the callers loop over seeds — so
one SamplingParams applies to the whole batch.

Note: under batching, per-episode wall-clock latency is not meaningful (many run
concurrently). Token counts and tool-call counts remain exact, and those are the
cost metrics the evaluation relies on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..env.hotpot_env import HotpotEnv, score_answer
from ..llm.vllm_client import VLLMClient
from .react_agent import (
    Step, Trajectory, parse_action, build_messages, STOP,
)


@dataclass
class Episode:
    """One in-flight ReAct rollout."""
    env: HotpotEnv
    steps: List[Step]
    max_steps: int
    temperature: float
    seed: Optional[int] = None
    nudge: Optional[str] = None          # consumed on the first generated step
    token_budget: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # cost accumulators
    total_gen_tokens: int = 0
    num_tool_calls: int = 0
    total_latency_s: float = 0.0
    gen_tokens_run: int = 0              # this run only (excludes replayed prefix)
    tool_calls_run: int = 0
    latency_run_s: float = 0.0
    n_prefix: int = 0

    done: bool = False
    terminated_reason: str = "unknown"
    final_answer: Optional[str] = None

    def to_trajectory(self) -> Trajectory:
        t = Trajectory(
            qid=self.env.qid, question=self.env.question,
            gold_answer=self.env.gold_answer, gold_titles=self.env.gold_titles,
            steps=self.steps, final_answer=self.final_answer,
            terminated_reason=self.terminated_reason,
            total_gen_tokens=self.total_gen_tokens,
            num_tool_calls=self.num_tool_calls,
            total_latency_s=self.total_latency_s,
            meta=dict(self.meta),
        )
        t.meta["recovery_gen_tokens"] = self.gen_tokens_run
        t.meta["recovery_tool_calls"] = self.tool_calls_run
        t.meta["recovery_latency_s"] = self.latency_run_s
        t.meta["n_prefix_steps"] = self.n_prefix
        sc = score_answer(t.final_answer, t.gold_answer)
        t.em, t.f1, t.success = sc["em"], sc["f1"], sc["correct"]
        return t


def _make_episode(record: Dict[str, Any], max_steps: int, temperature: float,
                  seed: Optional[int], prefix_steps: Optional[List[Step]] = None,
                  nudge: Optional[str] = None, token_budget: Optional[int] = None,
                  meta: Optional[Dict[str, Any]] = None) -> Episode:
    env = HotpotEnv(record=record)
    steps = list(prefix_steps or [])
    ep = Episode(env=env, steps=steps, max_steps=max_steps, temperature=temperature,
                 seed=seed, nudge=nudge, token_budget=token_budget, meta=meta or {})
    # Replay kept steps so tool state (current page, retrieved titles) is correct,
    # and carry their already-paid cost into the totals.
    for s in steps:
        if s.is_tool_call:
            env.step(s.action, s.action_input)
        ep.total_gen_tokens += s.n_gen_tokens
        if s.action in ("search", "lookup"):
            ep.num_tool_calls += 1
    ep.n_prefix = len(steps)
    return ep


def _drive(client: VLLMClient, episodes: List[Episode], max_tokens_per_step: int,
           progress: bool = False) -> None:
    """Step every active episode forward, batching all their prompts per round."""
    while True:
        # retire episodes that hit a cap
        for e in episodes:
            if e.done:
                continue
            if len(e.steps) >= e.max_steps:
                e.done, e.terminated_reason = True, "max_steps"
            elif e.token_budget is not None and e.gen_tokens_run >= e.token_budget:
                e.done, e.terminated_reason = True, "budget"

        active = [e for e in episodes if not e.done]
        if not active:
            break

        msgs = [build_messages(e.env.question, e.steps, e.nudge) for e in active]
        temp = active[0].temperature      # batches are homogeneous
        seed = active[0].seed

        t0 = time.time()
        gens = client.chat_batch(msgs, temperature=temp,
                                 max_tokens=max_tokens_per_step,
                                 stop=STOP, seed=seed, progress=progress)
        dt = time.time() - t0

        for e, gen in zip(active, gens):
            e.nudge = None                # nudge only affects the first regenerated step
            thought, action, action_input = parse_action(gen.text)
            idx = len(e.steps)

            e.total_gen_tokens += gen.num_tokens
            e.gen_tokens_run += gen.num_tokens
            e.total_latency_s += dt
            e.latency_run_s += dt

            if action not in ("search", "lookup", "finish"):
                e.steps.append(Step(idx, thought, action or "invalid", action_input,
                                    "Invalid action. Use search, lookup, or finish.",
                                    False, None, gen.num_tokens, dt, gen))
                e.done, e.terminated_reason = True, "error"
                continue

            res = e.env.step(action, action_input)
            e.steps.append(Step(idx, thought, action, action_input, res.observation,
                                res.is_tool_call, res.retrieved_title,
                                gen.num_tokens, dt, gen))
            if action in ("search", "lookup"):
                e.num_tool_calls += 1
                e.tool_calls_run += 1
            if res.finished:
                e.final_answer = res.answer
                e.done, e.terminated_reason = True, "finished"


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_generation_batch(client: VLLMClient, records: List[Dict[str, Any]],
                         max_steps: int, max_tokens_per_step: int,
                         temperature: float = 0.0, seed: Optional[int] = None,
                         progress: bool = False) -> List[Trajectory]:
    """Generate fresh trajectories for a batch of questions, concurrently."""
    eps = [_make_episode(r, max_steps, temperature, seed) for r in records]
    _drive(client, eps, max_tokens_per_step, progress)
    return [e.to_trajectory() for e in eps]


def run_repair_batch(client: VLLMClient, jobs: List[Dict[str, Any]],
                     max_steps: int, max_tokens_per_step: int,
                     temperature: float, seed: Optional[int],
                     progress: bool = False) -> List[Trajectory]:
    """Run a batch of repair episodes concurrently.

    Each job: {record, prefix_steps (List[Step]), nudge, token_budget, meta}
    All jobs in a batch must share `temperature` and `seed`.
    """
    eps = [_make_episode(j["record"], max_steps, temperature, seed,
                         prefix_steps=j.get("prefix_steps"),
                         nudge=j.get("nudge"),
                         token_budget=j.get("token_budget"),
                         meta=j.get("meta"))
           for j in jobs]
    _drive(client, eps, max_tokens_per_step, progress)
    return [e.to_trajectory() for e in eps]
