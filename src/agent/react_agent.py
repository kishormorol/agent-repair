"""ReAct agent for HotpotQA with a resumable execution loop.

A trajectory is a sequence of Steps. Each Step = one model turn producing
`Thought / Action / Action Input`, followed by the tool Observation. We store
the step's generated-token logprobs (for uncertainty) and timing/cost.

The SAME `run()` powers both:
  * fresh generation:  run(env)                       (prefix_steps=None)
  * repair-resume:     run(env, prefix_steps=steps[:k], nudge=...)
    -> keeps steps 0..k-1 verbatim and re-generates from step k onward.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..env.base_env import BaseEnv, score_answer
from ..env.hotpot_env import HotpotEnv
from ..llm.vllm_client import VLLMClient, GenerationResult


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a research assistant that answers multi-hop questions using a "
    "Wikipedia-like knowledge tool. Reason step by step.\n\n"
    "At EACH step output EXACTLY this format and then STOP:\n"
    "Thought: <your reasoning for this step>\n"
    "Action: <one of: search, lookup, finish>\n"
    "Action Input: <the argument>\n\n"
    "Tool semantics:\n"
    "- search: Action Input is an entity/title; returns that page's summary.\n"
    "- lookup: Action Input is a keyword; returns the next sentence containing "
    "it on the current page.\n"
    "- finish: Action Input is your final answer to the question.\n\n"
    "Do not write the Observation yourself; it will be given to you. "
    "Keep answers short and exact."
)

ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE | re.DOTALL)
BRACKET_RE = re.compile(r"(\w+)\s*\[\s*(.+?)\s*\]")  # fallback: search[Blade Runner]
STOP = ["Observation:", "\nObservation", "\nThought:"]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    index: int
    thought: str
    action: str
    action_input: str
    observation: str
    is_tool_call: bool
    retrieved_title: Optional[str]
    n_gen_tokens: int
    latency_s: float
    generation: Optional[GenerationResult] = None   # holds per-token logprobs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "thought": self.thought, "action": self.action,
            "action_input": self.action_input, "observation": self.observation,
            "is_tool_call": self.is_tool_call, "retrieved_title": self.retrieved_title,
            "n_gen_tokens": self.n_gen_tokens, "latency_s": self.latency_s,
            "generation": self.generation.to_dict() if self.generation else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Step":
        return Step(
            index=d["index"], thought=d["thought"], action=d["action"],
            action_input=d["action_input"], observation=d["observation"],
            is_tool_call=d["is_tool_call"], retrieved_title=d.get("retrieved_title"),
            n_gen_tokens=d.get("n_gen_tokens", 0), latency_s=d.get("latency_s", 0.0),
            generation=(GenerationResult.from_dict(d["generation"])
                        if d.get("generation") else None),
        )


@dataclass
class Trajectory:
    qid: str
    question: str
    gold_answer: str
    gold_titles: List[str]
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated_reason: str = "unknown"      # finished | max_steps | budget | error
    em: float = 0.0
    f1: float = 0.0
    success: bool = False
    total_gen_tokens: int = 0
    num_tool_calls: int = 0
    total_latency_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. strategy/seed for repairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid, "question": self.question, "gold_answer": self.gold_answer,
            "gold_titles": self.gold_titles, "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer, "terminated_reason": self.terminated_reason,
            "em": self.em, "f1": self.f1, "success": self.success,
            "total_gen_tokens": self.total_gen_tokens, "num_tool_calls": self.num_tool_calls,
            "total_latency_s": self.total_latency_s, "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trajectory":
        t = Trajectory(
            qid=d["qid"], question=d["question"], gold_answer=d["gold_answer"],
            gold_titles=d.get("gold_titles", []),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            final_answer=d.get("final_answer"),
            terminated_reason=d.get("terminated_reason", "unknown"),
            em=d.get("em", 0.0), f1=d.get("f1", 0.0), success=d.get("success", False),
            total_gen_tokens=d.get("total_gen_tokens", 0),
            num_tool_calls=d.get("num_tool_calls", 0),
            total_latency_s=d.get("total_latency_s", 0.0),
            meta=d.get("meta", {}),
        )
        return t


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_action(text: str) -> tuple[str, str, str]:
    """Return (thought, action, action_input) from a generated step."""
    # Thought = everything before the first "Action:".
    m_action = ACTION_RE.search(text)
    thought = text[: m_action.start()].strip() if m_action else text.strip()
    thought = re.sub(r"^Thought:\s*", "", thought, flags=re.IGNORECASE).strip()

    action, action_input = "", ""
    if m_action:
        action = m_action.group(1).strip().lower()
        m_input = INPUT_RE.search(text, m_action.end())
        if m_input:
            action_input = m_input.group(1).strip()
            # cut off anything after a following "Observation"/"Thought" if present
            action_input = re.split(r"\n(?:Observation|Thought)\b", action_input)[0].strip()
    if not action:  # fallback to bracket form  search[Blade Runner]
        mb = BRACKET_RE.search(text)
        if mb:
            action, action_input = mb.group(1).lower(), mb.group(2).strip()
    # strip stray surrounding brackets/quotes on the input
    action_input = action_input.strip().strip("[]").strip().strip('"').strip("'")
    return thought, action, action_input


# --------------------------------------------------------------------------- #
# Prompt builders (module-level so the batched runner can use them too)
# --------------------------------------------------------------------------- #
def build_scratchpad(question: str, steps: List[Step], nudge: Optional[str]) -> str:
    lines = [f"Question: {question}", ""]
    for s in steps:
        lines.append(f"Thought: {s.thought}")
        lines.append(f"Action: {s.action}")
        lines.append(f"Action Input: {s.action_input}")
        lines.append(f"Observation: {s.observation}")
    if nudge:
        lines.append(f"[Hint: {nudge}]")
    return "\n".join(lines).strip()


def build_messages(question: str, steps: List[Step],
                   nudge: Optional[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_scratchpad(question, steps, nudge)},
    ]


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ReActAgent:
    def __init__(self, client: VLLMClient, max_steps: int = 8,
                 max_tokens_per_step: int = 512):
        self.client = client
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step

    # ---- prompt construction (delegates to the module-level builders) ------ #
    @staticmethod
    def _scratchpad(question: str, steps: List[Step], nudge: Optional[str]) -> str:
        return build_scratchpad(question, steps, nudge)

    def _messages(self, question: str, steps: List[Step], nudge: Optional[str]):
        return build_messages(question, steps, nudge)

    # ---- main loop (fresh generation OR repair-resume) -------------------- #
    def run(self, env: HotpotEnv, temperature: float = 0.0,
            seed: Optional[int] = None,
            prefix_steps: Optional[List[Step]] = None,
            nudge: Optional[str] = None,
            token_budget: Optional[int] = None,
            meta: Optional[Dict[str, Any]] = None) -> Trajectory:
        """Run/resume a ReAct episode.

        prefix_steps: steps to keep verbatim (repair). Generation continues from
            index len(prefix_steps). The env must be REPLAYED for these kept
            steps first so tool state (current page, retrieved_titles) is correct.
        nudge: hint text injected before the first re-generated step.
        token_budget: stop early if generated tokens exceed this (matched budget).
        """
        traj = Trajectory(qid=env.qid, question=env.question,
                          gold_answer=env.gold_answer, gold_titles=env.gold_titles,
                          meta=meta or {})
        steps: List[Step] = list(prefix_steps or [])

        # Replay kept steps against the env so tool state is consistent, and
        # carry their cost forward into the trajectory totals.
        for s in steps:
            if s.is_tool_call:
                env.step(s.action, s.action_input)
            traj.total_gen_tokens += s.n_gen_tokens
            if s.action in ("search", "lookup"):
                traj.num_tool_calls += 1

        gen_tokens_this_run = 0
        tool_calls_this_run = 0
        latency_this_run = 0.0
        step_idx = len(steps)
        apply_nudge = nudge  # nudge only affects the first re-generated step

        while step_idx < self.max_steps:
            if token_budget is not None and gen_tokens_this_run >= token_budget:
                traj.terminated_reason = "budget"
                break

            messages = self._messages(env.question, steps, apply_nudge)
            apply_nudge = None  # consume after first use

            t0 = time.time()
            gens = self.client.chat(messages, temperature=temperature,
                                    max_tokens=self.max_tokens_per_step,
                                    n=1, stop=STOP, seed=seed)
            dt = time.time() - t0
            gen = gens[0]

            thought, action, action_input = parse_action(gen.text)

            if action not in ("search", "lookup", "finish"):
                # malformed -> record and stop to avoid loops
                obs = "Invalid action. Use search, lookup, or finish."
                steps.append(Step(step_idx, thought, action or "invalid", action_input,
                                  obs, False, None, gen.num_tokens, dt, gen))
                traj.total_gen_tokens += gen.num_tokens
                traj.total_latency_s += dt
                gen_tokens_this_run += gen.num_tokens
                latency_this_run += dt
                traj.terminated_reason = "error"
                break

            res = env.step(action, action_input)
            steps.append(Step(step_idx, thought, action, action_input,
                              res.observation, res.is_tool_call, res.retrieved_title,
                              gen.num_tokens, dt, gen))
            traj.total_gen_tokens += gen.num_tokens
            traj.total_latency_s += dt
            gen_tokens_this_run += gen.num_tokens
            latency_this_run += dt
            if action in ("search", "lookup"):
                traj.num_tool_calls += 1
                tool_calls_this_run += 1

            if res.finished:
                traj.final_answer = res.answer
                traj.terminated_reason = "finished"
                break
            step_idx += 1
        else:
            traj.terminated_reason = "max_steps"

        # Recovery-only costs = what was spent DURING this run (excludes the
        # replayed prefix, which was already paid for in the original episode).
        traj.meta["recovery_gen_tokens"] = gen_tokens_this_run
        traj.meta["recovery_tool_calls"] = tool_calls_this_run
        traj.meta["recovery_latency_s"] = latency_this_run
        traj.meta["n_prefix_steps"] = len(prefix_steps or [])

        traj.steps = steps
        sc = score_answer(traj.final_answer, traj.gold_answer)
        traj.em, traj.f1, traj.success = sc["em"], sc["f1"], sc["correct"]
        return traj
