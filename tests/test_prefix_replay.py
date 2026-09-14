from dataclasses import replace

import pytest

from src.agent.batch_runner import run_repair_batch
from src.agent.react_agent import ReActAgent, Step
from src.env.hotpot_env import HotpotEnv
from src.llm.vllm_client import GenerationResult, TokenInfo


RECORD = {"_id": "replay-q", "question": "Where?", "answer": "Paris",
          "context": [["Paris", ["Paris is a city.", "Paris has a river.",
                                  "Paris has a museum."]]], "supporting_facts": []}


def prefix():
    env = HotpotEnv(RECORD)
    steps = []
    for i, action in enumerate(["search", "lookup", "lookup"]):
        result = env.step(action, "Paris")
        steps.append(Step(i, "Look up Paris.", action, "Paris", result.observation,
                          result.is_tool_call, result.retrieved_title, 2, 0.0))
    return steps


class LookupClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return [GenerationResult("Thought: Next fact.\nAction: lookup\nAction Input: Paris",
                                 [1], [TokenInfo(1, "x", -1.0)])]

    def chat_batch(self, messages, **kwargs):
        return [self.chat(m)[0] for m in messages]


def resume(batched, client, steps, record=RECORD):
    if batched:
        return run_repair_batch(client, [{"record": record, "prefix_steps": steps}],
                                max_steps=1, max_tokens_per_step=20,
                                temperature=0.7, seed=0, step_budget_mode="new")[0]
    return ReActAgent(client, max_steps=1).run(
        HotpotEnv(record), prefix_steps=steps, step_budget_mode="new")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("field,value", [
    ("observation", "stale observation"), ("retrieved_title", "London"),
    ("is_tool_call", False), ("index", 7),
])
def test_replay_mismatch_fails_before_generating(batched, field, value):
    steps, client = prefix(), LookupClient()
    steps[1] = replace(steps[1], **{field: value})
    with pytest.raises(ValueError, match="[Rr]eplay"):
        resume(batched, client, steps)
    assert client.calls == 0


@pytest.mark.parametrize("batched", [False, True])
def test_changed_evidence_cannot_reuse_a_prefix(batched):
    record = dict(RECORD, context=[["Paris", ["Different evidence."]]])
    client = LookupClient()
    with pytest.raises(ValueError, match="[Rr]eplay"):
        resume(batched, client, prefix(), record)
    assert client.calls == 0


@pytest.mark.parametrize("batched", [False, True])
def test_replay_restores_page_and_lookup_cursor(batched):
    result = resume(batched, LookupClient(), prefix())
    assert result.steps[-1].observation == "[Paris] Paris has a museum."
    assert result.meta["n_prefix_steps"] == 3
    assert result.meta["recovery_gen_tokens"] == 1
    assert result.total_gen_tokens == 7
