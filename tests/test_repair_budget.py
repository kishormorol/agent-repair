import pytest

from src.agent.batch_runner import run_repair_batch
from src.agent.react_agent import ReActAgent, Step
from src.env.hotpot_env import HotpotEnv
from src.llm.vllm_client import GenerationResult, TokenInfo


RECORD = {
    "_id": "q1", "question": "Where?", "answer": "Paris",
    "context": [["Paris", ["Paris is a city."]]], "supporting_facts": [],
}


class BudgetClient:
    """Use the entire requested allowance to expose budget overshoot."""

    def __init__(self):
        self.calls = []

    def completion(self, max_tokens):
        self.calls.append(max_tokens)
        return GenerationResult(
            text="Thought: Look up the city.\nAction: search\nAction Input: Paris",
            token_ids=list(range(max_tokens)),
            tokens=[TokenInfo(i, "x", -1.0) for i in range(max_tokens)],
        )

    def chat(self, messages, *, max_tokens, **kwargs):
        return [self.completion(max_tokens)]

    def chat_batch(self, messages, *, max_tokens, **kwargs):
        limits = [max_tokens] * len(messages) if isinstance(max_tokens, int) else max_tokens
        return [self.completion(limit) for limit in limits]


@pytest.mark.parametrize("budget", [0, 3, 10, 13])
def test_sequential_repair_respects_remaining_budget(budget):
    client = BudgetClient()
    agent = ReActAgent(client, max_steps=8, max_tokens_per_step=10)
    trajectory = agent.run(HotpotEnv(record=RECORD), token_budget=budget)
    assert trajectory.meta["recovery_gen_tokens"] == budget
    assert trajectory.terminated_reason == "budget"
    assert all(0 < limit <= 10 for limit in client.calls)


def test_batched_repairs_respect_distinct_budgets():
    client = BudgetClient()
    budgets = [0, 3, 10, 13]
    trajectories = run_repair_batch(
        client, [{"record": dict(RECORD, _id=str(i)), "token_budget": b}
                 for i, b in enumerate(budgets)],
        max_steps=8, max_tokens_per_step=10, temperature=0.7, seed=0,
    )
    assert [t.meta["recovery_gen_tokens"] for t in trajectories] == budgets
    assert all(t.terminated_reason == "budget" for t in trajectories)


def test_prefix_tokens_do_not_consume_recovery_budget():
    prefix = [Step(0, "Search", "search", "Paris", "[Paris] Paris is a city.",
                   True, "Paris", 100, 0.0)]
    client = BudgetClient()
    trajectories = run_repair_batch(
        client, [{"record": RECORD, "token_budget": 3, "prefix_steps": prefix}],
        max_steps=8, max_tokens_per_step=10, temperature=0.7, seed=0,
    )
    assert trajectories[0].meta["recovery_gen_tokens"] == 3
    assert trajectories[0].total_gen_tokens == 103


class InvalidActionClient(BudgetClient):
    def completion(self, max_tokens):
        result = super().completion(min(3, max_tokens))
        result.text = "Thought: An incomplete response"
        return result


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("budget,reason", [(3, "budget"), (10, "error"), (None, "error")])
def test_invalid_action_at_token_cap_is_recorded_as_budget_exhaustion(batched, budget, reason):
    client = InvalidActionClient()
    if batched:
        result = run_repair_batch(
            client, [{"record": RECORD, "token_budget": budget}],
            max_steps=8, max_tokens_per_step=10, temperature=0.7, seed=0)[0]
    else:
        result = ReActAgent(client, max_steps=8, max_tokens_per_step=10).run(
            HotpotEnv(RECORD), token_budget=budget)
    assert result.terminated_reason == reason
    assert result.meta["invalid_action_step"] == 0
    assert result.meta["recovery_gen_tokens"] == 3
    assert result.steps[-1].action == "invalid" and not result.success
