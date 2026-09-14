import pytest

from src.agent.batch_runner import run_generation_batch
from src.agent.react_agent import ReActAgent
from src.env import get_dataset
from src.env.hotpot_env import HotpotEnv, f1_score, score_answer
from src.llm.vllm_client import GenerationResult, TokenInfo


@pytest.mark.parametrize("prediction,gold,expected", [
    ("yes no", "yes", 0.0),
    ("yes", "yes indeed", 0.0),
    ("no explanation", "no", 0.0),
    ("noanswer", "noanswer provided", 0.0),
    ("Yes!", "yes", 1.0),
    ("NO", "no", 1.0),
    ("noanswer", "noanswer", 1.0),
    ("Shane Meadows", "directed by Shane Meadows", 2 / 3),
    ("", "", 0.0),
])
def test_hotpot_answer_f1_matches_official_edge_cases(prediction, gold, expected):
    assert f1_score(prediction, gold) == pytest.approx(expected)
    assert score_answer(prediction, gold)["f1"] == pytest.approx(expected)
    assert get_dataset("hotpotqa")["score"](prediction, gold)["f1"] == pytest.approx(expected)


class AnswerClient:
    def chat(self, messages, **kwargs):
        return [GenerationResult("Thought: Answer.\nAction: finish\nAction Input: yes no",
                                 [1], [TokenInfo(1, "x", -1.0)])]

    def chat_batch(self, messages, **kwargs):
        return [self.chat(m)[0] for m in messages]


@pytest.mark.parametrize("batched", [False, True])
def test_agent_paths_use_hotpot_scorer(batched):
    record = {"_id": "q", "question": "Is it?", "answer": "yes",
              "context": [], "supporting_facts": []}
    if batched:
        result = run_generation_batch(AnswerClient(), [record], 2, 20)[0]
    else:
        result = ReActAgent(AnswerClient()).run(HotpotEnv(record))
    assert result.terminated_reason == "finished"
    assert result.f1 == 0 and not result.success
