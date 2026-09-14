from types import SimpleNamespace

import pytest

from src.llm.vllm_client import VLLMClient


def stub_client():
    client = VLLMClient("unused")
    received = []

    def generate(prompts, params, **kwargs):
        received.append(params)
        return [SimpleNamespace(outputs=[prompt]) for prompt in prompts]

    client._llm = SimpleNamespace(generate=generate)
    client._render = lambda messages: messages[0]["content"]
    client._sampling_params = lambda temperature, max_tokens, **kwargs: max_tokens
    client._to_result = lambda result, prompt_tokens=None: result
    return client, received


def test_independent_token_limits_reach_the_inference_engine_in_order():
    client, received = stub_client()
    messages = [[{"content": text}] for text in ("first", "second")]
    result = client.chat_batch(messages, max_tokens=[3, 7])
    assert received == [[3, 7]]
    assert result == ["first", "second"]


def test_shared_token_limit_remains_supported():
    client, received = stub_client()
    client.chat_batch([[{"content": "first"}]], max_tokens=5)
    assert received == [5]


@pytest.mark.parametrize("limits", [[1], [1, 0], [1, -3], [1, 1.5]])
def test_invalid_per_prompt_limits_fail_before_inference(limits):
    client, received = stub_client()
    with pytest.raises(ValueError):
        client.chat_batch([[{"content": "first"}], [{"content": "second"}]],
                          max_tokens=limits)
    assert received == []
