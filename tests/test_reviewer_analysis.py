import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.run_paired_analysis import load_verified_results
from src.agent.react_agent import ReActAgent
from src.agent.batch_runner import run_repair_batch
from src.analysis.causal import add_failure_mode_flags
from src.env.hotpot_env import HotpotEnv
from src.eval.metrics import paired_macro_comparison
from src.llm.vllm_client import GenerationResult, VLLMClient, resolve_judge_model
from src.localize.rules import get_step_scores
from src.repair.controlled import acquisition_cost, fingerprint
from src.utils import save_json, load_config
from src.utils.cohorts import select_cohort
from src.uncertainty.metrics import compute_math_metrics
import math

from test_reviewer_fixes import ROOT, RECORD, SearchClient


def test_macro_analysis_weights_datasets_not_question_counts():
    rows = [{"dataset": dataset, "qid": str(q), "strategy": strategy, "seed": seed,
             "success": int(dataset == "small" and strategy == "a")}
            for dataset, count in (("small", 1), ("large", 9)) for q in range(count)
            for strategy in ("a", "b") for seed in range(3)]
    result = paired_macro_comparison(pd.DataFrame(rows), "a", "b", iters=100)
    assert result["delta"] == 0.5
    assert result["delta_lo"] == result["delta_hi"] == 0.5
    assert result["n_questions"] == 10


def test_manifest_detects_a_question_missing_from_all_strategies(tmp_path):
    payload = {"configuration": {"dataset": {"name": "QA"}}, "question_ids": ["q1", "q2"]}
    digest = fingerprint(payload)
    save_json({"sha256": digest, "payload": payload}, tmp_path / "manifest.json")
    rows = [{"dataset": "QA", "qid": "q1", "strategy": strategy, "seed": seed,
             "success": 0, "multiplier": 1, "manifest_sha256": digest}
            for strategy in ("a", "b") for seed in range(3)]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="prespecified"):
        load_verified_results([path], "a", ["b"], [0, 1, 2], 1)


def test_test_cohorts_exclude_explored_ids_and_preserve_frozen_order(tmp_path):
    records = [dict(RECORD, _id=q) for q in ("q1", "q2", "q3")]
    save_json(["q3", "q2"], tmp_path / "test.json")
    save_json(["q1"], tmp_path / "explored.json")
    profile = {"split": "test", "id_manifest": "test.json",
               "exclude_ids_manifest": "explored.json"}
    assert [r["_id"] for r in select_cohort(records, profile, tmp_path)] == ["q3", "q2"]
    save_json(["q2"], tmp_path / "explored.json")
    with pytest.raises(ValueError, match="overlap"):
        select_cohort(records, profile, tmp_path)
    with pytest.raises(ValueError, match="require"):
        select_cohort(records, {"split": "test"}, tmp_path)


def test_exploration_flag_does_not_claim_action_correctness():
    data = pd.DataFrame({"oracle_step": [0], "pred_argmax": [2], "argmax_top1": [0],
                         "pred_step_action": ["search"]}, index=[8])
    result = add_failure_mode_flags(data)
    assert result.exploratory_action_at_missed_peak.iloc[0] == 1
    assert "uncertain_but_correct_explore" not in result


def test_nonfinite_uncertainty_cannot_select_an_origin():
    data = {"steps": [{"index": i, "uncertainty": {"perplexity": value}}
                       for i, value in enumerate([float("inf"), float("nan"), 3]) ]}
    assert get_step_scores(data, "perplexity") == [(2, 3)]


@pytest.mark.parametrize("batched", [False, True])
def test_recovery_records_prompt_tokens_and_request_counts(batched):
    class CountedClient(SearchClient):
        def chat(self, messages, **kwargs):
            result = super().chat(messages, **kwargs)[0]
            result.prompt_tokens = 21
            return [result]
    if batched:
        result = run_repair_batch(CountedClient(), [{"record": RECORD}],
                                  max_steps=2, max_tokens_per_step=10, temperature=0.7, seed=0)[0]
    else:
        result = ReActAgent(CountedClient(), max_steps=2).run(HotpotEnv(record=RECORD))
    assert result.meta["recovery_prompt_tokens"] == 42
    assert result.meta["recovery_model_requests"] == 2


def test_prompt_counts_are_extracted_and_legacy_unknown_is_not_zero():
    client = VLLMClient("unused")
    completion = SimpleNamespace(text="answer", token_ids=[1], logprobs=None)
    result = client._to_result(completion, client._prompt_tokens(SimpleNamespace(prompt_token_ids=[2, 3])))
    assert GenerationResult.from_dict(result.to_dict()).prompt_tokens == 2
    assert GenerationResult.from_dict({"text": "old"}).prompt_tokens is None


def test_policy_acquisition_cost_is_not_charged_per_shared_execution():
    unc = {"steps": [{"acquisition_cost": {"self_consistency": {
        "generated_tokens": 50, "prompt_tokens": 100, "requests": 1}}}]}
    assert acquisition_cost("full_restart", unc, None)["selection_gen_tokens"] == 0
    assert acquisition_cost("unc__perplexity__argmax", unc, None)["selection_gen_tokens"] == 0
    assert acquisition_cost("unc__self_consistency__argmax", unc, None)["selection_gen_tokens"] == 50
    assert acquisition_cost("oracle_targeted", unc, None)["selection_gen_tokens"] is None


def test_judge_is_never_silently_downgraded(monkeypatch, tmp_path):
    cfg = load_config(str(ROOT / "config/config_colab_hotpotqa.yaml"), base_override=str(tmp_path))
    monkeypatch.setattr("src.llm.vllm_client.gpu_vram_gb", lambda: 16)
    assert resolve_judge_model(cfg)["name"] == cfg.models.judge.name


def test_math_only_uncertainty_does_not_load_a_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("scripts.run_uncertainty")
    cfg = load_config(str(ROOT / "config/config_colab_hotpotqa.yaml"), base_override=str(tmp_path))
    cfg.raw["uncertainty"]["metrics"] = ["token_entropy", "perplexity"]
    log = SimpleNamespace(info=lambda *args: None)
    monkeypatch.setattr(module, "parse_args", lambda *args: SimpleNamespace(limit=None))
    monkeypatch.setattr(module, "boot", lambda *args: (cfg, log))
    monkeypatch.setattr(module, "load_agent", lambda *args: pytest.fail("Unexpected GPU model load"))
    module.main()


def test_missing_sampled_logprob_is_not_replaced_with_an_invented_probability():
    result = compute_math_metrics([{"logprob": None, "top_logprobs": {"1": -0.2}}])
    assert math.isnan(result["perplexity"])
    assert math.isnan(result["max_token_prob_max"])


def test_missing_entropy_distribution_is_not_perfect_certainty():
    result = compute_math_metrics([{"logprob": -1.0, "top_logprobs": {}}])
    assert math.isnan(result["token_entropy_max"])
