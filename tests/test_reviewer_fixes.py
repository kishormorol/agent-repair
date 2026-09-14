import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from scripts.run_experiment import (
    build_experiment_matrix, setup_experiment_dir, write_experiment_config,
)
from src.agent.batch_runner import run_repair_batch
from src.agent.react_agent import ReActAgent, Step
from src.env.hotpot_env import HotpotEnv
from src.eval import metrics
from src.llm.vllm_client import GenerationResult, TokenInfo, resolve_agent_model
from src.repair.strategies import get_nudge_text, select_target_step
from src.utils import load_config, save_json


ROOT = Path(__file__).resolve().parents[1]
RECORD = {"_id": "q1", "question": "Where?", "answer": "Paris",
          "context": [["Paris", ["Paris is a city."]]], "supporting_facts": []}


class SearchClient:
    def chat(self, messages, **kwargs):
        return [GenerationResult("Thought: Search.\nAction: search\nAction Input: Paris",
                                 [1], [TokenInfo(1, "x", -1.0)])]

    def chat_batch(self, messages, **kwargs):
        return [self.chat(m)[0] for m in messages]


@pytest.mark.parametrize("batched", [False, True])
def test_new_step_allowance_does_not_count_retained_prefix(batched):
    prefix = [Step(i, "Search", "search", "Paris", "[Paris] Paris is a city.",
                   True, "Paris", 1, 0.0) for i in range(2)]
    if batched:
        result = run_repair_batch(
            SearchClient(), [{"record": RECORD, "prefix_steps": prefix}],
            max_steps=3, max_tokens_per_step=10, temperature=0.7, seed=0,
            step_budget_mode="new")[0]
    else:
        result = ReActAgent(SearchClient(), max_steps=3).run(
            HotpotEnv(record=RECORD), prefix_steps=prefix, step_budget_mode="new")
    assert len(result.steps) == 5
    assert result.meta["recovery_gen_tokens"] == 3
    assert result.meta["step_budget_mode"] == "new"


def test_matched_restart_uses_identical_hint_at_origin_zero():
    hint = "Reconsider carefully."
    assert get_nudge_text("full_restart", None, hint, match_restart=True) == hint
    assert get_nudge_text("unc__perplexity__argmax__bt2", None, hint,
                          match_restart=True) == hint
    assert get_nudge_text("full_restart", None, hint) is None


def test_fixed_early_control_is_clamped_for_short_traces():
    assert select_target_step("fixed_early", 8, 6, None, None) == 1
    assert select_target_step("fixed_early", 1, 0, None, None) == 0


def test_generated_experiment_config_roundtrips_without_nested_paths(tmp_path):
    base = yaml.safe_load((ROOT / "config/config_experiment.yaml").read_text())
    original = copy.deepcopy(base)
    dirs = setup_experiment_dir(str(tmp_path), "hotpotqa", "qwen2.5-32b")
    path = Path(dirs["logs"]) / "experiment_config.yaml"
    write_experiment_config(base, {"name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                                   "dtype": "auto", "gpu_memory_utilization": 0.9},
                            {"raw_filename": "data.json", "pool_size": 10},
                            "hotpotqa", dirs, str(path))
    reloaded = load_config(str(path))
    assert Path(reloaded.path("trajectories")) == Path(dirs["trajectories"])
    assert Path(reloaded.path("data_processed")).is_relative_to(Path(dirs["logs"]).parent)
    assert reloaded.dataset.pool_size == 500
    assert base == original


def test_unknown_model_filter_fails_instead_of_succeeding_with_zero_runs():
    cfg = SimpleNamespace(raw={"experiment": {"models_file": str(ROOT / "config/models.yaml"),
                                              "datasets_file": str(ROOT / "config/datasets.yaml")}})
    with pytest.raises(ValueError, match="No experiments"):
        build_experiment_matrix(cfg, SimpleNamespace(model="missing-model", dataset="hotpotqa"))


def test_changed_model_configuration_cannot_reuse_a_stale_cache(tmp_path):
    cfg = load_config(str(ROOT / "config/config_colab_hotpotqa.yaml"), base_override=str(tmp_path))
    save_json({"name": "some-other-model", "dtype": "auto"},
              Path(cfg.path("data_processed")) / "agent_model.json")
    with pytest.raises(ValueError, match="model"):
        resolve_agent_model(cfg)


def paired_rows():
    return pd.DataFrame([{"qid": q, "strategy": s, "seed": r,
                          "success": int(q == "q1" and s == "a" and r == 0)}
                         for q in ("q1", "q2") for s in ("a", "b") for r in range(3)])


def test_paired_mean_success_is_not_majority_repeatability():
    result = metrics.paired_mean_comparison(paired_rows(), "a", "b", iters=1000)
    assert result["delta"] == pytest.approx(1 / 6)
    assert result["n_questions"] == 2
    assert result["delta_lo"] == 0
    assert result["delta_hi"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "null", "nonbinary"])
def test_paired_mean_comparison_rejects_incomplete_or_invalid_trials(defect):
    frame = paired_rows()
    if defect == "missing":
        frame = frame.iloc[1:]
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        # pandas 3 rejects fractional assignment into an integer column before
        # our validator can inspect the deliberately invalid trial.
        frame["success"] = frame["success"].astype(float)
        frame.loc[0, "success"] = float("nan") if defect == "null" else 0.5
    with pytest.raises(ValueError):
        metrics.paired_mean_comparison(frame, "a", "b", iters=100)
