import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.react_agent import ReActAgent
from src.env.hotpot_env import HotpotEnv
from src.repair.controlled import configured_strategies, budget_multipliers, plan_repairs
from src.utils import load_config, save_json, save_item, read_jsonl

from test_reviewer_fixes import RECORD, ROOT, SearchClient


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = importlib.import_module("scripts.run_repair")
    cfg = load_config(str(ROOT / "config/config_colab_hotpotqa.yaml"), base_override=str(tmp_path))
    cfg.raw["repair"].update(
        strategies=["full_restart", "random_step", "fixed_early", "unc__perplexity__argmax"],
        seeds=[0, 1], step_budget_mode="new", match_restart_hint=True,
        origin_sweep=True, run_id="unit-test")
    cfg.raw["repair"]["budget"]["multipliers"] = [0.5, 1.0]
    pool = [dict(RECORD, _id=f"q{i}") for i in range(2)]
    save_json(pool, Path(cfg.path("data_processed")) / "pool.json")
    save_json([r["_id"] for r in pool], Path(cfg.path("data_processed")) / "failed_ids.json")
    save_json({"name": cfg.models.agent.name, "dtype": cfg.models.agent.dtype},
              Path(cfg.path("data_processed")) / "agent_model.json")
    for record in pool:
        original = ReActAgent(SearchClient(), max_steps=2).run(HotpotEnv(record=record)).to_dict()
        save_item(cfg.path("trajectories"), record["_id"], original)
        save_item(cfg.path("uncertainty"), record["_id"],
                  {"qid": record["_id"], "steps": [
                      {"index": 0, "uncertainty": {"perplexity": 2}},
                      {"index": 1, "uncertainty": {"perplexity": 3}}]})
    loads = []

    def load_agent(*args):
        loads.append(True)
        client = SearchClient()
        client.model_name = cfg.models.agent.name
        return client

    monkeypatch.setattr(runner, "load_agent", load_agent)
    return runner, cfg, loads, SimpleNamespace(info=lambda *args: None), SimpleNamespace(
        limit=None, dry_run=False, dataset=None)


def test_origin_sweep_is_unique_budgeted_and_resumes_without_new_inference(frozen_run):
    runner, cfg, loads, log, args = frozen_run
    result = runner.run(cfg, log, args)
    assert result == {"planned_executions": 16, "strategy_rows": 32, "new_executions": 16}
    rows = list(read_jsonl(Path(cfg.path("repairs")) / "results.jsonl"))
    origins = list(read_jsonl(Path(cfg.path("repairs")) / "origins.jsonl"))
    assert len(rows) == 32 and len(origins) == 16
    assert len({row["execution_id"] for row in rows}) == 16
    assert {row["multiplier"] for row in rows} == {0.5, 1.0}
    assert all(row["recovery_gen_tokens"] <= row["budget"] for row in rows)
    assert all(row["targeted_oracle_match"] is None for row in rows)
    assert all(row["new_step_allowance"] == 8 for row in rows)
    assert runner.run(cfg, log, args)["new_executions"] == 0
    assert len(loads) == 1
    # Simulate an interruption during result fan-out, after execution persistence.
    path = Path(cfg.path("repairs")) / "results.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    assert runner.run(cfg, log, args)["new_executions"] == 0
    assert len(list(read_jsonl(path))) == 32
    assert len(loads) == 1


def test_changed_repair_settings_cannot_reuse_results(frozen_run):
    runner, cfg, _, log, args = frozen_run
    runner.run(cfg, log, args)
    cfg.raw["repair"]["nudge"]["retry_hint"] = "A different prompt"
    with pytest.raises(ValueError, match="changed"):
        runner.run(cfg, log, args)


def test_dry_run_does_not_load_model_or_write_results(frozen_run):
    runner, cfg, loads, log, args = frozen_run
    args.dry_run = True
    assert runner.run(cfg, log, args)["planned_executions"] == 16
    assert not loads
    assert not (Path(cfg.path("repairs")) / "manifest.json").exists()


def test_baseline_selection_and_budget_grid_are_explicit():
    assert configured_strategies({"baselines": ["fixed_early", "full_restart"]}) == [
        "fixed_early", "full_restart"]
    assert budget_multipliers({"budget": {"multipliers": [0.5, 1, 2]}}) == [0.5, 1, 2]
    with pytest.raises(ValueError):
        budget_multipliers({"budget": {"multipliers": [1, 1]}})


def test_backtracking_to_zero_shares_restart_execution(frozen_run):
    _, cfg, _, _, _ = frozen_run
    cfg.raw["repair"]["origin_sweep"] = False
    record = dict(RECORD, _id="q0")
    original = ReActAgent(SearchClient(), max_steps=2).run(HotpotEnv(record=record)).to_dict()
    uncertainty = {"steps": [{"index": 1, "uncertainty": {"perplexity": 3}}]}
    jobs = plan_repairs(cfg.raw, record, original, uncertainty, None, 0, 1,
                        ["full_restart", "unc__perplexity__argmax__bt2"])
    assert len(jobs) == 1
    assert jobs[0]["meta"]["target_step"] == 0
    assert len(jobs[0]["strategies"]) == 2


def test_sweep_rejects_informed_hints(frozen_run):
    runner, cfg, loads, log, args = frozen_run
    cfg.raw["repair"]["match_restart_hint"] = False
    with pytest.raises(ValueError, match="Origin sweeps require"):
        runner.run(cfg, log, args)
    assert not loads
