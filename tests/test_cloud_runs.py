from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.repair.controlled import fingerprint
from src.utils import load_config, load_json, save_json, read_jsonl
from src.utils.cloud_runs import (
    pin_model, prepare_config, primary_controls, repair_plan, run_directory, run_stage, write_once,
)
from test_controlled_repair import frozen_run
from test_position_matched import profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    save_json({}, snapshot / "config.json")
    return dict(dataset="hotpotqa", phase="pilot", run_id="pilot-v1",
                model_pin={"repo_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                           "revision": "a" * 40, "snapshot_path": str(snapshot)},
                strategy="unc__perplexity__argmax__bt2", pool_size=30, batch_size=2,
                origin_sweep=True, multipliers=[0.5, 1.0], code_sha256="b" * 64)


def test_cloud_profile_is_isolated_and_resumes_identically(tmp_path, settings):
    path = prepare_config(ROOT, tmp_path / "run", **settings)
    cfg = load_config(str(path))
    assert cfg.dataset.pool_size == 30
    assert cfg.dataset.split == "development"
    assert cfg.repair.step_budget_mode == "new" and cfg.repair.match_restart_hint
    assert cfg.models.agent.name == settings["model_pin"]["snapshot_path"]
    assert Path(cfg.path("data_processed")).is_relative_to(tmp_path / "run")
    assert prepare_config(ROOT, tmp_path / "run", **settings) == path
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_config(ROOT, tmp_path / "run", **dict(settings, batch_size=4))
    assert yaml.safe_load(path.read_text())["runtime"]["repair_batch_size"] == 2


@pytest.mark.parametrize("dataset,filename", [
    ("musique", "musique_dev.json"), ("2wikimultihopqa", "2wikimultihop_dev.json"),
])
def test_other_qa_profiles_do_not_reuse_hotpot_download_url(tmp_path, settings, dataset, filename):
    path = prepare_config(ROOT, tmp_path / dataset, **dict(settings, dataset=dataset))
    ds = yaml.safe_load(path.read_text())["dataset"]
    assert ds["raw_filename"] == filename and ds["url"] is None


def test_primary_random_control_has_same_backtracking_offset():
    strategies, baselines = primary_controls("unc__perplexity__argmax__bt2")
    assert baselines == ["full_restart", "random_step__bt2"]
    assert len(strategies) == len(set(strategies))
    assert {"fixed_early", "unc__perplexity__argmax", "random_step"} <= set(strategies)
    for strategy in ("oracle_targeted", "unc__perplexity__argmax__informed",
                     "unc__self_consistency__argmax", "unc__perplexity__argmax__bt3"):
        with pytest.raises(ValueError):
            primary_controls(strategy)


@pytest.mark.parametrize("strategy,random", [
    ("unc__perplexity__argmax", "random_step"),
    ("unc__perplexity__argmax__bt2", "random_step__bt2"),
])
def test_budget_profile_keeps_exactly_four_matched_conditions(tmp_path, settings, strategy, random):
    strategies, baselines = primary_controls(strategy, include_diagnostics=False)
    assert strategies == ["full_restart", random, "fixed_early", strategy]
    assert baselines == ["full_restart", random]
    path = prepare_config(ROOT, tmp_path / "budget-run", **dict(
        settings, strategy=strategy, include_diagnostics=False, origin_sweep=False,
        multipliers=[1.0], pool_size=10))
    cfg = load_config(str(path))
    assert cfg.raw["repair"]["strategies"] == strategies
    assert cfg.raw["repair"]["seeds"] == [0, 1, 2]
    assert cfg.raw["repair"]["origin_sweep"] is False
    assert cfg.raw["repair"]["budget"]["multipliers"] == [1.0]
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_config(ROOT, tmp_path / "budget-run", **dict(
            settings, strategy=strategy, include_diagnostics=True, origin_sweep=False,
            multipliers=[1.0], pool_size=10))
    with pytest.raises(ValueError, match="boolean"):
        primary_controls(strategy, include_diagnostics="false")


def test_cloud_test_mode_rejects_missing_or_unfrozen_cohorts(tmp_path, settings):
    with pytest.raises(ValueError, match="require test IDs"):
        prepare_config(ROOT, tmp_path / "run", **dict(settings, phase="test"))
    assert not (tmp_path / "run/config.yaml").exists()


def test_development_cohort_is_frozen_disjoint_and_complete(tmp_path, settings):
    ids, explored = tmp_path / "dev.json", tmp_path / "explored.json"
    save_json(["new2", "new1"], ids)
    save_json(["pilot1"], explored)
    args = dict(settings, phase="development", pool_size=2,
                development_ids=str(ids), explored_ids=str(explored))
    path = prepare_config(ROOT, tmp_path / "dev", **args)
    cfg = yaml.safe_load(path.read_text())
    assert load_json(cfg["dataset"]["id_manifest"]) == ["new2", "new1"]
    assert load_json(cfg["dataset"]["exclude_ids_manifest"]) == ["pilot1"]
    assert cfg["dataset"]["split"] == "development"
    with pytest.raises(ValueError, match="size"):
        prepare_config(ROOT, tmp_path / "wrong-size", **dict(args, pool_size=3))
    save_json(["new1"], explored)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_config(ROOT, tmp_path / "overlap", **args)


@pytest.mark.parametrize("include_diagnostics", [True, False])
def test_cloud_test_policy_enforces_disjoint_ids_selection_and_sample_size(tmp_path, settings, include_diagnostics):
    ids, explored = ["test2", "test1"], ["pilot1"]
    save_json(ids, tmp_path / "ids.json")
    save_json(explored, tmp_path / "explored.json")
    policy = {"strategy": settings["strategy"], "model_id": settings["model_pin"]["repo_id"],
              "model_revision": "a" * 40, "code_sha256": "b" * 64,
              "seeds": [0, 1, 2], "multipliers": [0.5, 1.0], "origin_sweep": True,
              "max_steps": 8, "max_tokens_per_step": 512, "batch_size": 2,
              "selection_note": "synthetic test fixture", "precision_target": "synthetic fixture",
              "n_questions": {"hotpotqa": 2}, "cohort_sha256": {"hotpotqa": {
                  "test": fingerprint(ids), "explored": fingerprint(explored)}}}
    if not include_diagnostics:
        policy["include_diagnostics"] = False
    save_json(policy, tmp_path / "policy.json")
    args = dict(settings, phase="test", test_ids=str(tmp_path / "ids.json"),
                explored_ids=str(tmp_path / "explored.json"), policy_file=str(tmp_path / "policy.json"),
                include_diagnostics=include_diagnostics)
    path = prepare_config(ROOT, tmp_path / "test", **args)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["dataset"]["pool_size"] == 2
    assert load_json(cfg["dataset"]["id_manifest"]) == ids
    with pytest.raises(ValueError, match="Diagnostic conditions"):
        prepare_config(ROOT, tmp_path / "changed-controls", **dict(args, include_diagnostics=not include_diagnostics))
    save_json(["test1"], tmp_path / "explored.json")
    with pytest.raises(ValueError, match="disjoint"):
        prepare_config(ROOT, tmp_path / "bad", **args)
    save_json(explored, tmp_path / "explored.json")
    policy["n_questions"]["hotpotqa"] = 3
    save_json(policy, tmp_path / "policy.json")
    with pytest.raises(ValueError, match="sample sizes"):
        prepare_config(ROOT, tmp_path / "bad", **args)


def test_model_pin_uses_same_commit_for_weights_and_tokenizer_on_resume(tmp_path, monkeypatch):
    import sys
    calls = []
    fake = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40)),
        snapshot_download=lambda repo, **kwargs: calls.append(kwargs) or str(tmp_path / "snapshot"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    first = pin_model(tmp_path / "run", tmp_path / "cache", "Qwen/model")
    assert first == pin_model(tmp_path / "run", tmp_path / "cache", "Qwen/model")
    assert all(call["revision"] == "a" * 40 for call in calls)
    with pytest.raises(ValueError, match="saved snapshot"):
        pin_model(tmp_path / "run", tmp_path / "cache", "Qwen/model", "c" * 40)


def test_stage_failure_is_not_swallowed_and_attempt_is_recorded(tmp_path, monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr("src.utils.cloud_runs.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_stage("python", ROOT, tmp_path, "run_generate.py", "config.yaml")
    rows = list(read_jsonl(tmp_path / "stage_attempts.jsonl"))
    assert len(rows) == 1 and rows[0]["status"] == "failed"


def test_cloud_paths_and_locks_reject_unsafe_or_changed_runs(tmp_path):
    with pytest.raises(ValueError):
        run_directory(tmp_path, "fever", "pilot", "run1")
    with pytest.raises(ValueError):
        run_directory(tmp_path, "hotpotqa", "pilot", "../run1")
    write_once(tmp_path / "lock.json", {"revision": "a"})
    with pytest.raises(ValueError):
        write_once(tmp_path / "lock.json", {"revision": "b"})
    assert load_json(tmp_path / "lock.json") == {"revision": "a"}


def test_notebook_repair_estimate_matches_actual_deduplicated_jobs(frozen_run):
    runner, cfg, _, log, args = frozen_run
    before = repair_plan(cfg)
    assert before["planned_unique_executions"] == before["missing_executions"] == 16
    assert before["strategy_rows"] == 32
    assert runner.run(cfg, log, args)["new_executions"] == 16
    after = repair_plan(cfg)
    assert after["cached_executions"] == 16 and after["missing_executions"] == 0


def test_budget_profile_counts_all_seeds_without_hidden_sweep_jobs(frozen_run):
    runner, cfg, _, log, args = frozen_run
    cfg.raw["repair"].update(origin_sweep=False, seeds=[0, 1, 2],
                            strategies=primary_controls("unc__perplexity__argmax__bt2",
                                                        include_diagnostics=False)[0])
    cfg.raw["repair"]["budget"]["multipliers"] = [1.0]
    plan = repair_plan(cfg)
    assert plan["per_question_execution_upper_bound"] == 12
    assert plan["planned_unique_executions"] <= 12 * plan["failed_questions"]
    assert plan["strategy_rows"] == 12 * plan["failed_questions"]
    assert runner.run(cfg, log, args)["new_executions"] == plan["planned_unique_executions"]
    assert repair_plan(cfg)["missing_executions"] == 0


def test_six_condition_profile_freezes_distribution_and_comparison_family(tmp_path, settings, profile):
    strategies, controls = primary_controls(settings["strategy"], include_diagnostics=False,
                                            include_position_control=True)
    assert len(strategies) == 6
    assert controls == ["full_restart", "random_step__bt2", "position_matched_random",
                        "unc__perplexity__argmax"]
    ids, excluded = ["new1", "new2"], ["a", "b", "successful", "older"]
    for name, value in (("ids", ids), ("excluded", excluded), ("profile", profile)):
        save_json(value, tmp_path / (name + ".json"))
    policy = {"strategy": settings["strategy"], "model_id": settings["model_pin"]["repo_id"],
        "model_revision": "a" * 40, "code_sha256": "b" * 64, "seeds": [0, 1, 2],
        "multipliers": [1.0], "origin_sweep": False, "max_steps": 8,
        "max_tokens_per_step": 512, "batch_size": 2, "selection_note": "prespecified fixture",
        "precision_target": "fixture", "include_diagnostics": False,
        "position_profile_sha256": profile["sha256"], "strategies": strategies,
        "primary_comparisons": controls,
        "n_questions": {"hotpotqa": 2}, "cohort_sha256": {"hotpotqa": {
            "test": fingerprint(ids), "explored": fingerprint(excluded)}}}
    save_json(policy, tmp_path / "policy.json")
    args = dict(settings, phase="test", origin_sweep=False, multipliers=[1.0], pool_size=2,
        include_diagnostics=False, test_ids=str(tmp_path / "ids.json"),
        explored_ids=str(tmp_path / "excluded.json"), policy_file=str(tmp_path / "policy.json"),
        position_profile=str(tmp_path / "profile.json"))
    path = prepare_config(ROOT, tmp_path / "six", **args)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["repair"]["strategies"] == strategies
    assert cfg["repair"]["position_matched_profile"] == profile
    assert load_json(path.parent / "position_profile.json") == profile
    with pytest.raises(ValueError, match="Position profile"):
        prepare_config(ROOT, tmp_path / "no-profile", **dict(args, position_profile=None))
    policy["strategies"] = strategies[:-1]
    save_json(policy, tmp_path / "policy.json")
    with pytest.raises(ValueError, match="Six-condition"):
        prepare_config(ROOT, tmp_path / "missing-control", **args)


def test_six_conditions_execute_deduplicate_and_resume(frozen_run, profile):
    runner, cfg, loads, log, args = frozen_run
    cfg.raw["repair"].update(origin_sweep=False, seeds=[0, 1, 2],
        strategies=primary_controls("unc__perplexity__argmax__bt2", include_diagnostics=False,
                                    include_position_control=True)[0], position_matched_profile=profile)
    cfg.raw["repair"]["budget"]["multipliers"] = [1.0]
    before = repair_plan(cfg)
    assert before["strategy_rows"] == 36 and before["per_question_execution_upper_bound"] == 18
    assert runner.run(cfg, log, args)["new_executions"] == before["planned_unique_executions"]
    assert runner.run(cfg, log, args)["new_executions"] == 0 and len(loads) == 1
    cfg.raw["dataset"]["split"] = "test"
    profile["payload"]["source_question_ids"].append("q0")
    profile["sha256"] = fingerprint(profile["payload"])
    with pytest.raises(ValueError, match="overlaps"):
        repair_plan(cfg)
