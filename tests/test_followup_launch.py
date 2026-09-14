import copy
import datetime as dt
import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts.prepare_diagnosis_followup import select_new_ids, freeze_cohort, ROOT
from scripts.run_aws_diagnosis_study import validate_live_checks
from src.repair.controlled import fingerprint


def test_complete_package_is_repeatable_disjoint_and_detects_tampering(tmp_path):
    from scripts.prepare_diagnosis_followup import prepare
    from scripts.run_aws_diagnosis_study import verified_package
    output = tmp_path/"package"
    result = prepare(output)
    assert prepare(output) == result
    launch, protocols, _ = verified_package(output)
    assert result["new_questions"] == 250 and result["excluded_questions"] == 799
    assert result["pilot_questions"] == 5 and result["gpu_executed"] is False
    assert len(protocols["main"]["batches"]) == 5
    assert not set(protocols["main"]["question_ids"]) & set(protocols["main"]["explored_ids"])
    assert launch["gpu_validation_status"] == "not_run"
    assert (output/"README.md").is_file()
    template = json.loads((output/"live-checks.template.json").read_text())
    with pytest.raises(ValueError, match="live-check template"):
        validate_live_checks(launch, template)
    (output/"main/batch01/config.yaml").write_text("changed")
    with pytest.raises(ValueError, match="package file mismatch"):
        verified_package(output)


def test_new_cohort_is_reproducible_disjoint_and_order_independent():
    records = [{"_id": str(i)} for i in range(40)]
    selected, n = select_new_ids(records, ["0", "1", "2"], n=20)
    reordered, _ = select_new_ids(list(reversed(records)), ["2", "0", "1"], n=20)
    assert selected == reordered and len(set(selected)) == 20 and n == 37
    assert not set(selected) & {"0", "1", "2"}
    with pytest.raises(ValueError, match="unique"):
        select_new_ids(records + records[:1], [], n=2)


def test_cohort_freeze_pins_absolute_configs_and_rejects_mutation(tmp_path):
    template = yaml.safe_load((ROOT/"config/config_iclr_pilot.yaml").read_text())
    kwargs = dict(template=template, storage=tmp_path/"remote", run_tag="new-study-v1", question_ids=["q1", "q2", "q3"],
                  explored=["old"], code_sha="a"*64, role="confirmatory_followup", batch_n=2)
    protocol = freeze_cohort(tmp_path/"package", **kwargs)
    assert freeze_cohort(tmp_path/"package", **kwargs) == protocol
    assert len(protocol["primary_comparisons"]) == 6
    assert [q for b in protocol["batches"] for q in b["ids"]] == kwargs["question_ids"]
    for batch in protocol["batches"]:
        cfg = yaml.safe_load((tmp_path/"package"/batch["directory"]/"config.yaml").read_text())
        assert all(Path(p).is_absolute() for p in cfg["paths"].values())
        assert fingerprint(cfg) == batch["configuration_sha256"]
        assert cfg["repair"]["origin_sweep"] is False
        assert cfg["repair"]["budget"]["multipliers"] == [0.5, 1.0, 2.0]
        assert cfg["repair"]["strategies"] == protocol["strategies"]
    changed = copy.deepcopy(kwargs)
    changed["template"]["repair"]["nudge"]["retry_hint"] = "changed"
    with pytest.raises(ValueError, match="Frozen"):
        freeze_cohort(tmp_path/"package", **changed)


@pytest.fixture
def live_budget():
    now = dt.datetime.now(dt.timezone.utc)
    launch = {"instance_id": "retained", "account_id": "account", "region": "eu-west-2", "instance_type": "g7e.2xlarge",
              "gross_cap_usd": 25, "total_allocation_usd": 119, "allocation_reserve_usd": 20,
              "session_overhead_reserve_usd": 1.4, "maximum_hourly_usd": 5.84531,
              "maximum_minutes_from_ec2_start": 240}
    checks = {k: launch[k] for k in ["instance_id", "account_id", "region", "instance_type"]}
    checks.update(observed_utc=now.isoformat(), instance_start_utc=(now-dt.timedelta(minutes=5)).isoformat(),
                  deadline_utc=(now+dt.timedelta(hours=3)).isoformat(), persistent_storage_confirmed=True,
                  external_stop_verified=True, credit_ec2_eligibility_verified=True, state="running",
                  instance_initiated_shutdown_behavior="stop", external_stop_deadline_utc=(now+dt.timedelta(hours=3)).isoformat(),
                  credit_expires_utc=(now+dt.timedelta(days=30)).isoformat(), gpu_hourly_usd=5.84531,
                  gross_spent_usd=24, eligible_credit_remaining_usd=85)
    return launch, checks, now


def test_live_budget_accounts_for_boot_time_and_preserves_allocation_reserve(live_budget):
    launch, checks, now = live_budget
    assert validate_live_checks(launch, checks, now).isoformat() == checks["deadline_utc"]
    checks["gross_spent_usd"] = 75  # 75 + 25 + 20 exceeds the existing 119 allocation.
    with pytest.raises(ValueError, match="allowance"):
        validate_live_checks(launch, checks, now)


@pytest.mark.parametrize("change", [
    {"observed_utc": "2020-01-01T00:00:00Z"},
    {"instance_start_utc": "2020-01-01T00:00:00Z"},
    {"instance_id": "different"},
    {"persistent_storage_confirmed": False},
    {"external_stop_verified": False},
    {"credit_ec2_eligibility_verified": False},
    {"instance_initiated_shutdown_behavior": "terminate"},
    {"eligible_credit_remaining_usd": 44},
    {"credit_expires_utc": "2020-01-01T00:00:00Z"},
    {"gpu_hourly_usd": 6.0},
    {"gpu_hourly_usd": float("nan")},
])
def test_live_launch_rejects_stale_or_incompatible_observations(live_budget, change):
    launch, checks, now = live_budget
    checks.update(change)
    with pytest.raises(ValueError):
        validate_live_checks(launch, checks, now)


@pytest.mark.parametrize("failure", [None, "exit", "exception"])
def test_controller_archives_failure_and_stops_without_starting_main(tmp_path, monkeypatch, live_budget, failure):
    from scripts import run_aws_diagnosis_study as driver
    from scripts import run_aws_study as previous_driver
    monkeypatch.setitem(sys.modules, "run_aws_study", previous_driver)
    session, storage, source = tmp_path/"session", tmp_path/"storage", tmp_path/"source"
    session.mkdir()
    launch, checks, _ = live_budget
    launch.update(storage=str(storage), remote_session=str(session), main_protocol_sha256="main", pilot_protocol_sha256="pilot")
    protocols = {role: {"model_revision": "revision", "batches": [{"directory": "batch01", "run_id": role,
        "run_directory": str(storage/"runs"/role)}]} for role in ["pilot", "main"]}
    for role in protocols:
        (session/role/"batch01").mkdir(parents=True)
        (session/role/"protocol.json").write_text("{}")
    for name in ["launch.json", "code.zip", "hotpot_evaluate_v1.reference.py"]:
        (session/name).write_text("fixture")
    (session/"package-manifest.json").write_text(json.dumps({"sha256": "package"}))
    (session/"checks.json").write_text(json.dumps(checks))
    snapshot = storage/"huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/revision"
    snapshot.mkdir(parents=True)
    for name in ["config.json", "tokenizer_config.json"]:
        (snapshot/name).write_text("{}")
    monkeypatch.setattr(driver, "verified_package", lambda *a, **kw: (launch, protocols, source))
    monkeypatch.setattr(driver, "gpu_preflight", lambda *a: None)
    commands, executions = [], []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kw: commands.append(command))

    def execute(command, log_path, deadline, cwd):
        executions.append(command)
        Path(log_path).write_text("fixture execution")
        if failure == "exception":
            raise OSError("fixture child error")
        return 1 if failure == "exit" else 0

    monkeypatch.setattr(previous_driver, "run_notebook", execute)
    if failure:
        with pytest.raises((RuntimeError, OSError)):
            driver.run_study(session, session/"checks.json", "gpu-python", "control-python")
        assert len(executions) == 1 and not (session/"main/batch01/status.json").exists()
    else:
        driver.run_study(session, session/"checks.json", "gpu-python", "control-python")
        assert len(executions) == 4 and (session/"completed.json").exists()
    assert commands[-1] == ["sudo", "-n", "shutdown", "-h", "+3"]
    status = json.loads((session/"pilot/batch01/status.json").read_text())
    assert driver.file_sha(session/status["archive"]) == status["archive_sha256"]
    assert status["returncode"] == (None if failure == "exception" else 1 if failure == "exit" else 0)
