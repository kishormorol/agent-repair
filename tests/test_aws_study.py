import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_aws_study import batch_deadline, run_study
from src.repair.controlled import fingerprint


def test_batch_deadline_never_extends_overall_allowance():
    now = dt.datetime(2026, 9, 11, tzinfo=dt.timezone.utc)
    assert batch_deadline(now + dt.timedelta(hours=4), now) == now + dt.timedelta(minutes=45)
    assert batch_deadline(now + dt.timedelta(minutes=10), now) == now + dt.timedelta(minutes=10)
    with pytest.raises(ValueError, match="Too little"):
        batch_deadline(now + dt.timedelta(minutes=2), now)


@pytest.mark.parametrize("failure", [False, True])
def test_study_archives_batches_and_always_schedules_stop(tmp_path, monkeypatch, failure):
    from scripts import run_aws_study as driver
    session, storage = tmp_path / "session", tmp_path / "storage"
    session.mkdir()
    study = {"code_sha256": "a" * 64, "dataset": "hotpotqa", "gross_cap_usd": 25,
             "gpu_hourly_usd": 5.84531, "batches": [{"run_id": f"run{i}", "directory": f"batch{i}"} for i in range(2)]}
    (session / "study-freeze.json").write_text(json.dumps({"sha256": fingerprint(study), "payload": study}))
    for name in ("position-profile.json", "run_iclr2027.ipynb", "agent-repair-iclr2027-code.zip"):
        (session / name).write_text("fixture")
    for batch in study["batches"]:
        (session / batch["directory"]).mkdir()
        (session / batch["directory"] / "parameters.json").write_text("{}")
    commands, executions = [], []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kw: commands.append(command))
    def notebook(command, log_path, deadline, cwd):
        executions.append(command)
        Path(log_path).write_text("synthetic execution")
        return 1 if failure else 0
    monkeypatch.setattr(driver, "run_notebook", notebook)
    deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
    if failure:
        with pytest.raises(RuntimeError, match="preserved"):
            run_study(session, storage, deadline, "python", "python")
    else:
        run_study(session, storage, deadline, "python", "python")
    assert len(executions) == (1 if failure else 2)
    assert commands[-1] == ["sudo", "-n", "shutdown", "-h", "+3"]
    assert len(list(session.glob("*.tar.gz"))) == len(executions)
    status = json.loads((session / "batch0/status.json").read_text())
    assert hashlib.sha256((session / status["archive"]).read_bytes()).hexdigest() == status["sha256"]
    if not failure:
        assert any("--results" in command for command in commands)
        run_study(session, storage, deadline, "python", "python")
        assert len(executions) == 2
