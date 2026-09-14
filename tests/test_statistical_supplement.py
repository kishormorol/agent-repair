import json
import subprocess
import sys
import zipfile

from scripts.build_statistical_supplement import build


def test_isolated_supplement_reproduces_statistics_and_rejects_tampering(tmp_path):
    archive = build(tmp_path / "build")
    isolated = tmp_path / "isolated"
    with zipfile.ZipFile(archive) as stream:
        assert all(name.startswith("statistical-supplement/") for name in stream.namelist())
        stream.extractall(isolated)
    bundle = isolated / "statistical-supplement"
    run = subprocess.run([sys.executable, "reproduce.py"], cwd=bundle, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((bundle / "verification.json").read_text())
    assert result["status"] == "passed" and result["trial_rows"] == 2034
    assert result["primary_contrasts_reproduced"] == 4 and result["secondary_contrasts_reproduced"] == 8
    (bundle / "repair-trials.csv").write_text("tampered")
    run = subprocess.run([sys.executable, "reproduce.py"], cwd=bundle, capture_output=True, text=True)
    assert run.returncode != 0
    assert "Checksum mismatch: repair-trials.csv" in run.stderr
