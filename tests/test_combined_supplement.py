import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_combined_supplement import FORBIDDEN, STUDIES, build

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"main": (4, 2034), "diagnosis": (6, 3186), "replication": (24, 8016),
            "position_pairs_qwen": (2, 738), "position_pairs_mistral": (2, 1098)}


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return build(tmp_path_factory.mktemp("supplement") / "build")


@pytest.fixture(scope="module")
def isolated(archive, tmp_path_factory):
    """Extract the bundle somewhere with no access to the working tree."""
    root = tmp_path_factory.mktemp("isolated")
    with zipfile.ZipFile(archive) as stream:
        assert all(name.startswith("combined-supplement/") for name in stream.namelist())
        stream.extractall(root)
    return root / "combined-supplement"


def shipped(archive):
    """Only what the archive actually contains; runtime caches are not shipped."""
    with zipfile.ZipFile(archive) as stream:
        return {name[len("combined-supplement/"):]: stream.read(name)
                for name in stream.namelist() if not name.endswith("/")}


def run_reproduce(bundle):
    return subprocess.run([sys.executable, "reproduce.py"], cwd=bundle,
                          capture_output=True, text=True)


def test_isolated_extraction_reproduces_every_primary_family(isolated):
    run = run_reproduce(isolated)
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((isolated / "verification.json").read_text())
    assert report["status"] == "passed"
    assert set(report["studies"]) == set(EXPECTED)
    for study, (family, rows) in EXPECTED.items():
        result = report["studies"][study]
        assert result["primary_tests"] == family, study
        assert result["trial_rows"] == rows, study
        assert result["holm_family_recomputed"] is True, study
        # Every contrast's means must be recomputed, not merely counted.
        assert result["contrast_means_recomputed"] == family, study


def test_bundle_carries_no_workspace_path_or_account_identifier(archive):
    for name, payload in sorted(shipped(archive).items()):
        text = payload.decode("utf-8", errors="ignore")
        for value in FORBIDDEN:
            assert value not in text, f"{value!r} leaked into {name}"


def test_bundle_needs_no_network_or_cloud_credentials(archive):
    for name, payload in sorted(shipped(archive).items()):
        if not name.endswith(".py"):
            continue
        source = payload.decode()
        for value in ["boto3", "requests", "urllib.request", "subprocess", "AWS_PROFILE"]:
            assert value not in source, f"{value} referenced in {name}"


def test_reproduce_rejects_a_tampered_trial_export(isolated):
    target = isolated / "replication" / "trials.csv"
    original = target.read_text()
    try:
        target.write_text("tampered")
        run = run_reproduce(isolated)
        assert run.returncode != 0
        assert "Checksum mismatch: replication/trials.csv" in run.stderr
    finally:
        target.write_text(original)
    assert run_reproduce(isolated).returncode == 0, "restoring the file must restore the pass"


def test_reproduce_detects_an_altered_reference_statistic(tmp_path):
    archive = build(tmp_path / "build")
    bundle = tmp_path / "iso" / "combined-supplement"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(tmp_path / "iso")
    analysis = json.loads((bundle / "position_pairs_qwen" / "analysis.json").read_text())
    analysis["primary_comparisons"][0]["delta"] += 0.05
    (bundle / "position_pairs_qwen" / "analysis.json").write_text(json.dumps(analysis))
    manifest = json.loads((bundle / "manifest.json").read_text())
    import hashlib
    manifest["sha256"]["position_pairs_qwen/analysis.json"] = hashlib.sha256(
        (bundle / "position_pairs_qwen" / "analysis.json").read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    run = run_reproduce(bundle)
    assert run.returncode != 0, "a changed delta must not pass reproduction"
    assert "delta differs" in run.stderr


def test_manifest_covers_every_packaged_file(archive):
    contents = shipped(archive)
    manifest = json.loads(contents["manifest.json"])
    assert set(contents) - {"manifest.json"} == set(manifest["sha256"])
    assert "raw replay and GPU regeneration are separate" in manifest["scope"]


def test_readme_separates_the_three_environments(isolated):
    readme = (isolated / "README.md").read_text()
    assert "Statistical reproduction | **yes**" in readme
    assert "Raw replay and reference scoring | no" in readme
    assert "GPU regeneration of trajectories | no" in readme
    for spec in STUDIES.values():
        assert spec["title"] in readme


def test_every_completed_study_is_packaged():
    """A new completed study must be added here, not silently omitted."""
    assert set(STUDIES) == set(EXPECTED)
    for study, spec in STUDIES.items():
        assert spec["trials"].is_file(), f"{study} trials missing"
        assert spec["analysis"].is_file(), f"{study} analysis missing"
