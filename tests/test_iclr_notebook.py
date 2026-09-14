import ast
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from scripts.build_iclr_notebook import ROOT, build_bundle, build_launch_folder, notebook


def test_notebook_is_valid_python_and_never_defaults_to_gpu_execution():
    nb = notebook("a" * 64)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            tree = ast.parse(cell.source, filename=f"cell_{index}")
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "call_python" and isinstance(node.args[0], ast.Constant)):
                    ast.parse(node.args[0].value, filename=f"subprocess_cell_{index}")
            assert cell.outputs == [] and cell.execution_count is None
    parameters = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    assert "EXECUTE_GPU = False" in parameters and 'PHASE = "pilot"' in parameters
    assert "ORIGIN_SWEEP = False" in parameters
    assert "INCLUDE_DIAGNOSTICS = False" in parameters
    assert "MULTIPLIERS = [1.0]" in parameters
    assert "MAX_REPAIR_EXECUTIONS = 120" in parameters
    assert 'PLATFORM = "aws"' in parameters
    assert "TOTAL_BUDGET_USD = 119.0" in parameters
    assert "NONCOMPUTE_RESERVE_USD = 20.0" in parameters
    assert "GPU_HOURLY_USD = None" in parameters
    assert "AWS_CREDITS_CONFIRMED = False" in parameters
    assert "PROVIDER_BUDGET_CONTROLS_CONFIRMED = False" in parameters
    source = "\n".join(c.source for c in nb.cells)
    assert "rmtree" not in source and "git pull" not in source and "git clone" not in source


@pytest.mark.parametrize("settings,error", [
    ({}, "live quote"),
    ({"GPU_HOURLY_USD": 6.0}, "live quote"),
    ({"GPU_HOURLY_USD": 6.0, "BILLING_RATE_CONFIRMED": True,
      "PROVIDER_BUDGET_CONTROLS_CONFIRMED": True}, "AWS credit eligibility"),
])
def test_notebook_budget_preflight_blocks_unconfirmed_spending_before_gpu_commands(tmp_path, monkeypatch,
                                                                                 settings, error):
    nb = notebook("a" * 64)
    parameters = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    preflight = next(c.source for c in nb.cells if "preflight" in c.metadata.get("tags", []))
    namespace = {}
    exec(parameters, namespace)
    namespace.update(EXECUTE_GPU=True, STORAGE_ROOT=tmp_path, REPO=ROOT, json=json)
    namespace.update(settings)
    def refuse_subprocess(*args, **kwargs):
        pytest.fail("An unconfirmed budget must be rejected before commands are launched")
    monkeypatch.setattr("subprocess.run", refuse_subprocess)
    with pytest.raises(ValueError, match=error):
        exec(preflight, namespace)
    assert not list(tmp_path.iterdir())


def test_aws_plan_uses_gross_119_allocation_without_spending(tmp_path, monkeypatch):
    nb = notebook("a" * 64)
    namespace = {}
    exec(next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", [])), namespace)
    namespace.update(REPO=ROOT, json=json, GPU_HOURLY_USD=6.0, SPENT_SO_FAR_USD=9.0,
                     STORAGE_ROOT=tmp_path)
    def refuse_subprocess(*args, **kwargs):
        pytest.fail("Plan-only must not launch commands")
    monkeypatch.setattr("subprocess.run", refuse_subprocess)
    exec(next(c.source for c in nb.cells if "preflight" in c.metadata.get("tags", [])), namespace)
    assert namespace["BUDGET"]["remaining_compute_usd"] == 90
    assert namespace["BUDGET"]["remaining_compute_hours"] == 15
    assert namespace["BUDGET"]["billing_enforced"] is False


def test_bundle_is_allowlisted_hashed_and_deterministic(tmp_path):
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    digest = build_bundle(ROOT, first)
    assert build_bundle(ROOT, second) == digest
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        assert manifest["code_sha256"] == digest
        assert "src/utils/cloud_runs.py" in manifest["files"]
        assert "scripts/run_repair.py" in manifest["files"]
        assert not any(p.startswith((".git/", "outputs/", "data/", "paper/")) or ".env" in p
                       for p in manifest["files"])
        for path, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected


def test_launch_folder_stages_matching_files_and_never_claims_cloud_results(tmp_path):
    bundle = tmp_path / "agent-repair-iclr2027-code.zip"
    digest = build_bundle(ROOT, bundle)
    notebook_path = tmp_path / "run_iclr2027.ipynb"
    nbformat.write(notebook(digest), notebook_path)
    (tmp_path / "START_HERE.md").write_text((ROOT / "START_HERE.md").read_text())
    launch = build_launch_folder(tmp_path, notebook_path, bundle)
    assert launch == tmp_path / "output/aws-upload"
    manifest = json.loads((launch / "launch_manifest.json").read_text())
    assert set(manifest["files"]) == {notebook_path.name, bundle.name, "START_HERE.md"}
    assert not manifest["gpu_provisioned"] and not manifest["results_generated"]
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((launch / name).read_bytes()).hexdigest() == expected
    assert (launch / notebook_path.name).read_bytes() == notebook_path.read_bytes()
    assert (launch / bundle.name).read_bytes() == bundle.read_bytes()


def test_entire_notebook_executes_plan_only_without_gpu_or_network(tmp_path):
    bundle = tmp_path / "agent-repair-iclr2027-code.zip"
    digest = build_bundle(ROOT, bundle)
    nb = notebook(digest)
    client = NotebookClient(nb, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    executed = client.execute()
    assert all(output.output_type != "error" for cell in executed.cells if cell.cell_type == "code"
               for output in cell.outputs)
    assert (tmp_path / ".iclr-notebook-check/code" / digest[:12] / "scripts/run_repair.py").exists()
    assert not list(tmp_path.rglob("results.jsonl"))
