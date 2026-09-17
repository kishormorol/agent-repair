import json

import pandas as pd
import pytest

from scripts.finalize_extension_study import compare_exports, finalize


@pytest.fixture
def exports(tmp_path):
    paths = [tmp_path / "local.json", tmp_path / "remote.json"]
    rows = [{"model_key": "model", "dataset": "dataset", "qid": "q", "mode": "token",
             "strategy": s, "seed": seed, "f1": .4, "final_answer": "NA"}
            for s in ["restart", "uncertainty"] for seed in [0, 1, 2]]
    for path in paths:
        path.write_text(json.dumps({"complete": True, "audits": [{"mismatches": 0}], "mean": .4}))
        pd.DataFrame(rows).to_csv(path.with_suffix(".trials.csv"), index=False)
    return paths


def test_extension_reproduction_compares_rows_by_identity_and_tolerates_rounding(exports):
    local, remote = exports
    remote.write_text(json.dumps({"complete": True, "audits": [{"mismatches": 0}], "mean": .4 + 1e-16}))
    path = remote.with_suffix(".trials.csv")
    pd.read_csv(path, keep_default_na=False).iloc[::-1].to_csv(path, index=False)
    result = compare_exports(local, remote)
    assert result["matched"] and result["trial_rows_compared"] == 6


def test_extension_reproduction_records_platform_roundoff_without_requiring_equal_counts(exports):
    local, remote = exports
    for path, count, error in [(local, 1006, 1.78e-15), (remote, 0, 0.)]:
        report = json.loads(path.read_text())
        report["audits"][0].update(model_key="model", dataset="dataset", uncertainty_abs_tolerance=1e-12,
                                   uncertainty_values_with_roundoff=count, uncertainty_max_abs_error=error)
        path.write_text(json.dumps(report))
    result = compare_exports(local, remote)
    assert result["matched"]
    diagnostics = result["uncertainty_roundoff_diagnostics"]
    assert diagnostics["local"][0]["count"] == 1006
    assert diagnostics["remote"][0]["count"] == 0


@pytest.mark.parametrize("field,value", [("uncertainty_max_abs_error", 1e-4),
                                         ("uncertainty_abs_tolerance", 1e-3),
                                         ("uncertainty_values_with_roundoff", -1)])
def test_extension_reproduction_rejects_invalid_roundoff_diagnostics(exports, field, value):
    for path in exports:
        report = json.loads(path.read_text())
        report["audits"][0].update(model_key="model", dataset="dataset", uncertainty_abs_tolerance=1e-12,
                                   uncertainty_values_with_roundoff=1, uncertainty_max_abs_error=1e-15)
        # Matching invalid diagnostics on both machines must still be rejected.
        report["audits"][0][field] = value
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="roundoff"):
        compare_exports(*exports)


@pytest.mark.parametrize("defect", ["incomplete", "analysis", "answer", "missing_row", "duplicate"])
def test_extension_reproduction_rejects_changed_or_incomplete_results(exports, defect):
    local, remote = exports
    if defect in {"incomplete", "analysis"}:
        report = json.loads(remote.read_text())
        report["complete" if defect == "incomplete" else "mean"] = False if defect == "incomplete" else .5
        remote.write_text(json.dumps(report))
    else:
        path = remote.with_suffix(".trials.csv")
        frame = pd.read_csv(path, keep_default_na=False)
        if defect == "answer":
            # Do not collapse literal answer strings such as NA into missing values.
            frame.loc[0, "final_answer"] = ""
        elif defect == "missing_row":
            frame = frame.iloc[:-1]
        else:
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        compare_exports(local, remote)


def test_finalization_refuses_an_active_or_failed_experiment(tmp_path):
    for state in ["executing", "failed"]:
        (tmp_path / "latest-status.json").write_text(json.dumps({"state": state}))
        with pytest.raises(ValueError, match="preserve partial records"):
            finalize(tmp_path)
