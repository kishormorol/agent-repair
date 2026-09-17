import copy
import hashlib
import json
import shutil

import pandas as pd
import pytest

from scripts.analyze_extension_study import analyze
from scripts.build_extension_paper import INPUTS, build_assets, load_extension, main, validate_coverage
from scripts.extension_position_balance import (CONTROL, TREATMENT, load_position_balance,
                                               position_distribution, summarize_positions)
from src.repair.controlled import fingerprint
from test_extension_study import extension_run


@pytest.fixture
def extension_evidence(extension_run, tmp_path):
    package, output, _, _ = extension_run
    base = tmp_path / "evidence"
    base.mkdir()
    shutil.copytree(package, base / "retrieved/package")
    shutil.copytree(output, base / "retrieved/results")
    local = base / "pooled-analysis-local.json"
    analysis = analyze(package, output, local)
    shutil.copy2(local, base / "remote-pooled-analysis.json")
    shutil.copy2(local.with_suffix(".trials.csv"), base / "remote-pooled-analysis.trials.csv")
    reproduced = {"matched": True, "mismatches": 0, "protocol_sha256": analysis["protocol_sha256"],
                  "audited_cells": 1, "main_model_question_evaluations": 2,
                  "unique_repairs": analysis["audits"][0]["unique_repairs"],
                  "trial_rows_compared": analysis["audits"][0]["trial_rows"],
                  "sha256": {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in INPUTS}}
    (base / "analysis-reproduction-check.json").write_text(json.dumps(reproduced))
    return base


def test_extension_tables_preserve_success_denominators_and_deadline_families(extension_evidence, tmp_path):
    analysis, protocol = load_extension(extension_evidence)
    out = tmp_path / "tables"
    build_assets(out, analysis, protocol)
    results = (out / "tables/iclr2027_extension_qwen32b_results.tex").read_text()
    assert "HotpotQA & Restart & 6/6 & 100.00" in results
    assert "HotpotQA & Diagnosis/replay & 6/6 & 100.00" in results
    runtime = (out / "tables/iclr2027_extension_runtime.tex").read_text()
    assert "3/3 (100.0)" in runtime and "By 10 s" in runtime
    contrasts = (out / "tables/iclr2027_extension_runtime_contrasts.tex").read_text()
    assert "5 & Restart & +0.00 & [+0.00, +0.00] & 1.000 & --" in contrasts
    assert "10 & Restart & +0.00 & [+0.00, +0.00] & 1.000 & 1.000" in contrasts
    assert (out / "figures/iclr2027_extension_contrasts.pdf").is_file()


@pytest.mark.parametrize("defect", ["incomplete", "missing_cell", "missing_trial", "changed_mean",
                                   "primary_family", "runtime_family", "runtime_cohort"])
def test_extension_tables_reject_incomplete_or_inconsistent_coverage(extension_evidence, defect):
    analysis, protocol = load_extension(extension_evidence)
    trials = pd.read_csv(extension_evidence / "pooled-analysis-local.trials.csv", keep_default_na=False)
    if defect == "incomplete":
        analysis["complete"] = False
    elif defect == "missing_cell":
        analysis["audits"].clear()
    elif defect == "missing_trial":
        trials = trials.iloc[:-1]
    elif defect == "changed_mean":
        next(r for r in analysis["summaries"] if r["mode"] == "token")["incremental_prompt_tokens"] += 1
    elif defect == "primary_family":
        analysis["primary_comparisons"].pop()
    elif defect == "runtime_family":
        next(r for r in analysis["runtime_comparisons"] if r["deadline_s"] == 5)["p_value_holm"] = 1.
    else:
        # Keep all counts and seed rows, but substitute the second failure for
        # the frozen first failure in the runtime cohort.
        trials.loc[trials["mode"] == "runtime", "qid"] = "q2"
    with pytest.raises(ValueError):
        validate_coverage(analysis, protocol, trials)


def test_extension_tables_reject_changed_evidence_after_reproduction(extension_evidence):
    path = extension_evidence / "pooled-analysis-local.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Evidence changed after independent reproduction"):
        load_extension(extension_evidence)


def test_extension_zero_failure_cell_has_no_conditional_repair_rate(extension_evidence, tmp_path):
    analysis, protocol = load_extension(extension_evidence)
    analysis = copy.deepcopy(analysis)
    audit = analysis["audits"][0]
    audit.update(main_failures=0, runtime_questions=0, trial_rows=0, unique_repairs=0)
    for key in ["summaries", "primary_comparisons", "runtime_comparisons"]:
        analysis[key] = []
    trials = pd.read_csv(extension_evidence / "pooled-analysis-local.trials.csv", keep_default_na=False).iloc[:0]
    validate_coverage(analysis, protocol, trials)
    build_assets(tmp_path, analysis, protocol)
    assert "Qwen32B & HotpotQA & 0 & -- & -- & --" in (tmp_path / "tables/iclr2027_extension_overview.tex").read_text()


def test_position_assets_use_failed_dev_questions_token_rows_and_verified_sources(extension_evidence, tmp_path):
    out = tmp_path / "paper"
    main(extension_evidence, out)
    diagnostics = json.loads((out / "iclr2027_extension_position_balance.json").read_text())
    cell = diagnostics["cells"][0]
    assert diagnostics["analysis_status"] == "exploratory_post_results"
    assert cell["development"]["n"] == 1
    assert cell["main_failures"] == 2 and cell["paired_seed_rows"] == 6
    assert cell["shared_execution_pairs"] == 6 and cell["questions_all_seeds_shared"] == 2
    assert cell["max_normalized_ecdf_gap"] == 0
    assert cell["policies"][CONTROL]["mean_retained_steps"] == 0
    rows = pd.read_csv(out / "iclr2027_extension_position_rows.csv")
    assert len(rows) == 12 and rows.n_steps.eq(2).all()
    assert "Qwen32B / HotpotQA & 1 & 2 & 100.00 & 100.00 & 0.00 & 0.00 & 0.000 & 6/6" in (
        out / "tables/iclr2027_extension_position_balance.tex").read_text()
    assert "Dev.-fitted random" in (out / "tables/iclr2027_extension_qwen32b_results.tex").read_text()
    assert (out / "figures/iclr2027_extension_position_ecdf.pdf").is_file()
    provenance = json.loads((out / "iclr2027_extension_provenance.json").read_text())
    for name, digest in provenance["asset_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert "iclr2027_extension_position_balance.json" in provenance["asset_sha256"]
    assert "iclr2027_extension_position_rows.csv" in provenance["asset_sha256"]
    assert any("main/originals" in name for name in diagnostics["source_sha256"])
    assert any("development/uncertainty" in name for name in diagnostics["source_sha256"])


def position_rows():
    rows = []
    for qid, control_origins in [("short", [0, 1, 2]), ("long", [2, 2, 2])]:
        for seed in range(3):
            for policy, origin in [(TREATMENT, 1), (CONTROL, control_origins[seed])]:
                rows.append({"qid": qid, "seed": seed, "strategy": policy, "origin": origin,
                             "execution_id": f"{qid}/{seed}/{origin}", "budget": 10,
                             "prompt_sha256": f"{qid}/{origin}", "mode": "token"})
    return pd.DataFrame(rows)


def test_position_balance_normalizes_each_trace_before_averaging_and_counts_shared_seeds():
    rows = position_rows()
    # An otherwise duplicate runtime record must not change the token diagnostic.
    rows = pd.concat([rows, rows.iloc[:1].assign(mode="runtime")], ignore_index=True)
    cell, selected = summarize_positions(rows, {"short": 3, "long": 5}, [0, 1, 2],
                                        [{"origin": 1, "n_steps": 3}, {"origin": 0, "n_steps": 8}])
    u, c = cell["policies"][TREATMENT], cell["policies"][CONTROL]
    assert u["mean_normalized_origin"] == .375  # mean(1/2, 1/4), not 2/(2+4)
    assert u["mean_retained_steps"] == 1
    assert [r["position"] for r in u["normalized_origin"]] == [.25, .5]
    assert [r["count"] for r in u["normalized_origin"]] == [3, 3]
    assert c["mean_normalized_origin"] == .5
    assert c["mean_retained_steps"] == 1.5
    assert c["origin_zero_fraction"] == 1/6
    assert cell["max_normalized_ecdf_gap"] == pytest.approx(1/3)
    assert cell["shared_execution_pairs"] == 1 and cell["shared_execution_fraction"] == 1/6
    assert cell["questions_all_seeds_shared"] == 0 and cell["questions_any_seed_differs"] == 2
    assert cell["development"]["n"] == 2 and cell["development"]["mean_normalized_origin"] == .25
    assert len(selected) == 12


def test_position_balance_handles_single_step_and_empty_failure_cohorts():
    single = position_distribution([{"origin": 0, "n_steps": 1}])
    assert single["mean_normalized_origin"] == 0 and single["origin_zero_fraction"] == 1
    cell, selected = summarize_positions(position_rows().iloc[:0], {}, [0, 1, 2], [])
    assert selected.empty and cell["paired_seed_rows"] == 0 and cell["development"]["n"] == 0
    assert cell["max_normalized_ecdf_gap"] is None and cell["shared_execution_fraction"] is None
    assert cell["policies"][TREATMENT]["origin_zero_fraction"] is None


@pytest.mark.parametrize("defect", ["missing_seed", "duplicate", "invalid_origin", "false_shared", "prompt", "budget"])
def test_position_balance_rejects_bad_coverage_and_execution_links(defect):
    rows = position_rows()
    if defect == "missing_seed":
        rows = rows.iloc[1:]
    elif defect == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif defect == "invalid_origin":
        rows.loc[0, "origin"] = 3
    elif defect == "false_shared":
        rows.loc[1, "execution_id"] = rows.loc[0, "execution_id"]
    elif defect == "prompt":
        rows.loc[3, "prompt_sha256"] = "different"  # The one shared pair.
    else:
        rows.loc[1, "budget"] = 11
    with pytest.raises(ValueError):
        summarize_positions(rows, {"short": 3, "long": 5}, [0, 1, 2], [])


@pytest.mark.parametrize("defect", ["missing_original", "original_checksum", "original_role", "refitted_profile", "input_data"])
def test_position_loader_rejects_changed_raw_evidence(extension_evidence, defect):
    analysis, protocol = load_extension(extension_evidence)
    cell = extension_evidence / "retrieved/results/qwen32b/hotpotqa"
    if defect == "missing_original":
        (cell / "main/originals/q1.json").unlink()
    elif defect in ["original_checksum", "original_role"]:
        path = cell / "main/originals/q1.json"
        saved = json.loads(path.read_text())
        if defect == "original_checksum":
            saved["payload"]["steps"].pop()
        else:
            saved["identity"]["role"] = "development"
        path.write_text(json.dumps(saved))
    elif defect == "refitted_profile":
        path = cell / "position-profile.json"
        saved = json.loads(path.read_text())
        # Even an internally consistent, newly hashed profile must match the
        # original development-only fit, not a post-results retuning.
        saved["payload"]["payload"]["origins"][0]["origin"] = 1
        saved["payload"]["sha256"] = fingerprint(saved["payload"]["payload"])
        saved["sha256"] = fingerprint(saved["payload"])
        path.write_text(json.dumps(saved))
    else:
        path = extension_evidence / "retrieved/package/data/hotpotqa.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        load_position_balance(extension_evidence, analysis, protocol)
