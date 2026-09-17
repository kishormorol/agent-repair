import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.build_iclr_draft import (
    DATASETS, NOTEBOOKS, MAIN_INPUTS, MAIN_STUDY, MAIN_STRATEGIES,
    build_main_assets, load_main_study, load_notebook_summaries,
    read_notebook_summary, read_rows,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset,random_rate", [
    ("HotpotQA", 8.1), ("MuSiQue", 3.8), ("2WikiMHQA", 19.0),
])
def test_frozen_notebook_table_excludes_obsolete_intervals(dataset, random_rate):
    directory, index = NOTEBOOKS[dataset]
    path = ROOT / f"results/{directory}/run_colab_{directory}.ipynb"
    summary = read_notebook_summary(path, index)
    assert len(summary) == 10
    assert "..." not in summary
    assert summary["random_step"]["fixed_%"] == random_rate
    assert all(set(row) == {"fixed_%", "avg_tokens", "avg_tool_calls"}
               for row in summary.values())


def test_frozen_notebook_snapshots_match_csv_baselines_and_record_sources():
    rows = {row["dataset"]: row for row in read_rows(
        ROOT / "results/cross_dataset/cross_dataset_traj_stats.csv")}
    summaries, sources = load_notebook_summaries(rows)
    assert set(summaries) == set(DATASETS)
    assert len(sources) == 3
    for relative_path, source in sources.items():
        assert source["sha256"] == hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    rows["HotpotQA"]["full_restart"] = "0.9"
    with pytest.raises(ValueError, match="snapshot mismatch"):
        load_notebook_summaries(rows)


@pytest.mark.parametrize("change,error", [
    ("duplicate", "Duplicate archived strategy"),
    ("missing", "Incomplete archived baseline"),
    ("invalid", "Invalid archived success"),
])
def test_notebook_extraction_rejects_corrupt_displays(tmp_path, change, error):
    strategies = ["random_step", "full_restart", "oracle_targeted",
                  "oracle_targeted__bt2", "oracle_targeted__informed",
                  "oracle_targeted__bt2__informed", "uncertainty_ensemble_any"]
    if change == "duplicate":
        strategies.append("random_step")
    elif change == "missing":
        strategies.remove("oracle_targeted")
    rate = "101" if change == "invalid" else "10"
    body = "".join(f"<tr><td>{s}</td><td>{rate}</td><td>20</td><td>2</td></tr>"
                   for s in strategies)
    html = ("<table><thead><tr><th>strategy</th><th>fixed_%</th>"
            "<th>avg_tokens</th><th>avg_tool_calls</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")
    path = tmp_path / "summary.ipynb"
    path.write_text(json.dumps({"cells": [{"outputs": [{"data": {"text/html": html}}]}]}))
    with pytest.raises(ValueError, match=error):
        read_notebook_summary(path, 0)


def test_restored_nudge_table_compares_the_same_origin():
    rows = {row["dataset"]: row for row in read_rows(
        ROOT / "results/cross_dataset/cross_dataset_traj_stats.csv")}
    summaries, _ = load_notebook_summaries(rows)
    table = (ROOT / "paper/generated/tables/iclr2027_nudges.tex").read_text()
    for dataset in DATASETS:
        for offset in (0, 2):
            key = "oracle_targeted" + ("__bt2" if offset else "")
            generic = summaries[dataset][key]["fixed_%"]
            informed = summaries[dataset][key + "__informed"]["fixed_%"]
            assert (f"{dataset} & {offset} & {generic:.1f} & {informed:.1f} & "
                    f"{informed - generic:+.1f}") in table


@pytest.fixture
def main_evidence(tmp_path):
    base = tmp_path / "main-study"
    for name in MAIN_INPUTS:
        destination = base / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAIN_STUDY / name, destination)
    return base


def test_main_tables_and_intervals_use_all_audited_conditions(tmp_path):
    records = load_main_study()
    build_main_assets(tmp_path, records)
    summary = records["main-summary.json"]
    table = (tmp_path / "tables/iclr2027_main_results.tex").read_text()
    for name, label in MAIN_STRATEGIES.items():
        row = summary["strategies"][name]
        assert f"{label} & {row['successful_seed_trials']}/339 & {100 * row['success_rate']:.2f}" in table
    contrasts = (tmp_path / "tables/iclr2027_main_contrasts.tex").read_text()
    assert "Full restart & -1.47 & [-4.13, +0.88] & 0.358 & 1.000" in contrasts
    assert "Position-matched random & +0.00 & [-2.95, +2.95] & 1.000 & 1.000" in contrasts
    assert (tmp_path / "figures/iclr2027_main_contrasts.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("filename,change,message", [
    ("main-summary.json", "rate", "strategy rate mismatch"),
    ("audits/batch03.json", "audit", "audit contains a mismatch"),
    ("study-freeze.json", "freeze", "frozen study identity mismatch"),
    ("pooled-analysis-local.json", "family", "primary comparison family changed"),
    ("cost-estimate.json", "cost", "cost component mismatch"),
])
def test_main_asset_loader_rejects_inconsistent_evidence(main_evidence, filename, change, message):
    path = main_evidence / filename
    record = json.loads(path.read_text())
    if change == "rate":
        record["strategies"]["full_restart"]["success_rate"] += 0.01
    elif change == "audit":
        record["mismatches"] = 1
    elif change == "freeze":
        record["payload"]["question_ids"].reverse()
    elif change == "family":
        del record["primary_contrasts"]["position_matched_random"]
    elif change == "cost":
        record["estimated_total_usd"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        load_main_study(main_evidence)


def test_main_asset_loader_requires_every_batch_audit(main_evidence):
    (main_evidence / "audits/batch05.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_main_study(main_evidence)


@pytest.fixture
def review_evidence(tmp_path):
    from scripts.build_iclr_draft import REVIEW_DIAGNOSTICS
    for path in [REVIEW_DIAGNOSTICS, *REVIEW_DIAGNOSTICS.parent.glob('diagnostics.*.csv')]:
        shutil.copy2(path, tmp_path / path.name)
    return tmp_path / REVIEW_DIAGNOSTICS.name


def test_review_assets_surface_costs_overlap_and_secondary_scoring(tmp_path):
    from scripts.build_iclr_draft import load_review_diagnostics, build_review_assets
    report = load_review_diagnostics(load_main_study())
    build_review_assets(tmp_path, report)
    assert "Full restart & 210 & 43 & 5 / 8 / 100" in (tmp_path / "tables/iclr2027_review_overlap.tex").read_text()
    costs = (tmp_path / "tables/iclr2027_review_costs.tex").read_text()
    assert "Full restart & 271.6 & 3153.3" in costs
    assert "Uncertainty + backtrack 2 & 269.9 & 4083.3" in costs
    sensitivity = (tmp_path / "tables/iclr2027_review_sensitivity.tex").read_text()
    assert "+2.95 [+0.29, +5.60]" in sensitivity
    assert "Uncertainty + backtrack 2 & 59.20 & 42.00" in (tmp_path / "tables/iclr2027_review_population.tex").read_text()
    cases = (tmp_path / "tables/iclr2027_review_cases.tex").read_text()
    assert cases.count(r"\paragraph{Case") == 3
    assert "Gate; F1, seeds 0/1/2" in cases


@pytest.mark.parametrize("defect", ["status", "primary", "cost", "code", "export"])
def test_review_loader_rejects_changed_or_inconsistent_diagnostics(review_evidence, defect):
    from scripts.build_iclr_draft import load_review_diagnostics
    report = json.loads(review_evidence.read_text())
    if defect == "status":
        report["analysis_status"] = "prespecified_primary"
    elif defect == "primary":
        report["execution_overlap"]["full_restart"]["paired_success"]["delta"] += 0.1
    elif defect == "cost":
        report["policy_accounting"]["full_restart"]["mean_cost_per_attempt"]["recovery_prompt_tokens"] += 1
    elif defect == "code":
        first = next(iter(report["analysis_code_sha256"]))
        report["analysis_code_sha256"][first] = "0" * 64
    else:
        first = next(iter(report["export_sha256"]))
        (review_evidence.parent / first).write_text("modified export")
    review_evidence.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_review_diagnostics(load_main_study(), review_evidence)


def test_every_generated_figure_is_saved_in_both_pdf_and_png():
    """LaTeX embeds the PDF; the PNG is for slides, issues and quick review."""
    figures = ROOT / "paper/generated/figures"
    stems = {path.stem for path in figures.glob("*.pdf")} | {path.stem for path in figures.glob("*.png")}
    assert stems, "no generated figures found"
    missing = {stem: [suffix for suffix in ("pdf", "png")
                      if not (figures / f"{stem}.{suffix}").is_file()]
               for stem in sorted(stems)}
    assert not any(missing.values()), f"figures missing a format: { {k: v for k, v in missing.items() if v} }"


def test_every_figure_builder_emits_both_formats():
    """A new savefig call must not reintroduce a single-format figure."""
    single = []
    for script in sorted((ROOT / "scripts").glob("*.py")):
        source = script.read_text()
        if "paper/generated" not in source and "figures" not in source:
            continue
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("fig.savefig("):
                continue
            # Dual-format calls interpolate the extension from the loop variable.
            if "{extension}" in stripped or "path" in stripped:
                continue
            if '.png"' in stripped or ".png'" in stripped:
                single.append(f"{script.name}: {stripped}")
    assert not single, "single-format figure savefig calls: " + "; ".join(single)


def test_new_experiment_figures_are_referenced_by_the_manuscript():
    manuscript = (ROOT / "paper/iclr2027.tex").read_text()
    for stem in ["iclr2027_position_pairs", "iclr2027_replication_diagnostics"]:
        assert f"generated/figures/{stem}.pdf" in manuscript, stem
        assert (ROOT / f"paper/generated/figures/{stem}.png").is_file(), stem


def test_generated_macro_names_are_valid_latex_control_sequences():
    """A digit in a macro name silently breaks the build at \\begin{document}."""
    import re
    bad = []
    for path in sorted((ROOT / "paper/generated").glob("iclr2027*.tex")):
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", path.read_text()):
            if not name.isalpha():
                bad.append(f"{path.name}: {name}")
    assert not bad, f"macro names must be letters only: {bad}"
