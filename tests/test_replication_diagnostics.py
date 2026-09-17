import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.paper_stats import sign_flip_resolution_floor, wins_losses_ties
from scripts.replication_diagnostics import (RESTART, build, build_diagnostics, execution_overlap,
                                             load_replication, macros, outcomes_table,
                                             overlap_table, paired_outcomes, restart_contrast,
                                             termination)
from src.repair.diagnosis import TREATMENT

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/iclr2027.tex"


@pytest.fixture(scope="module")
def diagnostics():
    return build_diagnostics()


def test_resolution_floor_is_two_over_two_to_the_m():
    assert sign_flip_resolution_floor(0) == 1.0
    assert sign_flip_resolution_floor(1) == 1.0
    for m, expected in [(2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (10, 2 / 1024)]:
        assert sign_flip_resolution_floor(m) == pytest.approx(expected)
    for bad in [-1, 1.5, "3", True]:
        with pytest.raises(ValueError):
            sign_flip_resolution_floor(bad)


def test_ties_stay_in_the_estimand_but_leave_the_exact_null():
    split = wins_losses_ties([0.33, -0.33, 0.0, 0.0, 1e-15, 0.67])
    assert split["wins"] == 2 and split["losses"] == 1
    assert split["ties"] == 3, "near-zero differences count as ties"
    assert split["questions"] == 6, "every paired question stays in the estimand"
    assert split["nonzero"] == 3 and split["resolution_floor"] == pytest.approx(0.25)


def test_every_cell_and_contrast_is_covered(diagnostics):
    assert len(diagnostics["cells"]) == 6
    assert {c["model_key"] for c in diagnostics["cells"]} == {"qwen32b", "mistral12b"}
    assert {c["dataset"] for c in diagnostics["cells"]} == {"hotpotqa", "musique", "2wikimultihopqa"}
    for cell in diagnostics["cells"]:
        assert len(cell["contrasts"]) == 4, "four primary controls per cell"
        for contrast in cell["contrasts"]:
            assert contrast["wins"] + contrast["losses"] + contrast["ties"] == contrast["questions"]


def test_diagnostics_reproduce_the_september_review_counts(diagnostics):
    """These exact figures are quoted in the gap review; they must not drift."""
    assert diagnostics["nonzero_question_range"] == [3, 23]
    assert (diagnostics["treatment_limit_stopped"], diagnostics["treatment_rows"]) == (621, 1113)
    expected = {("qwen32b", "hotpotqa"): 16, ("qwen32b", "musique"): 48,
                ("qwen32b", "2wikimultihopqa"): 11, ("mistral12b", "hotpotqa"): 25,
                ("mistral12b", "musique"): 38, ("mistral12b", "2wikimultihopqa"): 27}
    for (model_key, dataset), count in expected.items():
        contrast = restart_contrast(diagnostics, model_key, dataset)
        assert contrast["overlap"]["questions_with_any_differing_execution"] == count


def test_the_qwen_2wiki_example_cannot_reach_significance(diagnostics):
    """A positive-looking interval from four wins, no losses and 24 ties."""
    contrast = restart_contrast(diagnostics, "qwen32b", "2wikimultihopqa")
    assert (contrast["wins"], contrast["losses"], contrast["ties"]) == (4, 0, 24)
    assert contrast["delta_lo"] > 0, "the individual interval excludes zero"
    assert contrast["resolution_floor"] == pytest.approx(0.125)
    assert contrast["p_value"] >= contrast["resolution_floor"]
    assert contrast["p_value_holm"] == 1.0, "frozen primary output is retained unchanged"


def test_two_cells_are_below_the_resolution_of_their_own_test(diagnostics):
    blocked = [(c["model_key"], c["dataset"]) for c in diagnostics["cells"]
               if restart_contrast(diagnostics, c["model_key"], c["dataset"])["resolution_floor"] > 0.05]
    assert set(blocked) == {("qwen32b", "2wikimultihopqa"), ("qwen32b", "hotpotqa")}


def test_overlap_counts_attempts_and_questions_consistently(diagnostics):
    for cell in diagnostics["cells"]:
        for contrast in cell["contrasts"]:
            overlap = contrast["overlap"]
            assert 0 <= overlap["shared_attempts"] <= overlap["attempts"]
            assert 0 <= overlap["questions_with_any_differing_execution"] <= overlap["questions"]
            assert overlap["attempts"] == 3 * overlap["questions"], "three seeds per question"


def test_shared_execution_range_matches_the_review(diagnostics):
    shared = [restart_contrast(diagnostics, c["model_key"], c["dataset"])["overlap"]["shared_fraction"]
              for c in diagnostics["cells"]]
    assert min(shared) == pytest.approx(0.3333, abs=1e-3)
    assert max(shared) == pytest.approx(0.6795, abs=1e-3)


def test_termination_splits_limit_stops_from_finishes(diagnostics):
    for cell in diagnostics["cells"]:
        term = cell["termination"]
        assert term["limit_stopped"] == sum(term["by_reason"].get(r, 0) for r in ("budget", "max_steps"))
        assert sum(term["by_reason"].values()) == term["rows"]


def test_no_pooled_confirmatory_claim_is_produced(diagnostics):
    assert diagnostics["pooled_confirmatory_claim"] is False
    assert "shared across model cells" in diagnostics["pooled_note"]
    assert "primary Holm outputs unchanged" in diagnostics["status"]


def test_loader_rejects_an_unreproduced_analysis(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    source = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
    for name in ["pooled-analysis-local.json", "pooled-analysis-local.trials.csv"]:
        (base / name).write_text((source / name).read_text())
    check = json.loads((source / "analysis-reproduction-check.json").read_text())
    check["mismatches"] = 1
    (base / "analysis-reproduction-check.json").write_text(json.dumps(check))
    with pytest.raises(ValueError, match="do not agree"):
        load_replication(base)


def test_tables_render_from_the_diagnostics(diagnostics):
    outcomes, overlap = outcomes_table(diagnostics), overlap_table(diagnostics)
    assert outcomes.count(r"\\") == len(diagnostics["cells"]) + 1
    assert overlap.count(r"\\") == len(diagnostics["cells"]) + 1
    example = restart_contrast(diagnostics, "qwen32b", "2wikimultihopqa")
    assert f"{100 * example['delta']:+.2f}" in outcomes
    assert f"{example['resolution_floor']:.3f}" in outcomes


def test_build_emits_assets_and_manuscript_cites_only_defined_macros(tmp_path):
    diagnostics, record = build(out=tmp_path)
    for name in ["tables/iclr2027_replication_outcomes.tex", "tables/iclr2027_replication_overlap.tex",
                 "iclr2027_replication_diagnostics.tex", "iclr2027_replication_diagnostics.json"]:
        assert (tmp_path / name).is_file(), name
    assert record["source_sha256"] and record["code_sha256"]
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(macros(diagnostics))))
    cited = set(re.findall(r"\\(Rep\w+?)\{\}", MANUSCRIPT.read_text()))
    assert not cited - set(emitted), f"manuscript cites undefined macros: {sorted(cited - set(emitted))}"
