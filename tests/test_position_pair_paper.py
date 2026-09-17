import json
import re
import shutil
from pathlib import Path

import pytest

from scripts.position_pair_paper import (BASE, CELLS, blocks_table, build, load_cell,
                                         load_position_pairs, macros, ordered_rows,
                                         resolution_floor, results_table)

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/iclr2027.tex"


@pytest.fixture(scope="module")
def cells():
    return load_position_pairs()


def stage(tmp_path, model_key):
    """Copy just the files one cell's loader reads, so defects can be injected."""
    spec = CELLS[model_key]
    base = tmp_path / "run"
    for name in [f"{spec['package']}/protocol.json", spec["analysis"], spec["remote"]]:
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    return base


def test_both_cells_load_complete_and_exactly_balanced(cells):
    assert set(cells) == {"qwen32b", "mistral12b"}
    for key, cell in cells.items():
        assert cell["analysis"]["complete"]
        assert cell["protocol"]["model_key"] == key
        assert sum(a["mismatches"] for a in cell["analysis"]["audits"]) == 0
        for audit in cell["analysis"]["audits"]:
            assert audit["balance"]["raw_origin_distribution_equal"]
            assert audit["matched_questions"] == 2 * audit["pairs"]


def test_the_two_cells_face_an_identical_question_cohort(cells):
    """Shared identifiers make this a contrast, and forbid pooling."""
    qwen, mistral = cells["qwen32b"]["protocol"], cells["mistral12b"]["protocol"]
    assert set(qwen["cohorts"]) == set(mistral["cohorts"])
    for dataset in qwen["cohorts"]:
        assert qwen["cohorts"][dataset]["main_ids"] == mistral["cohorts"][dataset]["main_ids"]
    assert "pooled" in mistral["pooling_limitation"].lower()


@pytest.mark.parametrize("defect", [{"complete": False}, {"protocol_sha256": "0" * 64},
                                    {"primary_family_size": 3}])
def test_loader_rejects_incomplete_or_mismatched_analysis(tmp_path, defect):
    base = stage(tmp_path, "qwen32b")
    analysis = json.loads((base / CELLS["qwen32b"]["analysis"]).read_text())
    analysis.update(defect)
    (base / CELLS["qwen32b"]["analysis"]).write_text(json.dumps(analysis))
    with pytest.raises(ValueError):
        load_cell("qwen32b", base)


@pytest.mark.parametrize("field", ["raw_origin_distribution_equal",
                                   "normalized_origin_distribution_equal"])
def test_loader_rejects_an_analysis_that_lost_exact_balance(tmp_path, field):
    base = stage(tmp_path, "mistral12b")
    name = CELLS["mistral12b"]["analysis"]
    analysis = json.loads((base / name).read_text())
    analysis["audits"][0]["balance"][field] = False
    (base / name).write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="not exactly balanced"):
        load_cell("mistral12b", base)


def test_resolution_floor_matches_the_exact_sign_flip_minimum():
    for nonzero, expected in [(0, 1.0), (1, 1.0), (2, 0.5), (4, 0.125), (6, 0.03125)]:
        assert resolution_floor({"positive_blocks": nonzero, "negative_blocks": 0,
                                 "zero_blocks": 3}) == pytest.approx(expected)


def test_more_pairs_did_not_buy_more_resolution(cells):
    """Mistral/HotpotQA has 33 pairs to Qwen's 23 yet the same floor."""
    qwen = cells["qwen32b"]["primary"]["hotpotqa"]
    mistral = cells["mistral12b"]["primary"]["hotpotqa"]
    assert mistral["n_pairs"] > qwen["n_pairs"]
    assert resolution_floor(mistral) == resolution_floor(qwen) == pytest.approx(0.125)
    # Ties are the binding constraint, not cohort size.
    assert mistral["zero_blocks"] > qwen["zero_blocks"]


def test_the_qwen_hotpotqa_interval_could_not_have_reached_significance(cells):
    result = cells["qwen32b"]["primary"]["hotpotqa"]
    assert result["delta_lo"] > 0, "fixture assumption: the interval excludes zero"
    assert resolution_floor(result) > 0.05
    assert result["p_value"] >= resolution_floor(result)


def test_the_sign_follows_the_dataset_not_the_model(cells):
    for key in cells:
        assert cells[key]["primary"]["hotpotqa"]["delta"] > 0, key
        assert cells[key]["primary"]["2wikimultihopqa"]["delta"] < 0, key


def test_no_cell_reaches_significance_after_adjustment(cells):
    for _, _, result, _ in ordered_rows(cells):
        assert result["p_value_holm"] > 0.05


def test_tables_render_every_cell_from_its_analysis(cells):
    table, blocks = results_table(cells), blocks_table(cells)
    rows = list(ordered_rows(cells))
    assert table.count(r"\\") == len(rows) + 1
    assert blocks.count(r"\\") == len(rows) + 1
    for cell, dataset, result, _ in rows:
        assert f"{100 * result['delta']:+.2f}" in table
        assert f"{result['p_value_holm']:.3f}" in table
        assert cell["label"] in table and cell["label"] in blocks


def test_macros_cover_every_value_the_manuscript_cites(cells):
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(macros(cells))))
    cited = set(re.findall(r"\\(Pair\w+?)\{\}", MANUSCRIPT.read_text()))
    assert cited, "manuscript cites no position-pair macros"
    assert not cited - set(emitted), f"undefined macros cited: {sorted(cited - set(emitted))}"
    assert emitted["PairCells"] == "2"
    assert emitted["PairPairs"] == str(sum(a["pairs"] for c in cells.values()
                                           for a in c["analysis"]["audits"]))
    assert emitted["PairQwenHotpotFloor"] == f"{resolution_floor(cells['qwen32b']['primary']['hotpotqa']):.3f}"
    assert all(name.isalpha() for name in emitted), "LaTeX macro names must be letters only"


def test_build_emits_assets_and_provenance_for_both_cells(tmp_path):
    cells, record = build(out=tmp_path)
    for name in ["tables/iclr2027_position_pairs.tex", "tables/iclr2027_position_pair_blocks.tex",
                 "iclr2027_position_pairs.tex", "iclr2027_position_pairs_provenance.json",
                 "figures/iclr2027_position_pairs.pdf", "figures/iclr2027_position_pairs.png"]:
        assert (tmp_path / name).is_file(), name
    assert set(record["cells"]) == set(CELLS)
    for key, entry in record["cells"].items():
        assert entry["protocol_sha256"] == cells[key]["analysis"]["protocol_sha256"]
        assert entry["source_sha256"]
    assert "no pooled confirmatory statistic" in record["pooling_limitation"]


def test_manuscript_reports_both_cells_without_overclaiming():
    manuscript = MANUSCRIPT.read_text()
    prose = " ".join(manuscript.split())
    assert r"\label{sec:position_pairs}" not in manuscript, "study belongs in the appendix"
    assert r"\label{app:position_pairs}" in manuscript
    assert r"\input{generated/tables/iclr2027_position_pairs}" in manuscript
    assert r"\input{generated/tables/iclr2027_position_pair_blocks}" in manuscript
    body = manuscript[:manuscript.index(r"\appendix")]
    assert r"\PairPairs{}" in body, "the main text must carry the headline"
    assert "by construction" in prose and "identical" in prose
    assert "sign follows the dataset" in prose
    assert "should not be read as a positive effect" in prose
    assert "establishes neither equivalence" in prose
    assert "not a policy anyone could run online" in prose
    assert "forbids a pooled confirmatory statistic" in prose


def test_precision_limit_macros_are_generated_not_asserted(cells):
    """The count of cells below their own test's resolution must be derived."""
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(macros(cells))))
    blocked = [r for _, _, r, _ in ordered_rows(cells) if resolution_floor(r) > 0.05]
    assert emitted["PairBlockedCells"] == str(len(blocked))
    assert emitted["PairTotalCells"] == str(sum(1 for _ in ordered_rows(cells)))
    assert emitted["PairTiedPairs"] == str(sum(r["zero_blocks"] for _, _, r, _ in ordered_rows(cells)))
    # Recorded so the count cannot drift back to an earlier, wrong value.
    assert emitted["PairBlockedCells"] == "3", "three of four cells sit above the 0.05 floor"


def test_manuscript_states_what_remains_outstanding():
    prose = " ".join(MANUSCRIPT.read_text().split())
    assert "Precision, not effect size, binds these nulls" in prose
    assert "binding precision limit" in prose
    assert r"\PairBlockedCells{}" in MANUSCRIPT.read_text()
    for pending in ["Independent verification is assigned but not complete",
                    "not yet incorporated",
                    "were not re-verified",
                    "allocation is exhausted",
                    "pending author attestation"]:
        assert pending in prose, pending
