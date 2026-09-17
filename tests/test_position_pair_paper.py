import copy
import json
import re
from pathlib import Path

import pytest

from scripts.position_pair_paper import (blocks_table, build, load_position_pairs, macros,
                                         resolution_floor, results_table)

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/iclr2027.tex"


@pytest.fixture(scope="module")
def evidence():
    return load_position_pairs()


def test_loader_accepts_only_a_complete_exactly_balanced_analysis(evidence):
    protocol, analysis, primary, secondary = evidence
    assert analysis["complete"] and analysis["protocol_sha256"].startswith("9dba9b2b")
    assert sum(a["mismatches"] for a in analysis["audits"]) == 0
    for audit in analysis["audits"]:
        assert audit["balance"]["raw_origin_distribution_equal"]
        assert audit["matched_questions"] == 2 * audit["pairs"]
    assert set(primary) == set(secondary) == set(protocol["cohorts"])


@pytest.mark.parametrize("defect", [
    {"complete": False},
    {"protocol_sha256": "0" * 64},
    {"primary_family_size": 3},
])
def test_loader_rejects_incomplete_or_mismatched_analysis(tmp_path, defect):
    base = tmp_path / "run"
    (base / "prepared-v2").mkdir(parents=True)
    (base / "retrieved/results").mkdir(parents=True)
    source = ROOT / "output/aws-experiment/2026-09-16-position-pairs"
    (base / "prepared-v2/protocol.json").write_text((source / "prepared-v2/protocol.json").read_text())
    (base / "retrieved/results/analysis.json").write_text((source / "retrieved/results/analysis.json").read_text())
    analysis = json.loads((source / "analysis-local.json").read_text())
    analysis.update(defect)
    (base / "analysis-local.json").write_text(json.dumps(analysis))
    with pytest.raises(ValueError):
        load_position_pairs(base)


@pytest.mark.parametrize("field", ["raw_origin_distribution_equal", "normalized_origin_distribution_equal"])
def test_loader_rejects_an_analysis_that_lost_exact_balance(tmp_path, field):
    base = tmp_path / "run"
    (base / "prepared-v2").mkdir(parents=True)
    (base / "retrieved/results").mkdir(parents=True)
    source = ROOT / "output/aws-experiment/2026-09-16-position-pairs"
    for name in ["prepared-v2/protocol.json", "retrieved/results/analysis.json"]:
        (base / name).write_text((source / name).read_text())
    analysis = json.loads((source / "analysis-local.json").read_text())
    analysis["audits"][0]["balance"][field] = False
    (base / "analysis-local.json").write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="not exactly balanced"):
        load_position_pairs(base)


def test_resolution_floor_matches_the_exact_sign_flip_minimum():
    # With m non-zero blocks the smallest attainable two-sided p is 2/2**m.
    for nonzero, expected in [(0, 1.0), (1, 1.0), (2, 0.5), (4, 0.125), (6, 0.03125)]:
        result = {"positive_blocks": nonzero, "negative_blocks": 0, "zero_blocks": 3}
        assert resolution_floor(result) == pytest.approx(expected)


def test_hotpotqa_cell_could_not_have_reached_significance(evidence):
    """The interval excludes zero, so the floor must be reported beside it."""
    _, _, primary, _ = evidence
    result = primary["hotpotqa"]
    assert result["delta_lo"] > 0, "fixture assumption: the percentile interval excludes zero"
    assert resolution_floor(result) > 0.05
    assert result["p_value"] >= resolution_floor(result)


def test_tables_render_every_number_from_the_analysis(evidence):
    _, analysis, primary, secondary = evidence
    table = results_table(analysis, primary, secondary)
    blocks = blocks_table(primary)
    for dataset, label in [("hotpotqa", "HotpotQA"), ("2wikimultihopqa", "2Wiki")]:
        result = primary[dataset]
        assert f"{100 * result['mean_a']:.2f}" in table
        assert f"{100 * result['delta']:+.2f}" in table
        assert f"{result['p_value_holm']:.3f}" in table
        row = next(line for line in blocks.splitlines() if line.startswith(label))
        assert f"& {result['zero_blocks']} &" in row
    assert table.count(r"\\") == len(primary) + 1  # one header plus one row per cell


def test_macros_cover_every_value_the_manuscript_cites(evidence):
    protocol, analysis, primary, secondary = evidence
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                              "\n".join(macros(protocol, analysis, primary, secondary))))
    manuscript = MANUSCRIPT.read_text()
    cited = {name for name in re.findall(r"\\(Pair\w+?)\{\}", manuscript)}
    assert cited, "manuscript cites no position-pair macros"
    missing = cited - set(emitted)
    assert not missing, f"manuscript cites undefined macros: {sorted(missing)}"
    assert emitted["PairPairs"] == str(sum(a["pairs"] for a in analysis["audits"]))
    assert emitted["PairHotpotDelta"] == f"{100 * primary['hotpotqa']['delta']:+.2f}"
    assert emitted["PairHotpotFloor"] == f"{resolution_floor(primary['hotpotqa']):.3f}"


def test_build_writes_assets_and_provenance_over_verified_sources(tmp_path):
    protocol, analysis, primary, secondary, record = build(out=tmp_path)
    for name in ["tables/iclr2027_position_pairs.tex", "tables/iclr2027_position_pair_blocks.tex",
                 "iclr2027_position_pairs.tex", "iclr2027_position_pairs_provenance.json"]:
        assert (tmp_path / name).is_file(), name
    assert record["protocol_sha256"] == analysis["protocol_sha256"]
    assert record["source_sha256"] and record["code_sha256"]
    assert "analysis-local.json" in record["source_sha256"]
    assert record["precision_note"] == protocol["precision_note"]


def test_manuscript_reports_the_experiment_without_overclaiming():
    manuscript = MANUSCRIPT.read_text()
    # LaTeX source wraps, so match prose against whitespace-normalized text.
    prose = " ".join(manuscript.split())
    # The study lives in the appendix; the main text carries only the headline,
    # so the draft stays inside the 9-page main-text limit.
    assert r"\label{sec:position_pairs}" not in manuscript
    assert r"\label{app:position_pairs}" in manuscript
    body = manuscript[:manuscript.index(r"\appendix")]
    assert r"\PairHotpotHolm{}" in body and r"\PairTwoWikiHolm{}" in body
    assert r"\input{generated/iclr2027_position_pairs}" in manuscript
    assert r"\input{generated/tables/iclr2027_position_pairs}" in manuscript
    assert r"\input{generated/tables/iclr2027_position_pair_blocks}" in manuscript
    # The exact-balance claim and its limits must both be stated.
    assert "by construction" in prose and "identical" in prose
    assert "opposite" in prose
    assert "should not be read as a positive effect" in prose
    assert "establishes neither equivalence" in prose
    assert "not a policy anyone could run online" in prose
    # The superseded passage must now point at the follow-up.
    assert r"Appendix~\ref{app:position_pairs}" in manuscript
