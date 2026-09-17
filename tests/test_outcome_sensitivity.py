import json
import re
from pathlib import Path

import pytest

from scripts.outcome_sensitivity import (POLICIES, build, build_sensitivity, full_cohort_table,
                                         initial_table, macros)
from scripts.replication_diagnostics import RESTART
from src.repair.diagnosis import TREATMENT

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/iclr2027.tex"


@pytest.fixture(scope="module")
def sensitivity():
    return build_sensitivity()


def test_initial_cohort_reproduces_the_review_counts(sensitivity):
    """229 accepted, 177 strict EM, 52 accepted but never offered repair."""
    totals = sensitivity["totals"]
    assert totals["questions"] == 600
    assert totals["accepted"] == 229
    assert totals["exact_match"] == 177
    assert totals["accepted_not_exact"] == 52
    assert 100 * totals["accepted_not_exact"] / totals["questions"] == pytest.approx(8.67, abs=0.01)


def test_accepted_never_falls_below_strict_exact_match(sensitivity):
    """The gate is a superset of EM, so this ordering must hold everywhere."""
    for cell in sensitivity["cells"]:
        assert cell["exact_match"] <= cell["accepted"] <= cell["questions"]
        assert cell["accepted_not_exact"] == cell["accepted"] - cell["exact_match"]
        for policy in POLICIES:
            outcome = cell["policies"][policy]
            assert outcome["exact_match"] <= outcome["accepted"] <= outcome["attempts"]


def test_failures_offered_repair_complete_the_cohort(sensitivity):
    for cell in sensitivity["cells"]:
        assert cell["accepted"] + cell["failures_offered_repair"] == cell["questions"]


def test_a_third_of_accepted_repairs_fail_strict_exact_match(sensitivity):
    totals = sensitivity["totals"]
    assert totals["repair_accepted"] == 246
    assert totals["repair_accepted_not_exact"] == 84
    share = totals["repair_accepted_not_exact"] / totals["repair_accepted"]
    assert share == pytest.approx(0.341, abs=0.002), "the conclusion depends on the gate"


def test_full_cohort_preserves_already_accepted_answers(sensitivity):
    """Full-cohort outcomes can never fall below the initial accepted rate."""
    for cell in sensitivity["cells"]:
        initial = cell["accepted"] / cell["questions"]
        for policy in POLICIES:
            assert cell["policies"][policy]["full_cohort"] >= initial - 1e-12
            assert cell["policies"][policy]["full_cohort"] <= 1.0 + 1e-12


def test_full_cohort_deltas_stay_small_in_every_cell(sensitivity):
    deltas = [100 * cell["full_cohort_delta"] for cell in sensitivity["cells"]]
    assert max(abs(d) for d in deltas) < 3.0
    assert min(deltas) < 0 < max(deltas), "cells disagree in direction"


def test_full_cohort_matches_a_direct_recomputation(sensitivity):
    """Recompute one cell independently of the builder's own arithmetic."""
    import pandas as pd
    trials = pd.read_csv(ROOT / "output/aws-experiment/2026-09-15-extension-completion"
                                "/pooled-analysis-local.trials.csv")
    token = trials[trials["mode"] == "token"]
    cell = next(c for c in sensitivity["cells"]
                if (c["model_key"], c["dataset"]) == ("qwen32b", "2wikimultihopqa"))
    rows = token[(token.model_key == "qwen32b") & (token.dataset == "2wikimultihopqa")
                 & (token.strategy == TREATMENT)]
    expected = (cell["accepted"] + rows.groupby("qid").success.mean().sum()) / cell["questions"]
    assert cell["policies"][TREATMENT]["full_cohort"] == pytest.approx(expected)


def test_status_marks_rescoring_as_exploratory_not_a_new_cohort(sensitivity):
    assert "not a new failure cohort" in sensitivity["status"]
    assert "never offered repair" in sensitivity["unrepaired_note"]
    assert "separately frozen study" in sensitivity["unrepaired_note"]
    assert "never strict EM" in sensitivity["terminology"]


def test_tables_and_macros_render_from_the_analysis(sensitivity):
    gate, cohort = initial_table(sensitivity), full_cohort_table(sensitivity)
    assert "& 600 & 229 & 177 & 52" in gate.replace(r"\textbf{All} & ", "")
    assert gate.count(r"\\") == len(sensitivity["cells"]) + 2  # header, six cells, totals
    assert cohort.count(r"\\") == len(sensitivity["cells"]) + 1
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(macros(sensitivity))))
    assert emitted["GateOnly"] == "52" and emitted["GateOnlyPct"] == "8.67"
    assert emitted["GateRepairOnlyPct"] == "34.1"
    cited = set(re.findall(r"\\(Gate\w+?)\{\}", MANUSCRIPT.read_text()))
    assert not cited - set(emitted), f"undefined macros cited: {sorted(cited - set(emitted))}"


def test_build_emits_assets_in_both_figure_formats(tmp_path):
    sensitivity, record = build(out=tmp_path)
    for name in ["tables/iclr2027_outcome_gate.tex", "tables/iclr2027_outcome_full_cohort.tex",
                 "iclr2027_outcome_sensitivity.tex", "iclr2027_outcome_sensitivity.json",
                 "figures/iclr2027_outcome_sensitivity.pdf",
                 "figures/iclr2027_outcome_sensitivity.png"]:
        assert (tmp_path / name).is_file(), name
    assert record["source_sha256"] and record["code_sha256"]
