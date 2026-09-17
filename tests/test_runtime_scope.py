import re
from pathlib import Path

import pytest

from scripts.runtime_scope import LABELS, build, build_scope, latency_table, macros, runtime_rows
from src.repair.diagnosis import DIAGNOSIS_STRATEGY, TREATMENT
from scripts.replication_diagnostics import RESTART

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/iclr2027.tex"


@pytest.fixture(scope="module")
def scope():
    return build_scope()


def test_cohort_matches_the_frozen_runtime_design(scope):
    assert scope["totals"]["questions"] == 25
    assert scope["totals"]["attempts"] == 225
    assert {p["strategy"] for p in scope["policies"]} == {TREATMENT, RESTART, DIAGNOSIS_STRATEGY}
    for policy in scope["policies"]:
        assert policy["attempts"] == 75, "25 questions by three seeds"
        assert policy["questions"] == 25


def test_termination_counts_close_against_attempts(scope):
    for policy in scope["policies"]:
        total = (policy["finished"] + policy["step_limited"]
                 + policy["token_limited"] + policy["errored"])
        assert total == policy["attempts"], policy["label"]


def test_runtime_attempts_have_no_token_cap(scope):
    """The runtime design removes the generated-token allowance by construction."""
    assert scope["totals"]["token_limited"] == 0
    for policy in scope["policies"]:
        assert policy["token_limited"] == 0


def test_latency_quantiles_are_ordered_and_bounded(scope):
    for policy in scope["policies"]:
        assert 0 < policy["q25_s"] <= policy["q50_s"] <= policy["q75_s"] <= policy["q90_s"]
        assert policy["q90_s"] <= policy["max_s"]
        # Measured time is selection plus recovery, so neither part can exceed the mean.
        assert policy["selection_mean_s"] <= policy["mean_s"] + 1e-9
        assert policy["recovery_mean_s"] <= policy["mean_s"] + 1e-9


def test_scope_states_what_time_includes_and_what_a_deadline_does(scope):
    assert "prefix replay" in scope["measured_time"]
    assert "once per question" in scope["measured_time"]
    assert "never cancels work" in scope["deadline_role"]
    assert "applied afterwards" in scope["deadline_role"]
    for excluded in ["caching", "concurrency", "repeated acquisition"]:
        assert excluded in scope["excluded"]


def test_table_and_macros_render_from_the_measurements(scope):
    table = latency_table(scope)
    assert table.count(r"\\") == len(scope["policies"]) + 1
    for policy in scope["policies"]:
        assert f"{policy['q50_s']:.2f}" in table
    emitted = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(macros(scope))))
    assert emitted["RuntimeAttempts"] == "225" and emitted["RuntimeQuestions"] == "25"
    assert emitted["RuntimeTokenLimited"] == "0"
    cited = set(re.findall(r"\\(Runtime\w+?)\{\}", MANUSCRIPT.read_text()))
    assert not cited - set(emitted), f"undefined macros cited: {sorted(cited - set(emitted))}"


def test_manuscript_keeps_runtime_claims_inside_the_measured_setup():
    """Every efficiency mention must carry its limiting clause."""
    prose = " ".join(MANUSCRIPT.read_text().split())
    assert "does not measure savings from cancellation" in prose
    assert "not savings from active" in prose
    assert "do not establish wall-clock speedups" in prose
    # No unqualified efficiency claim for the tested policy.
    for overclaim in ["is faster than restart", "reduces latency", "saves time",
                      "more efficient than restart"]:
        assert overclaim not in prose, overclaim


def test_build_emits_assets_in_both_figure_formats(tmp_path):
    scope, record = build(out=tmp_path)
    for name in ["tables/iclr2027_runtime_latency.tex", "iclr2027_runtime_scope.tex",
                 "iclr2027_runtime_scope.json", "figures/iclr2027_runtime_latency.pdf",
                 "figures/iclr2027_runtime_latency.png"]:
        assert (tmp_path / name).is_file(), name
    assert record["source_sha256"] and record["code_sha256"]
