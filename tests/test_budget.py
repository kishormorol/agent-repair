import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.estimate_budget import budget_allocation


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(total_usd=200, spent_usd=0, reserve_usd=40, hourly_usd=1.59, session_usd=10)


def test_budget_reserves_funds_and_counts_previous_spending():
    first = budget_allocation(**DEFAULTS)
    assert first["remaining_compute_usd"] == 160
    assert first["remaining_compute_hours"] == pytest.approx(100.6289308)
    assert first["session_hours"] == pytest.approx(6.2893082)
    assert first["session_seconds"] == 22641
    assert first["billing_enforced"] is False
    later = budget_allocation(**dict(DEFAULTS, spent_usd=80))
    assert later["remaining_compute_usd"] == 80
    assert later["remaining_compute_hours"] == first["remaining_compute_hours"] / 2


@pytest.mark.parametrize("change", [
    {"spent_usd": 160}, {"spent_usd": 190}, {"reserve_usd": 200},
    {"spent_usd": 155, "session_usd": 10}, {"hourly_usd": 0},
    {"total_usd": 0}, {"session_usd": 0}, {"spent_usd": -1},
    {"hourly_usd": float("nan")}, {"total_usd": float("inf")},
    {"reserve_usd": True}, {"hourly_usd": "1.59"},
])
def test_budget_rejects_invalid_or_unaffordable_allocations(change):
    with pytest.raises(ValueError):
        budget_allocation(**dict(DEFAULTS, **change))


def test_budget_cli_runs_without_gpu_and_explains_no_billing_enforcement():
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/estimate_budget.py"),
                                "--hourly-usd", "1.59", "--spent-usd", "25"],
                               check=True, text=True, capture_output=True)
    payload, _ = json.JSONDecoder().raw_decode(completed.stdout)
    assert payload["total_usd"] == 119 and payload["reserve_usd"] == 20
    assert payload["remaining_compute_usd"] == 74
    assert "does not enforce a dollar cap" in completed.stdout
