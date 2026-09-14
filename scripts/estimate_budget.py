"""Offline spending allocation; this does not monitor or stop cloud billing."""
from __future__ import annotations

import argparse
import json
import math


def budget_allocation(*, total_usd, spent_usd, reserve_usd, hourly_usd, session_usd):
    values = dict(total_usd=total_usd, spent_usd=spent_usd, reserve_usd=reserve_usd,
                  hourly_usd=hourly_usd, session_usd=session_usd)
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if total_usd == 0 or hourly_usd == 0 or session_usd == 0:
        raise ValueError("Total, hourly rate and session allowance must be positive")
    remaining = total_usd - spent_usd - reserve_usd
    if remaining <= 0:
        raise ValueError("No compute allowance remains after actual spend and the reserve")
    if session_usd > remaining:
        raise ValueError("Session allowance exceeds the remaining compute allocation")
    return {**values, "remaining_compute_usd": remaining,
            "remaining_compute_hours": remaining / hourly_usd,
            "session_hours": session_usd / hourly_usd,
            "session_seconds": math.floor(session_usd / hourly_usd * 3600),
            "billing_enforced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-usd", type=float, default=119)
    parser.add_argument("--spent-usd", type=float, default=0)
    parser.add_argument("--reserve-usd", type=float, default=20)
    parser.add_argument("--hourly-usd", type=float, required=True,
                        help="Confirm the actual single-instance quote in the provider console")
    parser.add_argument("--session-usd", type=float, default=10)
    args = parser.parse_args()
    print(json.dumps(budget_allocation(**vars(args)), indent=2))
    print("Planning only: idle time, setup and downloads also consume billed hours. "
          "Update actual spend across every run. Stop compute in the provider console; "
          "storage may still accrue. This tool does not enforce a dollar cap.")


if __name__ == "__main__":
    main()
