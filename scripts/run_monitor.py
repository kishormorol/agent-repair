"""Polling policy for the cloud run monitor.

Both position-pair runs finished their studies and were missed. The monitor's
SSH polls degrade badly while the GPU is saturated, opening 16 to 18 minute
gaps, so the loop declared "deadline passed without a terminal state" when the
controller had in fact written `complete` minutes earlier. Each miss forced a
separate retrieval session at roughly $1.50.

Two corrections live here so they can be tested away from the SSH transport:
an archive list is itself a terminal signal, and polling tightens as the
deadline approaches instead of running at one relaxed cadence throughout.
"""
from __future__ import annotations

import datetime as dt

TERMINAL_STATES = ("complete", "failed")
# A hung poll must fail fast enough to leave room for another attempt.
POLL_TIMEOUT_S = 20
RELAXED_INTERVAL_S = 15
TIGHT_INTERVAL_S = 6
TIGHTEN_WITHIN = dt.timedelta(minutes=15)


def is_terminal(status):
    """True when the run has finished, by state or by having produced an archive.

    `archive_model` only runs once the controller has stopped working, on
    success or failure alike, so a populated archive list is as conclusive as
    the state field and survives a status write the monitor never managed to
    read.
    """
    if not isinstance(status, dict):
        raise ValueError("Status must be an object")
    if status.get("state") in TERMINAL_STATES:
        return True
    return bool(status.get("archives"))


def poll_interval(now, deadline, tighten_within=TIGHTEN_WITHIN):
    """Seconds to wait before the next poll, tightening near the deadline."""
    if now.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("Polling requires timezone-aware times")
    remaining = (deadline - now).total_seconds()
    if remaining <= 0:
        return 0
    interval = TIGHT_INTERVAL_S if remaining <= tighten_within.total_seconds() else RELAXED_INTERVAL_S
    return min(interval, remaining)


def should_keep_polling(now, deadline, failures, maximum_failures=8):
    """Keep going until the deadline passes or the transport looks truly dead."""
    return failures < maximum_failures and now < deadline
