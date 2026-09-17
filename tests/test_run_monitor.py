import datetime as dt

import pytest

from scripts.run_monitor import (POLL_TIMEOUT_S, RELAXED_INTERVAL_S, TIGHT_INTERVAL_S,
                                 is_terminal, poll_interval, should_keep_polling)

UTC = dt.timezone.utc


def test_terminal_state_is_recognised():
    for state in ["complete", "failed"]:
        assert is_terminal({"state": state, "archives": []})
    assert not is_terminal({"state": "executing", "archives": []})
    assert not is_terminal({"state": "preflight", "archives": []})


def test_an_archive_is_terminal_even_without_a_terminal_state():
    """Both real runs finished while the monitor still believed them executing."""
    missed = {"state": "auditing", "archives": [{"path": "/x/exports/a.tar.gz", "bytes": 1}]}
    assert is_terminal(missed), "a produced archive means the controller stopped working"


def test_status_must_be_an_object():
    for bad in ["complete", None, ["complete"]]:
        with pytest.raises(ValueError, match="must be an object"):
            is_terminal(bad)


def test_polling_tightens_as_the_deadline_approaches():
    now = dt.datetime(2026, 9, 17, 6, 0, tzinfo=UTC)
    assert poll_interval(now, now + dt.timedelta(minutes=60)) == RELAXED_INTERVAL_S
    assert poll_interval(now, now + dt.timedelta(minutes=10)) == TIGHT_INTERVAL_S
    assert poll_interval(now, now + dt.timedelta(minutes=15)) == TIGHT_INTERVAL_S


def test_polling_never_sleeps_past_the_deadline():
    now = dt.datetime(2026, 9, 17, 6, 0, tzinfo=UTC)
    assert poll_interval(now, now + dt.timedelta(seconds=3)) == 3
    assert poll_interval(now, now) == 0
    assert poll_interval(now, now - dt.timedelta(minutes=1)) == 0


def test_polling_requires_timezone_aware_times():
    naive = dt.datetime(2026, 9, 17, 6, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        poll_interval(naive, naive + dt.timedelta(minutes=5))


def test_poll_timeout_leaves_room_for_a_retry_inside_one_relaxed_interval():
    """A hung poll must not consume the whole window before the next attempt."""
    assert POLL_TIMEOUT_S < 30, "45s polls were what produced the 16-minute gaps"
    assert TIGHT_INTERVAL_S < RELAXED_INTERVAL_S < POLL_TIMEOUT_S


def test_polling_stops_on_a_dead_transport_or_a_passed_deadline():
    now = dt.datetime(2026, 9, 17, 6, 0, tzinfo=UTC)
    later = now + dt.timedelta(minutes=30)
    assert should_keep_polling(now, later, failures=0)
    assert not should_keep_polling(now, later, failures=8)
    assert not should_keep_polling(later, now, failures=0)


def test_the_two_observed_misses_would_now_be_caught():
    """Replay both runs' final observed status: each had finished unseen."""
    qwen = {"state": "complete", "archives": [{"path": "/s/exports/qwen32b.tar.gz"}]}
    mistral = {"state": "complete", "archives": [{"path": "/s/exports/mistral12b.tar.gz"}]}
    assert is_terminal(qwen) and is_terminal(mistral)
    # And the state the monitor last actually saw, one poll earlier.
    assert not is_terminal({"state": "executing", "archives": []})
    assert is_terminal({"state": "auditing", "archives": [{"path": "/s/exports/x.tar.gz"}]})
