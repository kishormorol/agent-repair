import datetime as dt

import pytest

from scripts.configure_aws_stop import timer_units


NOW = dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc)


@pytest.mark.parametrize("deadline", [
    "2026-09-11T11:59:00Z", "2026-09-11T12:00:00Z",
    "2026-09-11T13:00:01Z", "2026-09-11T12:30:00",
])
def test_unsafe_stop_deadlines_fail_before_installation(deadline):
    with pytest.raises(ValueError):
        timer_units(deadline, NOW)


def test_stop_uses_wall_and_elapsed_time_without_persistent_catchup():
    service, timer = timer_units("2026-09-11T08:30:00-04:00", NOW)
    assert "OnCalendar=2026-09-11 12:30:00 UTC" in timer
    assert "OnActiveSec=1800s" in timer
    assert "Persistent=false" in timer
    assert "shutdown -h now" in service
