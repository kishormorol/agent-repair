import datetime as dt

import pytest

from scripts.configure_aws_stop import timer_units


def test_verified_guest_timer_replaces_bootstrap_shutdown(monkeypatch, tmp_path):
    import sys
    import scripts.configure_aws_stop as stop

    calls=[]
    monkeypatch.setattr(sys,'argv',['configure_aws_stop','--deadline','unused','--install'])
    monkeypatch.setattr(stop,'timer_units',lambda _:('service','timer'))
    monkeypatch.setattr(stop.os,'geteuid',lambda:0)
    monkeypatch.setattr(stop,'Path',lambda _:tmp_path)
    monkeypatch.setattr(stop.subprocess,'run',lambda command,**_:calls.append(command))
    stop.main()
    assert ['shutdown','-c'] in calls
    assert calls.index(['shutdown','-c'])>calls.index(['systemctl','is-active','--quiet','aws119-budget-stop.timer'])


def test_failed_guest_timer_preserves_bootstrap_shutdown(monkeypatch, tmp_path):
    import sys
    import subprocess
    import scripts.configure_aws_stop as stop

    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[:2]==['systemctl','is-active']:
            raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(sys,'argv',['configure_aws_stop','--deadline','unused','--install'])
    monkeypatch.setattr(stop,'timer_units',lambda _:('service','timer'))
    monkeypatch.setattr(stop.os,'geteuid',lambda:0)
    monkeypatch.setattr(stop,'Path',lambda _:tmp_path)
    monkeypatch.setattr(stop.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):
        stop.main()
    assert ['shutdown','-c'] not in calls


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
