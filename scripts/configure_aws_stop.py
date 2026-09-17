"""Install a bounded guest stop without replaying expired deadlines on reboot."""
import argparse
import datetime as dt
import math
import os
from pathlib import Path
import subprocess


def timer_units(deadline, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    end = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if end.tzinfo is None or now.tzinfo is None:
        raise ValueError("Stop times require an explicit timezone")
    seconds = math.ceil((end - now).total_seconds())
    if not 0 < seconds <= 3600:
        raise ValueError("Stop deadline must be in the next 60 minutes")
    stop_at = end.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    service = """[Unit]
Description=Stop the bounded AWS agent-repair session
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
"""
    timer = f"""[Unit]
Description=Independent AWS agent-repair billing deadline
[Timer]
OnCalendar={stop_at}
OnActiveSec={seconds}s
Persistent=false
AccuracySec=1s
[Install]
WantedBy=timers.target
"""
    return service, timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    service, timer = timer_units(args.deadline)
    if not args.install:
        print(service + "\n" + timer)
        return
    if os.geteuid() != 0:
        raise PermissionError("Installing the guest timer requires root")
    unit = "aws119-budget-stop"
    subprocess.run(["systemctl", "stop", unit + ".timer"], check=True)
    root = Path("/etc/systemd/system")
    (root / (unit + ".service")).write_text(service)
    (root / (unit + ".timer")).write_text(timer)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", unit + ".timer"], check=True)
    subprocess.run(["systemctl", "is-active", "--quiet", unit + ".timer"], check=True)
    # Retire the short staging shutdown only after its replacement is active.
    # Otherwise it blocks SSH logins and powers off the running experiment.
    subprocess.run(["shutdown", "-c"], check=True)
    subprocess.run(["systemctl", "list-timers", unit + ".timer", "--no-pager"], check=True)


if __name__ == "__main__":
    main()
