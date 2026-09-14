"""Run frozen notebook batches with deadlines, archives, and a final guest stop."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

UTC = dt.timezone.utc


def batch_deadline(overall, now=None):
    now = now or dt.datetime.now(UTC)
    if overall.tzinfo is None or now.tzinfo is None:
        raise ValueError("Study deadlines need explicit timezones")
    if (overall - now).total_seconds() < 180:
        raise ValueError("Too little study allowance remains for another batch")
    return min(overall, now + dt.timedelta(minutes=45))


def run_notebook(command, log_path, deadline, cwd):
    seconds = (deadline - dt.datetime.now(UTC)).total_seconds() - 60
    if seconds <= 0:
        raise ValueError("Archive time must remain before the stop deadline")
    with Path(log_path).open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=cwd, start_new_session=True)
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def archive_batch(session, batch_dir, run, stamp):
    target = session / f"{batch_dir.name}-{stamp}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        archive.add(batch_dir, arcname="session/" + batch_dir.name)
        for name in ("study-freeze.json", "position-profile.json", "run_iclr2027.ipynb",
                     "agent-repair-iclr2027-code.zip", "execution-window.json"):
            archive.add(session / name, arcname="session/" + name)
        if run.exists():
            archive.add(run, arcname="run")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(digest + "  " + target.name + "\n")
    return target, digest


def run_study(session, storage, deadline, notebook_python, analysis_python):
    session, storage = Path(session).resolve(), Path(storage).resolve()
    frozen = json.loads((session / "study-freeze.json").read_text())
    study = frozen["payload"]
    canonical = json.dumps(study, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if frozen["sha256"] != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Frozen study checksum mismatch")
    now = dt.datetime.now(UTC)
    remaining = (deadline - now).total_seconds()
    if (not 0 < remaining <= 240 * 60 or study["gross_cap_usd"] != 25
            or remaining / 3600 * study["gpu_hourly_usd"] + 1.4 > study["gross_cap_usd"]):
        raise ValueError("Study deadline exceeds the frozen compute/recovery allowance")
    window = {"deadline_utc": deadline.isoformat(), "study_sha256": frozen["sha256"]}
    window_path = session / "execution-window.json"
    if window_path.exists() and json.loads(window_path.read_text()) != window:
        raise ValueError("Preserve the original study execution deadline")
    window_path.write_text(json.dumps(window, indent=2) + "\n")
    results = []
    try:
        for batch in study["batches"]:
            batch_dir = session / batch["directory"]
            run = storage / "runs/qwen32b/test" / batch["run_id"] / study["dataset"]
            status_path = batch_dir / "status.json"
            if status_path.exists():
                previous = json.loads(status_path.read_text())
                if previous["returncode"] == 0:
                    archive = session / previous["archive"]
                    if hashlib.sha256(archive.read_bytes()).hexdigest() != previous["sha256"]:
                        raise ValueError("Previously completed batch archive changed")
                    results.append(str(run / "outputs/repairs/results.jsonl"))
                    continue
            end = batch_deadline(deadline)
            subprocess.run(["sudo", "-n", str(notebook_python), str(session / "configure_aws_stop.py"),
                            "--deadline", end.isoformat(), "--install"], check=True)
            stamp = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            started = dt.datetime.now(UTC).isoformat()
            print(f"Starting {batch['run_id']}; stop deadline {end.isoformat()}", flush=True)
            command = [str(notebook_python), str(session / "execute_iclr_notebook.py"),
                       "--notebook", str(session / "run_iclr2027.ipynb"),
                       "--output", str(batch_dir / f"executed-{stamp}.ipynb"),
                       "--parameters", str(batch_dir / "parameters.json")]
            code = run_notebook(command, batch_dir / "run.log", end, session)
            (batch_dir / f"attempt-{stamp}.json").write_text(json.dumps({
                "started_utc": started, "ended_utc": dt.datetime.now(UTC).isoformat(),
                "returncode": code, "stop_deadline_utc": end.isoformat()}, indent=2) + "\n")
            archive, digest = archive_batch(session, batch_dir, run, stamp)
            status_path.write_text(json.dumps({"returncode": code, "archive": archive.name,
                                               "sha256": digest}, indent=2) + "\n")
            print(f"Archived {batch['run_id']}: exit {code}, {archive.name}, {digest}", flush=True)
            if code:
                raise RuntimeError(f"Batch {batch['run_id']} failed with exit {code}; preserved partial artifacts")
            results.append(str(run / "outputs/repairs/results.jsonl"))
        source = storage / "code" / study["code_sha256"][:12]
        subprocess.run([str(analysis_python), str(source / "scripts/analyze_study_batches.py"),
                        "--study", str(session / "study-freeze.json"), "--results", *results,
                        "--output", str(session / "pooled-analysis.json")], check=True)
        (session / "completed-utc.txt").write_text(dt.datetime.now(UTC).isoformat() + "\n")
        print("All frozen study batches completed and pooled.", flush=True)
    finally:
        subprocess.run(["sudo", "-n", "shutdown", "-h", "+3"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--notebook-python", required=True)
    parser.add_argument("--analysis-python", required=True)
    args = parser.parse_args()
    run_study(args.session, args.storage, dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00")),
              args.notebook_python, args.analysis_python)


if __name__ == "__main__":
    main()
