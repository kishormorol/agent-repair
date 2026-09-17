"""Bounded single-model GPU controller for the frozen position-pair study."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_aws_extension_study import archive_model, validate_live_checks
from scripts.run_extension_study import verify_package

PREFLIGHT = ["tests/test_position_pairs.py", "tests/test_position_pair_study.py",
             "tests/test_position_matched.py", "tests/test_prefix_replay.py",
             "tests/test_hotpot_scoring.py", "tests/test_vllm_limits.py"]


def execute(package, session, python, cache, checks):
    package, session = Path(package), Path(session)
    protocol, digest = verify_package(package)
    model = protocol["model_key"]
    deadline = validate_live_checks(protocol, checks)
    results, exports = session / "results", session / "exports"
    results.mkdir(parents=True, exist_ok=True)
    status = {"protocol_sha256": digest, "run_id": protocol["run_id"], "state": "preflight", "archives": []}
    status_path = session / "status.json"

    def update(**values):
        status.update(values, observed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    def arm():
        guest_deadline = min(deadline - dt.timedelta(seconds=60),
                             dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=55))
        subprocess.run(["sudo", python, str(ROOT / "scripts/configure_aws_stop.py"),
                        "--deadline", guest_deadline.isoformat(), "--install"], check=True)
        return time.monotonic()

    def run_bounded(command, log_path):
        armed = arm()
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if dt.datetime.now(dt.timezone.utc) >= deadline - dt.timedelta(minutes=7):
                        raise TimeoutError("Preserve results before the EC2 deadline")
                    if time.monotonic() - armed > 1200:
                        armed = arm()
                    update(active_log=str(log_path), active_pid=process.pid)
                    time.sleep(15)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)

    try:
        arm()
        probe = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        if len(probe.strip().splitlines()) != 1 or "RTX PRO 6000" not in probe:
            raise ValueError("Unexpected GPU identity")
        update(gpu=probe.strip())
        run_bounded([python, "-m", "pytest", "-q", *PREFLIGHT], session / "preflight.log")
        update(state="executing", active_model=model)
        run_bounded([python, "-u", str(ROOT / "scripts/run_position_pair_study.py"), "--package", str(package),
                     "--output", str(results), "--cache", str(cache), "--deadline-utc", deadline.isoformat()],
                    session / f"{model}.log")
        update(state="auditing")
        run_bounded([python, str(ROOT / "scripts/analyze_position_pair_study.py"), "--package", str(package),
                     "--output", str(results), "--destination", str(results / "analysis.json")],
                    session / "audit.log")
        status["archives"].append(archive_model(package, results, model, exports))
        update(state="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as error:
        update(state="failed", error=f"{type(error).__name__}: {error}")
        status["archives"].append(archive_model(package, results, model, exports))
        update(state="failed")
        raise
    finally:
        # Leave eight minutes for the active local retrieval, bounded by the
        # already armed guest timer and the separately verified AWS scheduler.
        subprocess.run(["sudo", "shutdown", "-h", "+8"], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--checks", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        protocol, digest = verify_package(args.package)
        print(json.dumps({"protocol_sha256": digest, "run_id": protocol["run_id"],
                          "model": protocol["model_key"],
                          "main_questions": protocol["main_n_per_dataset"] * len(protocol["cohorts"]),
                          "allocation": protocol["allocation"], "execution": False}, indent=2))
    else:
        if not all([args.session, args.cache, args.checks]):
            parser.error("Execution requires --session, --cache, and --checks")
        execute(args.package, args.session, args.python, args.cache, json.loads(args.checks.read_text()))
