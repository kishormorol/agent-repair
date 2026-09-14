"""Bounded sequential GPU controller with archives and independent-stop checks."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_extension_study import verify_package


def validate_live_checks(protocol, checks, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    observed = dt.datetime.fromisoformat(checks["observed_utc"])
    start = dt.datetime.fromisoformat(checks["instance_start_utc"])
    deadline = dt.datetime.fromisoformat(checks["deadline_utc"])
    external = dt.datetime.fromisoformat(checks["external_stop_deadline_utc"])
    expires = dt.datetime.fromisoformat(checks["credit_expires_utc"])
    if any(value.tzinfo is None for value in [observed, start, deadline, external, expires]):
        raise ValueError("Live observations require explicit timezones")
    limits = protocol["allocation"]
    rate = checks["gpu_hourly_usd"]
    duration = (deadline-start).total_seconds()
    estimated = duration/3600*rate + limits["overhead_allowance_usd"]
    if not (0 <= (now-observed).total_seconds() <= 900 and start <= now < deadline
            and now < external <= deadline and 0 < duration <= limits["maximum_minutes"]*60
            and 0 < rate <= limits["maximum_hourly_usd"] and expires > deadline):
        raise ValueError("Stale or invalid live execution/rate/credit window")
    if (estimated > limits["session_gross_ceiling_usd"]
            or checks["prior_gross_conservative_bound_usd"] < limits["prior_gross_conservative_bound_usd"]
            or checks["prior_gross_conservative_bound_usd"] + estimated > limits["total_usd"]-limits["reserve_usd"]
            or checks["eligible_credit_conservative_bound_usd"] < estimated):
        raise ValueError("Execution window exceeds the frozen allocation or eligible credits")
    expected = {"account_id": "692430448570", "instance_id": "i-03b33e00c47b11be6", "region": "eu-west-2",
                "instance_type": "g7e.2xlarge", "state": "running", "instance_initiated_shutdown_behavior": "stop",
                "persistent_storage_confirmed": True, "external_stop_verified": True, "credit_ec2_eligibility_verified": True}
    if any(checks.get(key) != value for key, value in expected.items()):
        raise ValueError("Live identity, storage, independent stop, or credit verification missing")
    return deadline


def archive_model(package, results, model, exports):
    exports.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = exports / f"{model}-{stamp}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        archive.add(package, arcname="package", filter=lambda info: None if "__pycache__" in Path(info.name).parts else info)
        if (results/model).exists():
            archive.add(results/model, arcname=f"results/{model}")
        for item in results.glob(f"{model}*.json"):
            archive.add(item, arcname=f"results/{item.name}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix+".sha256").write_text(f"{digest}  {target.name}\n")
    return {"path": str(target), "sha256": digest, "bytes": target.stat().st_size}


def execute(package, session, python, cache, checks):
    package, session = Path(package), Path(session)
    protocol, digest = verify_package(package)
    deadline = validate_live_checks(protocol, checks)
    results, exports = session/"results", session/"exports"
    results.mkdir(parents=True, exist_ok=True)
    status = {"protocol_sha256": digest, "state": "preflight", "archives": [], "completed_models": []}
    status_path = session/"status.json"
    current = None

    def update(**values):
        status.update(values, observed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2)+"\n")

    def arm():
        guest_deadline = min(deadline-dt.timedelta(seconds=60), dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=55))
        subprocess.run(["sudo", python, str(ROOT/"scripts/configure_aws_stop.py"), "--deadline", guest_deadline.isoformat(), "--install"], check=True)
        return time.monotonic()

    def run_bounded(command, log_path):
        armed = arm()
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if dt.datetime.now(dt.timezone.utc) >= deadline-dt.timedelta(minutes=7):
                        raise TimeoutError("Preserve results before the EC2 deadline")
                    if time.monotonic()-armed > 1200:
                        armed = arm()
                    update(active_log=str(log_path), active_pid=process.pid)
                    time.sleep(15)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)

    try:
        arm()
        probe = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        if len(probe.strip().splitlines()) != 1 or "RTX PRO 6000" not in probe:
            raise ValueError("Unexpected GPU identity")
        update(gpu=probe.strip())
        run_bounded([python, "-m", "pytest", "-q", "tests/test_extension_study.py", "tests/test_prefix_replay.py",
                     "tests/test_hotpot_scoring.py", "tests/test_vllm_limits.py"], session/"preflight.log")
        for model in protocol["models"]:
            current = model
            update(state="executing", active_model=model)
            run_bounded([python, "-u", str(ROOT/"scripts/run_extension_study.py"), "--package", str(package),
                         "--output", str(results), "--model", model, "--cache", str(cache), "--deadline-utc", deadline.isoformat()],
                        session/f"{model}.log")
            update(state="auditing")
            run_bounded([python, str(ROOT/"scripts/analyze_extension_study.py"), "--package", str(package), "--output", str(results),
                         "--destination", str(results/f"{model}-analysis.json"), "--model", model], session/f"{model}-audit.log")
            status["archives"].append(archive_model(package, results, model, exports))
            status["completed_models"].append(model)
            update(state="model_complete")
            current = None
        run_bounded([python, str(ROOT/"scripts/analyze_extension_study.py"), "--package", str(package), "--output", str(results),
                     "--destination", str(results/"pooled-analysis.json")], session/"pooled-audit.log")
        update(state="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as error:
        update(state="failed", error=f"{type(error).__name__}: {error}")
        if current:
            status["archives"].append(archive_model(package, results, current, exports))
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
        print(json.dumps({"protocol_sha256": digest, "models": list(protocol["models"]),
                          "main_questions": protocol["main_n_per_dataset_model"]*len(protocol["models"])*len(protocol["cohorts"]),
                          "allocation": protocol["allocation"], "execution": False}, indent=2))
    else:
        if not all([args.session, args.cache, args.checks]):
            parser.error("Execution requires --session, --cache, and --checks")
        execute(args.package, args.session, args.python, args.cache, json.loads(args.checks.read_text()))
