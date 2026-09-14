"""Run the frozen diagnosis follow-up on the retained AWS instance, with deadlines."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

UTC = dt.timezone.utc


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen execution artifact changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def verified_package(session, *, extract=False):
    session = Path(session).resolve()
    package = read(session/"package-manifest.json")
    if digest(package["files"]) != package["sha256"]:
        raise ValueError("Package manifest checksum mismatch")
    for name, checksum in package["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_sha(session/path) != checksum:
            raise ValueError("Frozen package file mismatch")
    launch = read(session/"launch.json")
    source = Path(launch["storage"])/"code"/launch["code_sha256"][:12]
    with zipfile.ZipFile(session/"code.zip") as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        checksum = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
        if checksum != launch["code_sha256"] or manifest["code_sha256"] != checksum:
            raise ValueError("Frozen code identity mismatch")
        if set(archive.namelist()) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("Unexpected bundle members")
        for name, checksum in manifest["files"].items():
            path = Path(name)
            data = archive.read(name)
            if path.is_absolute() or ".." in path.parts or hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError("Invalid bundled source")
            if name in ["scripts/run_aws_diagnosis_study.py", "scripts/configure_aws_stop.py"]:
                if file_sha(session/path.name) != checksum:
                    raise ValueError("Launch/stop script differs from frozen source")
            if extract:
                target = source/path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError("Previously extracted source changed")
                if not target.exists():
                    target.write_bytes(data)
    protocols = {}
    for role in ["pilot", "main"]:
        wrapper = read(session/role/"protocol.json")
        if digest(wrapper["payload"]) != wrapper["sha256"] or wrapper["sha256"] != launch[role+"_protocol_sha256"]:
            raise ValueError("Protocol identity mismatch")
        protocols[role] = wrapper["payload"]
    if (set(protocols["main"]["question_ids"]) & set(protocols["pilot"]["question_ids"])
            or not set(protocols["pilot"]["question_ids"]) <= set(protocols["main"]["explored_ids"])):
        raise ValueError("Development pilot leaked into the fresh follow-up")
    return launch, protocols, source


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Explicit UTC/timezone required")
    return parsed


def validate_live_checks(launch, checks, now=None):
    now = now or dt.datetime.now(UTC)
    timestamps = ["observed_utc", "instance_start_utc", "deadline_utc",
                  "external_stop_deadline_utc", "credit_expires_utc"]
    if any(not isinstance(checks.get(k), str) or not checks[k] for k in timestamps):
        raise ValueError("Complete the live-check template with measured, timezone-aware timestamps")
    observed, started, end = [timestamp(checks[k]) for k in ["observed_utc", "instance_start_utc", "deadline_utc"]]
    if not 0 <= (now-observed).total_seconds() <= 600:
        raise ValueError("AWS resource and billing observations must be fresh (within ten minutes)")
    if (not started <= now < end or (end-started).total_seconds() > launch["maximum_minutes_from_ec2_start"]*60
            or (end-now).total_seconds() < 180):
        raise ValueError("Deadline exceeds the original EC2 billing window")
    for key in ["instance_id", "account_id", "region", "instance_type"]:
        if checks.get(key) != launch[key]:
            raise ValueError("Live AWS identity differs from the authorized retained instance")
    for key in ["persistent_storage_confirmed", "external_stop_verified", "credit_ec2_eligibility_verified"]:
        if checks.get(key) is not True:
            raise ValueError("Missing verified storage, stop or credit eligibility observation")
    if (checks.get("state") != "running" or checks.get("instance_initiated_shutdown_behavior") != "stop"
            or not now < timestamp(checks["external_stop_deadline_utc"]) <= end
            or timestamp(checks["credit_expires_utc"]) <= end):
        raise ValueError("AWS state, independent stop, shutdown behavior or credit expiry is incompatible")
    for key in ["gpu_hourly_usd", "gross_spent_usd", "eligible_credit_remaining_usd"]:
        if type(checks[key]) not in [int, float] or not math.isfinite(checks[key]) or checks[key] < 0:
            raise ValueError("Invalid live rate, cumulative gross spending or credit balance")
    rate = checks["gpu_hourly_usd"]
    if (not 0 < rate <= launch["maximum_hourly_usd"]
            or (end-started).total_seconds()/3600*rate + launch["session_overhead_reserve_usd"] > launch["gross_cap_usd"]
            or checks["gross_spent_usd"]+launch["gross_cap_usd"]+launch["allocation_reserve_usd"] > launch["total_allocation_usd"]
            or checks["eligible_credit_remaining_usd"] < launch["gross_cap_usd"]+launch["allocation_reserve_usd"]):
        raise ValueError("Follow-up exceeds its frozen spending/credit allowance")
    return end


def copy_once(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if file_sha(source) != file_sha(target):
            raise ValueError("Previously staged input changed")
    else:
        import shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def execute_batch(session, role, index):
    """Child process: all stages share one parent-enforced batch deadline."""
    launch, protocols, source = verified_package(session)
    session = Path(session)
    window = read(session/"execution-window.json")
    if (window["package_sha256"] != read(session/"package-manifest.json")["sha256"]
            or dt.datetime.now(UTC) >= timestamp(window["deadline_utc"])):
        raise ValueError("A batch requires the current parent-controlled execution window")
    sys.path.insert(0, str(source))
    from src.utils import load_config, load_json
    from src.utils.cloud_runs import run_stage
    from src.llm.vllm_client import resolve_agent_model
    protocol = protocols[role]
    batch = protocol["batches"][index]
    folder = session/role/batch["directory"]
    run = Path(batch["run_directory"])
    for name in ["config.yaml", "test_ids.json", "explored_ids.json"]:
        copy_once(folder/name, run/name)
    cfg = load_config(str(run/"config.yaml"))
    if digest(cfg.raw) != batch["configuration_sha256"]:
        raise ValueError("Staged configuration differs from the frozen batch")
    write_once(run/"config_lock.json", {"sha256": batch["configuration_sha256"]})
    write_once(run/"model_snapshot.json", cfg.raw["notebook_provenance"]["model"])
    raw_file = session/"raw/hotpot_dev_distractor_v1.json"
    copy_once(raw_file, Path(cfg.path("data_raw"))/raw_file.name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", HF_HOME=str(Path(launch["storage"])/"huggingface"))
    if role == "pilot":
        records = {r["_id"]: r for r in load_json(raw_file)}
        write_once(Path(cfg.path("data_processed"))/"pool.json", [records[q] for q in batch["ids"]])
        write_once(Path(cfg.path("data_processed"))/"failed_ids.json", batch["ids"])
        for q in batch["ids"]:
            copy_once(session/"pilot/imported-trajectories"/(q+".json"), Path(cfg.path("trajectories"))/(q+".json"))
        resolve_agent_model(cfg)
        write_once(run/"imported_initials.json", {"scope": "Development-only smoke test; prior initial traces, no fresh test outcomes",
            "package_sha256": read(session/"package-manifest.json")["sha256"], "ids": batch["ids"]})
    else:
        for stage in ["run_setup.py", "run_generate.py"]:
            run_stage(sys.executable, source, run, stage, run/"config.yaml", env=env)
    run_stage(sys.executable, source, run, "run_uncertainty.py", run/"config.yaml", env=env)
    extra = ["--protocol", str(session/role/"protocol.json")]
    run_stage(sys.executable, source, run, "run_diagnosis_followup.py", run/"config.yaml", extra=[*extra,"--dry-run"], env=env)
    run_stage(sys.executable, source, run, "run_diagnosis_followup.py", run/"config.yaml", extra=extra, env=env)
    from scripts.analyze_diagnosis_followup import audit_batch
    _, _, audit = audit_batch(run, session/role/"protocol.json", session/"code.zip", session/"hotpot_evaluate_v1.reference.py")
    write_once(folder/"audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def archive_batch(session, role, batch, stamp):
    target = session/f"{role}-{batch['directory']}-{stamp}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for name in ["package-manifest.json", "launch.json", "code.zip", "hotpot_evaluate_v1.reference.py", "execution-window.json"]:
            archive.add(session/name, arcname="session/"+name)
        archive.add(session/role/"protocol.json", arcname="session/"+role+"/protocol.json")
        archive.add(session/role/batch["directory"], arcname="session/"+role+"/"+batch["directory"])
        run = Path(batch["run_directory"])
        if run.exists():
            archive.add(run, arcname="run")
    checksum = file_sha(target)
    target.with_suffix(target.suffix+".sha256").write_text(checksum+"  "+target.name+"\n")
    return target, checksum


def gpu_preflight(python, source, session, deadline):
    from run_aws_study import run_notebook
    probe = ("import json,torch,vllm; "
             "assert vllm.__version__ == '0.19.0', vllm.__version__; "
             "assert torch.cuda.is_available(); "
             "assert torch.cuda.device_count() == 1; "
             "p=torch.cuda.get_device_properties(0); assert p.total_memory/2**30 >= 70; "
             "print(json.dumps({'torch':torch.__version__,'vllm':vllm.__version__,'gpu':p.name,'vram_bytes':p.total_memory}))")
    commands = [[str(python), "-c", probe], [str(python), "-m", "pip", "check"],
                [str(python), "-m", "pip", "freeze"],
                [str(python), "-m", "pytest", "-q", "tests/test_diagnosis_replay.py",
                 "tests/test_controlled_repair.py", "tests/test_repair_budget.py", "tests/test_prefix_replay.py"]]
    for command in commands:
        code = run_notebook(command, session/"gpu-preflight.log", deadline, source)
        if code:
            raise RuntimeError(f"GPU environment/regression preflight failed with exit {code}")


def run_study(session, checks_path, gpu_python, control_python):
    session = Path(session).resolve()
    launch, protocols, source = verified_package(session)
    checks = read(checks_path)
    end = validate_live_checks(launch, checks)
    if str(session) != launch["remote_session"]:
        raise ValueError("Upload the frozen package to its declared remote session directory")
    write_once(session/"execution-window.json", {"package_sha256": read(session/"package-manifest.json")["sha256"],
        "instance_start_utc": checks["instance_start_utc"], "deadline_utc": end.isoformat()})
    write_once(session/("live-checks-"+timestamp(checks["observed_utc"]).strftime("%Y%m%dT%H%M%SZ")+".json"), checks)
    verified_package(session, extract=True)
    sys.path.insert(0, str(source/"scripts"))
    from run_aws_study import batch_deadline, run_notebook
    os.environ.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", HF_HOME=str(Path(launch["storage"])/"huggingface"))

    def arm(deadline):
        subprocess.run(["sudo", "-n", str(control_python), str(session/"configure_aws_stop.py"),
                        "--deadline", deadline.isoformat(), "--install"], check=True)

    try:
        arm(batch_deadline(end))
        snapshot = Path(launch["storage"])/"huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots"/protocols["main"]["model_revision"]
        if not (snapshot/"config.json").is_file() or not (snapshot/"tokenizer_config.json").is_file():
            raise ValueError("Pinned cached model snapshot is unavailable")
        gpu_preflight(gpu_python, source, session, batch_deadline(end))
        for role in ["pilot", "main"]:
            runs = []
            for index, batch in enumerate(protocols[role]["batches"]):
                folder = session/role/batch["directory"]
                status = folder/"status.json"
                if status.exists() and read(status)["returncode"] == 0:
                    previous = read(status)
                    if file_sha(session/previous["archive"]) != previous["archive_sha256"]:
                        raise ValueError("Completed batch archive changed")
                    runs.append(batch["run_directory"])
                    continue
                deadline = batch_deadline(end)
                arm(deadline)
                stamp = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                code = None
                try:
                    command = [str(gpu_python), str(source/"scripts/run_aws_diagnosis_study.py"),
                               "--session", str(session), "--batch", f"{role}:{index}"]
                    code = run_notebook(command, folder/"run.log", deadline, source)
                finally:
                    write_once(folder/f"attempt-{stamp}.json", {"returncode": code, "stop_deadline_utc": deadline.isoformat(),
                        "ended_utc": dt.datetime.now(UTC).isoformat()})
                    archive, checksum = archive_batch(session, role, batch, stamp)
                    status.write_text(json.dumps({"returncode": code, "archive": archive.name, "archive_sha256": checksum}, indent=2)+"\n")
                if code != 0:
                    raise RuntimeError(f"{role} {batch['run_id']} incomplete (exit {code}); partial archive preserved")
                runs.append(batch["run_directory"])
            # The development gate concerns execution/audit integrity, never a desired success rate.
            deadline = batch_deadline(end)
            arm(deadline)
            command = [str(gpu_python), str(source/"scripts/analyze_diagnosis_followup.py"),
                "--protocol", str(session/role/"protocol.json"), "--bundle", str(session/"code.zip"),
                "--reference", str(session/"hotpot_evaluate_v1.reference.py"), "--output", str(session/role/"pooled-analysis.json"),
                "--runs", *runs]
            code = run_notebook(command, session/role/"analysis.log", deadline, source)
            if code:
                raise RuntimeError(f"Complete {role} audit/analysis failed with exit {code}")
            write_once(session/role/"completed.json", {"protocol_sha256": launch[role+"_protocol_sha256"], "audit_complete": True})
        write_once(session/"completed.json", {"main_protocol_sha256": launch["main_protocol_sha256"], "complete": True})
    finally:
        # EBS and previous results remain retained. AWS must have verified shutdown behavior=stop.
        subprocess.run(["sudo", "-n", "shutdown", "-h", "+3"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--checks", type=Path)
    parser.add_argument("--gpu-python", default="/workspace/agent-repair-iclr2027/venv-vllm019/bin/python")
    parser.add_argument("--control-python", default="/workspace/jupyter-env/bin/python")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--batch", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan_only:
        launch, protocols, _ = verified_package(args.session)
        print(json.dumps({"gpu_executed": False, "main_questions": len(protocols["main"]["question_ids"]),
            "pilot_questions": len(protocols["pilot"]["question_ids"]), "main_batches": len(protocols["main"]["batches"]),
            "primary_family_size": len(protocols["main"]["primary_comparisons"]),
            "maximum_repair_executions": sum(len(p["question_ids"])*27 for p in protocols.values()),
            "gross_cap_usd": launch["gross_cap_usd"], "live_checks_required": True}, indent=2))
    elif args.batch:
        role, index = args.batch.split(":")
        execute_batch(args.session, role, int(index))
    elif args.checks:
        run_study(args.session, args.checks, args.gpu_python, args.control_python)
    else:
        parser.error("Use --plan-only or supply fresh --checks for the AWS launch")


if __name__ == "__main__":
    main()
