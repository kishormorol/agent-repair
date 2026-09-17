"""Reproduce downloaded extension archives and compare every remote result."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.build_followup_paper import compare_values
from src.repair.controlled import fingerprint

BASE = ROOT / "output/aws-experiment/2026-09-15-extension-completion"


def split_audit_roundoff(analysis):
    """Validate platform-dependent audit diagnostics without changing evidence."""
    result, diagnostics = copy.deepcopy(analysis), []
    fields = {"uncertainty_values_with_roundoff", "uncertainty_max_abs_error"}
    for audit in result["audits"]:
        present = fields & audit.keys()
        if not present:
            continue
        if present != fields or audit.get("uncertainty_abs_tolerance") != 1e-12:
            raise ValueError("Incomplete or changed uncertainty roundoff tolerance")
        count, error = audit["uncertainty_values_with_roundoff"], audit["uncertainty_max_abs_error"]
        if not (type(count) is int and count >= 0 and type(error) in (int, float)
                and math.isfinite(error) and 0 <= error <= 1e-12 and (count == 0) == (error == 0)):
            raise ValueError("Invalid or excessive uncertainty roundoff diagnostics")
        diagnostics.append({"model_key": audit["model_key"], "dataset": audit["dataset"],
                            "count": count, "maximum_absolute_error": error, "absolute_tolerance": 1e-12})
        for field in fields:
            del audit[field]
    return result, diagnostics


def compare_exports(local_json, remote_json):
    local_json, remote_json = Path(local_json), Path(remote_json)
    local, remote = [json.loads(p.read_text()) for p in [local_json, remote_json]]
    if local.get("complete") is not True or remote.get("complete") is not True:
        raise ValueError("The frozen extension remains incomplete")
    local_evidence, local_roundoff = split_audit_roundoff(local)
    remote_evidence, remote_roundoff = split_audit_roundoff(remote)
    compare_values(local_evidence, remote_evidence)
    if [(r["model_key"], r["dataset"]) for r in local_roundoff] != [(r["model_key"], r["dataset"]) for r in remote_roundoff]:
        raise ValueError("Changed uncertainty roundoff diagnostic coverage")
    keys = ["model_key", "dataset", "qid", "mode", "strategy", "seed"]
    frames = []
    for path in [local_json, remote_json]:
        frame = pd.read_csv(path.with_suffix(".trials.csv"), keep_default_na=False)
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate extension trial identities")
        frames.append(frame.sort_values(keys).reset_index(drop=True))
    try:
        pd.testing.assert_frame_equal(*frames, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as error:
        raise ValueError("Local and remote extension trial exports differ") from error
    return {"matched": True, "absolute_tolerance": 1e-12,
            "trial_rows_compared": len(frames[0]), "trial_columns_compared": len(frames[0].columns),
            "uncertainty_roundoff_diagnostics": {
                "note": "Validated separately per platform; all other analysis fields and trial columns are compared. Raw audits require identical recomputed profiles and policy origins.",
                "local": local_roundoff, "remote": remote_roundoff}}


def finalize(base=BASE):
    base = Path(base).resolve()
    status = json.loads((base / "latest-status.json").read_text())
    if status.get("state") != "complete":
        raise ValueError(f"Extension execution is {status.get('state')}; preserve partial records")
    package = base / "retrieved/package"
    frozen = json.loads((package / "protocol.json").read_text())
    protocol = frozen["payload"]
    if frozen["sha256"] != fingerprint(protocol) or status["protocol_sha256"] != frozen["sha256"]:
        raise ValueError("Downloaded protocol differs from the executed extension")
    if set(status["completed_models"]) != set(protocol["models"]):
        raise ValueError("Incomplete extension model coverage")
    archives = status["archives"]
    if len(archives) != len(protocol["models"]) or len({Path(a["path"]).name for a in archives}) != len(archives):
        raise ValueError("Missing or duplicate completed-model archives")
    archive_hashes = {}
    for item in archives:
        name = Path(item["path"]).name
        local = base / "archives" / name
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != item["sha256"] or local.stat().st_size != item["bytes"]:
            raise ValueError(f"Downloaded archive checksum/size mismatch: {name}")
        archive_hashes[name] = digest

    destination = base / "pooled-analysis-local.json"
    # Execute the downloaded auditor with its own source and package checks.
    command = [sys.executable, str(package / "code/scripts/analyze_extension_study.py"),
               "--package", str(package), "--output", str(base / "retrieved/results"),
               "--destination", str(destination)]
    subprocess.run(command, cwd=package / "code", check=True)
    reproduction = compare_exports(destination, base / "remote-pooled-analysis.json")
    analysis = json.loads(destination.read_text())
    cells = {(m, d) for m in protocol["models"] for d in protocol["cohorts"]}
    found = [(a["model_key"], a["dataset"]) for a in analysis["audits"]]
    if len(found) != len(cells) or set(found) != cells or any(a["mismatches"] for a in analysis["audits"]):
        raise ValueError("Incomplete extension cell audits")
    paths = [destination, destination.with_suffix(".trials.csv"), base / "remote-pooled-analysis.json",
             base / "remote-pooled-analysis.trials.csv", package / "protocol.json"]
    report = {"verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **reproduction,
              "protocol_sha256": frozen["sha256"], "audited_cells": len(cells),
              "main_model_question_evaluations": sum(a["main_questions"] for a in analysis["audits"]),
              "unique_repairs": sum(a["unique_repairs"] for a in analysis["audits"]),
              "mismatches": 0, "archive_sha256": archive_hashes,
              "sha256": {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "resource_state": "Verify EC2 stopped independently; analysis completion does not establish shutdown"}
    (base / "analysis-reproduction-check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    finalize(parser.parse_args().base)
