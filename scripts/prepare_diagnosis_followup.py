"""Freeze a new cohort, a development smoke test, and a bounded AWS follow-up."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import re
import shlex
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from scripts.build_iclr_notebook import build_bundle
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import DIAGNOSIS_PROMPT, DIAGNOSIS_SETTINGS, DIAGNOSIS_STRATEGY, TREATMENT
from src.utils import load_json
from src.utils.cloud_runs import write_once
from src.utils.cohorts import read_ids


MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"
REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
RAW_SHA = "a9b4c5906587ac0d20eb83dcb29d0240e268e342c170f7f22988f59f13fa82c3"
REFERENCE_SHA = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
MAIN_SHA = "e4ea08c8291b0f797d05b15110d762634ec48d57ae53e05aaf4373c805986308"
STRATEGIES = ["full_restart", TREATMENT, DIAGNOSIS_STRATEGY]
MULTIPLIERS = [0.5, 1.0, 2.0]
STORAGE = Path("/workspace/agent-repair-iclr2027")
SESSION = Path("/workspace/aws119-session/diagnosis-20260913-v1")
DEVELOPMENT = ROOT/"output/aws-experiment/2026-09-11-development"
DEV_RUN = DEVELOPMENT/"results/runs/qwen32b/development/aws119-development50-20260911-v1/hotpotqa"
MAIN = ROOT/"output/aws-experiment/2026-09-11-main"


def write_bytes_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen file changed: {path}. Use a new output and run tag.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def select_new_ids(records, explored, n=250, seed=20260913):
    ids = [r["_id"] for r in records]
    if len(ids) != len(set(ids)) or not set(explored) <= set(ids):
        raise ValueError("Dataset IDs must be unique and contain every documented exclusion")
    eligible = sorted(set(ids)-set(explored))
    if not 0 < n <= len(eligible):
        raise ValueError("Not enough eligible questions")
    return random.Random(seed).sample(eligible, n), len(eligible)


def make_configuration(template, *, storage, run_id, ids, explored, code_sha, role):
    raw = copy.deepcopy(template)
    run = Path(storage)/"runs/qwen32b"/("pilot" if role == "development_validation" else "test")/run_id/"hotpotqa"
    snapshot = Path(storage)/"huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots"/REVISION
    raw["project"]["seed"] = 42
    raw["models"]["agent"]["name"] = str(snapshot)
    raw["dataset"].update(name="hotpotqa", split="test", pool_size=len(ids), process_limit=None,
        id_manifest=str(run/"test_ids.json"), exclude_ids_manifest=str(run/"explored_ids.json"))
    raw["runtime"].update(gen_batch_size=2, repair_batch_size=2, uncertainty_batch_size=2)
    raw["agent"].update(max_steps=8, max_tokens_per_step=512, temperature=0.0, logprobs_topk=20)
    raw["repair"].update(run_id=run_id, strategies=STRATEGIES, seeds=[0, 1, 2],
        step_budget_mode="new", match_restart_hint=True, origin_sweep=False)
    raw["repair"]["nudge"].update(enabled=True, temperature=0.7)
    raw["repair"]["budget"].update(match_to="original_generated_tokens", multipliers=MULTIPLIERS)
    raw["notebook_provenance"] = {"code_sha256": code_sha, "phase": role, "primary_strategy": TREATMENT,
        "model": {"repo_id": MODEL_ID, "revision": REVISION, "snapshot_path": str(snapshot)}}
    for key in raw["paths"]:
        if key.endswith("_base"):
            raw["paths"][key] = str(run)
        elif key.startswith("data_"):
            raw["paths"][key] = str(run/"data"/key.removeprefix("data_"))
        else:
            raw["paths"][key] = str(run/"outputs"/key) if key != "outputs" else str(run/"outputs")
    return raw


def freeze_cohort(output, *, template, storage, run_tag, question_ids, explored, code_sha, role, batch_n):
    output = Path(output)
    if (not question_ids or len(question_ids) != len(set(question_ids)) or not explored
            or set(question_ids) & set(explored)):
        raise ValueError("Freeze a nonempty, unique cohort disjoint from documented exclusions")
    batches = []
    for i, start in enumerate(range(0, len(question_ids), batch_n), 1):
        ids = question_ids[start:start+batch_n]
        run_id = f"{run_tag}-b{i:02d}"
        raw = make_configuration(template, storage=storage, run_id=run_id, ids=ids, explored=explored,
                                 code_sha=code_sha, role=role)
        folder = output/f"batch{i:02d}"
        write_bytes_once(folder/"config.yaml", yaml.safe_dump(raw, sort_keys=False).encode())
        write_once(folder/"test_ids.json", ids)
        write_once(folder/"explored_ids.json", explored)
        batches.append({"directory": folder.name, "run_id": run_id, "ids": ids,
                        "configuration_sha256": fingerprint(raw), "run_directory": raw["paths"]["local_base"]})
    payload = {"schema": 1, "frozen_date_utc": "2026-09-13", "cohort_role": role,
        "dataset": "hotpotqa", "question_ids": question_ids, "explored_ids": explored,
        "cohort_sampling": "Uniform without replacement from sorted eligible IDs; Python Random seed 20260913."
            if role != "development_validation" else "First five sorted known development failures; original traces imported.",
        "stratification": None, "model_id": MODEL_ID, "model_revision": REVISION,
        "strategies": STRATEGIES, "seeds": [0, 1, 2], "multipliers": MULTIPLIERS,
        "initial_temperature": 0.0, "repair_temperature": 0.7, "initial_seed": 42,
        "max_new_steps": 8, "max_tokens_per_step": 512, "batch_size": 2,
        "retry_hint": template["repair"]["nudge"]["retry_hint"],
        "diagnosis_prompt_sha256": fingerprint(DIAGNOSIS_PROMPT), "diagnosis_settings": DIAGNOSIS_SETTINGS,
        "code_sha256": code_sha, "raw_dataset_sha256": RAW_SHA, "official_scorer_sha256": REFERENCE_SHA,
        "prior_main_study_sha256": MAIN_SHA, "max_repair_executions_per_batch": batch_n*27,
        "batches": batches,
        "primary_comparisons": [{"treatment": TREATMENT, "control": c, "multiplier": m}
                                for m in MULTIPLIERS for c in ["full_restart", DIAGNOSIS_STRATEGY]],
        "primary_outcome": "Repair success: official answer EM or answer F1 >= 0.5, on initial failures.",
        "analysis": {"unit": "Question mean across three seeds", "bootstrap_iters": 10000,
                     "ci": 0.95, "multiplicity": "One Holm family of six two-sided sign-flip tests",
                     "secondary": "Twelve EM/F1 contrasts with unadjusted intervals; exploratory",
                     "completion": "Analyze the complete frozen cohort; an interrupted study is incomplete, not a smaller confirmatory study."},
        "precision_note": "250 initial questions is a resource-bounded sample; no guarantee of a 2 percentage-point interval half-width or sufficient power.",
        "diagnosis_note": "Untrained same-model diagnosis/replay adaptation, not a Doctor-RAG reproduction. Diagnosis sees question and failed trace, excluding reference labels and stored uncertainty. Recovery receives only the selected prefix and common retry hint, not diagnosis prose or discarded suffix.",
        "cost_note": "Recovery allowances are matched; diagnosis adds measured cost. Charge one complete diagnosis to each policy attempt. Report generated tokens, prompt tokens, requests and tool calls; not FLOPs or a matched total-cost claim.",
        "selection_note": "Prospective follow-up motivated by review of the completed primary study; do not merge or replace its four-comparison primary family.",
        "history_limitation": "Disjoint from documented reconstructed history and the completed main cohort; the original full Drive pool has not been byte-compared."}
    write_once(output/"protocol.json", {"sha256": fingerprint(payload), "payload": payload})
    return payload


def prepare(output, remote_session=SESSION, storage=STORAGE, run_tag="aws119-diagnosis250-20260913-v1"):
    output, remote_session, storage = Path(output).resolve(), Path(remote_session), Path(storage)
    if (not remote_session.is_absolute() or not storage.is_absolute()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_tag)):
        raise ValueError("Use absolute remote paths and a simple new run tag")
    raw_file = DEV_RUN/"data/raw/hotpot_dev_distractor_v1.json"
    reference = DEVELOPMENT/"hotpot_evaluate_v1.reference.py"
    if file_fingerprint(raw_file) != RAW_SHA or file_fingerprint(reference) != REFERENCE_SHA:
        raise ValueError("Frozen dataset/reference checksum mismatch")
    main = load_json(MAIN/"study-freeze.json")
    if main["sha256"] != MAIN_SHA or fingerprint(main["payload"]) != MAIN_SHA:
        raise ValueError("The completed main study changed")
    explored = sorted(set(read_ids(MAIN/"explored-ids.json")) | set(read_ids(MAIN/"test-ids-250.json")))
    records = load_json(raw_file)
    ids, eligible = select_new_ids(records, explored)
    if (len(records), len(explored), eligible) != (7405, 799, 6606):
        raise ValueError("Documented cohort counts changed")
    pilot_ids = sorted(read_ids(DEV_RUN/"data/processed/failed_ids.json"))[:5]
    if len(pilot_ids) != 5 or not set(pilot_ids) <= set(explored):
        raise ValueError("The smoke test must use five already explored development failures")
    template = yaml.safe_load((ROOT/"config/config_iclr_pilot.yaml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)/"code.zip"
        code_sha = build_bundle(ROOT, bundle)
        write_bytes_once(output/"code.zip", bundle.read_bytes())
    main_protocol = freeze_cohort(output/"main", template=template, storage=storage, run_tag=run_tag,
        question_ids=ids, explored=explored, code_sha=code_sha, role="confirmatory_followup", batch_n=50)
    pilot_protocol = freeze_cohort(output/"pilot", template=template, storage=storage, run_tag=run_tag+"-pilot",
        question_ids=pilot_ids, explored=sorted(set(explored)-set(pilot_ids)), code_sha=code_sha,
        role="development_validation", batch_n=5)
    for q in pilot_ids:
        trace = load_json(DEV_RUN/"outputs/trajectories"/(q+".json"))
        dev_pin = load_json(DEV_RUN/"model_snapshot.json")
        if (trace["success"] or dev_pin["revision"] != REVISION or dev_pin["repo_id"] != MODEL_ID
                or trace["meta"]["model_name"] != dev_pin["snapshot_path"]):
            raise ValueError("Unexpected development trajectory/model")
        write_bytes_once(output/"pilot/imported-trajectories"/(q+".json"), (DEV_RUN/"outputs/trajectories"/(q+".json")).read_bytes())
    write_bytes_once(output/"raw/hotpot_dev_distractor_v1.json", raw_file.read_bytes())
    write_bytes_once(output/"hotpot_evaluate_v1.reference.py", reference.read_bytes())
    write_bytes_once(output/"configure_aws_stop.py", (ROOT/"scripts/configure_aws_stop.py").read_bytes())
    write_bytes_once(output/"run_aws_diagnosis_study.py", (ROOT/"scripts/run_aws_diagnosis_study.py").read_bytes())
    launch = {"schema": 1, "code_sha256": code_sha, "remote_session": str(remote_session), "storage": str(storage),
        "main_protocol_sha256": fingerprint(main_protocol), "pilot_protocol_sha256": fingerprint(pilot_protocol),
        "instance_id": "i-03b33e00c47b11be6", "account_id": "692430448570", "region": "eu-west-2",
        "instance_type": "g7e.2xlarge", "gross_cap_usd": 25.0, "total_allocation_usd": 119.0,
        "allocation_reserve_usd": 20.0, "session_overhead_reserve_usd": 1.4,
        "maximum_minutes_from_ec2_start": 240, "maximum_hourly_usd": 5.84531,
        "live_access_and_billing_check_required": True, "gpu_validation_status": "not_run"}
    write_once(output/"launch.json", launch)
    checks = {k: launch[k] for k in ["instance_id", "account_id", "region", "instance_type"]}
    checks.update({k: None for k in ["observed_utc", "instance_start_utc", "deadline_utc",
        "external_stop_deadline_utc", "credit_expires_utc", "gpu_hourly_usd", "gross_spent_usd",
        "eligible_credit_remaining_usd", "state", "instance_initiated_shutdown_behavior"]})
    checks.update({k: False for k in ["persistent_storage_confirmed", "external_stop_verified",
                                    "credit_ec2_eligibility_verified"]})
    write_once(output/"live-checks.template.json", checks)
    session_arg = shlex.quote(str(remote_session))
    driver_arg = shlex.quote(str(remote_session/"run_aws_diagnosis_study.py"))
    checks_arg = shlex.quote(str(remote_session/"live-checks.json"))
    instructions = f"""# Diagnosis/replay follow-up package

Prepared inputs only. No GPU validation or new study outcomes are included.
The frozen main cohort contains 250 new questions, excludes 799 documented
previous questions, and is split into five batches of 50. The development
check imports five known failed traces. Both use restart, uncertainty with
two-step backtracking, and same-model diagnosis/replay; three seeds; and
0.5x, 1x, and 2x original generated-token allowances. New-step allowance is
eight. The diagnosis policy is an adaptation, not a Doctor-RAG reproduction.

## Verify without inference

From this package directory, using Python's standard library:

```bash
python run_aws_diagnosis_study.py --session . --plan-only
```

The manifest covers every prepared input. Do not edit frozen files. Changed
code, policy or cohort needs a new package directory and run tag. Keep any
execution records separate from these original inputs.

## Execute on the retained instance

Upload the complete package contents to `{remote_session}`. The cached model
and existing GPU environment must be available under `{storage}`. The
controller expects one GPU, the pinned model revision and vLLM 0.19.0.

Create `live-checks.json` by copying `live-checks.template.json`. Replace all
null fields with actual AWS observations and set a verification flag true
only after its check succeeds. All timestamp values need explicit timezones.
Observe the resource identity, running state, persistent storage and Stop
shutdown behavior; record current EC2-eligible credits and expiry, current
hourly rate, and cumulative gross allocation usage including unbilled usage.
Record an independent external stop and its deadline. `observed_utc` must
be within ten minutes of launch. `instance_start_utc` is the actual start of
this EC2 billing session, including setup and upload time.

The package limits this session to $25 gross, preserves $20 of the original
$119 allocation, allows at most 240 minutes from instance start and caps
the hourly compute input at $5.84531. That ceiling is a frozen guard, not a
current price quote or a provider billing cap. The external stop deadline
must be no later than `deadline_utc`. The template deliberately cannot pass
the live checks without measured values.

After the launch scope and live values are confirmed, on the instance run:

```bash
/workspace/jupyter-env/bin/python {driver_arg} --session {session_arg} --checks {checks_arg}
```

The controller performs GPU preflight, the development check and its audit,
then all five main batches and the pooled audit. It archives completed and
interrupted batches and schedules a guest shutdown when it exits. A failed
development audit prevents the main cohort from starting. An interrupted
cohort remains incomplete; do not report a smaller confirmatory study.

## Preserve and analyze the outputs

Download this entire session directory, including every `.tar.gz` and
`.tar.gz.sha256`, `live-checks-*.json`, `execution-window.json`, logs, batch
status/audit files, and `pilot/` and `main/` pooled outputs. Verify downloaded
archive hashes before treating retrieval as complete. Independently confirm
EC2 reaches stopped state. Preserve the retained volume and prior results.

`main/pooled-analysis.json` reports six primary contrasts in one Holm family,
twelve exploratory EM/F1 contrasts, nine policy/allowance summaries and
unique repair plus diagnosis usage. Diagnosis cost is charged in full to
each independently evaluated policy attempt and counted once per physical
diagnosis in study totals. Token counts are not a runtime or FLOPs claim.

For local re-auditing, extract `code.zip` into a separate source directory,
extract each main batch archive into its own directory, and use the frozen
`scripts/analyze_diagnosis_followup.py` with `--protocol main/protocol.json`,
`--bundle code.zip`, `--reference hotpot_evaluate_v1.reference.py`, `--output`
and `--runs` followed by all five extracted `run/` directories. Pass absolute
paths when executing outside this package. The analysis requires the full
cohort and the matching frozen source.
"""
    write_bytes_once(output/"README.md", instructions.encode())
    files = {str(p.relative_to(output)): file_fingerprint(p) for p in sorted(output.rglob("*"))
             if p.is_file() and p.name != "package-manifest.json"}
    write_once(output/"package-manifest.json", {"sha256": fingerprint(files), "files": files})
    return {"output": str(output), "code_sha256": code_sha, "main_protocol_sha256": fingerprint(main_protocol),
            "new_questions": len(ids), "excluded_questions": len(explored), "eligible_questions": eligible,
            "pilot_questions": len(pilot_ids), "gpu_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-session", type=Path, default=SESSION)
    parser.add_argument("--storage", type=Path, default=STORAGE)
    parser.add_argument("--run-tag", default="aws119-diagnosis250-20260913-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.remote_session, args.storage, args.run_tag), indent=2))


if __name__ == "__main__":
    main()
