import json
from pathlib import Path

import pytest
import pandas as pd

from scripts.analyze_diagnosis_followup import audit_batch, summarize
from src.llm.vllm_client import GenerationResult, TokenInfo
from src.repair.controlled import fingerprint
from src.repair.diagnosis import DIAGNOSIS_PROMPT, DIAGNOSIS_STRATEGY
from src.utils import load_json, save_json
from test_diagnosis_replay import followup_run


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/"output/aws-experiment/2026-09-11-development/hotpot_evaluate_v1.reference.py"


def test_followup_audit_replays_every_execution_and_reproduces_family(followup_run):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    frame, originals, audit = audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)
    assert audit["unique_repairs"] == 36 and audit["unique_diagnoses"] == 2
    assert audit["trajectories_replayed_and_scored"] == 38 and audit["mismatches"] == 0
    assert audit["diagnosis_generated_tokens"] == 10
    result = summarize(load_json(args.protocol)["payload"], frame, originals)
    assert result["primary_family_size"] == len(result["primary_contrasts"]) == 6
    assert len(result["policy_accounting"]) == 9
    assert all(p["delta"] == 0 and p["p_value_holm"] == 1 for p in result["primary_contrasts"].values())
    assert result["policy_accounting"][DIAGNOSIS_STRATEGY+"@1"]["mean_per_attempt"]["selection_gen_tokens"] == 5
    usage = result["unique_study_usage"]
    assert usage["unique_diagnoses"] == 2
    assert usage["diagnosis_gen_tokens"] == 10
    assert usage["diagnosis_prompt_tokens"] == 60
    assert usage["diagnosis_model_requests"] == 2
    assert usage["repair_tool_calls"] == int(frame.drop_duplicates("execution_id").recovery_tool_calls.sum())


def test_followup_audit_counts_finish_as_request_but_not_retrieval(followup_run, monkeypatch):
    runner, cfg, args, log, _, client = followup_run
    original_chat = client.chat

    def finish_chat(messages, **kwargs):
        if messages[0]["content"] == DIAGNOSIS_PROMPT:
            return original_chat(messages, **kwargs)
        return [GenerationResult("Thought: Answer.\nAction: finish\nAction Input: Paris",
                                 [1], [TokenInfo(1, "x", -1.0)], 10)]

    monkeypatch.setattr(client, "chat", finish_chat)
    runner.run(cfg, log, args)
    frame, _, audit = audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)
    assert audit["mismatches"] == 0
    assert frame.terminated_reason.eq("finished").all()
    assert frame.recovery_model_requests.eq(1).all()
    assert frame.recovery_tool_calls.eq(0).all()
    assert frame.target_step.gt(0).any()  # Retained searches are still counted separately.


def test_followup_audit_rejects_semantic_replay_corruption(followup_run):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    executions = Path(cfg.path("repairs"))/"diagnosis-followup/executions"
    path = next(executions.glob("*.json"))
    saved = load_json(path)
    saved["trajectory"]["steps"][-1]["observation"] = "Fabricated evidence"
    # Even a self-consistent file checksum cannot substitute for replay verification.
    saved["trajectory_sha256"] = fingerprint(saved["trajectory"])
    save_json(saved, path)
    with pytest.raises(ValueError, match="replay mismatch|prefix changed"):
        audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)


def test_followup_audit_rejects_omitted_row(followup_run):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    path = Path(cfg.path("repairs"))/"diagnosis-followup/results.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:])+"\n")
    with pytest.raises(ValueError, match="trial coverage"):
        audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)


@pytest.mark.parametrize("field", ["recovery_tool_calls", "recovery_model_requests"])
def test_followup_audit_rejects_costs_inconsistent_with_replayed_steps(followup_run, field):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    out = Path(cfg.path("repairs"))/"diagnosis-followup"
    path = next((out/"executions").glob("*.json"))
    saved = load_json(path)
    saved["trajectory"]["meta"][field] += 1
    saved["trajectory_sha256"] = fingerprint(saved["trajectory"])
    save_json(saved, path)
    rows = [json.loads(line) for line in (out/"results.jsonl").read_text().splitlines()]
    for row in rows:
        if row["execution_id"] == path.stem:
            row[field] += 1
            if field == "recovery_model_requests":
                row["incremental_model_requests"] += 1
    (out/"results.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="request/tool accounting"):
        audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)


def test_followup_audit_rejects_answer_without_finish_action(followup_run):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    path = next((Path(cfg.path("repairs"))/"diagnosis-followup/executions").glob("*.json"))
    saved = load_json(path)
    assert all(s["action"] != "finish" for s in saved["trajectory"]["steps"])
    saved["trajectory"]["final_answer"] = saved["trajectory"]["gold_answer"]
    saved["trajectory"].update(em=1.0, f1=1.0, success=True)
    saved["trajectory_sha256"] = fingerprint(saved["trajectory"])
    save_json(saved, path)
    with pytest.raises(ValueError, match="final answer"):
        audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)


@pytest.mark.parametrize("corruption", ["missing_question", "duplicate_row", "shared_cost", "diagnosis_cost"])
def test_followup_summary_requires_complete_consistent_policy_trials(followup_run, corruption):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    frame, originals, _ = audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)
    if corruption == "missing_question":
        frame = frame[frame.qid.ne("q0")]
    elif corruption == "duplicate_row":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif corruption == "shared_cost":
        frame.loc[frame.index[0], "recovery_prompt_tokens"] += 1
    else:
        index = frame.index[frame.strategy.eq(DIAGNOSIS_STRATEGY)][0]
        frame.loc[index, "selection_gen_tokens"] += 1
        frame.loc[index, "incremental_gen_tokens"] += 1
    with pytest.raises(ValueError, match="trial coverage|Shared execution|diagnosis|cost accounting"):
        summarize(load_json(args.protocol)["payload"], frame, originals)


def test_no_initial_failures_is_not_reported_as_a_null_comparison():
    protocol = {"question_ids": ["q1"]}
    traces = {"q1": {"success": True, "em": 1.0, "f1": 1.0, "total_gen_tokens": 10}}
    result = summarize(protocol, pd.DataFrame(), traces)
    assert result["n_failed_questions"] == 0 and result["primary_contrasts"] == {}
    assert "not estimable" in result["status"]


def test_followup_pools_distinct_frozen_batch_ids(followup_run):
    runner, cfg, args, log, _, _ = followup_run
    runner.run(cfg, log, args)
    frame, originals, _ = audit_batch(Path(cfg.base), args.protocol, args.bundle, REFERENCE)
    protocol = load_json(args.protocol)["payload"]
    expected = summarize(protocol, frame, originals)["primary_contrasts"]
    frame.loc[frame.qid.eq("q1"), "run_id"] = "followup-batch-two"
    protocol["batches"] = [{"run_id": "followup-test", "ids": ["q0"]},
                           {"run_id": "followup-batch-two", "ids": ["q1"]}]
    assert summarize(protocol, frame, originals)["primary_contrasts"] == expected
    frame.loc[frame.qid.eq("q1"), "run_id"] = "undeclared-batch"
    with pytest.raises(ValueError, match="batch identity"):
        summarize(protocol, frame, originals)
