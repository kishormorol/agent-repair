import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.react_agent import ReActAgent
from src.env.hotpot_env import HotpotEnv
from src.llm.vllm_client import GenerationResult, TokenInfo
from src.repair.diagnosis import (
    DIAGNOSIS_STRATEGY, diagnosis_messages, parse_diagnosis,
    DIAGNOSIS_PROMPT, DIAGNOSIS_SETTINGS, TREATMENT,
    diagnose_original, plan_diagnosis_repairs,
)
from src.repair.controlled import fingerprint
from src.utils import load_config, save_json, save_item, read_jsonl
from test_reviewer_fixes import RECORD, ROOT, SearchClient


@pytest.fixture
def original():
    return ReActAgent(SearchClient(), max_steps=3).run(HotpotEnv(record=RECORD)).to_dict()


def test_diagnosis_input_excludes_reference_fields_and_uncertainty(original):
    original.update(gold_answer="ANSWER_CANARY", gold_titles=["TITLE_CANARY"],
                    reference_annotation="ANNOTATION_CANARY", f1=0.12345678)
    original["steps"][0]["gold_evidence"] = "EVIDENCE_CANARY"
    original["steps"][0]["generation"] = {"secret": "LOGPROB_CANARY"}
    messages = diagnosis_messages(original)
    text = json.dumps(messages)
    for word in ["ANSWER_CANARY", "TITLE_CANARY", "ANNOTATION_CANARY", "EVIDENCE_CANARY", "LOGPROB_CANARY", "0.12345678"]:
        assert word not in text
    task = json.loads(messages[1]["content"])
    assert task["question"] == RECORD["question"]
    assert task["steps"][0]["observation"] == original["steps"][0]["observation"]
    assert set(task["steps"][0]) == {"index", "thought", "action", "action_input", "observation"}


@pytest.mark.parametrize("response", [
    '{"origin":1,"reason":"Revisit the evidence."}',
    '```json\n{"origin":1}\n```',
])
def test_valid_diagnosis(response):
    result = parse_diagnosis(response, 3)
    assert result["target_step"] == 1 and not result["fallback"]


@pytest.mark.parametrize("response", [
    'origin: 1', '{"origin":true}', '{"origin":1.0}', '{"origin":-1}',
    '{"origin":3}', '{"origin":"1"}', '{}', '[1]',
    '{"origin":1} {"origin":2}', '{"origin":1,"reason":[]}',
])
def test_invalid_diagnosis_has_recorded_restart_fallback(response):
    result = parse_diagnosis(response, 3)
    assert result["target_step"] == 0 and result["fallback"] and result["fallback_reason"]


class DiagnosisClient:
    model_name = "test-model"

    def __init__(self, text='{"origin":1,"reason":"Revisit evidence."}', prompt_tokens=30):
        self.calls = []
        self.text = text
        self.prompt_tokens = prompt_tokens

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return [GenerationResult(self.text, list(range(5)),
                 [TokenInfo(i, "x", -1) for i in range(5)], self.prompt_tokens)]


def test_diagnosis_cache_preserves_one_request_and_rejects_changed_input(original, tmp_path):
    client = DiagnosisClient()
    target = tmp_path / 'diagnosis.json'
    first = diagnose_original(client, original, target, model_signature={"name": "test-model"})
    second = diagnose_original(client, original, target, model_signature={"name": "test-model"})
    assert first == second and len(client.calls) == 1
    assert first["selection_gen_tokens"] == 5
    assert first["selection_prompt_tokens"] == 30
    assert first["selection_model_requests"] == 1
    assert client.calls[0][1] == {"temperature": 0.0, "max_tokens": 512, "n": 1, "seed": 20260913}
    original["steps"][0]["thought"] = "Changed trace"
    with pytest.raises(ValueError, match="input"):
        diagnose_original(client, original, target, model_signature={"name": "test-model"})
    assert len(client.calls) == 1


def test_bad_cache_hash_and_missing_token_accounting_fail(original, tmp_path):
    client = DiagnosisClient()
    target = tmp_path / 'diagnosis.json'
    diagnose_original(client, original, target, model_signature={"name": "test-model"})
    saved = json.loads(target.read_text())
    saved["payload"]["target_step"] = 2
    target.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum"):
        diagnose_original(client, original, target, model_signature={"name": "test-model"})
    with pytest.raises(ValueError, match="prompt-token"):
        diagnose_original(DiagnosisClient(prompt_tokens=None), original, tmp_path/'missing.json',
                          model_signature={"name": "test-model"})


def test_selected_origin_reuses_execution_without_leaking_diagnosis_hint(original, tmp_path):
    cfg = load_config(str(ROOT/'config/config_colab_hotpotqa.yaml'), base_override=str(tmp_path))
    cfg.raw["repair"].update(step_budget_mode="new", match_restart_hint=True)
    uncertainty = {"qid": original["qid"], "steps": [
        {"index": i, "uncertainty": {"perplexity": i + 1}} for i in range(3)]}
    diagnosis = diagnose_original(DiagnosisClient('{"origin":0,"reason":"HINT_CANARY"}'),
        original, tmp_path/'diagnosis.json', model_signature={"name": "test-model"})
    before = copy.deepcopy(cfg.raw)
    jobs = plan_diagnosis_repairs(cfg.raw, RECORD, original, uncertainty, diagnosis, 0, 2)
    assert cfg.raw == before
    assert len(jobs) == 1 and jobs[0]["meta"]["target_step"] == 0
    assert set(jobs[0]["strategies"]) == {"full_restart", "unc__perplexity__argmax__bt2", DIAGNOSIS_STRATEGY}
    assert jobs[0]["token_budget"] == 2 * original["total_gen_tokens"]
    assert "HINT_CANARY" not in json.dumps(jobs)
    assert jobs[0]["policy_costs"][DIAGNOSIS_STRATEGY]["selection_gen_tokens"] == 5
    assert jobs[0]["policy_costs"]["full_restart"]["selection_model_requests"] == 0
    diagnosis["target_step"] = 1
    distinct = plan_diagnosis_repairs(cfg.raw, RECORD, original, uncertainty, diagnosis, 0, 2)
    assert len(distinct) == 2
    chosen = next(job for job in distinct if DIAGNOSIS_STRATEGY in job["strategies"])
    assert len(chosen["prefix_steps"]) == 1 and chosen["strategies"] == [DIAGNOSIS_STRATEGY]


class FollowupClient(DiagnosisClient, SearchClient):
    def chat(self, messages, **kwargs):
        if messages[0]["content"] == DIAGNOSIS_PROMPT:
            return DiagnosisClient.chat(self, messages, **kwargs)
        result = SearchClient.chat(self, messages, **kwargs)[0]
        result.prompt_tokens = 10
        return [result]


@pytest.fixture
def followup_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    runner = importlib.import_module("scripts.run_diagnosis_followup")
    cfg = load_config(str(ROOT/"config/config_colab_hotpotqa.yaml"), base_override=str(tmp_path))
    cfg.raw["repair"].update(seeds=[0, 1, 2], step_budget_mode="new", match_restart_hint=True,
                             run_id="followup-test")
    cfg.raw["repair"]["budget"]["multipliers"] = [0.5, 1.0, 2.0]
    cfg.raw["repair"]["nudge"].update(enabled=True, temperature=0.7)
    cfg.raw["runtime"].update(gen_batch_size=2, repair_batch_size=2)
    cfg.raw["paths"]["local_base"] = str(tmp_path)
    cfg.raw["notebook_provenance"] = {"model": {"revision": "test-revision", "repo_id": "test-model"}}
    records = [dict(RECORD, _id=f"q{i}") for i in range(2)]
    ids = [r["_id"] for r in records]
    save_json(records, Path(cfg.path("data_processed"))/"pool.json")
    save_json(ids, Path(cfg.path("data_processed"))/"failed_ids.json")
    save_json({"name": cfg.models.agent.name, "dtype": cfg.models.agent.dtype},
              Path(cfg.path("data_processed"))/"agent_model.json")
    for record in records:
        original = ReActAgent(SearchClient(), max_steps=3).run(HotpotEnv(record=record)).to_dict()
        save_item(cfg.path("trajectories"), record["_id"], original)
        save_item(cfg.path("uncertainty"), record["_id"], {"qid": record["_id"], "steps": [
            {"index": i, "uncertainty": {"perplexity": i+1}} for i in range(3)]})
    protocol = {"strategies": runner.STRATEGIES, "seeds": [0, 1, 2], "multipliers": [0.5, 1.0, 2.0],
                "max_new_steps": 8, "max_tokens_per_step": 512, "dataset": "hotpotqa",
                "diagnosis_prompt_sha256": fingerprint(DIAGNOSIS_PROMPT),
                "diagnosis_settings": DIAGNOSIS_SETTINGS, "question_ids": ids,
                "explored_ids": ["historical"], "model_revision": "test-revision", "model_id": "test-model",
                "batch_size": 2, "retry_hint": cfg.raw["repair"]["nudge"]["retry_hint"],
                "batches": [{"run_id": "followup-test", "ids": ids, "configuration_sha256": fingerprint(cfg.raw)}],
                "official_scorer_sha256": "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0",
                "max_repair_executions_per_batch": 54}
    from scripts.build_iclr_notebook import build_bundle
    bundle = tmp_path/"code.zip"
    protocol["code_sha256"] = build_bundle(ROOT, bundle)
    path = tmp_path/"protocol.json"
    save_json({"sha256": fingerprint(protocol), "payload": protocol}, path)
    loads = []
    client = FollowupClient()
    client.model_name = cfg.models.agent.name

    def load(*args):
        loads.append(True)
        return client

    monkeypatch.setattr(runner, "load_agent", load)
    return runner, cfg, SimpleNamespace(protocol=str(path), bundle=str(bundle), limit=None, dry_run=False, dataset=None), \
        SimpleNamespace(info=lambda *args: None), loads, client


def test_followup_costs_complete_coverage_and_interrupted_resume(followup_run):
    runner, cfg, args, log, loads, client = followup_run
    result = runner.run(cfg, log, args)
    assert result["strategy_rows"] == 54 and result["planned_executions"] == 36
    assert result["new_diagnoses"] == 2 and result["new_executions"] == 36
    assert len(loads) == 1 and len(client.calls) == 2
    path = Path(cfg.path("repairs"))/"diagnosis-followup/results.jsonl"
    rows = list(read_jsonl(path))
    assert all(row["recovery_gen_tokens"] <= row["budget"] for row in rows)
    assert all(row["new_step_allowance"] == 8 for row in rows)
    for row in rows:
        expected = 5 if row["strategy"] == DIAGNOSIS_STRATEGY else 0
        assert row["selection_gen_tokens"] == expected
        assert row["incremental_gen_tokens"] == row["recovery_gen_tokens"] + expected
    # Diagnosis is charged fully to every standalone policy attempt, across all seeds/caps.
    assert sum(r["selection_gen_tokens"] for r in rows) == 90
    # Interrupt fan-out after persisted executions; resume without any model reload/call.
    path.write_text(json.dumps(rows[0])+"\n")
    resumed = runner.run(cfg, log, args)
    assert resumed["new_executions"] == resumed["new_diagnoses"] == 0
    assert len(list(read_jsonl(path))) == 54 and len(loads) == 1 and len(client.calls) == 2
    rows = list(read_jsonl(path))
    rows[0]["success"] = 1
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError, match="differs from its cached"):
        runner.run(cfg, log, args)


def test_followup_dry_run_and_changed_inputs(followup_run):
    runner, cfg, args, log, loads, client = followup_run
    args.dry_run = True
    assert runner.run(cfg, log, args)["maximum_repair_executions"] == 54
    assert not loads and not client.calls
    assert not (Path(cfg.path("repairs"))/"diagnosis-followup/manifest.json").exists()
    args.dry_run = False
    runner.run(cfg, log, args)
    cfg.raw["repair"]["nudge"]["retry_hint"] = "changed"
    with pytest.raises(ValueError, match="frozen follow-up"):
        runner.run(cfg, log, args)


def test_followup_rejects_omitted_failure_before_model_loading(followup_run):
    runner, cfg, args, log, loads, _ = followup_run
    save_json(["q0"], Path(cfg.path("data_processed"))/"failed_ids.json")
    with pytest.raises(ValueError, match="failure cohort"):
        runner.run(cfg, log, args)
    assert not loads
