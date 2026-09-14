import copy
import datetime as dt
import json
from pathlib import Path

import pytest

from src.env.extension import ExtensionEnv, convert_2wiki, convert_musique, score_record
from src.llm.vllm_client import GenerationResult, TokenInfo
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import DIAGNOSIS_SETTINGS, TREATMENT, DIAGNOSIS_STRATEGY
from scripts.prepare_extension_study import HINT, MODELS, STRATEGIES
from scripts.run_extension_study import run_model, trial_rows
from scripts.analyze_extension_study import audit_cell, analyze


class ExtensionClient:
    model_name = "/snapshots/" + MODELS["qwen32b"]["revision"]
    calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        if "regeneration origin" in messages[0]["content"]:
            text = '{"origin": 1, "reason": "Retry after the search."}'
        elif kwargs.get("temperature") == .7:
            text = "Thought: Answer.\nAction: finish\nAction Input: Paris"
        elif "Observation:" in messages[-1]["content"]:
            text = "Thought: Answer.\nAction: finish\nAction Input: Rome"
        else:
            text = "Thought: Search.\nAction: search\nAction Input: Paris"
        return [GenerationResult(text, [1], [TokenInfo(1, "token", -1., {1: -1., 2: -2.})], prompt_tokens=100)]


@pytest.fixture
def extension_run(tmp_path):
    package, output = tmp_path/"package", tmp_path/"results"
    (package/"data").mkdir(parents=True)
    (package/"references").mkdir()
    records = [{"_id": q, "question": "Where?", "answer": "Paris", "context": [["Paris", ["Paris is a city."]]],
                "supporting_facts": []} for q in ["dev", "q1", "q2"]]
    (package/"data/hotpotqa.json").write_text(json.dumps(records))
    source = Path(__file__).resolve().parents[1]/"src/env/hotpot_env.py"
    reference = source.read_text().replace("def exact_match(", "def exact_match_score(")
    # The production audit reads the original official scorer, whose F1
    # function returns an (F1, precision, recall) tuple.
    reference = reference.replace("def f1_score(", "def scalar_f1(")
    reference += '\ndef f1_score(prediction, ground_truth):\n    return scalar_f1(prediction, ground_truth), 0., 0.\n'
    # Include the scalar implementation inside the required F1 function so
    # the official-function extractor can remain deliberately narrow.
    import inspect
    from src.env.hotpot_env import f1_score
    reference = reference[:reference.index('\ndef f1_score(prediction, ground_truth):')]
    scalar = inspect.getsource(f1_score)
    scalar = scalar.replace("return 0.0", "return (0.0, 0.0, 0.0)")
    scalar = scalar.replace("return 2 * precision * recall / (precision + recall)", "return (2 * precision * recall / (precision + recall), precision, recall)")
    reference += "\n" + scalar
    reference = reference.replace("collections.Counter", "Counter")
    (package/"references/hotpotqa.py").write_text(reference)
    protocol = {"run_id": "test-extension", "models": {"qwen32b": MODELS["qwen32b"]},
        "cohorts": {"hotpotqa": {"main_ids": ["q1", "q2"], "development_ids": ["dev"], "excluded_ids": ["old"]}},
        "max_new_steps": 8, "max_tokens_per_step": 512, "max_model_len": 16384, "initial_seed": 42,
        "logprobs_topk": 20, "repair_temperature": .7, "retry_hint": HINT, "strategies": STRATEGIES,
        "seeds": [0, 1, 2], "generated_token_multiplier": 1., "diagnosis_settings": DIAGNOSIS_SETTINGS,
        "primary_controls": ["full_restart", DIAGNOSIS_STRATEGY], "bootstrap_iters": 50, "analysis_seed": 4,
        "runtime": {"model": "qwen32b", "dataset": "hotpotqa", "n_failures_max": 1,
                    "strategies": ["full_restart", TREATMENT, DIAGNOSIS_STRATEGY],
                    "deadlines_s": [5., 10., 20.], "primary_deadline_s": 10., "measurement": "synthetic test"}}
    (package/"protocol.json").write_text(json.dumps({"sha256": fingerprint(protocol), "payload": protocol}))
    files = {str(p.relative_to(package)): file_fingerprint(p) for p in package.rglob("*") if p.is_file()}
    (package/"package-manifest.json").write_text(json.dumps({"sha256": fingerprint(files), "files": files}))
    client = ExtensionClient()
    deadline = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat()
    run_model(package, output, "qwen32b", tmp_path/"cache", deadline, client=client)
    return package, output, client, deadline


def test_extension_executes_audits_and_resumes(extension_run, tmp_path):
    package, output, client, deadline = extension_run
    audit, frame = audit_cell(package, output, "qwen32b", "hotpotqa")
    assert audit["main_questions"] == 2 and audit["main_failures"] == 2
    assert audit["trial_rows"] == 2*7*3 + 1*3*3
    assert audit["mismatches"] == 0 and audit["prefix_steps"] > 0
    assert frame[frame["mode"] == "runtime"].execution_id.nunique() == 9
    assert frame[frame["mode"] == "token"].execution_id.nunique() < 42
    before = client.calls
    run_model(package, output, "qwen32b", tmp_path/"cache", deadline, client=client)
    assert client.calls == before + 3  # warmup only; completed trials stay cached
    report = analyze(package, output, tmp_path/"analysis.json")
    assert report["complete"] and len(report["runtime_comparisons"]) == 6


@pytest.mark.parametrize("defect", ["observation", "missing_trial", "selection_cost", "budget", "score", "input"])
def test_extension_audit_rejects_altered_or_incomplete_records(extension_run, defect):
    package, output, _, _ = extension_run
    cell = output/"qwen32b/hotpotqa"
    if defect == "input":
        (package/"data/hotpotqa.json").write_text("[]")
    elif defect == "missing_trial":
        next((cell/"token/trials").glob("*.json")).unlink()
    else:
        folder = "token/trials" if defect == "selection_cost" else "token/executions"
        path = next((cell/folder).glob("*.json"))
        value = json.loads(path.read_text())
        payload = value["payload"]
        if defect == "observation": payload["steps"][-1]["observation"] = "Fabricated observation"
        elif defect == "selection_cost": payload[0]["selection_gen_tokens"] = 1000
        elif defect == "budget": payload["meta"]["recovery_gen_tokens"] = 100000
        elif defect == "score": payload["success"] = False
        value["sha256"] = fingerprint(payload)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit_cell(package, output, "qwen32b", "hotpotqa")


def test_musique_preserves_paragraphs_repeated_titles_and_aliases():
    raw = {"id": "q", "question": "Who?", "answer": "John Smith", "answer_aliases": ["J. Smith"],
           "answerable": True, "question_decomposition": [{}, {}], "paragraphs": [
               {"idx": 0, "title": "Person", "paragraph_text": "Dr. John lived in the U.S.  Really!", "is_supporting": False},
               {"idx": 1, "title": "Person", "paragraph_text": "His name was J. Smith.", "is_supporting": True}]}
    original = copy.deepcopy(raw)
    record = convert_musique(raw)
    env = ExtensionEnv(record)
    assert record["context"] == [["Person", [p["paragraph_text"] for p in raw["paragraphs"]]]]
    assert record["supporting_facts"] == [["Person", 1]]
    assert "U.S.  Really! His name" in env.step("search", "Person").observation
    assert env.step("lookup", "Smith").observation == "[Person] His name was J. Smith."
    assert env.score_answer("J. Smith", "John Smith")["em"] == 1
    assert raw == original


def test_2wiki_aliases_demonyms_and_boolean_scoring():
    raw = {"_id": "q", "answer": "United States", "answer_id": "Q30", "context": [["A", ["A. B."]]]}
    record = convert_2wiki(raw, {"Q30": {"aliases": ["USA"], "demonyms": ["American"]}})
    assert score_record("American", record)["em"] == 1
    assert score_record("USA", record)["f1"] == 1
    assert record["context"] == raw["context"]
    assert "answer_aliases" not in raw
    assert score_record("yes indeed", {**record, "answer": "yes", "answer_aliases": []})["f1"] == 0
    assert score_record("", {**record, "answer": "", "answer_aliases": []}) == {"em": 1., "f1": 0., "correct": True}
    assert score_record("", {"answer": "", "scoring": "musique"})["f1"] == 1
    assert score_record(None, record)["correct"] is False


def test_unanswerable_musique_is_not_silently_included():
    with pytest.raises(ValueError, match="answerable"):
        convert_musique({"answerable": False})


@pytest.mark.parametrize("defect", [None, "expired", "rate", "credit", "stop", "duration", "identity"])
def test_extension_cloud_budget_and_independent_stop_are_enforced(defect):
    from scripts.run_aws_extension_study import validate_live_checks
    now = dt.datetime.now(dt.timezone.utc)
    protocol = {"allocation": {"maximum_minutes": 330, "overhead_allowance_usd": 2.5,
                "maximum_hourly_usd": 5.84531, "session_gross_ceiling_usd": 35.,
                "prior_gross_conservative_bound_usd": 45., "total_usd": 119., "reserve_usd": 20.}}
    checks = {"observed_utc": now.isoformat(), "instance_start_utc": (now-dt.timedelta(minutes=3)).isoformat(),
        "deadline_utc": (now+dt.timedelta(minutes=320)).isoformat(),
        "external_stop_deadline_utc": (now+dt.timedelta(minutes=319)).isoformat(),
        "credit_expires_utc": (now+dt.timedelta(days=100)).isoformat(),
        "gpu_hourly_usd": 5.84531, "prior_gross_conservative_bound_usd": 45., "eligible_credit_conservative_bound_usd": 50.,
        "account_id": "692430448570", "instance_id": "i-03b33e00c47b11be6", "region": "eu-west-2", "instance_type": "g7e.2xlarge",
        "state": "running", "instance_initiated_shutdown_behavior": "stop", "persistent_storage_confirmed": True,
        "external_stop_verified": True, "credit_ec2_eligibility_verified": True}
    if defect == "expired": checks["observed_utc"] = (now-dt.timedelta(hours=1)).isoformat()
    elif defect == "rate": checks["gpu_hourly_usd"] = 6.
    elif defect == "credit": checks["eligible_credit_conservative_bound_usd"] = 5.
    elif defect == "stop": checks["external_stop_verified"] = False
    elif defect == "duration": checks["deadline_utc"] = (now+dt.timedelta(hours=10)).isoformat()
    elif defect == "identity": checks["instance_id"] = "wrong-instance"
    if defect is None:
        assert validate_live_checks(protocol, checks, now) > now
    else:
        with pytest.raises(ValueError):
            validate_live_checks(protocol, checks, now)
