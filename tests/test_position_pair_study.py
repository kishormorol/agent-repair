import datetime as dt
import inspect
import json
from pathlib import Path
import re

import pytest

from src.llm.vllm_client import GenerationResult, TokenInfo
from src.repair.controlled import fingerprint, file_fingerprint
from src.repair.diagnosis import TREATMENT
from src.repair.position_matched import fit_position_profile
from src.repair.position_pairs import DEV_CONTROL, POLICIES, SWAP
from scripts.prepare_extension_study import HINT, MODELS
from scripts.run_position_pair_study import run, select_pair_origins
from scripts.analyze_position_pair_study import analyze, audit_cell

DATASET = "hotpotqa"
# Four length-five questions form two pairs; the other lengths stay unmatched.
QUESTIONS = {"q1": (5, 4), "q2": (5, 1), "q3": (5, 3), "q4": (5, 0), "q5": (4, 2), "q6": (3, 1)}


class PairClient:
    """Deterministic trajectories whose length and peak uncertainty are scripted."""
    model_name = "/snapshots/" + MODELS["qwen32b"]["revision"]

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "numbers one through ten" in prompt:
            return [GenerationResult("one", [1], [TokenInfo(1, "one", -1., {1: -1.})], prompt_tokens=10)]
        length, peak = (int(re.search(rf"{name}=(\d+)", prompt).group(1)) for name in ["steps", "peak"])
        index = prompt.count("Observation:")
        if kwargs.get("temperature"):
            # Repair: answer at once, with success varying by origin and seed.
            answer = "Paris" if (index + kwargs["seed"]) % 2 else "Berlin"
            text = f"Thought: Answer.\nAction: finish\nAction Input: {answer}"
        elif index < length - 1:
            text = "Thought: Search.\nAction: search\nAction Input: Paris"
        else:
            text = "Thought: Answer.\nAction: finish\nAction Input: Berlin"
        logprob = -2. if index == peak else -1.
        return [GenerationResult(text, [1], [TokenInfo(1, "token", logprob, {1: logprob, 2: -3.})], prompt_tokens=100)]


def write_reference(path):
    """Reproduce the official HotpotQA scorer, whose F1 returns a tuple."""
    from src.env.hotpot_env import f1_score
    source = Path(__file__).resolve().parents[1] / "src/env/hotpot_env.py"
    reference = source.read_text().replace("def exact_match(", "def exact_match_score(")
    reference = reference.replace("def f1_score(", "def scalar_f1(")
    reference += "\ndef f1_score(prediction, ground_truth):\n    return scalar_f1(prediction, ground_truth), 0., 0.\n"
    reference = reference[:reference.index("\ndef f1_score(prediction, ground_truth):")]
    scalar = inspect.getsource(f1_score)
    scalar = scalar.replace("return 0.0", "return (0.0, 0.0, 0.0)")
    scalar = scalar.replace("return 2 * precision * recall / (precision + recall)",
                            "return (2 * precision * recall / (precision + recall), precision, recall)")
    reference = (reference + "\n" + scalar).replace("collections.Counter", "Counter")
    path.write_text(reference)


def frozen_profile():
    originals = [{"qid": "old-a", "success": False, "steps": [{"index": i} for i in range(5)]},
                 {"qid": "old-b", "success": False, "steps": [{"index": i} for i in range(3)]}]
    uncertainties = {
        "old-a": {"qid": "old-a", "steps": [{"index": i, "uncertainty": {"perplexity": i + 1}} for i in range(5)]},
        "old-b": {"qid": "old-b", "steps": [{"index": i, "uncertainty": {"perplexity": 3 - i}} for i in range(3)]}}
    return fit_position_profile(originals, uncertainties, source_ids=["old-a", "old-b"], dataset=DATASET,
        strategy=TREATMENT, source_phase="development", source_run_id="position-pairs-test",
        model=MODELS["qwen32b"], source_sha256={})


@pytest.fixture
def prepared(tmp_path):
    package, output = tmp_path / "package", tmp_path / "results"
    (package / "data").mkdir(parents=True)
    (package / "references").mkdir()
    (package / "calibration").mkdir()
    records = [{"_id": qid, "question": f"Where is it? steps={length} peak={peak}", "answer": "Paris",
                "context": [["Paris", ["Paris is a city."]]], "supporting_facts": []}
               for qid, (length, peak) in QUESTIONS.items()]
    (package / "data" / f"{DATASET}.json").write_text(json.dumps(records))
    write_reference(package / "references" / f"{DATASET}.py")
    (package / "calibration" / f"{DATASET}.json").write_text(json.dumps({"profile": frozen_profile()}))
    protocol = {"schema": 1, "run_id": "position-pairs-test", "model_key": "qwen32b",
        "models": {"qwen32b": MODELS["qwen32b"]},
        "cohorts": {DATASET: {"main_ids": list(QUESTIONS), "excluded_ids": ["old-a", "old-b"]}},
        "strategies": POLICIES, "seeds": [0, 1, 2], "main_n_per_dataset": len(QUESTIONS),
        "initial_seed": 42, "initial_temperature": 0., "repair_temperature": .7,
        "max_new_steps": 8, "max_tokens_per_step": 512, "max_model_len": 16384, "logprobs_topk": 20,
        "retry_hint": HINT, "generated_token_multiplier": 1., "prefix_caching": False,
        "pairing_seed": 20260916, "analysis_seed": 20260916, "bootstrap_iters": 50,
        "primary_control": SWAP, "primary_family_size": 2, "primary_family": "two exact sign-flip tests",
        "resampling_unit": "two-question pair", "precision_note": "bounded diagnostic",
        "scope": "prospective origin-assignment diagnostic",
        "secondary_control": "Expanded-development random; descriptive contrasts only"}
    (package / "protocol.json").write_text(json.dumps({"sha256": fingerprint(protocol), "payload": protocol}))
    files = {str(p.relative_to(package)): file_fingerprint(p) for p in package.rglob("*") if p.is_file()}
    (package / "package-manifest.json").write_text(json.dumps({"sha256": fingerprint(files), "files": files}))
    client = PairClient()
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    run(package, output, tmp_path / "cache", deadline, client=client)
    return package, output, client, deadline


def test_study_pairs_only_exact_lengths_and_reports_the_remainder(prepared):
    package, output, _, _ = prepared
    audit, frame, pairing = audit_cell(package, output, DATASET)
    assert audit["main_questions"] == 6 and audit["main_failures"] == 6
    assert audit["pairs"] == 2 and audit["matched_questions"] == 4
    assert sorted(pairing["unmatched_ids"]) == ["q5", "q6"]
    for pair in pairing["pairs"]:
        assert {QUESTIONS[q][0] for q in pair["qids"]} == {5}
    assert audit["mismatches"] == 0 and audit["trial_rows"] == 4 * 3 * len(POLICIES)


def test_swapped_origins_are_exactly_balanced_within_every_pair(prepared):
    package, output, _, _ = prepared
    audit, frame, pairing = audit_cell(package, output, DATASET)
    assert audit["balance"]["raw_origin_distribution_equal"]
    assert audit["balance"]["paired_attempts"] == 12
    for pair in pairing["pairs"]:
        sub = frame[frame.pair_id == pair["pair_id"]]
        own = sorted(sub[sub.strategy == TREATMENT].origin)
        assert sorted(sub[sub.strategy == SWAP].origin) == own
        # Backtracking two steps from the scripted peak drives each origin.
        assert set(own) == {max(0, QUESTIONS[q][1] - 2) for q in pair["qids"]}


def test_equal_origins_share_one_execution_and_differing_ones_do_not(prepared):
    package, output, _, _ = prepared
    _, frame, _ = audit_cell(package, output, DATASET)
    for (qid, seed), sub in frame.groupby(["qid", "seed"]):
        by_origin = sub.groupby("origin").execution_id.nunique()
        assert (by_origin == 1).all()
        assert sub.execution_id.nunique() == sub.origin.nunique()
    # A shared execution must carry an identical prompt for both policies.
    for execution_id, sub in frame.groupby("execution_id"):
        assert sub.prompt_sha256.nunique() == 1 and sub.success.nunique() == 1


def test_swap_borrows_the_partner_choice_and_the_control_ignores_test_scores(prepared):
    package, output, _, _ = prepared
    _, _, pairing = audit_cell(package, output, DATASET)
    protocol = json.loads((prepared[0] / "protocol.json").read_text())["payload"]
    cell = prepared[1] / "qwen32b" / DATASET
    originals = {q: json.loads((cell / "main/originals" / f"{q}.json").read_text())["payload"] for q in QUESTIONS}
    uncertainties = {q: json.loads((cell / "main/uncertainty" / f"{q}.json").read_text())["payload"] for q in QUESTIONS}
    pair = pairing["pairs"][0]
    choices = select_pair_origins(protocol, pair, originals, uncertainties, frozen_profile(), seed=0)
    left, right = pair["qids"]
    assert {c["strategy"]: c["origin"] for c in choices[left]}[SWAP] == \
           {c["strategy"]: c["origin"] for c in choices[right]}[TREATMENT]
    control = {c["strategy"]: c["origin"] for c in choices[left]}[DEV_CONTROL]
    assert 0 <= control < pair["n_steps"]


def test_run_resumes_without_recomputing_completed_trials(prepared, tmp_path):
    package, output, client, deadline = prepared
    before = client.calls
    run(package, output, tmp_path / "cache", deadline, client=client)
    assert client.calls == before + 3  # warmup only


def test_analysis_holm_adjusts_a_two_test_family_and_keeps_the_control_descriptive(prepared, tmp_path):
    package, output, _, _ = prepared
    report = analyze(package, output, tmp_path / "analysis.json")
    assert report["complete"] and report["primary_family_size"] == 2
    assert len(report["primary_comparisons"]) == 1 and len(report["secondary_comparisons"]) == 1
    primary = report["primary_comparisons"][0]
    assert primary["strategy_b"] == SWAP and primary["n_pairs"] == 2 and primary["n_questions"] == 4
    assert primary["p_value_holm"] >= primary["p_value"]
    assert sum(primary[k] for k in ["positive_blocks", "negative_blocks", "zero_blocks"]) == 2
    secondary = report["secondary_comparisons"][0]
    assert secondary["strategy_b"] == DEV_CONTROL and secondary["p_value_holm"] is None
    assert tmp_path.joinpath("analysis.trials.csv").exists()


def test_analysis_rejects_a_tampered_swap_origin(prepared, tmp_path):
    package, output, _, _ = prepared
    _, frame, pairing = audit_cell(package, output, DATASET)
    trial = next((output / "qwen32b" / DATASET / "trials").glob("*.json"))
    saved = json.loads(trial.read_text())
    saved["payload"][0]["origin"] += 1
    trial.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="identity/checksum mismatch"):
        audit_cell(package, output, DATASET)
