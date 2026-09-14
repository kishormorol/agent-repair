import copy
import hashlib
import io
import json
import tarfile

import pandas as pd
import pytest

from scripts.analyze_study_diagnostics import verify_archive_inputs
from src.analysis.review_diagnostics import summarize_diagnostics


TREATMENT = "unc__perplexity__argmax__bt2"
STRATEGIES = ["full_restart", "random_step__bt2", "fixed_early", TREATMENT,
              "position_matched_random", "unc__perplexity__argmax"]


@pytest.fixture
def diagnostic_inputs():
    study = {"question_ids": ["q0", "q1", "s0", "s1"], "strategy": TREATMENT,
             "strategies": STRATEGIES, "primary_comparisons": ["full_restart", "random_step__bt2",
                 "position_matched_random", "unc__perplexity__argmax"], "seeds": [0, 1, 2], "multiplier": 1}
    originals = {}
    for q in study["question_ids"]:
        accepted = q.startswith("s")
        originals[q] = {"qid": q, "question": "Which answer?", "gold_answer": "Alpha Beta" if q == "s1" else "Gamma",
            "final_answer": ("Alpha" if q == "s1" else "Gamma") if accepted else None,
            "success": accepted, "em": int(q == "s0"), "f1": 2/3 if q == "s1" else float(q == "s0"),
            "total_gen_tokens": 80, "steps": [{"index": i} for i in range(4)],
            "terminated_reason": "finished" if accepted else "max_steps"}
    uncertainty = {q: {"qid": q, "steps": [{"index": i, "uncertainty": {"perplexity": 1 + int(i == 3)}}
                                           for i in range(4)]} for q in ["q0", "q1"]}
    rows = []
    for q in ["q0", "q1"]:
        for strategy in STRATEGIES:
            for seed in study["seeds"]:
                origin = 0
                if q == "q1":
                    origin = 1 if strategy in [TREATMENT, "fixed_early"] else 0
                    if strategy == "unc__perplexity__argmax":
                        origin = 3
                    if strategy == "position_matched_random":
                        origin = int(seed > 0)
                success = int(q == "q1" and origin == 1 and seed == 0)
                rows.append({"qid": q, "strategy": strategy, "seed": seed,
                    "execution_id": f"{q}-{seed}-{origin}", "prompt_sha256": f"{q}-{origin}",
                    "target_step": origin, "n_prefix_steps": origin, "success": success,
                    "em": success, "f1": float(success), "final_answer": "Gamma" if success else None,
                    "terminated_reason": "finished" if success else "budget",
                    "budget": 80, "original_gen_tokens": 80, "new_step_allowance": 8,
                    "recovery_gen_tokens": 20 if success else 80, "recovery_prompt_tokens": 100 + 20 * origin,
                    "recovery_model_requests": 2, "recovery_tool_calls": 1})
    return study, pd.DataFrame(rows), originals, uncertainty


def test_overlap_decomposition_uses_questions_and_keeps_subgroup_weight(diagnostic_inputs):
    result = summarize_diagnostics(*diagnostic_inputs, iters=100)
    comparison = result["execution_overlap"]["full_restart"]
    assert comparison["same_execution_seed_rows"] == 3
    assert comparison["different_execution_seed_rows"] == 3
    assert comparison["questions_with_any_different_execution"] == 1
    assert (comparison["question_mean_wins"], comparison["question_mean_losses"], comparison["question_mean_ties"]) == (1, 0, 1)
    assert comparison["same_execution_contribution"] == 0
    assert comparison["different_execution_contribution"] == pytest.approx(1/6)
    active = result["nonzero_treatment_origin"]
    assert active["n_questions"] == 1
    assert active["paired_success_versus_restart"]["delta"] == pytest.approx(1/3)
    assert active["question_fraction"] * active["paired_success_versus_restart"]["delta"] == pytest.approx(1/6)


def test_policy_costs_charge_shared_work_but_study_total_deduplicates(diagnostic_inputs):
    study, frame, originals, uncertainty = diagnostic_inputs
    result = summarize_diagnostics(study, frame, originals, uncertainty, iters=100)
    policies = result["policy_accounting"]
    assert policies["full_restart"]["mean_cost_per_attempt"]["recovery_gen_tokens"] == 80
    assert policies[TREATMENT]["mean_cost_per_attempt"]["recovery_gen_tokens"] == 70
    assert sum(p["policy_trial_totals"]["recovery_gen_tokens"] for p in policies.values()) == frame.recovery_gen_tokens.sum()
    assert result["unique_study_usage"]["recovery_gen_tokens"] == frame.drop_duplicates("execution_id").recovery_gen_tokens.sum()
    assert result["unique_study_usage"]["recovery_gen_tokens"] < frame.recovery_gen_tokens.sum()
    assert policies[TREATMENT]["termination_counts"] == {"finished": 1, "budget": 5, "max_steps": 0}
    assert result["execution_overlap"]["full_restart"]["paired_cost_differences"]["recovery_gen_tokens"]["mean"] == -10
    assert result["descriptive_fixed_early"]["success"]["delta"] == 0


def test_reference_gate_keeps_accepted_non_exact_answers_unrepaired(diagnostic_inputs):
    result = summarize_diagnostics(*diagnostic_inputs, iters=100)
    assert result["cohort"]["accepted_without_exact_match"] == 1
    whole = result["policy_accounting"][TREATMENT]["reference_gated_full_cohort"]
    assert whole["success"]["n_questions"] == 4
    assert whole["success"]["mean"] == pytest.approx((2 + 1/3)/4)
    assert whole["em"]["mean"] == pytest.approx((1 + 1/3)/4)
    assert whole["f1"]["mean"] == pytest.approx((1 + 2/3 + 1/3)/4)
    assert len(result["illustrative_cases"][0]["rows"]) == 6
    assert result["analysis_status"] == "exploratory_review_stage"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "negative_cost", "shared_outcome", "gate", "uncertainty"])
def test_diagnostics_reject_invalid_evidence(diagnostic_inputs, defect):
    study, frame, originals, uncertainty = copy.deepcopy(diagnostic_inputs)
    if defect == "missing":
        frame = frame.iloc[1:]
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "negative_cost":
        frame.loc[0, "recovery_prompt_tokens"] = -1
    elif defect == "shared_outcome":
        frame.loc[0, "success"] = 1
    elif defect == "gate":
        originals["s1"]["em"] = 1
    else:
        uncertainty.pop("q0")
    with pytest.raises(ValueError):
        summarize_diagnostics(study, frame, originals, uncertainty, iters=100)


def test_zero_token_cap_is_counted_and_utilization_is_not_invented(diagnostic_inputs):
    study, frame, originals, uncertainty = copy.deepcopy(diagnostic_inputs)
    originals["q0"]["total_gen_tokens"] = 0
    frame.loc[frame.qid.eq("q0"), ["budget", "original_gen_tokens", "recovery_gen_tokens"]] = 0
    result = summarize_diagnostics(study, frame, originals, uncertainty, iters=100)
    restart = result["policy_accounting"]["full_restart"]
    assert restart["zero_token_caps"] == 3
    assert restart["token_cap_utilization"]["n"] == 3
    json.dumps(result, allow_nan=False)


def test_archive_input_verification_detects_mutation_and_missing_files(tmp_path):
    payload = b'{"frozen": true}\n'
    local = tmp_path / "input.json"
    local.write_bytes(payload)
    archive = tmp_path / "batch.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("run/input.json")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert verify_archive_inputs(archive, digest, {"run/input.json": local}) == {
        "run/input.json": hashlib.sha256(payload).hexdigest()}
    with pytest.raises(ValueError, match="omits"):
        verify_archive_inputs(archive, digest, {"run/absent.json": local})
    local.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        verify_archive_inputs(archive, digest, {"run/input.json": local})
    with pytest.raises(ValueError, match="checksum"):
        verify_archive_inputs(archive, "0" * 64, {"run/input.json": local})
