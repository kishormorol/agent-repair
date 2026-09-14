import random

import pytest

from src.repair.controlled import fingerprint
from src.repair.position_matched import (
    fit_position_profile, sample_position, validate_position_profile,
)
from src.repair.strategies import make_rng, select_target_step


@pytest.fixture
def profile():
    originals = [{"qid": "a", "success": False, "steps": [{"index": i} for i in range(5)]},
                 {"qid": "b", "success": False, "steps": [{"index": i} for i in range(3)]}]
    uncertainties = {
        "a": {"qid": "a", "steps": [{"index": i, "uncertainty": {"perplexity": i + 1}} for i in range(5)]},
        "b": {"qid": "b", "steps": [{"index": i, "uncertainty": {"perplexity": 3 - i}} for i in range(3)]},
    }
    return fit_position_profile(originals, uncertainties, source_ids=["a", "b", "successful"],
        dataset="hotpotqa", strategy="unc__perplexity__argmax__bt2", source_phase="development",
        source_run_id="development-fixture",
        model={"repo_id": "Qwen/Qwen2.5-32B-Instruct-AWQ", "revision": "a" * 40}, source_sha256={})


def test_profile_fits_one_origin_per_question_before_repair_outcomes(profile):
    assert profile["payload"]["origins"] == [
        {"qid": "a", "origin": 2, "n_steps": 5}, {"qid": "b", "origin": 0, "n_steps": 3}]
    assert validate_position_profile(profile, dataset="hotpotqa",
        strategy="unc__perplexity__argmax__bt2", excluded_ids=["a", "b", "successful", "older"])


def test_normalized_position_mapping_is_bounded_and_rounds_half_up(profile):
    class First:
        def choice(self, entries):
            return entries[0]
    assert [sample_position(profile, n, First()) for n in (1, 2, 3, 5, 9)] == [0, 1, 1, 2, 4]
    for n in range(1, 30):
        assert 0 <= sample_position(profile, n, random.Random(n)) < n


def test_position_draw_is_reproducible_and_does_not_use_test_uncertainty(profile):
    name = "position_matched_random"
    for seed in range(5):
        first = select_target_step(name, 8, None, None, make_rng("test", name, seed),
                                    position_profile=profile)
        assert first == select_target_step(name, 8, None, {"unrelated": "test scores"},
            make_rng("test", name, seed), position_profile=profile)


@pytest.mark.parametrize("field,value", [("source_phase", "test"), ("origins", []),
    ("origins", [{"qid": "a", "origin": 5, "n_steps": 5}]),
    ("origins", [{"qid": "a", "origin": -1, "n_steps": 5}])])
def test_invalid_position_profile_is_rejected_even_with_matching_checksum(profile, field, value):
    profile["payload"][field] = value
    profile["sha256"] = fingerprint(profile["payload"])
    with pytest.raises(ValueError):
        validate_position_profile(profile)


def test_profile_rejects_tampering_wrong_dataset_and_missing_exclusions(profile):
    with pytest.raises(ValueError, match="dataset"):
        validate_position_profile(profile, dataset="musique")
    with pytest.raises(ValueError, match="policy"):
        validate_position_profile(profile, strategy="unc__perplexity__argmax")
    with pytest.raises(ValueError, match="every"):
        validate_position_profile(profile, excluded_ids=["a", "b"])
    profile["payload"]["origins"][0]["origin"] = 1
    with pytest.raises(ValueError, match="checksum"):
        validate_position_profile(profile)


def test_position_control_requires_profile_and_rejects_second_backtrack(profile):
    with pytest.raises(ValueError, match="profile"):
        select_target_step("position_matched_random", 5, None, None, random.Random(0))
    with pytest.raises(ValueError, match="already"):
        select_target_step("position_matched_random__bt2", 5, None, None,
                           random.Random(0), position_profile=profile)


def test_profile_cannot_be_fitted_on_test_questions():
    with pytest.raises(ValueError, match="development"):
        fit_position_profile([], {}, source_ids=[], dataset="hotpotqa",
            strategy="unc__perplexity__argmax__bt2", source_phase="test", source_run_id="bad",
            model={}, source_sha256={})


def test_no_backtracking_control_retains_same_uncertainty_peak():
    uncertainty = {"steps": [{"index": i, "uncertainty": {"perplexity": v}}
                             for i, v in enumerate([1, 2, 3, 9, 2])]}
    assert select_target_step("unc__perplexity__argmax", 5, None, uncertainty, random.Random(0)) == 3
    assert select_target_step("unc__perplexity__argmax__bt2", 5, None, uncertainty, random.Random(0)) == 1
