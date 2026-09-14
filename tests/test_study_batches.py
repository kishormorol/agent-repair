import json

import pytest

from scripts.analyze_study_batches import analyze, load_study_batches
from src.repair.controlled import fingerprint
from src.utils.io import save_json, load_json


@pytest.fixture
def study_batches(tmp_path):
    study = {"dataset": "hotpotqa", "strategy": "a", "strategies": ["a", "b"],
             "primary_comparisons": ["b"], "seeds": [0, 1, 2], "multiplier": 1,
             "code_sha256": "a" * 64, "n_initial_questions": 4,
             "question_ids": ["q0", "s0", "q1", "s1"], "explored_ids": ["old"], "batches": []}
    paths = []
    for index in range(2):
        run_id = f"batch{index}"
        run = tmp_path / run_id
        ids = [f"q{index}", f"s{index}"]
        policy = {"run_id": run_id, "ids": ids}
        digest = fingerprint(policy)
        save_json(policy, run / "study_policy.json")
        save_json(ids, run / "test_ids.json")
        save_json([{"_id": q} for q in ids], run / "data/processed/pool.json")
        save_json(ids[:1], run / "data/processed/failed_ids.json")
        for q in ids:
            save_json({"qid": q, "success": q.startswith("s")}, run / "outputs/trajectories" / (q + ".json"))
        payload = {"configuration": {"dataset": {"name": "hotpotqa", "split": "test"},
            "repair": {"run_id": run_id, "strategies": ["a", "b"], "seeds": [0, 1, 2]},
            "notebook_provenance": {"phase": "test", "code_sha256": "a" * 64,
                "study_policy_sha256": digest}}, "question_ids": ids[:1]}
        manifest_digest = fingerprint(payload)
        path = run / "outputs/repairs/results.jsonl"
        save_json({"sha256": manifest_digest, "payload": payload}, path.with_name("manifest.json"))
        rows = [{"qid": ids[0], "dataset": "hotpotqa", "run_id": run_id, "strategy": name,
                 "seed": seed, "multiplier": 1, "success": int(name == "a"), "em": int(name == "a"),
                 "f1": float(name == "a"), "manifest_sha256": manifest_digest}
                for name in ("a", "b") for seed in range(3)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(path)
        study["batches"].append({"run_id": run_id, "ids": ids, "policy_sha256": digest})
    frozen = tmp_path / "study.json"
    save_json({"sha256": fingerprint(study), "payload": study}, frozen)
    return paths, frozen


def test_batches_pool_questions_in_one_dataset(study_batches, tmp_path):
    paths, study = study_batches
    _, frame = load_study_batches(paths, study)
    assert len(frame) == 12 and frame.dataset.nunique() == 1
    assert frame.run_id.nunique() == 1 and frame.batch_run_id.nunique() == 2
    result = analyze(paths, study, tmp_path / "analysis.json", iters=100)
    assert result["n_initial_questions"] == 4 and result["n_failed_questions"] == 2
    assert result["primary_contrasts"]["b"]["delta"] == 1


def test_pooled_analysis_requires_every_batch_once(study_batches):
    paths, study = study_batches
    with pytest.raises(ValueError, match="All frozen"):
        load_study_batches(paths[:1], study)
    with pytest.raises(ValueError, match="duplicated"):
        load_study_batches([paths[0], paths[0]], study)


def test_batch_policy_and_full_initial_coverage_are_checked(study_batches):
    paths, study = study_batches
    run = paths[0].parents[2]
    save_json(["q0"], run / "test_ids.json")
    with pytest.raises(ValueError, match="initial-question"):
        load_study_batches(paths, study)
    save_json(["q0", "s0"], run / "test_ids.json")
    save_json({"changed": True}, run / "study_policy.json")
    with pytest.raises(ValueError, match="configuration"):
        load_study_batches(paths, study)


def test_duplicate_rows_and_cross_batch_ids_are_rejected(study_batches):
    paths, frozen = study_batches
    original = paths[0].read_text()
    paths[0].write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_study_batches(paths, frozen)
    paths[0].write_text(original)
    study = load_json(frozen)["payload"]
    study["batches"][1]["ids"][0] = "q0"
    save_json({"sha256": fingerprint(study), "payload": study}, frozen)
    with pytest.raises(ValueError, match="unique"):
        load_study_batches(paths, frozen)
