import json
import shutil

import pandas as pd
import pytest

from scripts.build_followup_paper import BASE, INPUTS, LABELS, build_assets, load_followup


@pytest.fixture
def followup_evidence(tmp_path):
    for name in INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    return tmp_path


def test_followup_tables_use_all_audited_allowances_and_policy_costs(tmp_path):
    analysis = load_followup()
    build_assets(tmp_path, analysis)
    results = (tmp_path / "tables/iclr2027_diagnosis_results.tex").read_text()
    costs = (tmp_path / "tables/iclr2027_diagnosis_costs.tex").read_text()
    contrasts = (tmp_path / "tables/iclr2027_diagnosis_contrasts.tex").read_text()
    for multiplier in [.5, 1., 2.]:
        for strategy, label in LABELS.items():
            policy = analysis["policy_accounting"][f"{strategy}@{multiplier:g}"]
            mean = policy["mean_per_attempt"]
            prefix = f"${multiplier:g}\\times$ & {label} & "
            assert prefix + f"{round(mean['success']*354)}/354" in results
            assert prefix + f"{mean['incremental_gen_tokens']:.1f} & {mean['incremental_prompt_tokens']:.1f}" in costs
    assert "$1\\times$ & Restart & -0.56 & [-3.67, +2.54]" in contrasts
    assert "$0.5\\times$ & Diagnosis/replay & -1.41 & [-4.52, +1.41]" in contrasts


@pytest.mark.parametrize("defect,message", [
    ("protocol", "protocol identity"),
    ("audit", "audit is incomplete"),
    ("reproduction", "reproduction is incomplete"),
    ("family", "comparison family"),
    ("missing_trial", "trials differ"),
    ("policy_mean", "policy mean differs"),
])
def test_followup_tables_reject_incomplete_or_inconsistent_evidence(followup_evidence, defect, message):
    base = followup_evidence
    if defect == "missing_trial":
        path = base / "pooled-analysis-local.trials.csv"
        pd.read_csv(path, keep_default_na=False).iloc[:-1].to_csv(path, index=False)
    else:
        name = {"protocol": "prepared/main/protocol.json",
                "reproduction": "analysis-reproduction-check.json"}.get(defect, "pooled-analysis-local.json")
        path = base / name
        value = json.loads(path.read_text())
        if defect == "protocol":
            value["payload"]["question_ids"].reverse()
        elif defect == "audit":
            value["audits"][0]["mismatches"] = 1
        elif defect == "reproduction":
            value["matched"] = False
        elif defect == "family":
            del value["primary_contrasts"][next(iter(value["primary_contrasts"]))]
        elif defect == "policy_mean":
            # Matching two edited summaries must not bypass the measured trials.
            value["policy_accounting"]["diagnosis_replay@1"]["mean_per_attempt"]["incremental_prompt_tokens"] += 1
            (base / "retrieved/main/pooled-analysis.json").write_text(json.dumps(value))
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        load_followup(base)
