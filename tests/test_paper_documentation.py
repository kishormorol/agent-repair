import csv
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("HotpotQA", "MuSiQue", "2WikiMHQA")


def aggregate_rows():
    with (ROOT / "results/cross_dataset/cross_dataset_traj_stats.csv").open() as stream:
        return {row["dataset"]: row for row in csv.DictReader(stream)}


@pytest.mark.parametrize("key,label", [
    ("oracle", "Judge-targeted"),
    ("oracle_bt2", "Judge-targeted + backtrack 2"),
    ("full_restart", "Full restart"),
    ("best_unc", "Retrospective best uncertainty variant"),
])
def test_readme_rates_match_archived_mean_seed_source(key, label):
    rows = aggregate_rows()
    values = [f"{100 * float(rows[dataset][key]):.1f}" for dataset in DATASETS]
    expected = "| " + " | ".join([label, *values]) + " |"
    assert expected in (ROOT / "README.md").read_text()


def test_readme_failure_counts_match_source():
    rows = aggregate_rows()
    counts = [rows[dataset]["n_failed"] for dataset in DATASETS]
    readme = (ROOT / "README.md").read_text()
    assert "| Failed questions / 500 | " + " | ".join(counts) + " |" in readme
    assert f"There are {sum(map(int, counts))} distinct failed QA questions" in readme


@pytest.mark.parametrize("relative_path", [
    "README.md", "RUN_LOCAL.md", "paper_title_abstract.md",
    "docs/iclr2027_readiness.md", "docs/iclr2027_study_protocol.md",
    "docs/reviewer_report.md", "docs/experiments_to_run.md",
    "CLOUD_GPU_SETUP.md", "START_HERE.md",
    "docs/paper_section_review.md", "docs/aws_quota_appeal.md",
    "docs/iclr_expert_review_2026-09-13.md",
])
def test_documentation_relative_links_exist(relative_path):
    path = ROOT / relative_path
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        destination = target.split("#", 1)[0]
        assert (path.parent / destination).exists(), f"{relative_path}: {target}"


def test_standalone_abstract_matches_manuscript_and_generated_numbers():
    manuscript = (ROOT / "paper/iclr2027.tex").read_text()
    abstract = manuscript.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    macros = dict(re.findall(
        r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}",
        (ROOT / "paper/generated/iclr2027_numbers.tex").read_text(),
    ))
    # The abstract uses only these numeric macros and escaped percent signs.
    for name, value in macros.items():
        abstract = abstract.replace("\\" + name + "{}", value)
        abstract = abstract.replace("\\" + name + "\\%", value + "%")
    abstract = abstract.replace(r"\%", "%")
    assert "\\" not in abstract, "Handle any new TeX syntax before comparing prose"
    standalone = (ROOT / "paper_title_abstract.md").read_text()
    standalone = standalone.split("## Abstract\n", 1)[1].split("\n## ", 1)[0]
    assert " ".join(standalone.split()) == " ".join(abstract.split())


def test_active_paper_records_completed_main_study_and_deferred_scope():
    from scripts.build_iclr_notebook import notebook

    manuscript = (ROOT / "paper/iclr2027.tex").read_text()
    nb = notebook("a" * 64)
    parameters = next(cell.source for cell in nb.cells if "parameters" in cell.metadata.get("tags", []))
    assert "ORIGIN_SWEEP = False" in parameters and "INCLUDE_DIAGNOSTICS = False" in parameters
    assert "MULTIPLIERS = [1.0]" in parameters
    assert r"\section{Archived Results}" in manuscript
    assert r"\section{Controlled Repair Results}" in manuscript
    assert "Working draft: audited HotpotQA study" in manuscript
    assert "It has not been executed" not in manuscript
    assert "No new GPU experiments are reported" not in manuscript
    assert r"US\$119" in manuscript and r"$18N$" in manuscript
    assert "TOTAL_BUDGET_USD = 119.0" in parameters
    assert "A nonsignificant" in manuscript and "not evidence of equivalence" in manuscript
    assert "with the original full" in manuscript and "pool comparison unavailable" in manuscript
    assert "diagnosis/replay comparator" in manuscript


def test_manuscript_citations_resolve_and_include_closest_recent_work():
    manuscript = (ROOT / "paper/iclr2027.tex").read_text()
    bibliography = (ROOT / "paper/references.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys)), "Duplicate bibliography keys"
    cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript)
             for key in group.split(",")}
    assert cited <= set(keys), f"Missing bibliography entries: {cited - set(keys)}"
    assert {"jiao2026doctor", "luan2026symtrace", "zhuang2026agentrewind",
            "pothuru2026failures", "barke2026agentrx", "xinjie2025reagent"} <= cited
    labels = set(re.findall(r"\\label\{([^}]+)\}", manuscript))
    assert set(re.findall(r"\\ref\{([^}]+)\}", manuscript)) <= labels


def test_documented_matrix_filters_select_a_catalog_entry():
    config = yaml.safe_load((ROOT / "config/config_experiment.yaml").read_text())
    models = yaml.safe_load((ROOT / "config/models.yaml").read_text())
    for relative_path in ("README.md", "RUN_LOCAL.md"):
        text = (ROOT / relative_path).read_text()
        command = re.search(r"--model ([\w.-]+) --dataset ([\w.-]+) --dry-run", text)
        assert command is not None
        model, dataset = command.groups()
        assert any(model in models[tier] for tier in config["experiment"]["tiers"])
        assert dataset in config["experiment"]["datasets"]


def test_pilot_profile_activates_controlled_paths_without_privileged_hints(tmp_path):
    from src.repair.controlled import configured_strategies, budget_multipliers
    from src.repair.strategies import parse_strategy
    from src.utils import load_config

    cfg = load_config(str(ROOT / "config/config_iclr_pilot.yaml"), base_override=str(tmp_path))
    assert cfg.dataset.split == "development"
    assert cfg.dataset.pool_size == 30
    assert cfg.repair.step_budget_mode == "new"
    assert cfg.repair.match_restart_hint and cfg.repair.origin_sweep
    assert budget_multipliers(cfg.raw["repair"]) == [0.5, 1.0]
    strategies = configured_strategies(cfg.raw["repair"])
    assert {"full_restart", "random_step", "fixed_early"} <= set(strategies)
    assert all(not parse_strategy(s)["informed"] for s in strategies)
    assert set(cfg.uncertainty.metrics) == {"token_entropy", "perplexity", "max_token_prob"}
    for key in ("data_raw", "data_processed", "trajectories", "repairs", "logs"):
        assert "hotpotqa/qwen32b/pilot-v1/" in cfg.path(key)
