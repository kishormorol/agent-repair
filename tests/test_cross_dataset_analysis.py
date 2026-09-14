import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.metrics import ensemble_rows
from src.repair.strategies import parse_strategy


@pytest.mark.parametrize("precomputed_ensemble", [False, True])
def test_notebook_compares_same_seed_mean_for_every_category(precomputed_ensemble):
    strategy = "unc__perplexity__argmax"
    rows = []
    for name in ["full_restart", "random_step", "oracle_targeted", strategy]:
        for seed in range(3):
            rows.append({
                "qid": "q1", "strategy": name, "seed": seed,
                "success": int(seed == 0), "recovery_gen_tokens": 10,
                "recovery_tool_calls": 1, "recovery_latency_s": 0.1,
                "targeted_oracle_match": 0,
            })
    df = pd.DataFrame(rows)
    if precomputed_ensemble:
        df = pd.concat([df, ensemble_rows(df, [strategy])], ignore_index=True)
    path = Path(__file__).resolve().parents[1] / "run_cross_dataset_analysis.ipynb"
    notebook = json.loads(path.read_text())
    source = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                  if "# Key strategies to compare" in "".join(cell.get("source", [])))
    namespace = {
        "pd": pd, "all_results": {"HotpotQA": df},
        "parse_strategy": parse_strategy, "ensemble_rows": ensemble_rows,
        "display": lambda value: None,
    }
    exec(compile(source, str(path), "exec"), namespace)
    rates = namespace["cross"].set_index("strategy")["fix_rate"]
    for name in ("full_restart", "random_step", "oracle_targeted", "best_unc", "ensemble"):
        assert rates[name] == pytest.approx(1 / 3)
