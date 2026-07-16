"""Dataset environments: HotpotQA, FEVER, 2WikiMultiHopQA, MuSiQue.

All environments share the same tool interface (search/lookup/finish)
and data format (records with _id, question, answer, context, supporting_facts).
"""
from .base_env import (
    BaseEnv, ToolResult,
    normalize_answer, exact_match, f1_score, score_answer,
)
from .hotpot_env import HotpotEnv, load_dataset, sample_pool

# Dataset registry: maps dataset name -> (EnvClass, load_fn, sample_fn, download_fn, score_fn)
DATASET_REGISTRY = {}


def register_dataset(name, env_cls, load_fn, sample_fn, download_fn, score_fn=None):
    DATASET_REGISTRY[name] = {
        "env_cls": env_cls,
        "load": load_fn,
        "sample": sample_fn,
        "download": download_fn,
        "score": score_fn or score_answer,
    }


def get_dataset(name: str) -> dict:
    """Get the dataset module (env class, loaders, scorer) by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


# Register HotpotQA (default)
from .hotpot_env import (
    HotpotEnv,
    load_dataset as hotpot_load,
    sample_pool as hotpot_sample,
)
register_dataset(
    "hotpotqa",
    env_cls=HotpotEnv,
    load_fn=hotpot_load,
    sample_fn=hotpot_sample,
    download_fn=None,  # handled in run_setup
    score_fn=score_answer,
)

# Register FEVER
from .fever_env import (
    FEVEREnv,
    load_dataset as fever_load,
    sample_pool as fever_sample,
    download_fever_with_evidence,
    score_answer_fever,
)
register_dataset(
    "fever",
    env_cls=FEVEREnv,
    load_fn=fever_load,
    sample_fn=fever_sample,
    download_fn=download_fever_with_evidence,
    score_fn=score_answer_fever,
)

# Register 2WikiMultiHopQA
from .wikimultihop_env import (
    WikiMultiHopEnv,
    load_dataset as wiki_load,
    sample_pool as wiki_sample,
    download_2wikimultihop,
)
register_dataset(
    "2wikimultihopqa",
    env_cls=WikiMultiHopEnv,
    load_fn=wiki_load,
    sample_fn=wiki_sample,
    download_fn=download_2wikimultihop,
    score_fn=score_answer,
)

# Register MuSiQue
from .musique_env import (
    MuSiQueEnv,
    load_dataset as musique_load,
    sample_pool as musique_sample,
    download_musique,
)
register_dataset(
    "musique",
    env_cls=MuSiQueEnv,
    load_fn=musique_load,
    sample_fn=musique_sample,
    download_fn=download_musique,
    score_fn=score_answer,
)


__all__ = [
    "BaseEnv", "ToolResult",
    "normalize_answer", "exact_match", "f1_score", "score_answer",
    "HotpotEnv", "FEVEREnv", "WikiMultiHopEnv", "MuSiQueEnv",
    "load_dataset", "sample_pool",
    "DATASET_REGISTRY", "get_dataset", "register_dataset",
]
