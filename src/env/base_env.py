"""Abstract base environment for multi-step ReAct agents.

All dataset environments (HotpotQA, FEVER, 2WikiMultiHopQA, MuSiQue) implement
this interface so the pipeline stages, batch runner, and evaluation are
dataset-agnostic.
"""
from __future__ import annotations

import collections
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    observation: str
    is_tool_call: bool = True
    finished: bool = False
    answer: Optional[str] = None
    retrieved_title: Optional[str] = None


class BaseEnv(ABC):
    """Per-question environment with search/lookup/finish tools."""

    @property
    @abstractmethod
    def qid(self) -> str: ...

    @property
    @abstractmethod
    def question(self) -> str: ...

    @property
    @abstractmethod
    def gold_answer(self) -> str: ...

    @property
    @abstractmethod
    def gold_titles(self) -> List[str]: ...

    @abstractmethod
    def step(self, action: str, action_input: str) -> ToolResult: ...


# =========================================================================== #
# Shared answer normalization + EM / F1 (HotpotQA / SQuAD style)
# =========================================================================== #
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace."""
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: Optional[str], ground_truth: str) -> Dict[str, float]:
    """Return {'em':0/1, 'f1':float, 'correct':bool}."""
    if prediction is None:
        return {"em": 0.0, "f1": 0.0, "correct": False}
    em = exact_match(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)
    return {"em": float(em), "f1": f1, "correct": bool(em == 1 or f1 >= 0.5)}
