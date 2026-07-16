"""HotpotQA (distractor) environment: data loading, stratified sampling, the
agent's search/lookup/finish tools over each question's 10 local paragraphs,
and the official EM/F1 scorer.

Design choices (see experiment design):
  * Distractor setting -> the 10 candidate paragraphs ship with each question,
    so retrieval is fully offline and reproducible (no live Wikipedia).
  * `search(entity)`  -> returns the best title-matched paragraph's summary.
  * `lookup(keyword)` -> next sentence containing keyword in the current paragraph.
  * `finish(answer)`  -> terminates the episode.
  * Gold `supporting_facts` are retained for Stage-3 error annotation.
"""
from __future__ import annotations

import collections
import difflib
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.io import load_json


# =========================================================================== #
# Data loading & stratified sampling
# =========================================================================== #
def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the raw HotpotQA distractor JSON (a list of question records)."""
    return load_json(path)


def sample_pool(data: List[Dict[str, Any]], size: int,
                stratify_by: List[str], seed: int) -> List[Dict[str, Any]]:
    """Stratified sample of `size` questions, balanced across the given fields
    (e.g. ['level', 'type']). Falls back to random sampling if fields missing.
    """
    rng = random.Random(seed)
    if size >= len(data):
        pool = list(data)
        rng.shuffle(pool)
        return pool

    # Bucket by the composite stratification key.
    buckets: Dict[Tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    for rec in data:
        key = tuple(rec.get(f, "NA") for f in stratify_by)
        buckets[key].append(rec)

    # Proportional allocation per bucket.
    total = len(data)
    chosen: List[Dict[str, Any]] = []
    for key, recs in buckets.items():
        rng.shuffle(recs)
        n = max(1, round(size * len(recs) / total))
        chosen.extend(recs[:n])

    rng.shuffle(chosen)
    return chosen[:size]


# =========================================================================== #
# Answer normalization + official EM / F1 (HotpotQA / SQuAD style)
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
    """Return {'em':0/1, 'f1':float, 'correct':bool}. `correct` uses F1>=0.5 OR EM."""
    if prediction is None:
        return {"em": 0.0, "f1": 0.0, "correct": False}
    em = exact_match(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)
    return {"em": float(em), "f1": f1, "correct": bool(em == 1 or f1 >= 0.5)}


# =========================================================================== #
# The environment: tools over one question's local paragraphs
# =========================================================================== #
@dataclass
class ToolResult:
    observation: str
    is_tool_call: bool = True
    finished: bool = False
    answer: Optional[str] = None
    retrieved_title: Optional[str] = None   # for programmatic gold-retrieval check


@dataclass
class HotpotEnv:
    """Per-question environment. Construct with a HotpotQA record, then call
    `step(action, action_input)` for each ReAct action.
    """
    record: Dict[str, Any]
    max_sentences_return: int = 5

    # internal state
    _titles: List[str] = field(default_factory=list)
    _para_by_title: Dict[str, List[str]] = field(default_factory=dict)
    _current_title: Optional[str] = None
    _lookup_pos: int = 0
    _lookup_keyword: Optional[str] = None
    retrieved_titles: List[str] = field(default_factory=list)  # log for annotation

    def __post_init__(self):
        for title, sentences in self.record["context"]:
            self._titles.append(title)
            self._para_by_title[title] = sentences

    # --- accessors used by annotation / scoring ---------------------------- #
    @property
    def question(self) -> str:
        return self.record["question"]

    @property
    def gold_answer(self) -> str:
        return self.record["answer"]

    @property
    def gold_titles(self) -> List[str]:
        """Titles of the gold supporting-fact paragraphs (for retrieval check)."""
        return sorted({t for t, _ in self.record["supporting_facts"]})

    @property
    def qid(self) -> str:
        return self.record["_id"]

    # --- tools ------------------------------------------------------------- #
    def _best_title(self, query: str) -> Optional[str]:
        """Closest paragraph title to `query` (exact, then fuzzy)."""
        q = query.strip().lower()
        for t in self._titles:                      # exact (case-insensitive)
            if t.lower() == q:
                return t
        for t in self._titles:                      # substring containment
            if q in t.lower() or t.lower() in q:
                return t
        match = difflib.get_close_matches(query, self._titles, n=1, cutoff=0.3)
        return match[0] if match else None

    def search(self, entity: str) -> ToolResult:
        title = self._best_title(entity)
        if title is None:
            return ToolResult(observation=f"Could not find any page matching '{entity}'. "
                                          f"Try a different search term.",
                              retrieved_title=None)
        self._current_title = title
        self._lookup_pos = 0
        self._lookup_keyword = None
        self.retrieved_titles.append(title)
        summary = " ".join(self._para_by_title[title][: self.max_sentences_return])
        return ToolResult(observation=f"[{title}] {summary}", retrieved_title=title)

    def lookup(self, keyword: str) -> ToolResult:
        if self._current_title is None:
            return ToolResult(observation="No page is currently open. Use search first.")
        sentences = self._para_by_title[self._current_title]
        if self._lookup_keyword != keyword:
            self._lookup_keyword = keyword
            self._lookup_pos = 0
        kw = keyword.lower()
        while self._lookup_pos < len(sentences):
            sent = sentences[self._lookup_pos]
            self._lookup_pos += 1
            if kw in sent.lower():
                return ToolResult(observation=f"[{self._current_title}] {sent}",
                                  retrieved_title=self._current_title)
        return ToolResult(observation=f"No (more) sentences mentioning '{keyword}' "
                                      f"in [{self._current_title}].",
                          retrieved_title=self._current_title)

    def finish(self, answer: str) -> ToolResult:
        return ToolResult(observation=f"Episode finished with answer: {answer}",
                          finished=True, answer=answer)

    # --- dispatch ---------------------------------------------------------- #
    def step(self, action: str, action_input: str) -> ToolResult:
        action = (action or "").strip().lower()
        if action == "search":
            return self.search(action_input)
        if action == "lookup":
            return self.lookup(action_input)
        if action == "finish":
            return self.finish(action_input)
        return ToolResult(observation=f"Unknown action '{action}'. "
                                      f"Valid actions: search, lookup, finish.",
                          is_tool_call=False)
