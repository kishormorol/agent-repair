"""MuSiQue (Multihop Questions via Single-hop Composition) environment.

Task: 2-4 hop compositional QA designed to defeat shortcut reasoning.

Data source: https://github.com/StonyBrookNLP/musique or HuggingFace.

The environment uses the same search/lookup/finish tools as HotpotQA.
MuSiQue provides paragraphs (some supporting, some distractors) with
each question, making it compatible with our offline retrieval setup.
"""
from __future__ import annotations

import collections
import difflib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_env import BaseEnv, ToolResult, score_answer  # noqa: F401
from ..utils.io import load_json, save_json


# =========================================================================== #
# Data loading
# =========================================================================== #
def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load MuSiQue dataset (pre-processed JSON)."""
    return load_json(path)


def download_musique(dst: str) -> int:
    """Download MuSiQue-Ans dev set from HuggingFace.

    Reshapes to HotpotQA-compatible format:
    _id, question, answer, context, supporting_facts, type, level.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load("drt/musique", split="validation", trust_remote_code=True)

    recs = []
    for ex in ds:
        paragraphs = ex.get("paragraphs", [])

        # Build context and supporting_facts in HotpotQA format
        context = []
        sf = []

        if isinstance(paragraphs, dict):
            # HF format: {title: [...], paragraph_text: [...], is_supporting: [...]}
            titles = paragraphs.get("title", [])
            texts = paragraphs.get("paragraph_text", [])
            is_supp = paragraphs.get("is_supporting", [])
            for i, (title, text, supp) in enumerate(zip(titles, texts, is_supp)):
                sentences = [s.strip() + "." for s in text.split(".")
                             if s.strip()]
                if not sentences:
                    sentences = [text]
                context.append([title, sentences])
                if supp:
                    sf.append([title, 0])
        elif isinstance(paragraphs, list):
            for p in paragraphs:
                title = p.get("title", f"Paragraph {len(context)}")
                text = p.get("paragraph_text", "")
                sentences = [s.strip() + "." for s in text.split(".")
                             if s.strip()]
                if not sentences:
                    sentences = [text]
                context.append([title, sentences])
                if p.get("is_supporting", False):
                    sf.append([title, 0])

        # Determine hop count from decomposition if available
        decomp = ex.get("question_decomposition", [])
        if isinstance(decomp, dict):
            n_hops = len(decomp.get("question", []))
        elif isinstance(decomp, list):
            n_hops = len(decomp)
        else:
            n_hops = 2

        recs.append({
            "_id": ex.get("id", str(len(recs))),
            "question": ex["question"],
            "answer": ex["answer"],
            "type": f"{n_hops}hop",
            "level": f"{n_hops}hop",
            "context": context,
            "supporting_facts": sf,
        })

    save_json(recs, dst)
    return len(recs)


def sample_pool(data: List[Dict[str, Any]], size: int,
                stratify_by: List[str], seed: int) -> List[Dict[str, Any]]:
    """Stratified sample balanced across hop counts."""
    rng = random.Random(seed)
    if size >= len(data):
        pool = list(data)
        rng.shuffle(pool)
        return pool

    buckets: Dict[tuple, List] = collections.defaultdict(list)
    for rec in data:
        key = tuple(rec.get(f, "NA") for f in stratify_by)
        buckets[key].append(rec)

    total = len(data)
    chosen = []
    for key, recs in buckets.items():
        rng.shuffle(recs)
        n = max(1, round(size * len(recs) / total))
        chosen.extend(recs[:n])

    rng.shuffle(chosen)
    return chosen[:size]


# =========================================================================== #
# Environment (identical tools to HotpotQA)
# =========================================================================== #
@dataclass
class MuSiQueEnv(BaseEnv):
    """Per-question environment for MuSiQue compositional QA."""
    record: Dict[str, Any] = field(default_factory=dict)
    max_sentences_return: int = 5

    _titles: List[str] = field(default_factory=list)
    _para_by_title: Dict[str, List[str]] = field(default_factory=dict)
    _current_title: Optional[str] = None
    _lookup_pos: int = 0
    _lookup_keyword: Optional[str] = None
    retrieved_titles: List[str] = field(default_factory=list)

    def __post_init__(self):
        for title, sentences in self.record.get("context", []):
            self._titles.append(title)
            self._para_by_title[title] = sentences

    @property
    def question(self) -> str:
        return self.record["question"]

    @property
    def gold_answer(self) -> str:
        return self.record["answer"]

    @property
    def gold_titles(self) -> List[str]:
        return sorted({t for t, _ in self.record.get("supporting_facts", [])})

    @property
    def qid(self) -> str:
        return self.record["_id"]

    def _best_title(self, query: str) -> Optional[str]:
        q = query.strip().lower()
        for t in self._titles:
            if t.lower() == q:
                return t
        for t in self._titles:
            if q in t.lower() or t.lower() in q:
                return t
        match = difflib.get_close_matches(query, self._titles, n=1, cutoff=0.3)
        return match[0] if match else None

    def search(self, entity: str) -> ToolResult:
        title = self._best_title(entity)
        if title is None:
            return ToolResult(
                observation=f"Could not find any page matching '{entity}'. "
                            f"Try a different search term.",
                retrieved_title=None)
        self._current_title = title
        self._lookup_pos = 0
        self._lookup_keyword = None
        self.retrieved_titles.append(title)
        summary = " ".join(self._para_by_title[title][:self.max_sentences_return])
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
        return ToolResult(
            observation=f"No (more) sentences mentioning '{keyword}' "
                        f"in [{self._current_title}].",
            retrieved_title=self._current_title)

    def finish(self, answer: str) -> ToolResult:
        return ToolResult(observation=f"Episode finished with answer: {answer}",
                          finished=True, answer=answer)

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
