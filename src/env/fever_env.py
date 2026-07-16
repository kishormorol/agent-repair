"""FEVER (Fact Extraction and VERification) environment.

Task: given a claim, search Wikipedia evidence and classify as
SUPPORTS / REFUTES / NOT ENOUGH INFO.

Data source: https://fever.ai or HuggingFace ``fever/fever``.

The environment provides the same search/lookup/finish tools as HotpotQA,
operating over evidence documents shipped with each claim.  The agent's
``finish`` action should output one of the three labels.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv, ToolResult, score_answer  # noqa: F401
from ..utils.io import load_json, save_json


# =========================================================================== #
# Data loading
# =========================================================================== #
FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load FEVER dataset (our pre-processed JSON with evidence paragraphs)."""
    return load_json(path)


def download_fever(dst: str) -> int:
    """Download FEVER dev set from HuggingFace and reshape to our format.

    Each record gets: _id, question (=claim), answer (=label), context
    (list of [title, [sentences]]), supporting_facts, type, level.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load("fever/fever", "v1.0", split="labelled_dev",
                 trust_remote_code=True)

    # Group evidence by claim id
    recs = []
    for ex in ds:
        claim_id = str(ex["id"])
        label = ex["label"]
        claim = ex["claim"]

        # FEVER evidence is minimal — we store what's available
        # In practice, evidence_wiki / evidence_sentence_id may be used
        evidence_wiki = ex.get("evidence_wiki", "") or ""
        evidence_id = ex.get("evidence_sentence_id", -1)

        # Build a context entry if we have evidence
        context = []
        sf = []
        if evidence_wiki:
            title = evidence_wiki.replace("_", " ")
            # We store the evidence as a single-sentence paragraph
            context.append([title, [claim]])  # placeholder
            sf.append([title, 0])

        recs.append({
            "_id": claim_id,
            "question": claim,
            "answer": label if label in FEVER_LABELS else "NOT ENOUGH INFO",
            "type": "verification",
            "level": "claim",
            "context": context,
            "supporting_facts": sf,
        })

    save_json(recs, dst)
    return len(recs)


def download_fever_with_evidence(dst: str) -> int:
    """Download FEVER from HuggingFace with pre-retrieved evidence paragraphs.

    Uses the ``fever/fever`` dataset with the wiki_pages config for evidence.
    Falls back to a simpler version if wiki pages aren't available.
    """
    from datasets import load_dataset as hf_load

    # Load labelled dev claims
    ds = hf_load("fever/fever", "v1.0", split="labelled_dev",
                 trust_remote_code=True)

    # Try to load wiki pages for evidence context
    wiki_pages = {}
    try:
        wiki_ds = hf_load("fever/fever", "wiki_pages", split="wikipedia_pages",
                          trust_remote_code=True)
        for page in wiki_ds:
            pid = page.get("id", "")
            title = pid.replace("_", " ") if pid else ""
            lines = page.get("lines", "") or ""
            sentences = [s.strip() for s in lines.split("\n") if s.strip()]
            if title and sentences:
                wiki_pages[title] = sentences
    except Exception:
        pass

    recs = []
    seen_ids = set()
    for ex in ds:
        claim_id = str(ex["id"])
        if claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)

        label = ex["label"]
        claim = ex["claim"]

        # Build context from evidence annotations
        evidence_titles = set()
        sf = []
        ev_wiki = ex.get("evidence_wiki", "") or ""
        ev_sid = ex.get("evidence_sentence_id", -1)
        if ev_wiki:
            title = ev_wiki.replace("_", " ")
            evidence_titles.add(title)
            if ev_sid >= 0:
                sf.append([title, ev_sid])

        # Add distractor pages (sample from wiki_pages if available)
        context = []
        for title in evidence_titles:
            sents = wiki_pages.get(title, [f"Evidence for: {title}"])
            context.append([title, sents[:10]])

        # Add some distractor paragraphs if we have wiki pages
        if wiki_pages:
            rng = random.Random(int(claim_id) if claim_id.isdigit() else hash(claim_id))
            distractor_titles = [t for t in wiki_pages if t not in evidence_titles]
            if distractor_titles:
                chosen = rng.sample(distractor_titles,
                                    min(8, len(distractor_titles)))
                for dt in chosen:
                    context.append([dt, wiki_pages[dt][:10]])

        recs.append({
            "_id": claim_id,
            "question": claim,
            "answer": label if label in FEVER_LABELS else "NOT ENOUGH INFO",
            "type": "verification",
            "level": "claim",
            "context": context,
            "supporting_facts": sf,
        })

    save_json(recs, dst)
    return len(recs)


def sample_pool(data: List[Dict[str, Any]], size: int,
                stratify_by: List[str], seed: int) -> List[Dict[str, Any]]:
    """Stratified sample balanced across label classes."""
    import collections
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
# Environment
# =========================================================================== #
@dataclass
class FEVEREnv(BaseEnv):
    """Per-claim environment for FEVER fact verification."""
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
        return self.record["question"]  # = claim

    @property
    def gold_answer(self) -> str:
        return self.record["answer"]  # SUPPORTS / REFUTES / NOT ENOUGH INFO

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


# =========================================================================== #
# FEVER-specific scoring
# =========================================================================== #
def score_answer_fever(prediction: Optional[str], ground_truth: str) -> Dict[str, float]:
    """FEVER uses label accuracy (exact match on normalized label)."""
    if prediction is None:
        return {"em": 0.0, "f1": 0.0, "correct": False}
    # Normalize prediction to one of the three labels
    pred = prediction.strip().upper()
    gold = ground_truth.strip().upper()
    # Handle common variations
    label_map = {
        "SUPPORTS": "SUPPORTS", "SUPPORT": "SUPPORTS", "TRUE": "SUPPORTS",
        "REFUTES": "REFUTES", "REFUTE": "REFUTES", "FALSE": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO", "NEI": "NOT ENOUGH INFO",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH INFO",
    }
    pred = label_map.get(pred, pred)
    gold = label_map.get(gold, gold)
    em = int(pred == gold)
    return {"em": float(em), "f1": float(em), "correct": bool(em)}
