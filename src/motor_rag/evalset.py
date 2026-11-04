"""The labeled question set and the rule for "this chunk contains the evidence".

Labels are verbatim evidence phrases, not chunk ids, so they stay valid when
chunking changes. Matching ignores case, whitespace and dash/quote variants.
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from motor_rag.ingest import Chunk, load_sources

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "evals" / "questions.jsonl"


@dataclass
class Evidence:
    doc_id: str
    text: str


@dataclass
class Question:
    id: str
    category: str  # single | comparison | multi_hop | unanswerable
    question: str
    evidence: list[Evidence]
    reference: str
    source_chunk: str = ""  # generated questions only: the chunk they were written from

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", "", text)  # PDFs break lines and spaces inconsistently


@lru_cache
def _doc_meta() -> dict[str, tuple[str, str]]:
    return {s["doc_id"]: (s["insurer"], s["vehicle"]) for s in load_sources()}


def contains(chunk: Chunk, ev: Evidence) -> bool:
    """True if the chunk holds the evidence text from the same insurer and vehicle type.

    Insurers repeat clauses word for word across their own documents (a two wheeler
    package wording and its long-term version share the roadside assistance clause),
    so any of those copies answers the question equally well.
    """
    insurer, vehicle = _doc_meta().get(ev.doc_id, (None, None))
    same_source = chunk.doc_id == ev.doc_id or (chunk.insurer == insurer and chunk.vehicle == vehicle)
    return same_source and normalize(ev.text) in normalize(chunk.text)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Question(**{**r, "evidence": [Evidence(**e) for e in r["evidence"]]}) for r in rows]


def validate(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Problems with the dataset itself: evidence that no chunk contains, duplicate ids."""
    problems = []
    seen = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        for ev in q.evidence:
            if not any(contains(c, ev) for c in chunks):
                problems.append(f"{q.id}: evidence not found in {ev.doc_id}: {ev.text!r}")
    return problems
