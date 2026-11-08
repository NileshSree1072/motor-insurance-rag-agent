"""Semantic Cache for grounded policy answers.

Stores previously answered questions and their verified citations.
When a new query has cosine similarity >= SIMILARITY_THRESHOLD with
a cached question, returns the verified answer instantly with 0 tokens.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "index"
CACHE_FILE = CACHE_DIR / "semantic_cache.json"
DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass
class CachedResponse:
    question: str
    answer: str
    citations: list[dict]
    refused: bool
    mode: str
    trace: list[dict]
    embedding: list[float]
    created_at: float
    hits: int = 0


class SemanticCache:
    def __init__(self, path: Path | None = None, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.path = path or CACHE_FILE
        self.threshold = threshold
        self.entries: list[CachedResponse] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [CachedResponse(**item) for item in data]
            if self.entries:
                self._embeddings = np.array([e.embedding for e in self.entries], dtype=np.float32)
        except Exception:
            self.entries = []
            self._embeddings = None

    def _save(self):
        self.path.parent.mkdir(exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)
        temp_path.replace(self.path)

    def lookup(self, query_vector: np.ndarray) -> tuple[CachedResponse | None, float]:
        """Find the most similar cached answer if similarity >= threshold."""
        if not self.entries or self._embeddings is None:
            return None, 0.0

        # Cosine similarity for unit-normalized vectors is the dot product
        sims = self._embeddings @ query_vector
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            entry = self.entries[best_idx]
            entry.hits += 1
            return entry, best_sim

        return None, best_sim

    def store(self, question: str, vector: np.ndarray, answer: str,
              citations: list[dict], refused: bool, mode: str, trace: list[dict]):
        """Cache a grounded response. Refused or uncited answers are never cached."""
        if refused or not citations or not answer.strip():
            return

        # Avoid storing duplicate entries if a near-identical query is already cached
        existing, sim = self.lookup(vector)
        if existing and sim >= 0.98:
            return

        entry = CachedResponse(
            question=question,
            answer=answer,
            citations=citations,
            refused=refused,
            mode=mode,
            trace=trace,
            embedding=vector.tolist(),
            created_at=time.time(),
            hits=0,
        )
        self.entries.append(entry)
        if self._embeddings is None:
            self._embeddings = np.array([entry.embedding], dtype=np.float32)
        else:
            self._embeddings = np.vstack([self._embeddings, vector.astype(np.float32)])
        self._save()

    def clear(self):
        """Clear in-memory and on-disk cache."""
        self.entries.clear()
        self._embeddings = None
        if self.path.exists():
            self.path.unlink()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self.entries),
            "total_hits": sum(e.hits for e in self.entries),
            "threshold": self.threshold,
        }
