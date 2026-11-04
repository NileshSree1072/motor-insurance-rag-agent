"""Retrieval over policy chunks.

Four modes so the eval harness can measure each one:
  bm25           keyword search; strong on exact terms like "IDV" or "Rs. 1500"
  dense          embedding search; strong on paraphrases like "car flooded"
  hybrid         both, merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""
import re
from functools import cached_property
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from motor_rag.ingest import Chunk, load_chunks

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# bge models are trained with this prefix on queries (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

STOPWORDS = set("a an and are as at be by for from has have if in is it of on or that the this to under what when which will with my i me does do can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.bm25 = BM25Okapi([tokenize(f"{c.section} {c.text}") for c in self.chunks])

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANK_MODEL)

    @cached_property
    def embeddings(self) -> np.ndarray:
        """Chunk embeddings, cached on disk and rebuilt when the chunk set changes."""
        INDEX.mkdir(exist_ok=True)
        path = INDEX / "embeddings.npy"
        ids_path = INDEX / "embedding_ids.txt"
        ids = [c.chunk_id for c in self.chunks]
        if path.exists() and ids_path.exists() and ids_path.read_text(encoding="utf-8").splitlines() == ids:
            return np.load(path)
        vecs = self.embedder.encode(
            [f"{c.section}\n{c.text}" for c in self.chunks],
            batch_size=32, normalize_embeddings=True, show_progress_bar=True,
        )
        np.save(path, vecs)
        ids_path.write_text("\n".join(ids), encoding="utf-8")
        return vecs

    def _filter_mask(self, insurer: str | None, vehicle: str | None) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if insurer:
            mask &= np.array([c.insurer.lower() == insurer.lower() for c in self.chunks])
        if vehicle:
            mask &= np.array([c.vehicle == vehicle for c in self.chunks])
        return mask

    def _bm25_ranking(self, query: str, mask: np.ndarray) -> list[int]:
        scores = np.where(mask, self.bm25.get_scores(tokenize(query)), -np.inf)
        return [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

    def _dense_ranking(self, query: str, mask: np.ndarray) -> list[int]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = np.where(mask, self.embeddings @ q, -np.inf)
        return [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank",
               insurer: str | None = None, vehicle: str | None = None) -> list[Chunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        mask = self._filter_mask(insurer, vehicle)
        if mode == "bm25":
            return [self.chunks[i] for i in self._bm25_ranking(query, mask)[:k]]
        if mode == "dense":
            return [self.chunks[i] for i in self._dense_ranking(query, mask)[:k]]

        # Reciprocal Rank Fusion: score = sum of 1 / (60 + rank) over both rankings.
        # Uses ranks, not raw scores, so BM25 and cosine scales never need calibrating.
        fused: dict[int, float] = {}
        for ranking in (self._bm25_ranking(query, mask)[:50], self._dense_ranking(query, mask)[:50]):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        candidates = sorted(fused, key=fused.get, reverse=True)
        if mode == "hybrid":
            return [self.chunks[i] for i in candidates[:k]]

        candidates = candidates[:30]
        scores = self.reranker.predict([(query, f"{self.chunks[i].section}\n{self.chunks[i].text}") for i in candidates])
        order = np.argsort(-np.asarray(scores))
        return [self.chunks[candidates[j]] for j in order[:k]]
