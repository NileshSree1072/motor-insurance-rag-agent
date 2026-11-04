"""Retrieval over policy chunks.

Four modes so the eval harness can measure each one:
  bm25           keyword search; strong on exact terms like "IDV" or "Rs. 1500"
  dense          embedding search; strong on paraphrases like "car flooded"
  hybrid         both, merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""
import hashlib
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


# Everyday words -> the wording policies actually use. BM25 only matches exact
# tokens, so "drunk" never finds "under the influence of intoxicating liquor".
SYNONYMS = {
    "drunk": "intoxicating liquor drugs influence",
    "alcohol": "intoxicating liquor",
    "towing": "removal nearest repairer protection",
    "tow": "removal nearest repairer protection",
    "flood": "inundation water",
    "flooded": "inundation water ingression",
    "stolen": "theft burglary",
    "bike": "two wheeler",
    "scooter": "two wheeler",
    "truck": "goods carrying vehicle",
    "lorry": "goods carrying vehicle",
    "excess": "deductible",
    "zero": "nil depreciation",
    "nil": "zero depreciation",
    "breakdown": "immobilised roadside assistance",
    "cancel": "cancellation refund",
    "license": "driving licence",
}


def expand(query: str) -> str:
    extra = [SYNONYMS[t] for t in re.findall(r"[a-z]+", query.lower()) if t in SYNONYMS]
    return " ".join([query, *extra])


# How people actually name insurers in questions -> insurer name in sources.csv.
INSURER_ALIASES = {
    r"icici": "ICICI Lombard",
    r"\bsbi\b": "SBI General",
    r"universal\s+sompo|\busgi\b": "Universal Sompo",
    r"iffco": "IFFCO Tokio",
    r"\bdigit\b": "Go Digit",
    r"liberty": "Liberty General",
    r"raheja|\bqbe\b": "Raheja QBE",
    r"tata\s*aig|\btata\b": "Tata AIG",
    r"new\s+india": "New India Assurance",
    r"royal\s+sundaram": "Royal Sundaram",
    r"future\s+generali|generali": "Generali Central",
    r"\backo\b": "Acko",
}


def detect_insurers(text: str) -> list[str]:
    """Insurers named in the text, in order of first mention."""
    hits = []
    for pattern, name in INSURER_ALIASES.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m and name not in [h for _, h in hits]:
            hits.append((m.start(), name))
    return [name for _, name in sorted(hits)]


def index_text(c: Chunk) -> str:
    """What gets indexed for a chunk: its text plus where it comes from.

    A clause like "maximum of four claims" says nothing about which insurer or
    product it belongs to; that context lives in the document title. Prefixing it
    lets "ICICI two wheeler roadside assistance" match the right document's clause.
    """
    return f"{c.insurer} | {c.product} | {c.section}\n{c.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in self.chunks])

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
        """Chunk embeddings, cached on disk.

        The cache key is a hash of the model name and every indexed text, so any
        change to chunking or to index_text() rebuilds it. (Keying on chunk ids alone
        would silently reuse stale vectors when text changes but ids don't.)
        """
        texts = [index_text(c) for c in self.chunks]
        key = hashlib.sha256("\n\x00".join([EMBED_MODEL, *texts]).encode("utf-8")).hexdigest()[:16]
        INDEX.mkdir(exist_ok=True)
        path = INDEX / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        for old in INDEX.glob("embeddings*.npy"):
            old.unlink()
        np.save(path, vecs)
        return vecs

    def _filter_mask(self, insurer: str | None, vehicle: str | None) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if insurer:
            mask &= np.array([c.insurer.lower() == insurer.lower() for c in self.chunks])
        if vehicle:
            mask &= np.array([c.vehicle == vehicle for c in self.chunks])
        return mask

    def _bm25_ranking(self, query: str, mask: np.ndarray) -> list[int]:
        scores = np.where(mask, self.bm25.get_scores(tokenize(expand(query))), -np.inf)
        return [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

    def _dense_ranking(self, query: str, mask: np.ndarray) -> list[int]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = np.where(mask, self.embeddings @ q, -np.inf)
        return [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank",
               insurer: str | list[str] | None = None, vehicle: str | None = None) -> list[Chunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if isinstance(insurer, list):
            if len(insurer) > 1:
                return self._search_each(query, k, mode, insurer, vehicle)
            insurer = insurer[0] if insurer else None
        return self._search_one(query, k, mode, insurer, vehicle)

    def search_auto(self, query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[Chunk]:
        """Search restricted to the insurers named in the query, if any."""
        return self.search(query, k=k, mode=mode, insurer=detect_insurers(query) or None)

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Order given candidates by cross-encoder relevance, dropping near-duplicates.

        Overlapping chunks repeat text, so two candidates can be almost the same
        clause; keeping both wastes a result slot.
        """
        if not chunks:
            return []
        scores = self.reranker.predict([(expand(query), index_text(c)) for c in chunks])
        picked, seen = [], set()
        for j in np.argsort(-np.asarray(scores)):
            key = re.sub(r"\W+", "", chunks[j].text.lower())[:150]
            if key in seen:
                continue
            seen.add(key)
            picked.append(chunks[j])
            if len(picked) == k:
                break
        return picked

    def _search_each(self, query, k, mode, insurers, vehicle) -> list[Chunk]:
        """Comparisons: give every named insurer a fair share of the k results."""
        per = -(-k // len(insurers))  # ceil
        lists = [self._search_one(query, per, mode, ins, vehicle) for ins in insurers]
        merged = [c for group in zip(*[l + [None] * (per - len(l)) for l in lists]) for c in group if c]
        return merged[:max(k, len(insurers))]

    def _search_one(self, query, k, mode, insurer, vehicle) -> list[Chunk]:
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
        scores = self.reranker.predict([(expand(query), index_text(self.chunks[i])) for i in candidates])
        order = np.argsort(-np.asarray(scores))
        return [self.chunks[candidates[j]] for j in order[:k]]
