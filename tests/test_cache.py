"""Tests for semantic cache memory."""
import numpy as np
import pytest

from motor_rag.cache import SemanticCache


def test_empty_cache_returns_none(tmp_path):
    cache = SemanticCache(path=tmp_path / "cache.json")
    v = np.random.randn(384).astype(np.float32)
    v /= np.linalg.norm(v)
    hit, sim = cache.lookup(v)
    assert hit is None
    assert sim == 0.0
    assert cache.stats()["entries"] == 0


def test_store_and_lookup_exact_and_similar(tmp_path):
    cache = SemanticCache(path=tmp_path / "cache.json", threshold=0.92)

    # Base unit vector
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0

    cache.store(
        question="What is SBI depreciation?",
        vector=v1,
        answer="SBI applies 50% on plastic parts [1].",
        citations=[{"n": 1, "insurer": "SBI General", "product": "Package", "section": "Depreciation", "page": 5, "text": "50%", "url": ""}],
        refused=False,
        mode="agent",
        trace=[],
    )
    assert cache.stats()["entries"] == 1

    # Exact lookup
    hit, sim = cache.lookup(v1)
    assert hit is not None
    assert hit.question == "What is SBI depreciation?"
    assert sim >= 0.999
    assert hit.hits == 1

    # Similar vector (sim ~ 0.95)
    v2 = v1.copy()
    v2[1] = 0.32
    v2 /= np.linalg.norm(v2)
    hit2, sim2 = cache.lookup(v2)
    assert hit2 is not None
    assert sim2 >= 0.92
    assert hit2.hits == 2

    # Dissimilar vector (orthogonal, sim = 0.0)
    v3 = np.zeros(384, dtype=np.float32)
    v3[10] = 1.0
    hit3, sim3 = cache.lookup(v3)
    assert hit3 is None
    assert sim3 < 0.92


def test_refused_or_uncited_answers_are_never_cached(tmp_path):
    cache = SemanticCache(path=tmp_path / "cache.json")
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0

    # Refused
    cache.store("Out of scope?", v, "The provided policy documents do not cover this.", [], True, "agent", [])
    assert cache.stats()["entries"] == 0

    # Uncited
    cache.store("Uncited answer", v, "Some statement without brackets.", [], False, "agent", [])
    assert cache.stats()["entries"] == 0


def test_cache_persistence_across_reloads(tmp_path):
    file_path = tmp_path / "cache.json"
    cache1 = SemanticCache(path=file_path)

    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0

    cache1.store(
        question="Towing limit?",
        vector=v,
        answer="Rs 2,500 [1].",
        citations=[{"n": 1, "insurer": "Universal Sompo", "product": "Package", "section": "Towing", "page": 14, "text": "2500", "url": ""}],
        refused=False,
        mode="agent",
        trace=[],
    )
    assert file_path.exists()

    # Load in a fresh instance
    cache2 = SemanticCache(path=file_path)
    assert cache2.stats()["entries"] == 1
    hit, sim = cache2.lookup(v)
    assert hit is not None
    assert hit.question == "Towing limit?"
    assert sim >= 0.999


def test_clear_removes_entries_and_file(tmp_path):
    file_path = tmp_path / "cache.json"
    cache = SemanticCache(path=file_path)
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0
    cache.store("Q", v, "A [1]", [{"n": 1}], False, "rag", [])
    assert file_path.exists()

    cache.clear()
    assert cache.stats()["entries"] == 0
    assert not file_path.exists()
