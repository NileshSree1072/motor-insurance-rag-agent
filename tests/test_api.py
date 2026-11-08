import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai import RateLimitError

from motor_rag import api
from motor_rag.cache import SemanticCache
from motor_rag.ingest import Chunk
from motor_rag.rag import Answer


def chunk(n):
    return Chunk(f"d#{n}", "sbi_pc_saod_pw", "SBI General", "Own Damage", "private_car", "wording", n,
                 "7. DEDUCTIBLE", f"clause {n}")


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        v = np.zeros((len(texts), 384), dtype=np.float32)
        v[:, 0] = 1.0
        return v


class FakeRetriever:
    def __init__(self):
        self.embedder = FakeEmbedder()
        self.chunks = []


@pytest.fixture
def client(monkeypatch):
    # Skip the lifespan (it loads the real index and models); set the state it would set.
    monkeypatch.setitem(api.state, "retriever", FakeRetriever())
    monkeypatch.setitem(api.state, "urls", {"sbi_pc_saod_pw": "https://example.com/sbi.pdf"})
    monkeypatch.setitem(api.state, "cache", None)
    return TestClient(api.app)


def test_only_cited_sources_are_returned_with_their_answer_numbers(client, monkeypatch):
    sources = [chunk(1), chunk(2), chunk(3)]
    monkeypatch.setattr(api.rag, "answer", lambda q, r: Answer("Deductible applies [3].", sources, [sources[2]]))
    data = client.post("/ask", json={"question": "What is the deductible?", "mode": "rag"}).json()
    assert [c["n"] for c in data["citations"]] == [3]
    assert data["citations"][0]["url"] == "https://example.com/sbi.pdf#page=3"
    assert data["refused"] is False and data["trace"] == []


def test_question_length_is_validated(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422


def test_rate_limit_becomes_a_429_with_a_readable_message(client, monkeypatch):
    def limited(q, r):
        resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
        raise RateLimitError("rate limited", response=resp, body=None)
    monkeypatch.setattr(api.rag, "answer", limited)
    res = client.post("/ask", json={"question": "What is the deductible?", "mode": "rag"})
    assert res.status_code == 429 and "rate limit" in res.json()["detail"]


def test_chat_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Motor Insurance Policy Assistant" in res.text


def test_semantic_cache_hit_bypasses_llm(client, monkeypatch, tmp_path):
    cache = SemanticCache(path=tmp_path / "cache.json", threshold=0.90)
    monkeypatch.setitem(api.state, "cache", cache)

    # First query: cache miss, runs RAG, populates cache
    sources = [chunk(1)]
    called = []
    def fake_answer(q, r):
        called.append(q)
        return Answer("SBI deductible is Rs 1,000 [1].", sources, [sources[0]])
    monkeypatch.setattr(api.rag, "answer", fake_answer)

    r1 = client.post("/ask", json={"question": "What is the deductible?", "mode": "rag"}).json()
    assert r1["cached"] is False
    assert len(called) == 1
    assert cache.stats()["entries"] == 1

    # Second query: exact same question embedding -> cache hit!
    r2 = client.post("/ask", json={"question": "What is the deductible?", "mode": "rag"}).json()
    assert r2["cached"] is True
    assert r2["tokens"] == 0
    assert len(called) == 1  # LLM was NOT called again!
    assert r2["answer"] == "SBI deductible is Rs 1,000 [1]."
    assert len(r2["citations"]) == 1
    assert r2["trace"][0]["tool"] == "semantic_cache"

    # Cache endpoints
    assert client.get("/cache").json()["entries"] == 1
    assert client.delete("/cache").json()["status"] == "cleared"
    assert client.get("/cache").json()["entries"] == 0
