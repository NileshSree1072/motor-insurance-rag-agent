"""HTTP API and chat page.

  uv run uvicorn motor_rag.api:app --port 8000
  open http://localhost:8000
"""
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import RateLimitError
from pydantic import BaseModel, Field

from motor_rag import agent, graph, rag
from motor_rag.cache import SemanticCache
from motor_rag.ingest import Chunk, load_sources
from motor_rag.llm import model_name, usage
from motor_rag.retrieve import QUERY_PREFIX, Retriever

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and models once; the first request would otherwise take ~30 s.
    retriever = Retriever()
    retriever.search("warm up", k=1)
    state["retriever"] = retriever
    state["urls"] = {s["doc_id"]: s["url"] for s in load_sources()}
    state["cache"] = SemanticCache()
    yield


app = FastAPI(title="Motor Insurance Policy Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["rag", "agent"] = "agent"


class Citation(BaseModel):
    n: int
    insurer: str
    product: str
    section: str
    page: int
    text: str
    url: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    trace: list[dict]
    latency_s: float
    tokens: int
    cached: bool = False
    match_similarity: float | None = None


def citation(n: int, c: Chunk) -> Citation:
    url = state["urls"].get(c.doc_id, "")
    return Citation(n=n, insurer=c.insurer, product=c.product, section=c.section, page=c.page,
                    text=c.text, url=f"{url}#page={c.page}" if url else "")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start, tokens_before = time.perf_counter(), usage[model_name()]
    retriever: Retriever = state["retriever"]
    cache: SemanticCache | None = state.get("cache")

    # 1. Embed query for semantic cache check
    q_vec = retriever.embedder.encode([QUERY_PREFIX + req.question], normalize_embeddings=True)[0]

    # 2. Check semantic cache before calling Groq
    if cache:
        hit, sim = cache.lookup(q_vec)
        if hit:
            cache_trace = [{
                "tool": "semantic_cache",
                "args": {"matched_question": hit.question, "similarity": round(sim, 3)},
                "ms": round((time.perf_counter() - start) * 1000, 1),
                "note": f"instant cache hit ({round(sim * 100, 1)}% match)",
            }]
            return AskResponse(
                answer=hit.answer,
                refused=hit.refused,
                citations=[Citation(**c) for c in hit.citations],
                trace=cache_trace,
                latency_s=round(time.perf_counter() - start, 3),
                tokens=0,
                cached=True,
                match_similarity=round(sim, 3),
            )

    try:
        if req.mode == "agent":
            result = agent.run(req.question, retriever)
            trace = [asdict(s) for s in result.trace]
        else:
            result = rag.answer(req.question, retriever)
            trace = []
    except RateLimitError:
        raise HTTPException(429, "The language model's rate limit was reached. Try again in a few minutes.")

    cited_ids = {c.chunk_id for c in result.cited}
    citations = [citation(n, c) for n, c in enumerate(result.sources, start=1) if c.chunk_id in cited_ids]

    # 3. Cache grounded verified answers
    if cache and not result.refused and citations:
        cache.store(
            question=req.question,
            vector=q_vec,
            answer=result.text,
            citations=[c.model_dump() for c in citations],
            refused=result.refused,
            mode=req.mode,
            trace=trace,
        )

    return AskResponse(
        answer=result.text,
        refused=result.refused,
        citations=citations,
        trace=trace,
        latency_s=round(time.perf_counter() - start, 2),
        tokens=usage[model_name()] - tokens_before,
        cached=False,
        match_similarity=None,
    )


@app.get("/cache")
def cache_stats():
    cache: SemanticCache | None = state.get("cache")
    return cache.stats() if cache else {"entries": 0, "total_hits": 0}


@app.delete("/cache")
def clear_cache():
    cache: SemanticCache | None = state.get("cache")
    if cache:
        cache.clear()
    return {"status": "cleared"}


@app.get("/health")
def health():
    cache: SemanticCache | None = state.get("cache")
    cache_info = cache.stats() if cache else {"entries": 0, "total_hits": 0}
    return {
        "status": "ok",
        "chunks": len(state["retriever"].chunks),
        "graph": graph.available(),
        "model": model_name(),
        "cache": cache_info,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
