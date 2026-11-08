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
from motor_rag.ingest import Chunk, load_sources
from motor_rag.llm import model_name, usage
from motor_rag.retrieve import Retriever

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and models once; the first request would otherwise take ~30 s.
    retriever = Retriever()
    retriever.search("warm up", k=1)
    state["retriever"] = retriever
    state["urls"] = {s["doc_id"]: s["url"] for s in load_sources()}
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


def citation(n: int, c: Chunk) -> Citation:
    url = state["urls"].get(c.doc_id, "")
    return Citation(n=n, insurer=c.insurer, product=c.product, section=c.section, page=c.page,
                    text=c.text, url=f"{url}#page={c.page}" if url else "")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the blocking
    # model and LLM calls don't stall the event loop.
    start, tokens_before = time.perf_counter(), usage[model_name()]
    try:
        if req.mode == "agent":
            result = agent.run(req.question, state["retriever"])
            trace = [asdict(s) for s in result.trace]
        else:
            result = rag.answer(req.question, state["retriever"])
            trace = []
    except RateLimitError:
        raise HTTPException(429, "The language model's rate limit was reached. Try again in a few minutes.")
    # Only excerpts the answer actually cited are returned, numbered as in the answer text.
    cited_ids = {c.chunk_id for c in result.cited}
    citations = [citation(n, c) for n, c in enumerate(result.sources, start=1) if c.chunk_id in cited_ids]
    return AskResponse(answer=result.text, refused=result.refused, citations=citations, trace=trace,
                       latency_s=round(time.perf_counter() - start, 2),
                       tokens=usage[model_name()] - tokens_before)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks), "graph": graph.available(), "model": model_name()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
