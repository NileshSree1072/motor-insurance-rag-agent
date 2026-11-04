"""Single-shot RAG: retrieve once, answer once, cite every claim."""
import re
from dataclasses import dataclass, field

from motor_rag.ingest import Chunk
from motor_rag.llm import chat
from motor_rag.retrieve import Retriever, detect_insurers

NOT_FOUND = "The provided policy documents do not cover this."

SYSTEM_PROMPT = f"""You answer questions about Indian motor insurance policies using ONLY the numbered excerpts provided.

Rules:
- Every factual sentence must end with the excerpt number(s) it comes from, like [2] or [1][3].
- Name the insurer when terms differ between insurers; never merge two insurers' terms into one answer.
- Quote numbers (percentages, rupee limits, days) exactly as written in the excerpt.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge, even if you are confident."""


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    cited: list[Chunk] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{n}] {c.insurer} | {c.product} | {c.section} | page {c.page}\n{c.text}"
        for n, c in enumerate(chunks, start=1)
    )


def cited_chunks(text: str, chunks: list[Chunk]) -> list[Chunk]:
    """Excerpts cited as [2], [1, 3] or 【2】 (some models use full-width brackets)."""
    groups = re.findall(r"[\[【]\s*(\d+(?:\s*[,，]\s*\d+)*)\s*[\]】]", text)
    nums = sorted({int(n) for g in groups for n in re.findall(r"\d+", g)})
    return [chunks[n - 1] for n in nums if 1 <= n <= len(chunks)]


def tidy(text: str) -> str:
    """Normalise model output: narrow/no-break spaces to spaces, 【n】 to [n]."""
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"【\s*([\d,\s]+?)\s*】", r"[\1]", text).strip()


def answer(question: str, retriever: Retriever, k: int = 6, mode: str = "hybrid_rerank") -> Answer:
    chunks = retriever.search(question, k=k, mode=mode, insurer=detect_insurers(question) or None)
    resp = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"},
    ])
    text = tidy(resp.choices[0].message.content)
    return Answer(text=text, sources=chunks, cited=cited_chunks(text, chunks))
