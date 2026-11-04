"""Tool-calling agent harness.

The model decides which tool to call; this module runs it, keeps the loop safe
and records what happened:
  - a hard step limit, so a confused model cannot loop forever
  - tool arguments validated before running; bad calls return an error the model can fix
  - an identical repeated call is answered from cache and flagged instead of re-run
  - every excerpt any tool returns gets one global number, so [n] citations in the
    final answer point at a specific chunk no matter which step produced it
  - a trace of every step (tool, arguments, duration, result size) for debugging and evals
"""
import json
import time
from dataclasses import dataclass, field

from motor_rag.ingest import Chunk, load_sources
from motor_rag.llm import chat
from motor_rag.rag import NOT_FOUND, cited_chunks
from motor_rag.retrieve import Retriever

MAX_STEPS = 6
EXCERPTS_PER_SEARCH = 5
VEHICLES = ("private_car", "two_wheeler", "goods_carrying", "misc")

SYSTEM_PROMPT = f"""You are an assistant for Indian motor insurance policy questions. You can only know what the tools return.

How to work:
- Search before answering. For a question about one insurer, pass that insurer to the search tool.
- For a comparison, search each insurer separately; never assume two insurers have the same terms.
- If a result mentions another clause that decides the answer (an exclusion, a definition, "subject to ..."), look that up too.
- Stop searching once you have the clauses you need.

How to answer:
- Every factual sentence ends with the excerpt number(s) it relies on, like [3] or [2][7].
- Quote numbers (percentages, rupee limits, days) exactly as written.
- If the tools do not return the answer, reply exactly: "{NOT_FOUND}"
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_policy_clauses",
        "description": "Search policy wordings for clauses relevant to a query. Returns numbered excerpts.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "insurer": {"type": "string", "description": "Optional exact insurer name from list_insurers."},
            "vehicle": {"type": "string", "enum": list(VEHICLES), "description": "Optional vehicle type."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "list_insurers",
        "description": "List the insurers and products whose documents are indexed.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_section",
        "description": "Read the full text of one section of one document, when an excerpt is cut off or refers to the rest of its section.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "section": {"type": "string", "description": "Section heading exactly as shown in an excerpt."},
        }, "required": ["doc_id", "section"]},
    }},
]


@dataclass
class Step:
    tool: str
    args: dict
    ms: float
    result_chars: int
    note: str = ""


@dataclass
class AgentAnswer:
    text: str
    sources: list[Chunk]
    cited: list[Chunk] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    stopped_by_limit: bool = False

    @property
    def refused(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()

    @property
    def steps(self) -> int:
        return len(self.trace)


class ToolError(Exception):
    """Returned to the model as an error message instead of crashing the run."""


class Harness:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.excerpts: list[Chunk] = []  # global numbering: excerpt n is self.excerpts[n-1]
        self.cache: dict[str, str] = {}
        self.insurers = sorted({s["insurer"] for s in load_sources()
                                if any(c.insurer == s["insurer"] for c in retriever.chunks)})

    def _number(self, chunks: list[Chunk]) -> str:
        lines = []
        for c in chunks:
            if c not in self.excerpts:
                self.excerpts.append(c)
            n = self.excerpts.index(c) + 1
            lines.append(f"[{n}] {c.insurer} | {c.product} | doc_id={c.doc_id} | {c.section} | page {c.page}\n{c.text}")
        return "\n\n".join(lines) if lines else "No matching clauses found."

    def search_policy_clauses(self, query: str, insurer: str | None = None, vehicle: str | None = None) -> str:
        if not query.strip():
            raise ToolError("query must not be empty")
        if insurer:
            match = [i for i in self.insurers if i.lower() == insurer.lower()]
            if not match:
                raise ToolError(f"unknown insurer {insurer!r}; valid insurers: {', '.join(self.insurers)}")
            insurer = match[0]
        if vehicle and vehicle not in VEHICLES:
            raise ToolError(f"vehicle must be one of {VEHICLES}")
        return self._number(self.retriever.search(query, k=EXCERPTS_PER_SEARCH, insurer=insurer, vehicle=vehicle))

    def list_insurers(self) -> str:
        products = {}
        for s in load_sources():
            if s["insurer"] in self.insurers:
                products.setdefault(s["insurer"], []).append(f"{s['product']} ({s['vehicle']})")
        return "\n".join(f"{ins}: {'; '.join(p)}" for ins, p in sorted(products.items()))

    def get_section(self, doc_id: str, section: str) -> str:
        chunks = [c for c in self.retriever.chunks if c.doc_id == doc_id and c.section.lower() == section.lower()]
        if not chunks:
            raise ToolError(f"no section {section!r} in {doc_id!r}; copy doc_id and section exactly from an excerpt")
        return self._number(chunks[:6])  # a section can be long; six chunks is plenty for one clause

    def call(self, name: str, raw_args: str) -> tuple[str, str]:
        """Run one tool call. Returns (result text, trace note)."""
        key = f"{name}:{raw_args}"
        if key in self.cache:
            return self.cache[key] + "\n\n(You already made this exact call; use the results above.)", "repeated call"
        if name not in {t["function"]["name"] for t in TOOLS}:
            return f"Error: unknown tool {name!r}", "unknown tool"
        try:
            args = json.loads(raw_args or "{}")
            result = getattr(self, name)(**args)
        except (json.JSONDecodeError, TypeError) as e:
            return f"Error: invalid arguments ({e})", "bad arguments"
        except ToolError as e:
            return f"Error: {e}", "tool error"
        self.cache[key] = result
        return result, ""


def run(question: str, retriever: Retriever, max_steps: int = MAX_STEPS) -> AgentAnswer:
    harness = Harness(retriever)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    trace: list[Step] = []

    for _ in range(max_steps):
        msg = chat(messages, tools=TOOLS, tool_choice="auto").choices[0].message
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            return AgentAnswer(text=text, sources=harness.excerpts,
                               cited=cited_chunks(text, harness.excerpts), trace=trace)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            start = time.perf_counter()
            result, note = harness.call(tc.function.name, tc.function.arguments)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}
            trace.append(Step(tc.function.name, args, round((time.perf_counter() - start) * 1000, 1), len(result), note))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Out of steps: ask for a final answer from what was gathered, with tools disabled.
    messages.append({"role": "user", "content": "Step limit reached. Answer now using only the excerpts above, with citations."})
    text = (chat(messages).choices[0].message.content or "").strip()
    return AgentAnswer(text=text, sources=harness.excerpts, cited=cited_chunks(text, harness.excerpts),
                       trace=trace, stopped_by_limit=True)
