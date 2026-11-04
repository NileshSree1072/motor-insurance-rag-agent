"""Eval harness.

  python -m motor_rag.evaluate retrieval            # no LLM needed; compares all search modes
  python -m motor_rag.evaluate answers [--agent]    # needs GROQ_API_KEY

Results are printed as a table and saved to evals/runs/<timestamp>-<kind>.json.
"""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from motor_rag.evalset import QUESTIONS, Question, contains, load_questions, validate
from motor_rag.ingest import Chunk
from motor_rag.retrieve import MODES, Retriever

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "evals" / "runs"
PRODUCTION_SETUP = "hybrid_rerank + insurer filter"  # what rag.answer() uses


# ---------- retrieval ----------

def retrieval_scores(q: Question, results: list[Chunk]) -> dict:
    """recall: share of evidence items found; all_found: every item found; rr: 1/rank of first hit."""
    found = [any(contains(c, ev) for c in results) for ev in q.evidence]
    first_hit = next((rank for rank, c in enumerate(results, start=1)
                      if any(contains(c, ev) for ev in q.evidence)), None)
    return {"recall": sum(found) / len(found), "all_found": all(found), "rr": 1 / first_hit if first_hit else 0.0}


def eval_retrieval(retriever: Retriever, questions: list[Question], k: int) -> dict:
    answerable = [q for q in questions if q.answerable]
    # Load models and build the index first so timings measure queries, not start-up.
    retriever.search("warm up", k=1, mode="hybrid_rerank")
    report = {}
    for mode, filtered in [(m, f) for m in MODES for f in (False, True)]:
        search = retriever.search_auto if filtered else retriever.search
        rows, start = [], time.perf_counter()
        for q in answerable:
            s = retrieval_scores(q, search(q.question, k=k, mode=mode))
            rows.append({"id": q.id, "category": q.category, **s})
        n = len(rows)
        report[f"{mode}{' + insurer filter' if filtered else ''}"] = {
            f"recall@{k}": sum(r["recall"] for r in rows) / n,
            f"all_found@{k}": sum(r["all_found"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
            "ms_per_query": (time.perf_counter() - start) * 1000 / n,
            "misses": [r["id"] for r in rows if r["recall"] < 1],
        }
    return report


def print_retrieval(report: dict, k: int, n: int):
    print(f"\nRetrieval on {n} answerable questions (k={k})")
    print(f"{'mode':<32}{'recall@'+str(k):>10}{'all@'+str(k):>8}{'MRR':>7}{'ms/q':>8}   misses")
    for mode, r in report.items():
        print(f"{mode:<32}{r[f'recall@{k}']:>10.1%}{r[f'all_found@{k}']:>8.1%}{r['mrr']:>7.2f}"
              f"{r['ms_per_query']:>8.0f}   {', '.join(r['misses']) or '-'}")


# ---------- answers ----------

JUDGE_PROMPT = """You grade answers from an insurance question-answering system.

Question: {question}
Reference answer: {reference}
Excerpts the system was given:
{context}

System answer:
{answer}

Grade two things:
1. faithful: is every factual claim in the system answer supported by the excerpts? (true/false)
2. correct: does the system answer agree with the reference answer on the key facts? Extra correct detail is fine. (true/false)

Reply with JSON only: {{"faithful": true/false, "correct": true/false, "reason": "<one sentence>"}}"""


def judge(q: Question, answer_text: str, context: str) -> dict:
    from motor_rag.llm import chat, judge_model_name
    resp = chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(
            question=q.question, reference=q.reference, context=context, answer=answer_text)}],
        model=judge_model_name(),
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"faithful": False, "correct": False, "reason": "judge returned invalid JSON"}


def eval_answers(retriever: Retriever, questions: list[Question], use_agent: bool, use_graph: bool | None = None) -> dict:
    from motor_rag.rag import answer, format_context
    run = None
    if use_agent:
        from motor_rag.agent import run as run_agent
        run = run_agent
    from motor_rag.llm import model_name, usage
    rows = []
    for q in questions:
        start, tokens_before = time.perf_counter(), usage[model_name()]
        a = run(q.question, retriever, use_graph=use_graph) if use_agent else answer(q.question, retriever)
        latency = time.perf_counter() - start
        row = {"id": q.id, "category": q.category, "answer": a.text, "refused": a.refused,
               "latency_s": round(latency, 2), "steps": getattr(a, "steps", 1),
               "tokens": usage[model_name()] - tokens_before,
               "tools": [s.tool for s in getattr(a, "trace", [])]}
        if q.answerable:
            # Citation accuracy: of the evidence items, how many are in chunks the answer actually cited.
            row["cited_evidence"] = sum(any(contains(c, ev) for c in a.cited) for ev in q.evidence) / len(q.evidence)
            if not a.refused:
                row.update(judge(q, a.text, format_context(a.sources)))
        rows.append(row)
        print(f"  {q.id:<5} {'REFUSED' if a.refused else 'answered':<9} {latency:5.1f}s  {a.text[:90]!r}")
    return {"summary": summarize(rows), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    judged = [r for r in answerable if "faithful" in r]
    summary = {
        "questions": len(rows),
        "correct": sum(bool(r.get("correct")) for r in answerable) / len(answerable),
        "faithful": sum(bool(r.get("faithful")) for r in judged) / max(len(judged), 1),
        "citation_accuracy": sum(r["cited_evidence"] for r in answerable) / len(answerable),
        "false_refusal_rate": sum(r["refused"] for r in answerable) / len(answerable),
        "correct_refusal_rate": sum(r["refused"] for r in unanswerable) / max(len(unanswerable), 1),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
        "avg_tokens": sum(r["tokens"] for r in rows) / len(rows),
        "avg_steps": sum(r["steps"] for r in rows) / len(rows),
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in answerable}):
        rs = [r for r in answerable if r["category"] == cat]
        summary["by_category"][cat] = {"n": len(rs), "correct": sum(bool(r.get("correct")) for r in rs) / len(rs)}
    return summary


def merge(paths: list[str]) -> dict:
    """Combine answer runs done in parts (e.g. across days, because of token quotas).

    A question answered in more than one file keeps its latest result.
    """
    rows = {}
    for path in sorted(paths):  # run files are timestamped, so sorted = oldest first
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]:
            rows[row["id"]] = row
    return {"summary": summarize(list(rows.values())), "rows": list(rows.values()), "merged_from": sorted(paths)}


def print_answers(summary: dict, label: str):
    print(f"\nAnswer quality ({label}, {summary['questions']} questions)")
    for key in ("correct", "faithful", "citation_accuracy", "false_refusal_rate", "correct_refusal_rate"):
        print(f"  {key:<22}{summary[key]:>7.1%}")
    print(f"  {'avg_latency_s':<22}{summary['avg_latency_s']:>7.2f}")
    print(f"  {'avg_tokens':<22}{summary['avg_tokens']:>7.0f}")
    print(f"  {'avg_steps':<22}{summary['avg_steps']:>7.1f}")
    for cat, s in summary["by_category"].items():
        print(f"  correct on {cat:<11}{s['correct']:>7.1%}  (n={s['n']})")


def save(kind: str, data: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["retrieval", "answers", "merge"])
    parser.add_argument("files", nargs="*", help="merge: answer-run JSON files to combine")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--agent", action="store_true", help="answer with the tool-calling agent")
    parser.add_argument("--no-graph", action="store_true", help="agent without the Neo4j tools (ablation)")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--min-recall", type=float,
                        help="retrieval: exit with an error if the production setup scores below this (CI quality gate)")
    parser.add_argument("--set", default="questions", choices=["questions", "heldout"],
                        help="questions: hand-written (used while tuning); heldout: generated, never tuned on")
    args = parser.parse_args()

    if args.kind == "merge":
        result = merge(args.files)
        print_answers(result["summary"], f"merged from {len(args.files)} runs")
        print(f"\nsaved {save('merged', result).relative_to(ROOT)}")
        return

    retriever = Retriever()
    questions = load_questions(QUESTIONS.parent / f"{args.set}.jsonl")
    problems = validate(questions, retriever.chunks)
    if problems:
        raise SystemExit("Eval set is out of date with the documents:\n" + "\n".join(problems))
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted]

    if args.kind == "retrieval":
        report = eval_retrieval(retriever, questions, args.k)
        print_retrieval(report, args.k, sum(q.answerable for q in questions))
        print(f"\nsaved {save('retrieval', report).relative_to(ROOT)}")
        if args.min_recall is not None:
            score = report[PRODUCTION_SETUP][f"recall@{args.k}"]
            if score < args.min_recall:
                raise SystemExit(f"FAIL: {PRODUCTION_SETUP} recall@{args.k} {score:.1%} < {args.min_recall:.0%}")
            print(f"PASS: {PRODUCTION_SETUP} recall@{args.k} {score:.1%} >= {args.min_recall:.0%}")
    else:
        kind = ("agent-nograph" if args.no_graph else "agent") if args.agent else "rag"
        result = eval_answers(retriever, questions, args.agent, use_graph=False if args.no_graph else None)
        print_answers(result["summary"], kind)
        print(f"\nsaved {save(kind, result).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
