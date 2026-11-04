"""Generate a held-out question set from randomly sampled chunks.

The hand-written set in evals/questions.jsonl was used while tuning retrieval
(synonyms, chunk size), so its scores are optimistic. This set is generated
once with a fixed seed and is never used for tuning; it measures whether the
improvements generalise.

Each generated item is kept only if:
  - its evidence is a verbatim span of the source chunk
  - the question names the insurer (real users say whose policy they mean)
  - the question does not copy a long run of words from the evidence

  python -m motor_rag.gen_questions --n 60
"""
import argparse
import json
import random
import re

from motor_rag.evalset import QUESTIONS, normalize
from motor_rag.ingest import load_chunks
from motor_rag.llm import chat
from motor_rag.retrieve import detect_insurers

HELDOUT = QUESTIONS.parent / "heldout.jsonl"

PROMPT = """You write test questions for a motor insurance question-answering system.

Insurer: {insurer}
Product: {product}
Excerpt from the policy document:
\"\"\"{text}\"\"\"

Write ONE question a real policyholder might ask that this excerpt answers.
- Use everyday words, not the policy's legal wording.
- Mention the insurer by name.
- It must be answerable from this excerpt alone, about a specific rule, limit, percentage, exclusion or condition.
- If the excerpt is only contact details, a table of contents, a list of names or other filler, skip it.

Reply with JSON only:
{{"skip": false, "question": "...", "evidence": "<shortest verbatim span of the excerpt, 4 to 20 words, that contains the answer>", "reference": "<one sentence answer>"}}
or {{"skip": true}}"""


def copies_evidence(question: str, evidence: str, run: int = 5) -> bool:
    """True if the question repeats `run` consecutive words of the evidence."""
    words = re.findall(r"[a-z0-9]+", evidence.lower())
    q = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    return any(" ".join(words[i:i + run]) in q for i in range(len(words) - run + 1))


def generate(n: int, seed: int) -> list[dict]:
    chunks = load_chunks()
    rng = random.Random(seed)
    order = rng.sample(range(len(chunks)), len(chunks))
    kept, rejected = [], {"skip": 0, "not verbatim": 0, "no insurer": 0, "copies wording": 0, "bad json": 0}
    for idx in order:
        if len(kept) >= n:
            break
        c = chunks[idx]
        resp = chat([{"role": "user", "content": PROMPT.format(insurer=c.insurer, product=c.product, text=c.text)}],
                    response_format={"type": "json_object"})
        try:
            item = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            rejected["bad json"] += 1
            continue
        if item.get("skip"):
            rejected["skip"] += 1
            continue
        q, ev = item.get("question", ""), item.get("evidence", "")
        if len(ev.split()) < 3 or normalize(ev) not in normalize(c.text):
            rejected["not verbatim"] += 1
            continue
        if c.insurer not in detect_insurers(q):
            rejected["no insurer"] += 1
            continue
        if copies_evidence(q, ev):
            rejected["copies wording"] += 1
            continue
        kept.append({"id": f"h{len(kept) + 1:02d}", "category": "single", "question": q,
                     "evidence": [{"doc_id": c.doc_id, "text": ev}], "reference": item.get("reference", ""),
                     "source_chunk": c.chunk_id})
        print(f"  kept {len(kept):>2}: {q}")
    print(f"\nkept {len(kept)}; rejected {rejected}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    items = generate(args.n, args.seed)
    with HELDOUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {HELDOUT}")


if __name__ == "__main__":
    main()
