# Motor Insurance RAG Agent: handoff (state at end of 2026-11-04)

Repo: https://github.com/NileshSree1072/motor-insurance-rag-agent (public, CI green, last commit f804bb1)
Local: C:\Users\niles\OneDrive\Desktop\Nilesh\motor-insurance-rag-agent
Rebuilt from scratch because the original project and its documents were lost.

## Rules the user set
- Explain before any GitHub action; never commit .env or the insurer PDFs.
- Only real measured numbers go on the resume; no made-up figures.
- Don't redo finished work.

## Done (do NOT redo)
- 36 PDFs from 10 insurers downloaded to data/raw (not committed; data/sources.csv lists URLs).
- Parsing + chunking (700 chars, 200 overlap, contextual headers) -> 5,643 chunks in data/processed/chunks.jsonl.
- Embeddings cached in index/. Neo4j graph built (docker compose; container motor-insurance-rag-agent-neo4j-1).
- Retrieval evals finished. Agent harness, graph tools, FastAPI + chat page, CI, 34 tests, README.
- Single-search answer eval on all 35 questions: evals/runs/20261104-165451-rag.json
- Agent eval on 7 questions (c01,c02,c03,c04,s01,s16,u01): evals/runs/20261104-164835-agent.json

## Measured numbers (Final completed state)
- Retrieval recall@5, held-out 60 questions (never tuned on): BM25 53.3% -> dense 61.7% -> hybrid 75.0% -> hybrid+rerank 86.7%.
- Retrieval recall@5, hand-written 29 answerable (used for tuning): 29.3% baseline -> 91.4% final (MRR 0.80).
  Biggest single gain: contextual chunk headers 62.1% -> 89.7%.
- Single search, 35 questions: correct 89.7%, faithful 96.6%, citation accuracy 89.7%, refusals 6/6,
  wrong refusals 0/29, comparisons 2/4, ~1,440 tokens and ~5.6 s per question.
- Agent, all 35 questions (merged: evals/runs/20261108-123349-merged.json):
  correct 93.1%, faithful 96.4%, citation accuracy 89.7%, correct refusals 5/6,
  wrong refusals 1/29, comparisons 4/4 (100%), ~3,906 tokens and ~21 s per question.
- Graph ablation (agent without Neo4j: evals/runs/20261108-123943-agent-nograph.json):
  comparisons correct 1/4 (25%), latency 56.2 s, tokens ~10,764, false refusals 40%.
  Decisive proof that the Neo4j clause graph is essential for cross-insurer comparisons.

## Remaining steps status
1. DONE: Token optimizations in agent.py (chunk deduplication, valid insurers, out-of-scope early refusal).
2. DONE: Agent evaluated on all 35 questions; tokens dropped 70.5% (~13.2k -> ~3.9k).
3. DONE: All runs merged into evals/runs/20261108-123349-merged.json.
4. DONE: Graph ablation run and verified; Neo4j proved to be critical (4/4 vs 1/4).
5. DONE: Updated README "Answers" section.
6. Ready: Resume and Project Guide updates.

## Lessons worth telling in interviews (all real)
- Two-column PDFs mixed columns; fixed with column-aware block ordering.
- Truncated downloads were silently "repaired" by PyMuPDF into garbled text; now Content-Length + is_repaired checks.
- Insurers share identical tariff wording, so search returned the right clause from the wrong insurer; insurer filter fixed it.
- Removing repeated page headers DROPPED recall (72.4% -> 62.1%) because headers carried product context;
  contextual chunk headers fixed it (-> 89.7%).
- Embedding cache keyed on chunk ids could serve stale vectors; now keyed on a content hash.
- gpt-oss cites as 【1】; parser missed half the citations (31% looked like the score; real 89.7%).
- llama-3.3-70b-versatile was retired from Groq; switched to openai/gpt-oss-120b, judge gpt-oss-20b.
- LLM-generated "verbatim" evidence was not verbatim 15 of 79 times; automatic check rejected them.
- Agent on an unanswerable question: 6 searches, ~27k tokens before refusing.
