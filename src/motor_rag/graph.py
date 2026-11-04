"""Clause graph in Neo4j.

  (:Insurer)-[:PUBLISHES]->(:Document)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:NEXT]->(:Chunk)              reading order inside a document
  (:Chunk)-[:MENTIONS]->(:Concept)        "towing", "depreciation", "no claim bonus", ...
  (:Chunk)-[:DEFINES]->(:Clause)          the chunk that carries "IMT. 22. COMPULSORY DEDUCTIBLE"
  (:Chunk)-[:REFERS_TO]->(:Clause)        "subject to IMT 22", "under Section II"

Vector search finds text that *sounds like* the question. The graph answers
questions vector search cannot:
  - "what do all insurers say about towing?"  -> Concept -> Chunks, grouped by Insurer
  - "this clause says 'subject to IMT 22'"    -> follow REFERS_TO to the clause that decides it

Concepts and references are extracted with rules, not an LLM, so the graph is
reproducible and free to rebuild.

  python -m motor_rag.graph build
"""
import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv

from motor_rag.ingest import Chunk, load_chunks, load_sources

load_dotenv()

# Concept -> patterns that signal it. Kept to concepts people actually ask about.
CONCEPTS = {
    "depreciation": r"depreciation",
    "zero depreciation": r"zero depreciation|nil depreciation|depreciation (?:protect|shield|waiver|cover)",
    "idv": r"\bidv\b|insured'?s declared value",
    "no claim bonus": r"no claim bonus|\bncb\b",
    "deductible": r"deductible|\bexcess\b",
    "towing": r"towing|removal to the nearest repairer|protection and removal",
    "roadside assistance": r"road[- ]?side assistance|\brsa\b|emergency assistance",
    "engine protection": r"engine (?:protect|guard|secure)|water ingress|hydrostatic|lubricating oil",
    "consumables": r"consumable",
    "return to invoice": r"return to invoice|invoice (?:price|value)",
    "key replacement": r"key (?:protect|replacement|& lock)|loss of keys?",
    "tyre cover": r"tyre (?:protect|secure|and rim)|tread depth",
    "personal accident": r"personal accident|owner[- ]driver",
    "third party liability": r"third part(?:y|ies)|section ii",
    "theft": r"\btheft\b|burglary|housebreaking",
    "total loss": r"total loss|constructive total loss|\bctl\b",
    "drunk driving": r"intoxicating liquor|influence of (?:alcohol|drugs)",
    "driving licence": r"driving licen[cs]e|effective (?:and valid )?licen[cs]e",
    "geographical area": r"geographical area",
    "overloading": r"overload",
    "cancellation": r"cancel",
    "claims procedure": r"notice of claim|intimat\w+ (?:of )?(?:the )?claim|claim form|cashless",
    "grievance": r"grievance|ombudsman",
    "personal belongings": r"personal (?:belongings|baggage)",
    "electric vehicle": r"electric vehicle|\bev\b|charger|drive motor|hybrid electric",
    "daily allowance": r"daily (?:allowance|conveyance|benefit|cash)|alternate (?:car|travel)",
}
_CONCEPT_RE = {name: re.compile(p, re.IGNORECASE) for name, p in CONCEPTS.items()}

# Policy-internal references: India Motor Tariff endorsements ("IMT. 22") and policy
# sections ("Section II"). Sections are limited to I-X / 1-9 so "Section 56 of the Act" is ignored.
REFERENCE_RE = re.compile(r"\b(IMT\.?\s*\d{1,2}[A-Z]?\b|Section\s+(?:[IVX]{1,4}|[1-9])\b)", re.IGNORECASE)
# A clause's own definition: the reference followed by its upper-case title,
# e.g. "IMT. 22. COMPULSORY DEDUCTIBLE" or "SECTION II - LIABILITY TO THIRD PARTIES".
DEFINITION_RE = re.compile(
    r"\b(IMT\.?\s*\d{1,2}[A-Z]?|SECTION\s+(?:[IVX]{1,4}|[1-9]))\b[\s.:\-–]*(?=[A-Z][A-Z ,/&()'’\-–]{6,})")
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII", "8": "VIII", "9": "IX"}


def concepts_in(text: str) -> list[str]:
    return [name for name, rx in _CONCEPT_RE.items() if rx.search(text)]


def normalize_ref(ref: str) -> str:
    """'IMT. 22' -> 'IMT 22'; 'section 2' and 'Section ii' -> 'SECTION II'."""
    ref = re.sub(r"\s+", " ", ref.replace(".", " ")).strip().upper()
    m = re.fullmatch(r"SECTION (\d)", ref)
    return f"SECTION {ROMAN[m.group(1)]}" if m else ref


def references_in(text: str) -> list[str]:
    return sorted({normalize_ref(m) for m in REFERENCE_RE.findall(text)})


def definitions_in(text: str) -> list[str]:
    """References this text is the definition of (it carries the clause's title)."""
    return sorted({normalize_ref(m) for m in DEFINITION_RE.findall(text)})


@lru_cache
def driver():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "change-me-locally"))
    return GraphDatabase.driver(uri, auth=auth)


def available() -> bool:
    try:
        driver().verify_connectivity()
        return True
    except Exception:
        return False


SCHEMA = [
    "CREATE CONSTRAINT insurer_name IF NOT EXISTS FOR (i:Insurer) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT section_key IF NOT EXISTS FOR (s:Section) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT clause_key IF NOT EXISTS FOR (c:Clause) REQUIRE c.key IS UNIQUE",
]


def chunk_rows(chunks: list[Chunk]) -> list[dict]:
    rows, prev, seen_sections = [], {}, set()
    for c in chunks:
        defines = set(definitions_in(c.text))
        # A section heading is defined by the first chunk under it, not by every chunk that inherits it.
        section_key = f"{c.doc_id}::{c.section}"
        if section_key not in seen_sections:
            seen_sections.add(section_key)
            defines |= set(definitions_in(c.section))
        rows.append({
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "insurer": c.insurer, "page": c.page, "text": c.text,
            "section_key": section_key, "section": c.section,
            "prev": prev.get(c.doc_id), "concepts": concepts_in(f"{c.section} {c.text}"),
            "defines": sorted(defines),
            "refs": [r for r in references_in(c.text) if r not in defines],
        })
        prev[c.doc_id] = c.chunk_id
    return rows


def build(batch: int = 500):
    chunks = load_chunks()
    docs = {c.doc_id for c in chunks}
    sources = [s for s in load_sources() if s["doc_id"] in docs]
    rows = chunk_rows(chunks)
    with driver().session() as s:
        s.run("MATCH (n) DETACH DELETE n")  # full rebuild: the graph is derived data
        for stmt in SCHEMA:
            s.run(stmt)
        s.run("""
            UNWIND $docs AS d
            MERGE (i:Insurer {name: d.insurer})
            MERGE (doc:Document {doc_id: d.doc_id})
              SET doc.product = d.product, doc.vehicle = d.vehicle, doc.doc_type = d.doc_type, doc.url = d.url
            MERGE (i)-[:PUBLISHES]->(doc)
        """, docs=sources)
        for i in range(0, len(rows), batch):
            s.run("""
                UNWIND $rows AS r
                MATCH (doc:Document {doc_id: r.doc_id})
                MERGE (sec:Section {key: r.section_key})
                  SET sec.name = r.section, sec.doc_id = r.doc_id
                MERGE (doc)-[:HAS_SECTION]->(sec)
                CREATE (c:Chunk {chunk_id: r.chunk_id, doc_id: r.doc_id, insurer: r.insurer,
                                 page: r.page, text: r.text, section: r.section})
                MERGE (sec)-[:HAS_CHUNK]->(c)
                WITH c, r
                UNWIND r.concepts AS concept
                MERGE (k:Concept {name: concept})
                MERGE (c)-[:MENTIONS]->(k)
            """, rows=rows[i:i + batch])
        s.run("""
            UNWIND $rows AS r
            WITH r WHERE r.prev IS NOT NULL
            MATCH (a:Chunk {chunk_id: r.prev}), (b:Chunk {chunk_id: r.chunk_id})
            MERGE (a)-[:NEXT]->(b)
        """, rows=rows)
        # Clauses are per document: "IMT 22" in ICICI's wording is ICICI's IMT 22.
        s.run("""
            UNWIND $rows AS r
            UNWIND r.defines AS ref
            MATCH (c:Chunk {chunk_id: r.chunk_id})
            MERGE (k:Clause {key: r.doc_id + '::' + ref}) SET k.ref = ref, k.doc_id = r.doc_id
            MERGE (c)-[:DEFINES]->(k)
        """, rows=rows)
        # Only references to clauses this document actually defines become edges.
        s.run("""
            UNWIND $rows AS r
            UNWIND r.refs AS ref
            MATCH (k:Clause {key: r.doc_id + '::' + ref})
            MATCH (c:Chunk {chunk_id: r.chunk_id})
            MERGE (c)-[:REFERS_TO]->(k)
        """, rows=rows)
        counts = s.run("""
            MATCH (n) WITH labels(n)[0] AS label, count(*) AS n RETURN label, n ORDER BY label
        """).data()
        rels = s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type").data()
    print("nodes:", {r["label"]: r["n"] for r in counts})
    print("relationships:", {r["type"]: r["n"] for r in rels})


# ---------- queries used by the agent ----------

def related_clauses(chunk_id: str, limit: int = 4) -> list[str]:
    """Chunks a given chunk depends on: definitions of clauses it refers to, then the next chunk."""
    with driver().session() as s:
        refs = s.run("""
            MATCH (:Chunk {chunk_id: $id})-[:REFERS_TO]->(:Clause)<-[:DEFINES]-(t:Chunk)
            RETURN DISTINCT t.chunk_id AS id ORDER BY id LIMIT $limit
        """, id=chunk_id, limit=limit).value()
        nxt = s.run("MATCH (:Chunk {chunk_id: $id})-[:NEXT]->(n:Chunk) RETURN n.chunk_id",
                    id=chunk_id).value()
    return list(dict.fromkeys(refs + nxt))[:limit + 1]


def concept_by_insurer(concept: str, insurers: list[str] | None = None, vehicle: str | None = None) -> dict[str, list[str]]:
    """Every chunk that mentions a concept, grouped by insurer.

    This is a candidate set, not a ranking: the caller reranks each insurer's
    candidates against the actual question.
    """
    with driver().session() as s:
        rows = s.run("""
            MATCH (i:Insurer)-[:PUBLISHES]->(d:Document)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
                  -[:MENTIONS]->(:Concept {name: $concept})
            WHERE ($insurers IS NULL OR i.name IN $insurers) AND ($vehicle IS NULL OR d.vehicle = $vehicle)
            RETURN i.name AS insurer, collect(c.chunk_id) AS ids
        """, concept=concept, insurers=insurers, vehicle=vehicle).data()
    return {r["insurer"]: r["ids"] for r in rows}


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
