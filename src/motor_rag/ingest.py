"""Parse the policy PDFs into section-aware chunks.

Most Indian motor policy wordings are printed in two columns. Reading a page
line by line mixes the two columns into nonsense, so text blocks are ordered
column by column before chunking.

Each chunk carries the section heading it sits under, so answers can cite
"SBI General, Stand-Alone OD, GENERAL EXCEPTIONS, p.2" instead of a bare chunk id.
"""
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "data" / "sources.csv"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"

MAX_CHARS = 700
MIN_CHARS = 150
OVERLAP_CHARS = 200

# "6. GENERAL EXCEPTIONS", "SECTION II - LIABILITY TO THIRD PARTIES", "EXCLUSIONS"
HEADING = re.compile(
    r"^(?:(?:section|part)\s+[ivx\d]+[\s.:\-–]*)?(?:\d{1,2}\.\s*)?[A-Z][A-Z0-9 ,&/()'’\-–]{3,80}$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    insurer: str
    product: str
    vehicle: str
    doc_type: str
    page: int
    section: str
    text: str


# Single-word headings that are real sections. Other lone upper-case words are
# usually table cells (city names in branch-office lists, column labels).
SECTION_WORDS = {
    "DEFINITIONS", "EXCLUSIONS", "EXCEPTIONS", "CONDITIONS", "COVERAGE", "SCOPE",
    "PREAMBLE", "SCHEDULE", "ENDORSEMENTS", "ANNEXURE", "GRIEVANCES", "OMBUDSMAN",
    "CANCELLATION", "CLAIMS", "DEPRECIATION", "DEDUCTIBLES", "BENEFITS", "LIMITATIONS",
}


def is_heading(line: str) -> bool:
    line = line.strip()
    letters = [c for c in line if c.isalpha()]
    # Headings in these documents are upper case; ordinary sentences are not.
    if not (len(letters) >= 4
            and sum(c.isupper() for c in letters) / len(letters) > 0.8
            and HEADING.match(line)):
        return False
    numbered = re.match(r"^(?:(?:section|part)\s+[ivx\d]+|\d{1,2}\.|[a-z]\))", line, re.IGNORECASE)
    words = re.findall(r"[A-Za-z]+", line)
    return bool(numbered) or len(words) >= 2 or line.upper().strip(" :") in SECTION_WORDS


def page_blocks(page) -> list[str]:
    """Text blocks in reading order.

    Blocks that span the page middle (titles, footers, wide tables) cut the page
    into horizontal bands; inside a band the left column is read before the right.
    """
    mid = page.rect.width / 2
    blocks = sorted(
        (b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()),
        key=lambda b: b[1],
    )
    ordered, band = [], []

    def flush_band():
        band.sort(key=lambda b: (b[2] > mid + 20, b[1]))  # left column first, then by height
        ordered.extend(b[4] for b in band)
        band.clear()

    for b in blocks:
        if b[0] < mid - 20 and b[2] > mid + 20:  # full-width block
            flush_band()
            ordered.append(b[4])
        else:
            band.append(b)
    flush_band()
    return ordered


def clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi")
    text = text.replace("�", "-")  # dashes some PDFs encode without a glyph mapping
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def tail(text: str, size: int = OVERLAP_CHARS) -> str:
    """The end of a chunk to repeat at the start of the next, beginning at a sentence or word."""
    if len(text) <= size:
        return text
    end = text[-size:]
    sentence = re.search(r"[.;:]\s+", end)
    if sentence and sentence.end() < len(end) - 20:
        return end[sentence.end():]
    return end.split(" ", 1)[-1]


def split_long(text: str, limit: int = int(MAX_CHARS * 1.5)) -> list[str]:
    """Split text longer than `limit` at sentence ends, keeping pieces near MAX_CHARS.

    Each piece after the first starts with the tail of the one before, so a
    sentence on the boundary is whole in at least one piece.
    """
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + len(sentence) > MAX_CHARS:
            pieces.append(current)
            current = tail(current)
        current = f"{current} {sentence}".strip()
    if current:
        if pieces and len(current) < MIN_CHARS:
            pieces[-1] += " " + current  # a short tail belongs with the sentences before it
        else:
            pieces.append(current)
    # A single sentence with no breaks (a long table row) is cut hard as a last resort.
    return [p[i:i + limit] for p in pieces for i in range(0, len(p), limit)]


def is_boilerplate(text: str) -> bool:
    """Registered-office footers repeat on every page and only add noise to search."""
    t = text.lower()
    return ("cin:" in t or "irdai reg" in t or "registered office" in t or "regd. office" in t) and len(t) < 900


def header_key(block: str) -> str:
    # Page numbers differ from page to page ("Page 3 of 36"), so ignore digits.
    return re.sub(r"[\d\s]+", "", block.lower())[:120]


def margin_blocks(page, margin: float = 0.12) -> list[str]:
    """Text blocks in the top or bottom `margin` of the page, where headers and footers live."""
    h = page.rect.height
    return [b[4] for b in page.get_text("blocks")
            if b[6] == 0 and b[4].strip() and (b[3] < h * margin or b[1] > h * (1 - margin))]


def running_headers(margins: list[list[str]]) -> set[str]:
    """Margin blocks repeated on at least 30% of a document's pages (min 3) are headers/footers.

    Only margin blocks count: phrases like "Provided always that" repeat just as
    often in the body, and dropping them would cut real conditions.
    """
    if len(margins) < 3:
        return set()
    counts: dict[str, int] = {}
    for blocks in margins:
        for key in {header_key(b) for b in blocks}:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(len(margins) * 0.3))
    # Very short repeats ("Provided always that") are body phrases that landed near a page edge.
    return {key for key, n in counts.items() if n >= threshold and len(key) >= 20}


def chunk_document(path: Path, meta: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "GENERAL"
    buf: list[str] = []
    buf_page = 1

    def flush():
        nonlocal buf
        text = clean(" ".join(buf))
        if text:
            if chunks and len(text) < MIN_CHARS and chunks[-1].section == section:
                chunks[-1].text += " " + text  # fold short tails into the previous chunk
            else:
                for piece in split_long(text):
                    chunks.append(Chunk(
                        chunk_id=f"{meta['doc_id']}#{len(chunks)}",
                        doc_id=meta["doc_id"], insurer=meta["insurer"], product=meta["product"],
                        vehicle=meta["vehicle"], doc_type=meta["doc_type"],
                        page=buf_page, section=section, text=piece,
                    ))
        buf = []

    prev_was_cell = False
    with fitz.open(path) as doc:
        pages = [page_blocks(page) for page in doc]
        margins = [margin_blocks(page) for page in doc]
        running = running_headers(margins)
        for page_no, blocks in enumerate(pages, start=1):
            in_margin = {header_key(b) for b in margins[page_no - 1]}
            for block in blocks:
                key = header_key(block)
                if is_boilerplate(block) or (key in running and key in in_margin):
                    continue
                lines = block.strip().splitlines()
                # Some insurers print the number and title of a heading on separate lines ("2." / "COVERAGE").
                if len(lines) > 1 and re.fullmatch(r"\s*\d{1,2}\.\s*", lines[0]):
                    lines = [f"{lines[0].strip()} {lines[1].strip()}"] + lines[2:]
                    block = "\n".join(lines)
                first_line = lines[0]
                buffered = len(clean(" ".join(buf)))
                # Consecutive tiny blocks are table rows ("ADILABAD / Andhrapradesh"),
                # even when a cell happens to look like an upper-case heading.
                is_cell = len(clean(block)) < 40
                in_table = is_cell and prev_was_cell and not re.match(r"^\s*\d{1,2}\.", block)
                prev_was_cell = is_cell
                # A heading starts a new chunk only once the current one has real content;
                # runs of short headings (add-on lists, table captions) stay together.
                if is_heading(first_line) and not in_table and (buffered == 0 or buffered >= MIN_CHARS):
                    flush()
                    section = clean(first_line).upper()
                    buf_page = page_no
                    rest = lines[1:]
                    if rest:
                        buf.append("\n".join(rest))
                    continue
                if not buf:
                    buf_page = page_no
                buf.append(block)
                if sum(len(b) for b in buf) > MAX_CHARS:
                    flush()
                    # Cut for length, not at a heading: carry the tail over so a
                    # sentence crossing the cut is whole in the next chunk.
                    buf = [tail(chunks[-1].text)] if chunks else []
                    buf_page = page_no
        flush()
    return chunks


def load_sources() -> list[dict]:
    with SOURCES.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ingest() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for meta in load_sources():
        path = RAW / f"{meta['doc_id']}.pdf"
        if not path.exists():
            continue
        try:
            doc_chunks = chunk_document(path, meta)
        except Exception as e:  # a truncated download shouldn't stop the whole run
            print(f"{meta['doc_id']:<28} SKIPPED (unreadable PDF: {e}); re-download it")
            continue
        all_chunks += doc_chunks
        print(f"{meta['doc_id']:<28} {len(doc_chunks):>4} chunks")
    CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"\n{len(all_chunks)} chunks written to {CHUNKS.relative_to(ROOT)}")
    return all_chunks


def load_chunks() -> list[Chunk]:
    with CHUNKS.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


if __name__ == "__main__":
    ingest()
