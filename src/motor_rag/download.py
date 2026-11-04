"""Download the policy PDFs listed in data/sources.csv into data/raw/.

Each file must open as a PDF with extractable text. That rejects HTML error
pages saved under a .pdf name and downloads truncated by a dropped connection
(those still start with %PDF, so a magic-byte check alone lets them through).
"""
import csv
import sys
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (motor-rag research downloader)"}


def load_sources():
    with SOURCES.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pdf(data: bytes) -> None:
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages (truncated download?)")
        if not any(page.get_text().strip() for page in doc):
            raise ValueError("PDF has no text layer (scanned image); needs OCR")


def is_valid(path: Path) -> bool:
    try:
        check_pdf(path.read_bytes())
        return True
    except Exception:
        return False


def download_all(force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        for src in load_sources():
            target = RAW / f"{src['doc_id']}.pdf"
            if target.exists() and not force and is_valid(target):
                ok += 1
                continue
            try:
                resp = client.get(src["url"])
                resp.raise_for_status()
                expected = resp.headers.get("content-length")
                if expected and "content-encoding" not in resp.headers and int(expected) != len(resp.content):
                    raise ValueError(f"truncated: got {len(resp.content)} of {expected} bytes")
                check_pdf(resp.content)
                target.write_bytes(resp.content)
                ok += 1
                print(f"ok      {src['doc_id']}")
            except Exception as e:  # report and keep going; one dead link shouldn't stop the run
                failed.append(src["doc_id"])
                print(f"FAILED  {src['doc_id']}: {e}")
    print(f"\n{ok} downloaded, {len(failed)} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if download_all(force="--force" in sys.argv) else 0)
