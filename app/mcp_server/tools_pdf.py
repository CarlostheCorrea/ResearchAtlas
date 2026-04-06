"""
PDF download and extraction tools for the MCP server.
Phase 3 — Week 9: Custom MCP server — bridges agents to local PDF files via MCP resources.
Agents call these tools; they never import fitz or pdfplumber directly.
"""
import os
import time
import urllib.request
import app.database as db
from app.config import PDF_DIR
from app.rag.parser import extract_text_from_pdf
from app.rag.cleaner import clean_pdf_text as _clean
from app.rag.chunker import chunk_text


def _ensure_pdf_dir():
    os.makedirs(PDF_DIR, exist_ok=True)


def download_pdf(arxiv_id: str) -> dict:
    """Download a paper's PDF from arXiv. Skips re-download if already on disk."""
    _ensure_pdf_dir()
    path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")

    if os.path.exists(path):
        size = os.path.getsize(path)
        return {"arxiv_id": arxiv_id, "path": path, "size_bytes": size, "already_existed": True}

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(2):
        try:
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            if size < 1000:
                os.remove(path)
                raise ValueError(f"Downloaded file too small ({size} bytes)")
            db.cache_pdf(arxiv_id, path, size)
            return {"arxiv_id": arxiv_id, "path": path, "size_bytes": size, "already_existed": False}
        except Exception as e:
            print(f"[tools_pdf] download attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)

    return {"arxiv_id": arxiv_id, "error": "PDF download failed after 2 attempts"}


def extract_pdf_text(arxiv_id: str) -> dict:
    """Extract raw text from a downloaded PDF. Returns text and metadata."""
    path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    result = extract_text_from_pdf(path)
    if result.get("error"):
        return result
    db.cache_extracted_text(
        arxiv_id,
        result["text"],
        result["char_count"],
        result["page_count"],
        result["method"],
    )
    result["arxiv_id"] = arxiv_id
    return result


def clean_pdf_text(arxiv_id: str, raw_text: str) -> dict:
    """Clean extracted PDF text — removes noise, fixes hyphenation, normalizes whitespace."""
    result = _clean(raw_text)
    result["arxiv_id"] = arxiv_id
    return result


def chunk_paper(arxiv_id: str, cleaned_text: str) -> dict:
    """Split cleaned text into overlapping chunks with section metadata."""
    result = chunk_text(arxiv_id, cleaned_text)
    if result["chunks"]:
        db.insert_chunks(result["chunks"])
    return result
