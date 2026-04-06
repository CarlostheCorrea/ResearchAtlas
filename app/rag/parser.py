"""
PyMuPDF PDF text extraction with pdfplumber fallback.
Phase 3 — Week 9: MCP Server (called by extract_pdf_text MCP tool).
Pure function — no side effects, easy to test.
"""
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Per-method extraction timeout in seconds (some PDFs hang PyMuPDF indefinitely)
_EXTRACT_TIMEOUT = 60


def _pymupdf_extract(pdf_path: str) -> dict:
    import fitz
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return {"text": text, "page_count": page_count}


def _pdfplumber_extract(pdf_path: str) -> dict:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    return {"text": text, "page_count": page_count}


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract raw text from a PDF file.
    Tries PyMuPDF first (60 s timeout); falls back to pdfplumber (60 s timeout).
    Returns: { text, char_count, page_count, method }
    """
    if not os.path.exists(pdf_path):
        return {"text": "", "char_count": 0, "page_count": 0, "method": "error",
                "error": f"File not found: {pdf_path}"}

    # Try PyMuPDF with timeout
    print(f"[parser] Extracting text with PyMuPDF: {os.path.basename(pdf_path)}")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_pymupdf_extract, pdf_path)
            result = future.result(timeout=_EXTRACT_TIMEOUT)
        if len(result["text"]) >= 100:
            print(f"[parser] PyMuPDF OK — {len(result['text'])} chars, {result['page_count']} pages")
            return {
                "text": result["text"],
                "char_count": len(result["text"]),
                "page_count": result["page_count"],
                "method": "pymupdf",
            }
        print(f"[parser] PyMuPDF returned too little text ({len(result['text'])} chars), trying fallback")
    except FuturesTimeoutError:
        print(f"[parser] PyMuPDF timed out after {_EXTRACT_TIMEOUT}s, trying pdfplumber fallback")
    except Exception as e:
        print(f"[parser] PyMuPDF failed: {e}")

    # Fallback to pdfplumber with timeout
    print("[parser] Extracting text with pdfplumber...")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_pdfplumber_extract, pdf_path)
            result = future.result(timeout=_EXTRACT_TIMEOUT)
        print(f"[parser] pdfplumber OK — {len(result['text'])} chars, {result['page_count']} pages")
        return {
            "text": result["text"],
            "char_count": len(result["text"]),
            "page_count": result["page_count"],
            "method": "pdfplumber",
        }
    except FuturesTimeoutError:
        print(f"[parser] pdfplumber also timed out after {_EXTRACT_TIMEOUT}s")
        return {"text": "", "char_count": 0, "page_count": 0, "method": "failed",
                "error": f"PDF extraction timed out after {_EXTRACT_TIMEOUT}s — PDF may be malformed or encrypted"}
    except Exception as e:
        print(f"[parser] pdfplumber failed: {e}")
        return {"text": "", "char_count": 0, "page_count": 0, "method": "failed",
                "error": str(e)}
