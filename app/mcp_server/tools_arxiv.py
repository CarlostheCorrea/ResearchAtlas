"""
arXiv discovery tools for the MCP server.
Phase 3 — Week 9: Custom MCP server — bridges agents to the arXiv API.
No API key required. Official arXiv export endpoint.
"""
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional
import app.database as db

ARXIV_API = "https://export.arxiv.org/api/query"
# ArXiv ToS: minimum 3 s between requests for automated access.
# Identify the client as required by ArXiv usage policy.
_ARXIV_MIN_DELAY = 3.5
_ARXIV_USER_AGENT = "ResearchAtlas/1.0 (academic-project; no-reply@example.com)"
_last_request_time: float = 0.0


def _arxiv_get(url: str, timeout: int = 30, max_retries: int = 4) -> ET.Element:
    """
    Fetch an ArXiv API URL with rate-limit compliance and retry on 429/5xx.

    • Enforces a minimum 3.5 s gap between every request (ArXiv ToS).
    • Sends a descriptive User-Agent (ArXiv ToS requirement).
    • On 429/503: waits 15 s, 30 s, 60 s … before retrying.
    """
    global _last_request_time

    for attempt in range(max_retries):
        # Enforce minimum inter-request gap
        elapsed = time.monotonic() - _last_request_time
        wait = _ARXIV_MIN_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": _ARXIV_USER_AGENT})
            _last_request_time = time.monotonic()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return ET.parse(resp).getroot()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < max_retries - 1:
                # Start at 15 s — 3 s is not enough after a real 429
                backoff = 15 * (2 ** attempt)
                print(f"[tools_arxiv] Rate-limited ({exc.code}), waiting {backoff}s before retry {attempt + 1}/{max_retries - 1}")
                time.sleep(backoff)
            else:
                raise
    raise RuntimeError("ArXiv request failed after all retries")
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _parse_entries(root) -> list[dict]:
    """Parse arXiv Atom feed entries into Paper-compatible dicts."""
    papers = []
    for entry in root.findall("atom:entry", NS):
        try:
            raw_id = (entry.findtext("atom:id", "", NS) or "").strip()
            arxiv_id = raw_id.split("/abs/")[-1].replace("v1", "").replace("v2", "").strip()

            title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", NS) or "").strip().replace("\n", " ")
            published_raw = (entry.findtext("atom:published", "", NS) or "")[:10]

            authors = [
                (a.findtext("atom:name", "", NS) or "").strip()
                for a in entry.findall("atom:author", NS)
            ]

            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            categories = [
                c.get("term", "") for c in entry.findall("arxiv:primary_category", NS)
            ]
            categories += [
                c.get("term", "") for c in entry.findall("atom:category", NS)
                if c.get("term", "") not in categories
            ]

            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published_raw,
                "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                "categories": categories,
            })
        except Exception as e:
            print(f"[tools_arxiv] Failed to parse entry: {e}")

    return papers


def search_papers(
    query: str,
    max_results: int = 20,
    sort_by: str = "relevance",
    start: int = 0,
) -> list[dict]:
    """Search arXiv for papers matching query. Returns list of Paper dicts."""
    try:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"
        root = _arxiv_get(url)
        papers = _parse_entries(root)
        # Cache each paper metadata
        for p in papers:
            try:
                db.cache_paper_metadata(p)
            except Exception:
                pass
        return papers
    except Exception as e:
        print(f"[tools_arxiv] search_papers error: {e}")
        return []


def get_paper_metadata(arxiv_id: str) -> Optional[dict]:
    """Fetch metadata for a single paper by ID. Checks cache first."""
    cached = db.get_cached_paper(arxiv_id)
    if cached:
        return cached

    results = search_papers(f"id:{arxiv_id}", max_results=1)
    return results[0] if results else None


def get_paper_abstract(arxiv_id: str) -> dict:
    """Return abstract for a paper."""
    paper = get_paper_metadata(arxiv_id)
    if not paper:
        return {"arxiv_id": arxiv_id, "abstract": ""}
    return {"arxiv_id": arxiv_id, "abstract": paper["abstract"]}


def shortlist_paper(arxiv_id: str, session_id: str) -> dict:
    """Add a paper to the user's shortlist."""
    db.shortlist_paper(arxiv_id, session_id)
    return {"shortlisted": True, "arxiv_id": arxiv_id}
