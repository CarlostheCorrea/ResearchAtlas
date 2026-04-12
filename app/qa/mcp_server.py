"""
Q/A-focused MCP server for ResearchAtlas.

This server is used internally by the app's Q/A experience over stdio.
"""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import json
import os
import re
from pathlib import Path
import sys
from typing import Any

import fitz
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from openai import BadRequestError, OpenAI

import app.database as db
from app.config import (
    DB_PATH,
    OPENAI_API_KEY,
    OPENAI_IMAGE_MODEL,
    OPENAI_MODEL,
    PDF_DIR,
    QA_ASSETS_DIR,
    VECTORSTORE_DIR,
)
from app.mcp_server.tools_arxiv import get_paper_abstract, get_paper_metadata
from app.mcp_server.tools_pdf import clean_pdf_text, download_pdf, extract_pdf_text, chunk_paper
from app.mcp_server.tools_rag import get_paper_section, index_paper, retrieve_paper_chunks
from app.qa.assets import ensure_assets_root, ensure_session_dir, record_asset
from app.rag.vectorstore import is_collection_compatible

server = FastMCP(
    name="researchatlas-qa-mcp",
    instructions=(
        "Q/A MCP server for ResearchAtlas. It prepares paper context, finds evidence, "
        "compares sections, creates citations, and generates downloadable answer assets."
    ),
)
client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)


def _ensure_runtime() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    os.makedirs(QA_ASSETS_DIR, exist_ok=True)
    db.init_db()
    ensure_assets_root()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _safe_call(fn, *args, **kwargs):
    """Route non-protocol stdout from underlying helpers to stderr for stdio MCP safety."""
    with redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower())).strip()


def _snippet_from_text(text: str, query: str, limit: int = 320) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text or "") if p.strip()]
    if not parts:
        return (text or "")[:limit].strip()

    query_tokens = {tok for tok in _normalize(query).split() if len(tok) > 2}
    best = parts[0]
    best_score = -1
    for part in parts:
        part_tokens = set(_normalize(part).split())
        overlap = len(query_tokens & part_tokens)
        if overlap > best_score:
            best = part
            best_score = overlap

    return best[:limit].strip()


def _evidence_from_chunks(arxiv_id: str, query: str, chunks: list[dict], max_items: int = 4) -> list[dict]:
    evidence = []
    seen = set()
    for chunk in chunks:
        quote = _snippet_from_text(chunk.get("text", ""), query)
        key = (chunk.get("page"), quote.lower())
        if not quote or key in seen:
            continue
        seen.add(key)
        evidence.append({
            "arxiv_id": arxiv_id,
            "page": chunk.get("page", 1),
            "section": chunk.get("section", "Unknown"),
            "quote": quote,
            "source_chunk": chunk.get("text", "")[:800],
        })
        if len(evidence) >= max_items:
            break
    return evidence


def _wrap_text(text: str, width: int = 88) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += f" {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _slugify(text: str, fallback: str = "document") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return cleaned or fallback


def _export_suffix(question: str) -> str:
    lowered = (question or "").lower()
    if "workflow" in lowered:
        return "workflow"
    if any(token in lowered for token in ("summary", "summarize", "overview")):
        return "summary"
    if any(token in lowered for token in ("evidence", "citation", "cite", "proof")):
        return "evidence"
    if "compare" in lowered:
        return "comparison"
    return "qa-response"


def _export_filename(title: str, question: str, extension: str) -> str:
    title_slug = _slugify(title, fallback="researchatlas")
    suffix = _export_suffix(question)
    stem = f"{title_slug}-{suffix}"
    return f"{stem[:80].rstrip('-')}.{extension}"


def _markdown_heading(question: str) -> str:
    lowered = (question or "").lower()
    if any(token in lowered for token in ("key findings", "main findings", "findings")):
        return "Key Findings"
    if any(token in lowered for token in ("summary", "summarize", "overview")):
        return "Summary"
    if "workflow" in lowered:
        return "Workflow"
    if any(token in lowered for token in ("evidence", "citation", "cite", "proof")):
        return "Evidence Summary"
    return "Response"


def _extract_points(text: str) -> list[str]:
    numbered_matches = re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+(.*?)(?=(?:\n\s*(?:\d+[.)]|[-*•])\s+)|\Z)", text or "", flags=re.S)
    if numbered_matches:
        return [re.sub(r"\s+", " ", item).strip() for item in numbered_matches if item.strip()]

    cleaned = (text or "").strip()
    if not cleaned:
        return [""]

    intro_match = re.match(r"^(.*?:)\s+(.*)$", cleaned, flags=re.S)
    if intro_match and any(token in intro_match.group(1).lower() for token in ("findings", "follows", "list")):
        cleaned = intro_match.group(2).strip()

    inline_numbered_matches = re.findall(r"(?:^|\s)\d+[.)]\s+(.*?)(?=(?:\s+\d+[.)]\s+)|\Z)", cleaned, flags=re.S)
    if inline_numbered_matches:
        return [re.sub(r"\s+", " ", item).strip() for item in inline_numbered_matches if item.strip()]

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
    return sentence_parts or [cleaned]


def _parse_export_style(question: str) -> dict[str, Any]:
    lowered = (question or "").lower()

    return {
        "heading": _markdown_heading(question),
        "list_style": any(token in lowered for token in ("list format", "bullet", "bulleted", "bullet list", "make a list", "as a list", "key findings")),
    }


def _build_markdown_document(title: str, question: str, answer: str, citations: list[dict]) -> str:
    style = _parse_export_style(question)
    lines = [f"# {title}", "", f"## {style['heading']}", ""]
    if style["list_style"]:
        for index, point in enumerate(_extract_points(answer), start=1):
            lines.append(f"{index}. {point}")
    else:
        lines.append(answer.strip())

    if citations:
        lines += ["", "## Citations"]
        if style["list_style"]:
            for item in citations:
                section = item.get("section", "Unknown")
                page = item.get("page", "?")
                quote = item.get("quote", "").strip()
                lines.append(f"- **{section} (page {page})**: {quote}")
        else:
            for item in citations:
                section = item.get("section", "Unknown")
                page = item.get("page", "?")
                quote = item.get("quote", "").strip()
                lines += ["", f"### {section} (page {page})", "", quote]
    return "\n".join(lines).strip() + "\n"


def _write_wrapped_lines(page, x: int, y: int, text: str, fontsize: int, width_chars: int, color=(0, 0, 0), bullet: bool = False) -> int:
    lines = _wrap_text(text, width=width_chars)
    for line in lines:
        prefix = "- " if bullet else ""
        page.insert_text((x, y), f"{prefix}{line}", fontsize=fontsize, fontname="helv", color=color)
        y += fontsize + 5
    return y


def _write_pdf(path: Path, title: str, question: str, answer: str, citations: list[dict]) -> None:
    style = _parse_export_style(question)
    doc = fitz.open()
    page = doc.new_page()
    y = 48

    for line in _wrap_text(title, width=42):
        page.insert_text((48, y), line, fontsize=20, fontname="helv", color=(0, 0, 0))
        y += 26
    y += 8

    page.insert_text((48, y), style["heading"], fontsize=14, fontname="helv", color=(0.2, 0.2, 0.2))
    y += 22

    if style["list_style"]:
        for index, point in enumerate(_extract_points(answer), start=1):
            y = _write_wrapped_lines(page, 60, y, f"{index}. {point}", fontsize=11, width_chars=82)
            y += 4
            if y > 760:
                page = doc.new_page()
                y = 48
    else:
        y = _write_wrapped_lines(page, 48, y, answer, fontsize=11, width_chars=90)

    if citations:
        y += 10
        if y > 760:
            page = doc.new_page()
            y = 48
        page.insert_text((48, y), "Citations", fontsize=14, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 22
        for item in citations:
            section = item.get("section", "Unknown")
            page_num = item.get("page", "?")
            quote = item.get("quote", "").strip()
            y = _write_wrapped_lines(page, 48, y, f"{section} (page {page_num})", fontsize=11, width_chars=88, color=(0.18, 0.36, 0.73))
            y = _write_wrapped_lines(page, 60, y, quote, fontsize=10, width_chars=82)
            y += 8
            if y > 760:
                page = doc.new_page()
                y = 48

    doc.save(path)
    doc.close()


def _llm_compare(arxiv_id: str, section_a: str, section_b: str, content_a: str, content_b: str, focus: str | None) -> dict:
    prompt = (
        f"Paper: {arxiv_id}\n"
        f"Section A: {section_a}\n{content_a}\n\n"
        f"Section B: {section_b}\n{content_b}\n\n"
        f"Focus: {focus or 'general comparison'}"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Compare two sections from the same research paper using only the supplied excerpts. "
                    "Return JSON with keys summary, section_a_points, and section_b_points."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


@server.tool(
    name="ensure_paper_context",
    description=(
        "Prepare a paper for evidence-based Q/A by ensuring its PDF is downloaded, text is extracted, "
        "chunks are stored, and the semantic index exists. Use this before evidence lookup when the paper "
        "may not have been analyzed yet."
    ),
)
def ensure_paper_context(arxiv_id: str) -> str:
    _ensure_runtime()
    if db.is_paper_indexed(arxiv_id) and is_collection_compatible(arxiv_id):
        pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
        return _json({
            "arxiv_id": arxiv_id,
            "ready": True,
            "already_indexed": True,
            "pdf_path": pdf_path if os.path.exists(pdf_path) else None,
        })

    download = _safe_call(download_pdf, arxiv_id)
    if download.get("error"):
        raise ToolError(download["error"])

    extract = _safe_call(extract_pdf_text, arxiv_id)
    if extract.get("error") or not extract.get("text"):
        raise ToolError(extract.get("error", "PDF extraction failed"))

    cleaned = _safe_call(clean_pdf_text, arxiv_id, extract["text"])
    chunks = _safe_call(chunk_paper, arxiv_id, cleaned.get("cleaned_text", ""))
    index = _safe_call(index_paper, arxiv_id)
    if index.get("error"):
        raise ToolError(index["error"])

    return _json({
        "arxiv_id": arxiv_id,
        "ready": True,
        "already_indexed": False,
        "pdf_path": download.get("path"),
        "page_count": extract.get("page_count"),
        "chunk_count": chunks.get("chunk_count", len(chunks.get("chunks", []))),
        "chunks_indexed": index.get("chunks_indexed", 0),
    })


@server.tool(
    name="retrieve_paper_chunks",
    description=(
        "Retrieve semantically relevant paper chunks for a question. Use this for general Q/A when you "
        "need the most relevant excerpts from the selected paper."
    ),
)
def retrieve_paper_chunks_tool(arxiv_id: str, question: str, k: int = 5) -> str:
    chunks = _safe_call(retrieve_paper_chunks, arxiv_id, question, k)
    return _json({"arxiv_id": arxiv_id, "question": question, "chunks": chunks})


@server.tool(
    name="find_evidence",
    description=(
        "Find exact supporting evidence in the selected paper for a user question or claim. Returns short "
        "quote-level evidence items with page numbers, section names, and text suitable for PDF highlighting."
    ),
)
def find_evidence(arxiv_id: str, question: str, max_quotes: int = 4) -> str:
    chunks = _safe_call(retrieve_paper_chunks, arxiv_id, question, k=max(max_quotes * 2, 6))
    evidence = _evidence_from_chunks(arxiv_id, question, chunks, max_items=max_quotes)
    return _json({
        "arxiv_id": arxiv_id,
        "question": question,
        "evidence": evidence,
        "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
    })


@server.tool(
    name="cite_evidence",
    description=(
        "Return quote-level citations for a specific claim or answer draft using the selected paper as the "
        "only source of truth."
    ),
)
def cite_evidence(arxiv_id: str, claim: str, max_quotes: int = 4) -> str:
    chunks = _safe_call(retrieve_paper_chunks, arxiv_id, claim, k=max(max_quotes * 2, 6))
    evidence = _evidence_from_chunks(arxiv_id, claim, chunks, max_items=max_quotes)
    return _json({
        "arxiv_id": arxiv_id,
        "claim": claim,
        "citations": evidence,
        "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
    })


@server.tool(
    name="compare_sections",
    description=(
        "Compare two named sections inside the current paper, such as methods versus results or introduction "
        "versus conclusion. Use this when the user explicitly asks for a comparison."
    ),
)
def compare_sections(arxiv_id: str, section_a: str, section_b: str, focus: str | None = None) -> str:
    chunks_a = _safe_call(get_paper_section, arxiv_id, section_a).get("chunks", [])[:3]
    chunks_b = _safe_call(get_paper_section, arxiv_id, section_b).get("chunks", [])[:3]
    if not chunks_a or not chunks_b:
        raise ToolError("Could not find both requested sections in the paper.")

    content_a = "\n\n".join(c.get("text", "")[:700] for c in chunks_a)
    content_b = "\n\n".join(c.get("text", "")[:700] for c in chunks_b)
    comparison = _llm_compare(arxiv_id, section_a, section_b, content_a, content_b, focus)
    citations = _evidence_from_chunks(arxiv_id, focus or f"{section_a} {section_b}", chunks_a + chunks_b, max_items=4)
    return _json({
        "arxiv_id": arxiv_id,
        "section_a": section_a,
        "section_b": section_b,
        "comparison": comparison,
        "citations": citations,
        "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
    })


@server.tool(
    name="create_md",
    description=(
        "Create a downloadable Markdown document for the current Q/A response, including the question, "
        "answer, and supporting citations."
    ),
)
def create_md(session_id: str, title: str, question: str, answer: str, citations_json: str = "[]") -> str:
    ensure_session_dir(session_id)
    filename = _export_filename(title, question, "md")
    asset = record_asset(session_id, filename, "markdown", "Download Markdown")
    citations = json.loads(citations_json or "[]")
    markdown = _build_markdown_document(title, question, answer, citations)
    Path(asset["path"]).write_text(markdown, encoding="utf-8")
    return _json(asset)


@server.tool(
    name="create_pdf",
    description=(
        "Create a downloadable PDF document for the current Q/A response, including the question, answer, "
        "and supporting citations."
    ),
)
def create_pdf(session_id: str, title: str, question: str, answer: str, citations_json: str = "[]") -> str:
    ensure_session_dir(session_id)
    filename = _export_filename(title, question, "pdf")
    asset = record_asset(session_id, filename, "pdf", "Download PDF")
    citations = json.loads(citations_json or "[]")
    _write_pdf(Path(asset["path"]), title, question, answer, citations)
    return _json(asset)


@server.tool(
    name="create_graphic",
    description=(
        "Generate a downloadable image, chart, diagram, or workflow illustration using the OpenAI image API. "
        "Use this when the user explicitly asks for an image, graphic, chart, or visual explanation."
    ),
)
def create_graphic(session_id: str, prompt: str, title: str = "Generated Graphic") -> str:
    ensure_session_dir(session_id)
    asset = record_asset(session_id, "graphic.png", "image", title)

    # Always emphasise correct spelling in the prompt
    base_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Every word of text rendered in the image must be spelled correctly. "
        "Use simple, common English words only. Double-check all labels before rendering."
    )

    image_bytes = _generate_image_bytes(base_prompt, asset)
    asset["revised_prompt"] = asset.get("revised_prompt")

    # Validate text via GPT-4o vision; retry once if misspellings are found
    retries = 0
    corrections: list[str] = []
    for _attempt in range(1):
        misspellings = _validate_image_text(image_bytes)
        if not misspellings:
            break
        corrections.extend(misspellings)
        retries += 1
        correction_hint = "; ".join(misspellings)
        retry_prompt = (
            f"{base_prompt}\n\n"
            f"Previous version contained spelling errors: {correction_hint}. "
            "Regenerate with all text spelled correctly."
        )
        image_bytes = _generate_image_bytes(retry_prompt, asset)

    Path(asset["path"]).write_bytes(image_bytes)
    asset["validation"] = {
        "retries": retries,
        "corrections": corrections,
        "ok": retries == 0,
    }
    return _json(asset)


def _generate_image_bytes(prompt: str, asset: dict) -> bytes:
    """Call the image API and return raw PNG bytes."""
    try:
        response = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
        )
    except BadRequestError as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise ToolError(f"Image generation failed: {message}") from exc
    except Exception as exc:
        raise ToolError(f"Image generation failed: {exc}") from exc

    if not response.data:
        raise ToolError("Image generation returned no image data.")

    image = response.data[0]
    asset["revised_prompt"] = getattr(image, "revised_prompt", None)

    if getattr(image, "b64_json", None):
        return base64.b64decode(image.b64_json)
    if getattr(image, "url", None):
        downloaded = httpx.get(image.url, timeout=60.0)
        downloaded.raise_for_status()
        return downloaded.content
    raise ToolError("Image generation returned neither b64_json nor a URL.")


def _validate_image_text(image_bytes: bytes) -> list[str]:
    """Use GPT-4o vision to detect misspelled words in the image.

    Returns a list of correction strings like ``"wildlif -> wildlife"`` or an
    empty list if everything looks correct.
    """
    b64 = base64.b64encode(image_bytes).decode()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "List every word or phrase of text visible in this image, exactly as written. "
                                "Then identify any misspelled English words. "
                                "Reply ONLY as JSON: "
                                "{\"misspellings\": [\"wrong -> correct\", ...]}. "
                                "If there are no misspellings return {\"misspellings\": []}."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return data.get("misspellings", [])
    except Exception:
        # Validation is best-effort; never block image delivery
        return []


@server.resource(
    "researchatlas://paper/{arxiv_id}/metadata",
    name="paper_metadata",
    description="Metadata for a selected paper, including title, authors, categories, and PDF URL.",
    mime_type="application/json",
)
def paper_metadata(arxiv_id: str) -> str:
    _ensure_runtime()
    paper = _safe_call(get_paper_metadata, arxiv_id)
    if not paper:
        raise ResourceError(f"No metadata found for {arxiv_id}")
    return _json(paper)


@server.resource(
    "researchatlas://paper/{arxiv_id}/abstract",
    name="paper_abstract",
    description="Abstract text for a selected paper.",
    mime_type="application/json",
)
def paper_abstract(arxiv_id: str) -> str:
    _ensure_runtime()
    abstract = _safe_call(get_paper_abstract, arxiv_id)
    if not abstract:
        raise ResourceError(f"No abstract found for {arxiv_id}")
    return _json(abstract)


def main() -> None:
    _ensure_runtime()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
