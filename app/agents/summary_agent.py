"""
Summary agent — generates structured summary from retrieved chunks.
Uses ONLY retrieved chunks — never invents details.
Follows a fixed 10-section template.

Phase 3 — Week 10: Multi-Agent Workflows. Single responsibility: produce Summary.
"""
import json
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts import SUMMARY_SYSTEM_PROMPT
from app.schemas import Summary
from app.mcp_client import get_mcp_client

# 90-second timeout — gpt-4o with a large context can take 30-60 s
client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)

# Max characters of chunk context to send — prevents overly large prompts
_MAX_CONTEXT_CHARS = 12_000


def run_summary_agent(state: dict) -> dict:
    """Generate a structured summary from paper chunks via MCP retrieval."""
    mcp = get_mcp_client()
    arxiv_id = state.get("selected_arxiv_id")
    if not arxiv_id:
        return {"error": "summary_agent: selected_arxiv_id is missing from state"}

    # Retrieve representative chunks from each major section
    sections_to_cover = ["introduction", "method", "results", "conclusion", "limitations"]
    all_chunks = []
    for section in sections_to_cover:
        chunk_result = mcp.call_tool("get_paper_section", {
            "arxiv_id": arxiv_id,
            "section_hint": section,
        })
        all_chunks.extend(chunk_result.get("chunks", [])[:2])

    # Also retrieve chunks most relevant to the paper's own title
    metadata = mcp.call_tool("get_paper_metadata", {"arxiv_id": arxiv_id})
    # MCP may return an error dict or None — normalise to empty dict
    if not isinstance(metadata, dict) or "error" in metadata:
        metadata = {}

    title_chunks = mcp.call_tool("retrieve_paper_chunks", {
        "arxiv_id": arxiv_id,
        "question": metadata.get("title", arxiv_id),
        "k": 3,
    })
    if isinstance(title_chunks, list):
        all_chunks.extend(title_chunks)

    # Deduplicate by chunk_id
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        cid = c.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(c)

    # Build context string with section labels — cap total size to avoid huge prompts
    context_parts = [
        f"[{c.get('section', 'UNKNOWN').upper()}]\n{c.get('text', '')}"
        for c in unique_chunks
    ]
    context = "\n\n".join(context_parts)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[...context truncated for length...]"

    print(f"[summary_agent] Sending {len(context):,} chars of context to {OPENAI_MODEL}")

    if not context.strip():
        # Fallback: use abstract if no chunks available
        abstract = metadata.get("abstract", "Abstract not available.")
        context = f"[ABSTRACT]\n{abstract}"
        confidence_note = "Based on abstract only — PDF analysis unavailable"
    else:
        confidence_note = "Based on full PDF analysis"

    try:
        print(f"[summary_agent] Calling {OPENAI_MODEL} for summary...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}"
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        summary_data = json.loads(response.choices[0].message.content)
        print(f"[summary_agent] Summary generated ({len(response.choices[0].message.content)} chars)")
    except Exception as e:
        print(f"[summary_agent] LLM call failed: {type(e).__name__}: {e}")
        summary_data = {
            "overview": "Summary generation failed.",
            "problem_addressed": "N/A",
            "main_contribution": "N/A",
            "method": "N/A",
            "datasets_experiments": "N/A",
            "results": "N/A",
            "limitations": "N/A",
            "why_it_matters": "N/A",
            "confidence_note": "Error during generation",
        }

    summary_data["confidence_note"] = summary_data.get("confidence_note", confidence_note)

    revision_note = state.get("revision_note", "")
    if revision_note:
        # Append revision context and regenerate
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}\n\n"
                        f"Revision requested: {revision_note}"
                    )},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            summary_data = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[summary_agent] revision LLM call failed: {e}")

    summary = Summary(
        arxiv_id=arxiv_id,
        title=metadata.get("title") or arxiv_id,
        **{k: v for k, v in summary_data.items() if k in Summary.model_fields and v is not None},
    )

    return {"draft_summary": summary.model_dump()}
