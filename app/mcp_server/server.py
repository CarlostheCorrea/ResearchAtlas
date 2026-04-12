"""
MCP server for the arXiv Research Assistant.
Week 9 deliverable: custom MCP server bridging LLM agents to arXiv, PDF files,
ChromaDB vector store, and SQLite research library.

Run standalone: uvicorn app.mcp_server.server:app --port 8001

All tool implementations live in the tools_* modules — this file only wires them up
into FastAPI endpoints that agents can call via HTTP.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import traceback
import app.database as db
from app.mcp_server.tools_arxiv import (
    search_papers, get_paper_metadata, get_paper_abstract, shortlist_paper
)
from app.mcp_server.tools_pdf import (
    download_pdf, extract_pdf_text, clean_pdf_text, chunk_paper
)
from app.mcp_server.tools_rag import (
    index_paper, retrieve_paper_chunks, get_paper_section
)
from app.mcp_server.tools_memory import (
    save_to_library, list_saved_papers, save_user_preference,
    get_user_preferences, log_feedback, get_feedback_history,
    create_pending_review, resolve_review
)

app = FastAPI(title="arXiv Research MCP Server", version="1.0.0")


# ── Health + tool listing ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "server": "arxiv-research-mcp"}


@app.get("/tools")
def list_tools():
    """List all registered MCP tools."""
    return {
        "tools": [
            # Discovery
            {"name": "search_papers", "group": "arxiv", "type": "tool"},
            {"name": "get_paper_metadata", "group": "arxiv", "type": "resource"},
            {"name": "get_paper_abstract", "group": "arxiv", "type": "resource"},
            {"name": "shortlist_paper", "group": "arxiv", "type": "tool"},
            # PDF
            {"name": "download_pdf", "group": "pdf", "type": "tool"},
            {"name": "extract_pdf_text", "group": "pdf", "type": "tool"},
            {"name": "clean_pdf_text", "group": "pdf", "type": "tool"},
            {"name": "chunk_paper", "group": "pdf", "type": "tool"},
            # RAG
            {"name": "index_paper", "group": "rag", "type": "tool"},
            {"name": "retrieve_paper_chunks", "group": "rag", "type": "tool"},
            {"name": "get_paper_section", "group": "rag", "type": "resource"},
            # Memory
            {"name": "save_to_library", "group": "memory", "type": "tool"},
            {"name": "list_saved_papers", "group": "memory", "type": "resource"},
            {"name": "save_user_preference", "group": "memory", "type": "tool"},
            {"name": "get_user_preferences", "group": "memory", "type": "resource"},
            {"name": "log_feedback", "group": "memory", "type": "tool"},
            {"name": "get_feedback_history", "group": "memory", "type": "resource"},
            {"name": "create_pending_review", "group": "memory", "type": "tool"},
            {"name": "resolve_review", "group": "memory", "type": "tool"},
        ]
    }


# ── Tool dispatcher ────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    tool: str
    params: dict = {}


@app.post("/call")
def call_tool(call: ToolCall) -> Any:
    """Generic tool dispatcher. Agents POST { tool: name, params: {...} }."""
    name = call.tool
    p = call.params

    dispatch = {
        "search_papers":          lambda: search_papers(
            p["query"], p.get("max_results", 20), p.get("sort_by", "relevance"),
            p.get("start", 0), p.get("year_from"), p.get("search_mode", "topic"),
        ),
        "get_paper_metadata":     lambda: get_paper_metadata(p["arxiv_id"]),
        "get_paper_abstract":     lambda: get_paper_abstract(p["arxiv_id"]),
        "shortlist_paper":        lambda: shortlist_paper(p["arxiv_id"], p["session_id"]),
        "download_pdf":           lambda: download_pdf(p["arxiv_id"]),
        "extract_pdf_text":       lambda: extract_pdf_text(p["arxiv_id"]),
        "clean_pdf_text":         lambda: clean_pdf_text(p["arxiv_id"], p["raw_text"]),
        "chunk_paper":            lambda: chunk_paper(p["arxiv_id"], p["cleaned_text"]),
        "index_paper":            lambda: index_paper(p["arxiv_id"]),
        "retrieve_paper_chunks":  lambda: retrieve_paper_chunks(
            p["arxiv_id"], p["question"], p.get("k", 5)
        ),
        "get_paper_section":      lambda: get_paper_section(p["arxiv_id"], p["section_hint"]),
        "save_to_library":        lambda: save_to_library(
            p["arxiv_id"], p["title"], p["summary"], p.get("notes", "")
        ),
        "list_saved_papers":      lambda: list_saved_papers(),
        "save_user_preference":   lambda: save_user_preference(p["topic"], p["weight"]),
        "get_user_preferences":   lambda: get_user_preferences(),
        "log_feedback":           lambda: log_feedback(
            p["arxiv_id"], p["rating"], p.get("comment", "")
        ),
        "get_feedback_history":   lambda: get_feedback_history(p.get("limit", 50)),
        "create_pending_review":  lambda: create_pending_review(p["arxiv_id"], p["draft_summary"]),
        "resolve_review":         lambda: resolve_review(
            p["summary_id"], p["decision"], p.get("revision_note", "")
        ),
    }

    handler = dispatch.get(name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")

    try:
        return handler()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[mcp_server] ERROR in tool '{name}':\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
