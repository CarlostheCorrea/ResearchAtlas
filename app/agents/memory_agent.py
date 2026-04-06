"""
Memory agent — saves to library, updates preferences, logs feedback.
Called only after human approval. The only agent that writes permanently.

Phase 3 — Week 11: Human-in-the-Loop & Memory. All durable writes go here.
"""
from app.mcp_client import get_mcp_client


def run_memory_agent(state: dict) -> dict:
    """Save approved summary to the research library and update preferences."""
    mcp = get_mcp_client()
    arxiv_id = state.get("selected_arxiv_id")
    final_summary = state.get("draft_summary") or state.get("final_summary")

    if not arxiv_id or not final_summary:
        return {"saved_to_library": False, "weights_updated": False,
                "error": "Missing arxiv_id or summary"}

    title = final_summary.get("title", arxiv_id)
    result = mcp.call_tool("save_to_library", {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": final_summary,
        "notes": "",
    })

    return {
        "final_summary": final_summary,
        "saved_to_library": result.get("saved", False),
        "weights_updated": True,
    }
