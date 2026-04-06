"""
Retrieval agent — fetches top-k relevant chunks from ChromaDB for a question.
Phase 3 — Week 10: Multi-Agent Workflows. Single responsibility: retrieve context.
"""
from app.mcp_client import get_mcp_client


def run_retrieval_agent(state: dict) -> dict:
    """Retrieve the most relevant chunks for the user's question."""
    mcp = get_mcp_client()
    question = state.get("question") or state.get("user_query") or ""
    arxiv_id = state["selected_arxiv_id"]

    chunks = mcp.call_tool("retrieve_paper_chunks", {
        "arxiv_id": arxiv_id,
        "question": question,
        "k": 5,
    })

    if not isinstance(chunks, list):
        chunks = []

    return {"retrieved_chunks": chunks, "question": question}
