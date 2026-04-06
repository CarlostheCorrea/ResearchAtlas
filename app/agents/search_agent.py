"""
Search agent — calls the search_papers MCP tool and optionally rewrites queries.
Phase 3 — Week 10: Multi-Agent Workflows. Single responsibility: fetch raw results.
"""
from app.mcp_client import get_mcp_client
from app.config import DEFAULT_MAX_RESULTS


def run_search_agent(state: dict) -> dict:
    """Fetch raw search results from arXiv via MCP tool."""
    mcp = get_mcp_client()
    query = state.get("user_query", "")
    max_results = state.get("max_results", DEFAULT_MAX_RESULTS)

    result = mcp.call_tool("search_papers", {
        "query": query,
        "max_results": max_results,
        "sort_by": "relevance",
    })

    if isinstance(result, list):
        papers = result
    else:
        papers = result if isinstance(result, list) else []

    return {"raw_search_results": papers}
