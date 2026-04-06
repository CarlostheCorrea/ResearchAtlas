"""
Search API routes — triggers the discovery flow.
Phase 3 — Week 10: Multi-Agent Workflows. Manager → search → pre_filter → ranking.
"""
import uuid
from fastapi import APIRouter
from app.schemas import SearchRequest
from app.graph.build_graph import get_graph

router = APIRouter()

# In-memory session store for filter reports (cleared on restart — ephemeral)
_filter_reports: dict[str, dict] = {}


@router.post("/api/search")
async def search_papers(request: SearchRequest):
    """
    Trigger the discovery flow.
    Creates a new LangGraph session and runs manager → search → pre_filter → ranking.
    Returns ranked results immediately (no interrupt in this flow).
    """
    session_id = str(uuid.uuid4())
    graph = get_graph()

    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id": session_id,
        "user_query": request.query,
        "intent": "discover",
        "max_results": request.max_results,
        "year_from": request.year_from,
        "required_categories": request.categories,
        "user_preferences": {},
    }

    final_state = None
    try:
        for chunk in graph.stream(initial_state, config=config):
            final_state = chunk
    except Exception as e:
        return {"error": str(e), "session_id": session_id}

    # Collect the last node's output
    if final_state:
        last_node_output = list(final_state.values())[-1]
    else:
        last_node_output = {}

    # Get full state snapshot
    snapshot = graph.get_state(config)
    state_values = snapshot.values if snapshot else {}

    filter_report = state_values.get("filter_report", {})
    ranked_results = state_values.get("ranked_results", [])
    error = state_values.get("error")

    if filter_report:
        _filter_reports[session_id] = filter_report

    return {
        "session_id": session_id,
        "ranked_results": ranked_results,
        "filter_report": filter_report,
        "error": error,
    }


@router.get("/api/search/filter-report/{session_id}")
async def get_filter_report(session_id: str):
    """Return the pre-filter report for a session — shows what was dropped and why."""
    report = _filter_reports.get(session_id)
    if not report:
        return {"error": "No filter report found for this session"}
    return report
