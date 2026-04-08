"""
Analysis API routes — triggers the summary/analyze flow and review checkpoints.
Phase 3 — Week 11: Human-in-the-Loop. Interrupt handling and session polling.
"""
import uuid
import traceback
from fastapi import APIRouter, BackgroundTasks
from app.schemas import ChatRequest
from app.graph.build_graph import get_graph
import app.database as db
from app.rag.vectorstore import is_collection_compatible

router = APIRouter()

# In-memory session state tracker
_sessions: dict[str, dict] = {}


def _extract_interrupt_payload(snapshot) -> dict:
    """
    Extract the interrupt() payload from a graph snapshot.
    LangGraph stores it in snapshot.tasks[i].interrupts[j].value
    """
    try:
        for task in getattr(snapshot, "tasks", []):
            interrupts = getattr(task, "interrupts", [])
            for intr in interrupts:
                if hasattr(intr, "value") and isinstance(intr.value, dict):
                    return intr.value
                elif isinstance(intr, dict):
                    return intr
    except Exception as e:
        print(f"[routes_chat] _extract_interrupt_payload error: {e}")
    return {}


def _run_graph_background(session_id: str, initial_state: dict, config: dict):
    """Run graph in background thread and update session state."""
    graph = get_graph()
    _sessions[session_id]["status"] = "running"

    try:
        for chunk in graph.stream(initial_state, config=config):
            node_name = list(chunk.keys())[0]
            _sessions[session_id]["last_node"] = node_name

        snapshot = graph.get_state(config)
        state_vals = snapshot.values if snapshot else {}

        # Check for interrupt() pause — payload lives in snapshot.tasks[].interrupts
        interrupt_payload = _extract_interrupt_payload(snapshot)
        if interrupt_payload:
            _sessions[session_id].update({
                "status": "interrupted",
                "interrupt_payload": interrupt_payload,
            })
            return

        # Also check snapshot.next (for interrupt_before style)
        if snapshot and snapshot.next:
            next_node = snapshot.next[0] if snapshot.next else None
            if next_node in ("human_gate_before_download", "human_gate_before_save"):
                _sessions[session_id].update({
                    "status": "interrupted",
                    "interrupt_payload": {},
                    "next_node": next_node,
                })
                return

        # Completed normally
        _sessions[session_id].update({
            "status": "completed",
            "final_answer": state_vals.get("final_answer"),
            "ranked_results": state_vals.get("ranked_results"),
            "summary": state_vals.get("final_summary") or state_vals.get("draft_summary"),
            "error": state_vals.get("error"),
        })
    except Exception as e:
        traceback.print_exc()
        _sessions[session_id].update({"status": "error", "error": str(e)})


@router.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Trigger the summary/analyze flow.
    Q/A now runs through a dedicated MCP-backed route.
    """
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    paper_indexed = (
        db.is_paper_indexed(request.arxiv_id) and is_collection_compatible(request.arxiv_id)
        if request.arxiv_id else False
    )
    message_lower = (request.message or "").strip().lower()

    if request.arxiv_id:
        intent = "analyze_paper"
    else:
        intent = "discover"

    initial_state = {
        "session_id": session_id,
        "user_query": request.message,
        "question": None,
        "selected_arxiv_id": request.arxiv_id,
        "intent": intent,
        "paper_indexed": paper_indexed,
        "user_preferences": {},
        # Explicitly clear stale fields so the SqliteSaver checkpoint from a previous
        # run on this thread_id doesn't contaminate the new flow.
        "final_answer": None,
        "retrieved_chunks": [],
        "draft_summary": None,
        "ranked_results": [],
        "error": None,
    }

    _sessions[session_id] = {"status": "starting", "session_id": session_id}

    background_tasks.add_task(_run_graph_background, session_id, initial_state, config)

    return {"session_id": session_id, "status": "started"}


@router.get("/api/chat/status/{session_id}")
async def get_session_status(session_id: str):
    """Poll for graph status."""
    session = _sessions.get(session_id)
    if not session:
        graph = get_graph()
        config = {"configurable": {"thread_id": session_id}}
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return {"status": "completed", "session_id": session_id}
        return {"status": "unknown", "session_id": session_id}
    return session
