"""
Review API routes — resumes a paused LangGraph graph after human approval.
Phase 3 — Week 11: Human-in-the-Loop. This is how the frontend resumes interrupt() nodes.
"""
from fastapi import APIRouter, BackgroundTasks
from langgraph.types import Command
from app.schemas import ReviewDecision
from app.graph.build_graph import get_graph
from api.routes_chat import _sessions, _extract_interrupt_payload

router = APIRouter()


@router.post("/api/review/decide")
async def submit_review_decision(decision: ReviewDecision, background_tasks: BackgroundTasks):
    """
    Resume a paused LangGraph graph after human review.
    Must use Command(resume=...) — passing raw state would restart the graph.
    """
    session_id = decision.session_id
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    resume_value = {"decision": decision.decision}
    if decision.revision_note:
        resume_value["revision_note"] = decision.revision_note

    if session_id in _sessions:
        _sessions[session_id]["status"] = "resuming"

    def resume_graph():
        _sessions[session_id]["status"] = "running"
        try:
            # Command(resume=...) tells LangGraph to resume the interrupted node
            # and pass resume_value as the return value of interrupt()
            for chunk in graph.stream(Command(resume=resume_value), config=config):
                node_name = list(chunk.keys())[0]
                _sessions[session_id]["last_node"] = node_name

            snapshot = graph.get_state(config)
            state_vals = snapshot.values if snapshot else {}

            # Check if we hit a second interrupt (before_save gate)
            if snapshot and snapshot.next:
                next_node = snapshot.next[0] if snapshot.next else None
                if next_node in ("human_gate_before_download", "human_gate_before_save"):
                    interrupt_payload = _extract_interrupt_payload(snapshot)
                    _sessions[session_id].update({
                        "status": "interrupted",
                        "interrupt_payload": interrupt_payload,
                        "next_node": next_node,
                    })
                    return

            # Also check for interrupt() pause (snapshot.tasks has interrupts)
            interrupt_payload = _extract_interrupt_payload(snapshot)
            if interrupt_payload:
                _sessions[session_id].update({
                    "status": "interrupted",
                    "interrupt_payload": interrupt_payload,
                })
                return

            _sessions[session_id].update({
                "status": "completed",
                "final_answer": state_vals.get("final_answer"),
                "summary": state_vals.get("final_summary") or state_vals.get("draft_summary"),
                "saved_to_library": state_vals.get("saved_to_library"),
                "error": state_vals.get("error"),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            _sessions[session_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(resume_graph)

    return {
        "session_id": session_id,
        "decision": decision.decision,
        "status": "resuming",
    }
