"""
Q/A API routes backed by the dedicated MCP-driven Q/A orchestrator.
"""
from __future__ import annotations

import asyncio
import traceback
import uuid

from fastapi import APIRouter, BackgroundTasks

from app.qa.mcp_host import qa_mcp_session
from app.qa.orchestrator import run_qa_orchestrator
from app.qa.assets import clear_all_assets
from app.schemas import QARequest

router = APIRouter()

_qa_sessions: dict[str, dict] = {}


def _log_qa_event(session_id: str, event: dict) -> None:
    title = event.get("title") or event.get("tool") or "Q/A event"
    status = event.get("status") or event.get("kind") or "info"
    tool = event.get("tool")
    details = event.get("details") or ""
    prefix = f"[qa] {session_id[:8]} {status}: {title}"
    if tool:
        prefix += f" [{tool}]"
    if details:
        prefix += f" - {details}"
    print(prefix, flush=True)


def _push_timeline_event(session_id: str, event: dict) -> None:
    session = _qa_sessions.setdefault(session_id, {"session_id": session_id, "tool_timeline": []})
    session.setdefault("tool_timeline", []).append(event)
    _log_qa_event(session_id, event)
    if event.get("kind") == "rationale":
        session.setdefault("thought_log", []).append(event)


def _run_qa_background(session_id: str, question: str, arxiv_id: str) -> None:
    session_state = _qa_sessions[session_id]
    session_state["status"] = "running"
    print(f"[qa] {session_id[:8]} running: started Q/A for {arxiv_id} - {question[:120]}", flush=True)
    continuation_context = {
        "latest_answer": session_state.get("latest_answer"),
        "recent_turns": session_state.get("recent_turns", []),
    }
    try:
        result = asyncio.run(
            run_qa_orchestrator(
                session_id=session_id,
                question=question,
                arxiv_id=arxiv_id,
                progress=lambda event: _push_timeline_event(session_id, event),
                continuation_context=continuation_context,
            )
        )
        recent_turns = session_state.setdefault("recent_turns", [])
        if result.get("final_answer"):
            turn = {
                "question": question,
                "answer": result["final_answer"],
                "citations": result.get("answer_citations", []),
                "assets": result.get("assets", []),
            }
            recent_turns.append(turn)
            del recent_turns[:-5]
            session_state["latest_answer"] = turn

        session_state.update({
            "status": "completed",
            **result,
        })
        print(
            f"[qa] {session_id[:8]} completed: answer ready "
            f"({len(result.get('tool_timeline', []))} timeline events, {len(result.get('assets', []))} assets)",
            flush=True,
        )
    except Exception as exc:
        traceback.print_exc()
        session_state.update({
            "status": "error",
            "error": str(exc),
        })
        print(f"[qa] {session_id[:8]} error: {exc}", flush=True)


@router.post("/api/qa")
async def ask_question(request: QARequest, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())
    existing = _qa_sessions.get(session_id)
    if existing and existing.get("arxiv_id") == request.arxiv_id:
        existing.update({
            "status": "starting",
            "tool_timeline": [],
            "thought_log": [],
            "assets": [],
            "generated_image": None,
            "answer_citations": [],
            "tracking": None,
            "error": None,
        })
    else:
        _qa_sessions[session_id] = {
            "status": "starting",
            "session_id": session_id,
            "arxiv_id": request.arxiv_id,
            "tool_timeline": [],
            "thought_log": [],
            "recent_turns": [],
            "latest_answer": None,
            "assets": [],
            "answer_citations": [],
            "tracking": None,
        }
    print(f"[qa] {session_id[:8]} accepted: queued Q/A request for {request.arxiv_id}", flush=True)
    background_tasks.add_task(_run_qa_background, session_id, request.message, request.arxiv_id)
    return {"session_id": session_id, "status": "started"}


@router.get("/api/qa/status/{session_id}")
async def qa_status(session_id: str):
    return _qa_sessions.get(session_id, {"status": "unknown", "session_id": session_id})


@router.get("/api/qa/tools")
async def qa_tools():
    async with qa_mcp_session() as session:
        tools_result = await session.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_result.tools
        ]
    return {"tools": tools}


@router.delete("/api/qa/assets")
async def delete_qa_assets():
    removed = clear_all_assets()
    return {"status": "ok", "removed_sessions": removed}
