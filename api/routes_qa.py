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
from app.schemas import QARequest

router = APIRouter()

_qa_sessions: dict[str, dict] = {}


def _push_timeline_event(session_id: str, event: dict) -> None:
    session = _qa_sessions.setdefault(session_id, {"session_id": session_id, "tool_timeline": []})
    session.setdefault("tool_timeline", []).append(event)


def _run_qa_background(session_id: str, question: str, arxiv_id: str) -> None:
    _qa_sessions[session_id]["status"] = "running"
    try:
        result = asyncio.run(
            run_qa_orchestrator(
                session_id=session_id,
                question=question,
                arxiv_id=arxiv_id,
                progress=lambda event: _push_timeline_event(session_id, event),
            )
        )
        _qa_sessions[session_id].update({
            "status": "completed",
            **result,
        })
    except Exception as exc:
        traceback.print_exc()
        _qa_sessions[session_id].update({
            "status": "error",
            "error": str(exc),
        })


@router.post("/api/qa")
async def ask_question(request: QARequest, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())
    _qa_sessions[session_id] = {
        "status": "starting",
        "session_id": session_id,
        "tool_timeline": [],
        "assets": [],
        "answer_citations": [],
    }
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
