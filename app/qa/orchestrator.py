"""
Dedicated MCP-driven Q/A orchestration runtime.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from openai import OpenAI

import app.database as db
from app.config import OPENAI_API_KEY, OPENAI_MODEL, QA_MAX_TOOL_STEPS
from app.prompts import QA_MCP_PLANNER_PROMPT, QA_MCP_SYNTHESIS_PROMPT
from app.qa.mcp_host import decode_tool_result, qa_mcp_session, read_json_resource
from app.rag.vectorstore import is_collection_compatible

client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)

ProgressFn = Callable[[dict[str, Any]], None]


def _requested_download_tools(question: str) -> list[str]:
    lowered = question.lower()
    wants_md = any(token in lowered for token in ("markdown", ".md", " md ", "md file"))
    wants_pdf = "pdf" in lowered
    wants_download = any(token in lowered for token in ("download", "downloadable", "printable", "export", "file"))

    if wants_md and not wants_pdf:
        return ["create_md"]
    if wants_pdf and not wants_md:
        return ["create_pdf"]
    if wants_md and wants_pdf:
        return ["create_md", "create_pdf"]
    if wants_download:
        return ["create_md"]
    return []


def _needs_evidence(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in (
            "evidence",
            "cite",
            "citation",
            "support",
            "proof",
            "quote",
            "quoted",
            "source",
            "where in the paper",
            "highlight",
        )
    )


def _needs_graphic(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("image", "graphic", "diagram", "chart", "workflow", "visual"))


def _build_tool_catalog(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in tools
    ]


def _planner_tool_catalog(tool_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    asset_tools = {"create_md", "create_pdf", "create_graphic"}
    return [tool for tool in tool_catalog if tool["name"] not in asset_tools]


def _latest_evidence(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in reversed(tool_results):
        data = item.get("result") or {}
        if isinstance(data, dict):
            if "evidence" in data and data["evidence"]:
                return data["evidence"]
            if "citations" in data and data["citations"]:
                return data["citations"]
    return []


def _used_explicit_evidence_tool(tool_results: list[dict[str, Any]]) -> bool:
    return any(item.get("tool") in {"find_evidence", "cite_evidence"} for item in tool_results)


def _timeline_entry(kind: str, title: str, details: str, **extra) -> dict[str, Any]:
    base = {"kind": kind, "title": title, "details": details}
    base.update(extra)
    return base


def _tool_error_message(raw_result: Any, decoded: Any) -> str:
    if getattr(raw_result, "isError", False):
        if isinstance(decoded, dict):
            return (
                decoded.get("error")
                or decoded.get("text")
                or decoded.get("message")
                or "Tool call failed."
            )
        return str(decoded or "Tool call failed.")
    return ""


def _plan_next_action(question: str, metadata: dict[str, Any], abstract: dict[str, Any], tool_catalog: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": QA_MCP_PLANNER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "paper_metadata": metadata,
                        "paper_abstract": abstract,
                        "available_tools": tool_catalog,
                        "prior_tool_results": tool_results[-4:],
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def _synthesize_answer(question: str, metadata: dict[str, Any], tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": QA_MCP_SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "paper_metadata": metadata,
                        "tool_results": tool_results,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def _graphic_prompt(question: str, answer: str, metadata: dict[str, Any]) -> str:
    return (
        f"Create a clean, information-rich visual for this research Q/A response.\n"
        f"Paper: {metadata.get('title', metadata.get('arxiv_id', 'Unknown paper'))}\n"
        f"User request: {question}\n"
        f"Grounded answer summary: {answer}\n"
        f"Style: academic, legible, presentation-ready, high contrast."
    )


async def run_qa_orchestrator(session_id: str, question: str, arxiv_id: str, progress: ProgressFn | None = None) -> dict[str, Any]:
    progress = progress or (lambda event: None)

    async with qa_mcp_session() as session:
        tools_result = await session.list_tools()
        tool_catalog = _build_tool_catalog(tools_result.tools)
        planner_catalog = _planner_tool_catalog(tool_catalog)

        metadata = await read_json_resource(session, f"researchatlas://paper/{arxiv_id}/metadata")
        abstract = await read_json_resource(session, f"researchatlas://paper/{arxiv_id}/abstract")

        tool_results: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        assets: list[dict[str, Any]] = []

        if not db.is_paper_indexed(arxiv_id) or not is_collection_compatible(arxiv_id):
            progress(_timeline_entry("tool", "Preparing paper context", "Downloading and indexing the paper for grounded Q/A.", tool="ensure_paper_context", status="running"))
            prep_raw = await session.call_tool("ensure_paper_context", {"arxiv_id": arxiv_id})
            prep = decode_tool_result(prep_raw)
            step = _timeline_entry("tool", "Prepared paper context", "Paper context is ready for evidence lookup.", tool="ensure_paper_context", status="completed")
            timeline.append(step)
            progress(step)
            tool_results.append({"tool": "ensure_paper_context", "result": prep})

        for _ in range(QA_MAX_TOOL_STEPS):
            plan = _plan_next_action(question, metadata, abstract, planner_catalog, tool_results)
            if plan.get("action") != "tool":
                break

            tool_name = plan.get("tool")
            arguments = plan.get("arguments", {})
            reason = plan.get("reason", "Model selected this tool.")
            if tool_name not in {tool["name"] for tool in planner_catalog}:
                break

            start_event = _timeline_entry("tool", f"Running {tool_name}", reason, tool=tool_name, status="running")
            progress(start_event)
            raw = await session.call_tool(tool_name, arguments)
            result = decode_tool_result(raw)
            error_message = _tool_error_message(raw, result)
            if error_message:
                failed = _timeline_entry("tool", f"Failed {tool_name}", error_message, tool=tool_name, status="failed")
                timeline.append(failed)
                progress(failed)
                continue
            finished = _timeline_entry("tool", f"Finished {tool_name}", reason, tool=tool_name, status="completed")
            timeline.append(finished)
            progress(finished)
            tool_results.append({"tool": tool_name, "arguments": arguments, "result": result})

        if _needs_evidence(question) and not _latest_evidence(tool_results):
            raw = await session.call_tool("find_evidence", {"arxiv_id": arxiv_id, "question": question, "max_quotes": 4})
            result = decode_tool_result(raw)
            fallback = _timeline_entry("tool", "Finished find_evidence", "Captured supporting evidence for the final answer.", tool="find_evidence", status="completed")
            timeline.append(fallback)
            progress(fallback)
            tool_results.append({"tool": "find_evidence", "arguments": {"arxiv_id": arxiv_id, "question": question}, "result": result})

        synthesized = _synthesize_answer(question, metadata, tool_results)
        answer = synthesized.get("answer", "I could not generate an answer from the available evidence.")
        evidence_requested = _needs_evidence(question)
        evidence_used = _used_explicit_evidence_tool(tool_results)
        graphic_requested = _needs_graphic(question)
        citations = synthesized.get("citations") or _latest_evidence(tool_results)
        if not (evidence_requested or evidence_used):
            citations = []

        requested_download_tools = _requested_download_tools(question)
        if requested_download_tools:
            citations_json = json.dumps(citations, ensure_ascii=True)
            for tool_name in requested_download_tools:
                raw = await session.call_tool(
                    tool_name,
                    {
                        "session_id": session_id,
                        "title": metadata.get("title", arxiv_id),
                        "question": question,
                        "answer": answer,
                        "citations_json": citations_json,
                    },
                )
                asset = decode_tool_result(raw)
                error_message = _tool_error_message(raw, asset)
                if error_message:
                    failed = _timeline_entry("tool", f"Failed {tool_name}", error_message, tool=tool_name, status="failed")
                    timeline.append(failed)
                    progress(failed)
                    continue
                assets.append(asset)
                done = _timeline_entry("tool", f"Finished {tool_name}", "Generated a downloadable answer artifact.", tool=tool_name, status="completed")
                timeline.append(done)
                progress(done)

        generated_image = None
        if graphic_requested:
            raw = await session.call_tool(
                "create_graphic",
                {
                    "session_id": session_id,
                    "prompt": _graphic_prompt(question, answer, metadata),
                    "title": "Generated Graphic",
                },
            )
            generated_image = decode_tool_result(raw)
            error_message = _tool_error_message(raw, generated_image)
            if error_message:
                failed = _timeline_entry("tool", "Failed create_graphic", error_message, tool="create_graphic", status="failed")
                timeline.append(failed)
                progress(failed)
                generated_image = None
            else:
                assets.append(generated_image)
                done = _timeline_entry("tool", "Finished create_graphic", "Generated an image based on the current answer.", tool="create_graphic", status="completed")
                timeline.append(done)
                progress(done)

        evidence_bundle = {
            "enabled": evidence_requested or evidence_used,
            "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
            "items": citations,
        }

        chat_message: str | None
        if graphic_requested:
            chat_message = "Generated an image representing the paper. See the Generated Graphic panel." if generated_image else None
        else:
            chat_message = answer

        return {
            "final_answer": answer,
            "chat_message": chat_message,
            "answer_citations": citations,
            "tool_timeline": timeline,
            "available_tools": tool_catalog,
            "assets": assets,
            "generated_image": generated_image,
            "evidence_bundle": evidence_bundle,
            "paper_metadata": metadata,
        }
