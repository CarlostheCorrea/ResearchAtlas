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
    # Only trigger on explicit image/visual requests — not generic words like "workflow"
    return any(token in lowered for token in ("make me an image", "generate an image", "create an image",
               "make an image", "make that an image", "turn that into an image", "turn it into an image",
               "make this an image", "draw", "graphic", "diagram", "chart", "visualize", "visualise",
               "illustration", "infographic", "show me a picture", "picture of"))


def _is_continuation_request(question: str) -> bool:
    lowered = question.lower()
    references_previous = any(
        token in lowered
        for token in (
            "that",
            "this",
            "it",
            "previous answer",
            "last answer",
            "your previous",
            "your last",
            "above answer",
            "same answer",
        )
    )
    transform_requested = bool(_requested_download_tools(question)) or _needs_graphic(question)
    return references_previous and transform_requested


def _latest_context_answer(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    latest = context.get("latest_answer")
    if latest and latest.get("answer"):
        return latest
    for turn in reversed(context.get("recent_turns", [])):
        if turn.get("answer"):
            return {
                "question": turn.get("question", ""),
                "answer": turn["answer"],
                "citations": turn.get("citations", []),
            }
    return None


def _recent_turns(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (context or {}).get("recent_turns", []) or []


def _answer_memory_query(question: str, context: dict[str, Any] | None) -> str | None:
    lowered = question.lower().strip()
    turns = _recent_turns(context)
    if not turns:
        if any(token in lowered for token in ("last question", "previous question", "what did i ask", "what was my question")):
            return "I do not have a previous question in this Q/A thread yet."
        if any(token in lowered for token in ("last answer", "previous answer", "what did you answer", "your last response")):
            return "I do not have a previous answer in this Q/A thread yet."
        return None

    latest_turn = turns[-1]
    if any(token in lowered for token in ("last question", "previous question", "what did i ask", "what was my question")):
        return f'Your last question was: "{latest_turn.get("question", "")}"'
    if any(token in lowered for token in ("last answer", "previous answer", "what did you answer", "your last response")):
        return f'Your last answer was: "{latest_turn.get("answer", "")}"'
    return None


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


def _rationale_entry(title: str, details: str) -> dict[str, Any]:
    return _timeline_entry("rationale", title, details, status="completed")


def _emit(timeline: list[dict[str, Any]], progress: ProgressFn, event: dict[str, Any]) -> dict[str, Any]:
    timeline.append(event)
    progress(event)
    return event


def _synthesis_trace(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    graphic_requested: bool,
    requested_download_tools: list[str],
) -> str:
    """Build a user-facing trace from actual state, not model free text."""
    if graphic_requested:
        return "The answer content is ready, so the next step is to generate a visual from the current Q/A context."

    if requested_download_tools:
        formats = ", ".join(tool.replace("create_", "").upper() for tool in requested_download_tools)
        return f"The answer content is ready, so the next step is to create the requested {formats} download."

    if citations:
        citation_label = "citation" if len(citations) == 1 else "citations"
        return f"The final answer is grounded in {len(citations)} supporting {citation_label} from the paper."

    if tool_results:
        used_tools = ", ".join(dict.fromkeys(str(item.get("tool")) for item in tool_results if item.get("tool")))
        if used_tools:
            return f"The final answer was synthesized from the available paper context returned by {used_tools}."
        return "The final answer was synthesized from the available paper context."

    if answer:
        return "The final answer was synthesized from the selected paper metadata and recent Q/A context."

    return "No final answer was available to synthesize."


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


def _plan_next_action(question: str, metadata: dict[str, Any], abstract: dict[str, Any], tool_catalog: list[dict[str, Any]], tool_results: list[dict[str, Any]], continuation_context: dict[str, Any] | None = None) -> dict[str, Any]:
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
                        "recent_qa_context": {
                            "latest_answer": (continuation_context or {}).get("latest_answer"),
                            "recent_turns": (continuation_context or {}).get("recent_turns", [])[-3:],
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def _synthesize_answer(question: str, metadata: dict[str, Any], tool_results: list[dict[str, Any]], continuation_context: dict[str, Any] | None = None) -> dict[str, Any]:
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
                        "recent_qa_context": {
                            "latest_answer": (continuation_context or {}).get("latest_answer"),
                            "recent_turns": (continuation_context or {}).get("recent_turns", [])[-3:],
                        },
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
    """Fallback prompt when the brief builder fails."""
    return (
        f"Create a clean, information-rich visual for this research Q/A response.\n"
        f"Paper: {metadata.get('title', metadata.get('arxiv_id', 'Unknown paper'))}\n"
        f"User request: {question}\n"
        f"Grounded answer summary: {answer}\n"
        f"Style: academic, legible, presentation-ready, high contrast."
    )


def _build_image_brief(question: str, answer: str, metadata: dict[str, Any]) -> str:
    """Use GPT-4o to convert the research answer into a precise, structured image brief.

    The brief explicitly names every label, stage, and connection so the image
    model renders the correct terms instead of inventing generic ones.
    """
    title = metadata.get("title") or metadata.get("arxiv_id", "Research Paper")
    system = (
        "You are a visual design assistant. Convert research findings into a precise, "
        "structured brief for an AI image generator.\n"
        "Rules:\n"
        "1. Choose the most appropriate diagram type.\n"
        "   IMPORTANT: for pipelines or flows with 3–5 stages, ALWAYS use a 2×2 grid "
        "   (or 2-row layout) — never a single horizontal row, as boxes overflow the canvas.\n"
        "2. List EVERY label, box title, and annotation EXACTLY as they should appear "
        "— spell out the full, correct text for each element verbatim from the research content.\n"
        "3. Describe layout, arrows, and connections unambiguously.\n"
        "4. Leave generous margins (at least 80px on every edge) so no text is cropped.\n"
        "5. Specify: Style: clean academic diagram, black text on white background, "
        "sans-serif font, high contrast, no decorative clip-art.\n"
        "Output ONLY the image brief. No preamble, no markdown."
    )
    user = (
        f"Paper: {title}\n"
        f"User request: {question}\n\n"
        f"Research content to visualise:\n{answer[:2500]}\n\n"
        "Write the image brief:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=600,
            temperature=0.1,
        )
        brief = (resp.choices[0].message.content or "").strip()
        if brief:
            return brief
    except Exception:
        pass
    # Fall back to simple prompt if GPT-4o call fails
    return _graphic_prompt(question, answer, metadata)


async def run_qa_orchestrator(session_id: str, question: str, arxiv_id: str, progress: ProgressFn | None = None, continuation_context: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = progress or (lambda event: None)

    async with qa_mcp_session() as session:
        tool_results: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        assets: list[dict[str, Any]] = []
        _emit(
            timeline,
            progress,
            _timeline_entry(
                "tool",
                "Started MCP Q/A host",
                f"Connecting to the local MCP server for paper {arxiv_id}.",
                tool="qa_orchestrator",
                status="running",
            ),
        )

        tools_result = await session.list_tools()
        tool_catalog = _build_tool_catalog(tools_result.tools)
        planner_catalog = _planner_tool_catalog(tool_catalog)
        _emit(
            timeline,
            progress,
            _timeline_entry(
                "tool",
                "Discovered MCP tools",
                f"Loaded {len(tool_catalog)} available tools and {len(planner_catalog)} planner-callable tools.",
                tool="tools/list",
                status="completed",
            ),
        )

        metadata = await read_json_resource(session, f"researchatlas://paper/{arxiv_id}/metadata")
        abstract = await read_json_resource(session, f"researchatlas://paper/{arxiv_id}/abstract")
        _emit(
            timeline,
            progress,
            _timeline_entry(
                "tool",
                "Loaded paper resources",
                "Read paper metadata and abstract through MCP resources.",
                tool="resources/read",
                status="completed",
            ),
        )
        previous_answer = _latest_context_answer(continuation_context)

        memory_answer = _answer_memory_query(question, continuation_context)
        if memory_answer:
            rationale = _rationale_entry(
                "CoT Trace",
                "This is a conversation-memory question, so I can answer from the recent Q/A thread without calling paper tools.",
            )
            timeline.append(rationale)
            progress(rationale)
            return {
                "final_answer": memory_answer,
                "chat_message": memory_answer,
                "answer_citations": [],
                "tool_timeline": timeline,
                "available_tools": tool_catalog,
                "assets": assets,
                "generated_image": None,
                "evidence_bundle": {
                    "enabled": False,
                    "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
                    "items": [],
                },
                "paper_metadata": metadata,
            }

        if _is_continuation_request(question) and previous_answer:
            rationale = _rationale_entry(
                "CoT Trace",
                "This request refers to the most recent Q/A answer, so I can transform that answer directly without re-running paper retrieval.",
            )
            timeline.append(rationale)
            progress(rationale)

            answer = previous_answer["answer"]
            citations = previous_answer.get("citations", [])
            requested_download_tools = _requested_download_tools(question)
            graphic_requested = _needs_graphic(question)
            generated_image = None
            chat_message = None

            for tool_name in requested_download_tools:
                raw = await session.call_tool(
                    tool_name,
                    {
                        "session_id": session_id,
                        "title": metadata.get("title", arxiv_id),
                        "question": question,
                        "answer": answer,
                        "citations_json": json.dumps(citations, ensure_ascii=True),
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
                done = _timeline_entry("tool", f"Finished {tool_name}", "Generated a downloadable artifact from the previous answer.", tool=tool_name, status="completed")
                timeline.append(done)
                progress(done)

            if graphic_requested:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        "Building image brief",
                        "Converting the previous answer into an image-generation prompt.",
                        tool="create_graphic",
                        status="running",
                    ),
                )
                raw = await session.call_tool(
                    "create_graphic",
                    {
                        "session_id": session_id,
                        "prompt": _build_image_brief(question, answer, metadata),
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
                    done = _timeline_entry("tool", "Finished create_graphic", "Generated an image from the previous answer.", tool="create_graphic", status="completed")
                    timeline.append(done)
                    progress(done)
                    chat_message = "Generated an image from the previous answer. See the Generated Graphic panel."

            if requested_download_tools and not chat_message:
                chat_message = "Generated the requested file from the previous answer. See the Downloads panel."

            return {
                "final_answer": answer,
                "chat_message": chat_message,
                "answer_citations": citations,
                "tool_timeline": timeline,
                "available_tools": tool_catalog,
                "assets": assets,
                "generated_image": generated_image,
                "evidence_bundle": {
                    "enabled": bool(citations),
                    "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
                    "items": citations,
                },
                "paper_metadata": metadata,
            }

        if not db.is_paper_indexed(arxiv_id) or not is_collection_compatible(arxiv_id):
            _emit(
                timeline,
                progress,
                _timeline_entry("tool", "Preparing paper context", "Downloading and indexing the paper for grounded Q/A.", tool="ensure_paper_context", status="running"),
            )
            prep_raw = await session.call_tool("ensure_paper_context", {"arxiv_id": arxiv_id})
            prep = decode_tool_result(prep_raw)
            step = _timeline_entry("tool", "Prepared paper context", "Paper context is ready for evidence lookup.", tool="ensure_paper_context", status="completed")
            timeline.append(step)
            progress(step)
            tool_results.append({"tool": "ensure_paper_context", "result": prep})

        for step_index in range(QA_MAX_TOOL_STEPS):
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "tool",
                    f"Planning MCP step {step_index + 1}",
                    "Asking the planner whether another MCP tool is needed.",
                    tool="planner",
                    status="running",
                ),
            )
            plan = _plan_next_action(question, metadata, abstract, planner_catalog, tool_results, continuation_context)
            if plan.get("rationale"):
                rationale = _rationale_entry("CoT Trace", str(plan.get("rationale", "")).strip())
                timeline.append(rationale)
                progress(rationale)
            if plan.get("action") != "tool":
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        "Planner selected final answer",
                        "No additional MCP tool was needed before synthesis.",
                        tool="planner",
                        status="completed",
                    ),
                )
                break

            tool_name = plan.get("tool")
            arguments = plan.get("arguments", {})
            reason = plan.get("reason", "Model selected this tool.")
            if tool_name not in {tool["name"] for tool in planner_catalog}:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        "Planner stopped",
                        f"Skipped unavailable or non-planner tool: {tool_name}.",
                        tool="planner",
                        status="completed",
                    ),
                )
                break

            start_event = _timeline_entry("tool", f"Running {tool_name}", reason, tool=tool_name, status="running")
            _emit(timeline, progress, start_event)
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
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "tool",
                    "Running evidence fallback",
                    "Evidence was requested, so the orchestrator is calling find_evidence directly.",
                    tool="find_evidence",
                    status="running",
                ),
            )
            raw = await session.call_tool("find_evidence", {"arxiv_id": arxiv_id, "question": question, "max_quotes": 4})
            result = decode_tool_result(raw)
            fallback = _timeline_entry("tool", "Finished find_evidence", "Captured supporting evidence for the final answer.", tool="find_evidence", status="completed")
            timeline.append(fallback)
            progress(fallback)
            tool_results.append({"tool": "find_evidence", "arguments": {"arxiv_id": arxiv_id, "question": question}, "result": result})

        graphic_requested = _needs_graphic(question)

        # For graphic requests the planner often skips RAG (it sees "make an image"
        # and returns "final"). Force a chunk retrieval so the image brief is
        # grounded in the actual paper content rather than just the abstract.
        if graphic_requested and not any(r.get("tool") == "retrieve_paper_chunks" for r in tool_results):
            try:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        "Retrieving image context",
                        "Loading paper chunks so the generated visual is grounded in the paper.",
                        tool="retrieve_paper_chunks",
                        status="running",
                    ),
                )
                rag_raw = await session.call_tool("retrieve_paper_chunks", {
                    "arxiv_id": arxiv_id,
                    "query": question,
                    "top_k": 6,
                })
                rag = decode_tool_result(rag_raw)
                if not _tool_error_message(rag_raw, rag):
                    rag_step = _timeline_entry("tool", "Retrieved paper chunks for image", "Loading paper content to ground the graphic.", tool="retrieve_paper_chunks", status="completed")
                    timeline.append(rag_step)
                    progress(rag_step)
                    tool_results.append({"tool": "retrieve_paper_chunks", "result": rag})
            except Exception:
                pass

        _emit(
            timeline,
            progress,
            _timeline_entry(
                "tool",
                "Synthesizing answer",
                f"Combining {len(tool_results)} MCP tool result(s) into the final Q/A response.",
                tool="synthesis",
                status="running",
            ),
        )
        synthesized = _synthesize_answer(question, metadata, tool_results, continuation_context)
        answer = synthesized.get("answer", "I could not generate an answer from the available evidence.")
        evidence_requested = _needs_evidence(question)
        evidence_used = _used_explicit_evidence_tool(tool_results)
        citations = synthesized.get("citations") or _latest_evidence(tool_results)
        if not (evidence_requested or evidence_used):
            citations = []

        requested_download_tools = _requested_download_tools(question)
        synthesis_trace = _synthesis_trace(
            question,
            answer,
            citations,
            tool_results,
            graphic_requested,
            requested_download_tools,
        )
        rationale = _rationale_entry("Synthesis Trace", synthesis_trace)
        timeline.append(rationale)
        progress(rationale)

        if requested_download_tools:
            citations_json = json.dumps(citations, ensure_ascii=True)
            for tool_name in requested_download_tools:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        f"Running {tool_name}",
                        "Creating the requested downloadable artifact.",
                        tool=tool_name,
                        status="running",
                    ),
                )
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
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "tool",
                    "Building image brief",
                    "Converting the answer into an image-generation prompt.",
                    tool="create_graphic",
                    status="running",
                ),
            )
            raw = await session.call_tool(
                "create_graphic",
                {
                    "session_id": session_id,
                    "prompt": _build_image_brief(question, answer, metadata),
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
                validation = generated_image.get("validation", {}) if isinstance(generated_image, dict) else {}
                retries = validation.get("retries", 0)
                corrections = validation.get("corrections", [])
                if retries == 0:
                    detail = "Image text validated — no spelling errors found."
                elif corrections:
                    detail = f"Image text corrected after {retries} retry — fixed: {'; '.join(corrections[:3])}."
                else:
                    detail = f"Image regenerated {retries}× to fix spelling."
                done = _timeline_entry("tool", "Finished create_graphic", detail, tool="create_graphic", status="completed")
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
