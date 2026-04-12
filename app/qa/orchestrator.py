"""
Dedicated MCP-driven Q/A orchestration runtime.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from openai import OpenAI

import app.database as db
from app.config import OPENAI_API_KEY, OPENAI_MODEL, QA_MAX_TOOL_STEPS, QA_REPAIR_ENABLED
from app.observability import trace_span
from app.prompts import QA_MCP_PLANNER_PROMPT, QA_MCP_SYNTHESIS_PROMPT
from app.qa.evaluation import evaluate_artifact_request, evaluate_qa_response, should_repair, tracking_score
from app.qa.mcp_host import decode_tool_result, qa_mcp_session, read_json_resource
from app.rag.vectorstore import is_collection_compatible

client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)

ProgressFn = Callable[[dict[str, Any]], None]


def _requested_download_tools(question: str) -> list[str]:
    lowered = question.lower()
    wants_presentation = any(token in lowered for token in ("presentation", "slide", "slides", "slide deck", "deck"))
    wants_md = any(token in lowered for token in ("markdown", ".md", " md ", "md file"))
    wants_pdf = "pdf" in lowered
    wants_download = any(token in lowered for token in ("download", "downloadable", "printable", "export", "file"))

    if wants_presentation:
        return ["create_presentation"]
    if wants_md and not wants_pdf:
        return ["create_md"]
    if wants_pdf and not wants_md:
        return ["create_pdf"]
    if wants_md and wants_pdf:
        return ["create_md", "create_pdf"]
    if wants_download:
        return ["create_md"]
    return []


def _presentation_options(question: str) -> dict[str, Any]:
    lowered = question.lower()
    slide_count = 1 if any(token in lowered for token in ("1 slide", "one slide", "one-slide", "single slide", "one-page slide")) else 3
    audience = "class"
    if "general audience" in lowered or "non-technical" in lowered or "beginner" in lowered:
        audience = "general"
    elif "technical" in lowered or "expert" in lowered or "graduate" in lowered:
        audience = "technical"
    include_speaker_notes = not any(token in lowered for token in ("no speaker notes", "without speaker notes", "no notes"))
    return {
        "slide_count": slide_count,
        "audience": audience,
        "include_speaker_notes": include_speaker_notes,
    }


def _asset_tool_arguments(tool_name: str, session_id: str, metadata: dict[str, Any], question: str, answer: str, citations_json: str) -> dict[str, Any]:
    arguments = {
        "session_id": session_id,
        "title": metadata.get("title", metadata.get("arxiv_id", "ResearchAtlas")),
        "question": question,
        "answer": answer,
        "citations_json": citations_json,
    }
    if tool_name == "create_presentation":
        arguments.update(_presentation_options(question))
    return arguments


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


def _graphic_source_question(question: str) -> str:
    lowered = question.lower()
    if "methodolog" in lowered:
        return "What methodology details from the paper should be used as source content?"
    if "workflow" in lowered or "pipeline" in lowered:
        return "What workflow or pipeline details from the paper should be used as source content?"
    if "result" in lowered or "finding" in lowered:
        return "What result or finding details from the paper should be used as source content?"
    return "What paper details should be used as source content for this visual explanation?"


# Matches pronoun DIRECTLY coupled to a transform verb — the pronoun must be
# the object being transformed, not a relative clause word like "that summarises".
# Examples that match:  "make this a PDF"  "turn it into a PDF"  "make that an image"
#                       "make the previous question answer a presentation"
# Examples that don't:  "PDF that summarises methods"  "limitations of the research"
_CONTINUATION_RE = re.compile(
    r"\b(?:make|turn|convert|export|save|put|get|render|create)\s+(?:this|that|it)\b"
    r"|\b(?:this|that|it)\s+(?:into|as|a|to)\s+(?:pdf|md|markdown|image|graphic|picture|presentation|slide)"
    # "make the previous/last answer/question [into] a presentation"
    r"|\b(?:make|turn|convert|export|render)\s+the\s+(?:previous|last)\s+(?:answer|question(?:\s+answer)?)\b",
    re.IGNORECASE,
)
_EXPLICIT_BACK_REFS = (
    "previous answer", "last answer", "your previous", "your last",
    "above answer", "same answer", "the answer above",
    "previous response", "last response",
    "what you just said", "what you wrote",
    # Users often say "previous question answer" meaning "the answer to the previous question"
    "previous question answer", "last question answer",
    "previous question into", "last question into",
    # With transform requests, "previous question" usually means "the answer to the previous question".
    # This is safe because _is_continuation_request checks transform_requested first.
    "previous question", "last question",
)


def _is_continuation_request(question: str) -> bool:
    """Return True only when the user is clearly asking to reformat/transform
    the *previous* answer — not when they are requesting new content that
    happens to contain words like 'it', 'that', or 'this' as part of another
    phrase (e.g. 'limitations' contains 'it'; 'that summarises' is a relative clause)."""
    transform_requested = bool(_requested_download_tools(question)) or _needs_graphic(question)
    if not transform_requested:
        return False
    lowered = question.lower()
    # Unambiguous multi-word references to the prior turn
    if any(ref in lowered for ref in _EXPLICIT_BACK_REFS):
        return True
    # Pronoun tightly coupled with a transform verb (positional, not substring)
    if _CONTINUATION_RE.search(question):
        return True
    return False


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


# ── Metadata fast-path ────────────────────────────────────────────────────────

_METADATA_PATTERNS: list[tuple[list[str], str]] = [
    # authors
    (["who are the author", "who wrote", "who is the author", "list the author",
      "name the author", "what are the author", "authors of this", "author of this"], "authors"),
    # title
    (["what is the title", "what's the title", "paper title", "title of this", "title of the paper"], "title"),
    # year / publication date
    (["what year", "when was this published", "when was it published", "publication date",
      "when was this paper", "published in", "year of publication", "release date"], "year"),
    # venue / journal / conference
    (["what journal", "what conference", "where was it published", "what venue", "published at"], "venue"),
    # categories
    (["what category", "what categories", "what field", "arxiv category", "subject area"], "categories"),
]


def _is_metadata_question(question: str) -> bool:
    """Return True for questions answerable entirely from paper metadata fields."""
    lowered = question.lower()
    for patterns, _ in _METADATA_PATTERNS:
        if any(p in lowered for p in patterns):
            return True
    return False


def _answer_from_metadata(question: str, metadata: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Build a direct answer + proper metadata citation from paper_metadata."""
    lowered = question.lower()

    def _meta_citation(field_label: str, value: str) -> dict[str, Any]:
        return {"section": "Paper Metadata", "page": None, "quote": f"{field_label}: {value}"}

    # Authors
    if any(p in lowered for p in _METADATA_PATTERNS[0][0]):
        raw = metadata.get("authors", [])
        author_str = ", ".join(str(a) for a in raw) if isinstance(raw, list) else str(raw or "")
        if author_str:
            answer = f"The authors of this paper are {author_str}."
        else:
            answer = "Author information is not available for this paper."
        return answer, [_meta_citation("Authors", author_str or "Unknown")]

    # Title
    if any(p in lowered for p in _METADATA_PATTERNS[1][0]):
        title = metadata.get("title", "")
        answer = f'The title of this paper is "{title}".' if title else "Title information is not available."
        return answer, [_meta_citation("Title", title or "Unknown")]

    # Year / date
    if any(p in lowered for p in _METADATA_PATTERNS[2][0]):
        year = metadata.get("published", metadata.get("year", ""))
        answer = f"This paper was published in {year}." if year else "Publication date information is not available."
        return answer, [_meta_citation("Published", year or "Unknown")]

    # Venue
    if any(p in lowered for p in _METADATA_PATTERNS[3][0]):
        venue = metadata.get("journal_ref", metadata.get("venue", ""))
        answer = f"This paper was published at/in: {venue}." if venue else "Venue information is not available in the paper metadata."
        return answer, [_meta_citation("Venue", venue or "Unknown")]

    # Categories
    if any(p in lowered for p in _METADATA_PATTERNS[4][0]):
        cats = metadata.get("categories", [])
        cat_str = ", ".join(cats) if isinstance(cats, list) else str(cats or "")
        answer = f"This paper's arXiv categories are: {cat_str}." if cat_str else "Category information is not available."
        return answer, [_meta_citation("Categories", cat_str or "Unknown")]

    # Generic fallback — shouldn't normally reach here
    title = metadata.get("title", "Unknown")
    raw = metadata.get("authors", [])
    author_str = ", ".join(str(a) for a in raw) if isinstance(raw, list) else str(raw or "")
    year = metadata.get("published", metadata.get("year", "Unknown"))
    answer = f'Paper: "{title}" by {author_str} ({year}).'
    return answer, [_meta_citation("Metadata", answer)]


def _answer_memory_query(question: str, context: dict[str, Any] | None) -> str | None:
    lowered = question.lower().strip()

    # Never intercept when the user is asking to TRANSFORM the previous answer
    # (e.g. "make the previous question answer a presentation").  Those requests
    # are handled by the continuation path; returning memory text here would
    # swallow the tool call.
    if bool(_requested_download_tools(question)) or _needs_graphic(question):
        return None

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
    asset_tools = {"create_md", "create_pdf", "create_graphic", "create_presentation"}
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
    with trace_span("planner", {"qa.question": question[:200], "qa.tool_result_count": len(tool_results)}):
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
    with trace_span("synthesis", {"qa.question": question[:200], "qa.tool_result_count": len(tool_results)}):
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
        with trace_span("artifact_generation", {"qa.artifact": "image_brief"}):
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


def _add_tracking_event(timeline: list[dict[str, Any]], progress: ProgressFn, tracking: dict[str, Any]) -> None:
    status = tracking.get("overall_status", "needs_review")
    score = tracking_score(tracking)
    event = _timeline_entry(
        "tool",
        "Finished judge_eval",
        f"LLM-as-judge completed with status {status} and average score {score:.2f}.",
        tool="judge_eval",
        status="completed" if status in {"passed", "repaired"} else "failed",
    )
    timeline.append(event)
    progress(event)


def _graphic_source_failed(tracking: dict[str, Any] | None) -> bool:
    """Only block image generation when core content quality still needs repair."""
    if not tracking or tracking.get("overall_status") != "needs_review":
        return False
    if not should_repair(tracking):
        return False
    answer_relevance = tracking.get("answer_relevance", {})
    if answer_relevance.get("status") == "fail":
        return True
    # For graphics, citation gaps should trigger repair/evidence gathering, but
    # should not block image generation when the source answer is otherwise
    # strong. The final Tracking tab still shows needs_review.
    return tracking_score(tracking) < 0.75


async def _call_tool(session, tool_name: str, arguments: dict[str, Any]):
    span_name = "mcp_tool_call"
    if tool_name in {"ensure_paper_context", "retrieve_paper_chunks", "find_evidence", "cite_evidence", "compare_sections"}:
        span_name = "retrieval"
    elif tool_name.startswith("create_"):
        span_name = "artifact_generation"
    with trace_span(span_name, {"mcp.tool": tool_name, "mcp.arguments": json.dumps(arguments, ensure_ascii=True)[:500]}):
        return await session.call_tool(tool_name, arguments)


async def _evaluate_and_repair(
    session,
    session_id: str,
    question: str,
    arxiv_id: str,
    metadata: dict[str, Any],
    timeline: list[dict[str, Any]],
    progress: ProgressFn,
    tool_results: list[dict[str, Any]],
    answer: str,
    citations: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    generated_image: dict[str, Any] | None,
    evidence_bundle: dict[str, Any],
    chat_message: str | None,
    continuation_context: dict[str, Any] | None,
    allow_repair: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any], str | None]:
    _emit(
        timeline,
        progress,
        _timeline_entry("tool", "Running judge_eval", "Evaluating answer quality, grounding, citations, tool choice, and artifacts.", tool="judge_eval", status="running"),
    )
    tracking = evaluate_qa_response(
        question,
        answer,
        citations,
        timeline,
        assets,
        generated_image,
        evidence_bundle,
        metadata,
    )
    _add_tracking_event(timeline, progress, tracking)

    if not (allow_repair and QA_REPAIR_ENABLED and should_repair(tracking)):
        return answer, citations, evidence_bundle, tracking, chat_message

    _emit(
        timeline,
        progress,
        _timeline_entry("tool", "Running repair", tracking.get("repair_reason") or "Judge requested one grounded repair pass.", tool="repair", status="running"),
    )

    repaired_tool_results = list(tool_results)
    repaired_citations = list(citations)
    needs_more_evidence = (
        not repaired_citations
        or tracking.get("groundedness", {}).get("status") == "fail"
        or tracking.get("citation_quality", {}).get("status") == "fail"
    )
    if needs_more_evidence:
        _emit(
            timeline,
            progress,
            _timeline_entry("tool", "Repair finding evidence", "Citations were missing or weak, so repair is forcing find_evidence.", tool="find_evidence", status="running"),
        )
        raw = await _call_tool(session, "find_evidence", {"arxiv_id": arxiv_id, "question": question, "max_quotes": 6})
        result = decode_tool_result(raw)
        error_message = _tool_error_message(raw, result)
        if error_message:
            failed = _timeline_entry("tool", "Repair evidence failed", error_message, tool="find_evidence", status="failed")
            timeline.append(failed)
            progress(failed)
        else:
            repaired_tool_results.append({"tool": "find_evidence", "arguments": {"arxiv_id": arxiv_id, "question": question}, "result": result})
            repaired_citations = _latest_evidence(repaired_tool_results)
            done = _timeline_entry("tool", "Repair evidence captured", "Captured evidence for the repair pass.", tool="find_evidence", status="completed")
            timeline.append(done)
            progress(done)

    repaired = _synthesize_answer(question, metadata, repaired_tool_results, continuation_context)
    repaired_answer = repaired.get("answer") or answer
    repaired_citations = repaired.get("citations") or repaired_citations
    repaired_evidence_bundle = {
        "enabled": bool(repaired_citations),
        "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
        "items": repaired_citations,
    }
    repaired_tracking = evaluate_qa_response(
        question,
        repaired_answer,
        repaired_citations,
        timeline,
        assets,
        generated_image,
        repaired_evidence_bundle,
        metadata,
        repair_attempted=True,
    )

    if repaired_tracking.get("overall_status") == "passed":
        repaired_tracking["overall_status"] = "repaired"
        done = _timeline_entry("tool", "Finished repair", "Repair improved the judged answer, so the repaired answer is being returned.", tool="repair", status="completed")
        timeline.append(done)
        progress(done)
        return repaired_answer, repaired_citations, repaired_evidence_bundle, repaired_tracking, repaired_answer

    if tracking_score(repaired_tracking) > tracking_score(tracking):
        repaired_tracking["overall_status"] = "needs_review"
        done = _timeline_entry(
            "tool",
            "Finished repair with review needed",
            "Repair improved the score, but the judge still found quality issues, so the improved answer is returned with needs_review status.",
            tool="repair",
            status="failed",
        )
        timeline.append(done)
        progress(done)
        return repaired_answer, repaired_citations, repaired_evidence_bundle, repaired_tracking, repaired_answer

    tracking["overall_status"] = "needs_review"
    failed = _timeline_entry("tool", "Repair did not improve answer", "The original answer remains the best available response; review the judge notes.", tool="repair", status="failed")
    timeline.append(failed)
    progress(failed)
    return answer, citations, evidence_bundle, tracking, chat_message


async def run_qa_orchestrator(session_id: str, question: str, arxiv_id: str, progress: ProgressFn | None = None, continuation_context: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = progress or (lambda event: None)

    with trace_span("qa_request", {"qa.session_id": session_id, "qa.arxiv_id": arxiv_id, "qa.question": question[:200]}):
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

        # ── Continuation check MUST come before memory check ─────────────────
        # "Now use the previous answer to make a presentation" contains the
        # phrase "previous answer", which _answer_memory_query would incorrectly
        # treat as a recall query and short-circuit. Always check for a
        # transform/export request first so continuation requests are never
        # swallowed by the memory fast-path.
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
            # Only embed citations in the exported file when the user explicitly
            # asked for them (e.g. "with citations", "cited", "evidence").
            citations_for_export = citations if _needs_evidence(question) else []
            citations_json = json.dumps(citations_for_export, ensure_ascii=True)

            for tool_name in requested_download_tools:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        f"Running {tool_name}",
                        "Creating the requested artifact from the previous answer.",
                        tool=tool_name,
                        status="running",
                    ),
                )
                args = _asset_tool_arguments(tool_name, session_id, metadata, question, answer, citations_json)
                raw = await _call_tool(session, tool_name, args)
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
                raw = await _call_tool(
                    session,
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

            # Evidence Viewer should not open for pure artifact continuation
            # requests — the user is reformatting a previous answer, not asking
            # for new evidence. Keep items available for the answer_citations
            # badges but disable the panel.
            evidence_bundle = {
                "enabled": False,
                "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
                "items": citations,
            }
            # Use artifact-only evaluation — the "answer" here is a status
            # confirmation, not research content, so the full LLM judge would
            # produce meaningless answer_relevance/groundedness failures.
            tracking = evaluate_artifact_request(assets, timeline)
            _add_tracking_event(timeline, progress, tracking)

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
                "tracking": tracking,
            }

        # ── Memory fast-path (must come AFTER continuation check) ────────────
        # "What was your last answer?" → report from history.
        # This runs after _is_continuation_request so that "use the previous
        # answer to make a presentation" is never swallowed here.
        memory_answer = _answer_memory_query(question, continuation_context)
        if memory_answer:
            rationale = _rationale_entry(
                "CoT Trace",
                "This is a conversation-memory question, so I can answer from the recent Q/A thread without calling paper tools.",
            )
            timeline.append(rationale)
            progress(rationale)
            evidence_bundle = {
                "enabled": False,
                "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
                "items": [],
            }
            tracking = evaluate_qa_response(
                question,
                memory_answer,
                [],
                timeline,
                assets,
                None,
                evidence_bundle,
                metadata,
            )
            _add_tracking_event(timeline, progress, tracking)
            return {
                "final_answer": memory_answer,
                "chat_message": memory_answer,
                "answer_citations": [],
                "tool_timeline": timeline,
                "available_tools": tool_catalog,
                "assets": assets,
                "generated_image": None,
                "evidence_bundle": evidence_bundle,
                "paper_metadata": metadata,
                "tracking": tracking,
            }

        # ── Metadata fast path ────────────────────────────────────────────────
        # For questions answerable from paper_metadata alone (authors, title,
        # year, venue, categories) we skip the planner loop entirely and attach
        # a proper "Paper Metadata" citation so the judge can verify groundedness.
        if _is_metadata_question(question):
            meta_answer, meta_citations = _answer_from_metadata(question, metadata)
            rationale = _rationale_entry(
                "CoT Trace",
                "This question asks about a paper metadata field (authors, title, year, venue, or categories). "
                "The metadata is already available, so no retrieval tool is needed.",
            )
            timeline.append(rationale)
            progress(rationale)
            evidence_bundle = {
                "enabled": True,
                "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
                "items": meta_citations,
            }
            tracking = evaluate_qa_response(
                question,
                meta_answer,
                meta_citations,
                timeline,
                assets,
                None,
                evidence_bundle,
                metadata,
            )
            _add_tracking_event(timeline, progress, tracking)
            return {
                "final_answer": meta_answer,
                "chat_message": meta_answer,
                "answer_citations": meta_citations,
                "tool_timeline": timeline,
                "available_tools": tool_catalog,
                "assets": assets,
                "generated_image": None,
                "evidence_bundle": evidence_bundle,
                "paper_metadata": metadata,
                "tracking": tracking,
            }

        if not db.is_paper_indexed(arxiv_id) or not is_collection_compatible(arxiv_id):
            _emit(
                timeline,
                progress,
                _timeline_entry("tool", "Preparing paper context", "Downloading and indexing the paper for grounded Q/A.", tool="ensure_paper_context", status="running"),
            )
            prep_raw = await _call_tool(session, "ensure_paper_context", {"arxiv_id": arxiv_id})
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
            raw = await _call_tool(session, tool_name, arguments)
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
            raw = await _call_tool(session, "find_evidence", {"arxiv_id": arxiv_id, "question": question, "max_quotes": 4})
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
                rag_raw = await _call_tool(session, "retrieve_paper_chunks", {
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
        if not (evidence_requested or evidence_used or graphic_requested):
            citations = []

        graphic_source_question = _graphic_source_question(question) if graphic_requested else question
        if graphic_requested and not citations:
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "tool",
                    "Finding image evidence",
                    "Graphic requests need grounded source data, so the orchestrator is collecting quote-level evidence before image generation.",
                    tool="find_evidence",
                    status="running",
                ),
            )
            raw = await _call_tool(session, "find_evidence", {"arxiv_id": arxiv_id, "question": graphic_source_question, "max_quotes": 4})
            result = decode_tool_result(raw)
            error_message = _tool_error_message(raw, result)
            if error_message:
                failed = _timeline_entry("tool", "Image evidence failed", error_message, tool="find_evidence", status="failed")
                timeline.append(failed)
                progress(failed)
            else:
                tool_results.append({"tool": "find_evidence", "arguments": {"arxiv_id": arxiv_id, "question": graphic_source_question}, "result": result})
                citations = _latest_evidence(tool_results)
                done = _timeline_entry("tool", "Image evidence captured", "Captured evidence that will ground the image source answer.", tool="find_evidence", status="completed")
                timeline.append(done)
                progress(done)
                repaired_source = _synthesize_answer(graphic_source_question, metadata, tool_results, continuation_context)
                answer = repaired_source.get("answer") or answer
                citations = repaired_source.get("citations") or citations

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

        # ── Initial download artifacts (pre-repair) ──────────────────────────
        # We create them now so the judge can score artifact_match. If repair
        # later improves the answer we will regenerate them with the better text.
        initial_download_assets: list[dict[str, Any]] = []
        if requested_download_tools:
            # Only include citations in the exported file when the user
            # explicitly asked for them. A "Make me a PDF" request should
            # produce a clean document without appended reference lists.
            citations_for_export = citations if _needs_evidence(question) else []
            citations_json = json.dumps(citations_for_export, ensure_ascii=True)
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
                args = _asset_tool_arguments(tool_name, session_id, metadata, question, answer, citations_json)
                raw = await _call_tool(session, tool_name, args)
                asset = decode_tool_result(raw)
                error_message = _tool_error_message(raw, asset)
                if error_message:
                    failed = _timeline_entry("tool", f"Failed {tool_name}", error_message, tool=tool_name, status="failed")
                    timeline.append(failed)
                    progress(failed)
                    continue
                assets.append(asset)
                initial_download_assets.append(asset)
                done = _timeline_entry("tool", f"Finished {tool_name}", "Generated a downloadable answer artifact.", tool=tool_name, status="completed")
                timeline.append(done)
                progress(done)

        evidence_bundle = {
            "enabled": bool(citations),
            "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
            "items": citations,
        }
        graphic_source_tracking = None
        if graphic_requested:
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "tool",
                    "Validating image source",
                    "Running the judge on the source answer before image generation.",
                    tool="judge_eval",
                    status="running",
                ),
            )
            answer, citations, evidence_bundle, graphic_source_tracking, _ = await _evaluate_and_repair(
                session=session,
                session_id=session_id,
                question=graphic_source_question,
                arxiv_id=arxiv_id,
                metadata=metadata,
                timeline=timeline,
                progress=progress,
                tool_results=tool_results,
                answer=answer,
                citations=citations,
                assets=assets,
                generated_image=None,
                evidence_bundle=evidence_bundle,
                chat_message=answer,
                continuation_context=continuation_context,
                allow_repair=True,
            )

        generated_image = None
        if graphic_requested and _graphic_source_failed(graphic_source_tracking):
            failed = _timeline_entry(
                "tool",
                "Skipped create_graphic",
                "The source answer still failed groundedness/relevance checks after repair, so no image was generated from weak data.",
                tool="create_graphic",
                status="failed",
            )
            timeline.append(failed)
            progress(failed)
        elif graphic_requested:
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
            raw = await _call_tool(
                session,
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
            "enabled": evidence_requested or evidence_used or graphic_requested,
            "pdf_url": f"/paper-pdfs/{arxiv_id}.pdf",
            "items": citations,
        }

        pre_repair_answer = answer
        chat_message: str | None
        if graphic_requested:
            chat_message = "Generated an image representing the paper. See the Generated Graphic panel." if generated_image else None
        else:
            chat_message = answer

        # ── Post-image evaluation ─────────────────────────────────────────────
        # When a graphic was successfully generated the source answer was already
        # validated (and possibly repaired) by the pre-image judge above.
        # Running a full LLM judge again on the text-answer against the graphic
        # question always looks borderline (the "answer" is source prose, not an
        # image), so we score it the same way as continuation artifact requests:
        # mark content metrics not_applicable and only check artifact_match.
        # If the image FAILED to generate, fall through to the normal judge so
        # the user sees a proper quality score on the text answer.
        if graphic_requested and generated_image is not None:
            tracking = evaluate_artifact_request(assets, timeline)
            _emit(
                timeline,
                progress,
                _timeline_entry(
                    "judge",
                    "Finished judge_eval",
                    f"Graphic generated — scored as artifact request "
                    f"(status={tracking.get('overall_status', '?')}).",
                    tool="judge_eval",
                    status="completed" if tracking.get("overall_status") == "passed" else "failed",
                ),
            )
        else:
            answer, citations, evidence_bundle, tracking, chat_message = await _evaluate_and_repair(
                session=session,
                session_id=session_id,
                question=question,
                arxiv_id=arxiv_id,
                metadata=metadata,
                timeline=timeline,
                progress=progress,
                tool_results=tool_results,
                answer=answer,
                citations=citations,
                assets=assets,
                generated_image=generated_image,
                evidence_bundle=evidence_bundle,
                chat_message=chat_message,
                continuation_context=continuation_context,
                allow_repair=True,
            )

        # ── Regenerate download artifacts if repair improved the answer ───────
        # The pre-repair artifacts contain the original (lower-quality) text.
        # Replace them so the user always downloads the best available answer.
        # Include "needs_review" — repair may improve the answer without fully
        # passing the judge (e.g. going from 0.17 → 0.50). The file should
        # still contain the improved text, not the original failure message.
        if (
            tracking.get("overall_status") in ("repaired", "needs_review")
            and requested_download_tools
            and answer != pre_repair_answer
            and initial_download_assets
        ):
            # Drop the stale pre-repair files from the assets list
            stale_ids = {id(a) for a in initial_download_assets}
            assets = [a for a in assets if id(a) not in stale_ids]

            repaired_citations_for_export = citations if _needs_evidence(question) else []
            repaired_citations_json = json.dumps(repaired_citations_for_export, ensure_ascii=True)
            for tool_name in requested_download_tools:
                _emit(
                    timeline,
                    progress,
                    _timeline_entry(
                        "tool",
                        f"Regenerating {tool_name}",
                        "Updating the downloadable file with the repaired, higher-quality answer.",
                        tool=tool_name,
                        status="running",
                    ),
                )
                args = _asset_tool_arguments(tool_name, session_id, metadata, question, answer, repaired_citations_json)
                raw = await _call_tool(session, tool_name, args)
                asset = decode_tool_result(raw)
                error_message = _tool_error_message(raw, asset)
                if error_message:
                    failed = _timeline_entry("tool", f"Failed regenerating {tool_name}", error_message, tool=tool_name, status="failed")
                    timeline.append(failed)
                    progress(failed)
                else:
                    assets.append(asset)
                    done = _timeline_entry("tool", f"Regenerated {tool_name}", "Downloadable file updated with repaired answer.", tool=tool_name, status="completed")
                    timeline.append(done)
                    progress(done)

        # Evidence Viewer panel should only open when the user explicitly asked
        # for evidence/citations/quotes. Repair may add citations to ground the
        # answer but that shouldn't force open the PDF viewer on every question.
        evidence_bundle["enabled"] = evidence_requested and bool(evidence_bundle.get("items"))

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
            "tracking": tracking,
        }
