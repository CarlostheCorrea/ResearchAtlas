"""
LLM-as-judge evaluation for MCP-driven Q/A.

Two independent judge calls eliminate correlation bias:
  - Quality judge : answer_relevance + groundedness  (highest stakes)
  - Supporting judge: citation_quality + retrieval_relevance +
                      tool_choice_quality + artifact_match

Each judge uses chain-of-thought reasoning ("thought" field) before scoring,
which produces better-calibrated scores than forcing the model to score
without reasoning first.

Phoenix span attributes are set after each judge call so the Phoenix UI
can filter / aggregate by metric score.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL, QA_JUDGE_ENABLED
from app.observability import phoenix_status, trace_span

client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)

# All metric names — order matters for the UI
METRICS = (
    "answer_relevance",
    "groundedness",
    "citation_quality",
    "retrieval_relevance",
    "tool_choice_quality",
    "artifact_match",
)

# ── Judge prompts ─────────────────────────────────────────────────────────────

_QUALITY_JUDGE_PROMPT = """
You are a strict evaluator for a research-paper Q/A application.
Evaluate ONLY these two dimensions from the payload:

  answer_relevance — Does the answer directly and completely address the question?
  groundedness     — Are ALL factual claims in the answer supported by the
                     provided citations/evidence? Fail this if the answer
                     asserts paper content that no citation supports.

IMPORTANT — METADATA GROUNDING RULE:
If the question asks about authorship, title, publication year/date, venue, journal,
conference, or arXiv categories, AND the answer is consistent with the paper_metadata
field in the payload, mark groundedness as PASS even if citations only reference
"Paper Metadata" (not RAG page chunks). The paper_metadata object is always
an authoritative source for these fields — no RAG page citation is required.

Rules:
- Reason step-by-step in the "thought" field BEFORE assigning scores.
- Do not use outside knowledge — judge only from the supplied payload.
- score is a float from 0.0 (worst) to 1.0 (best).
- note must be one concise sentence explaining the score.

Return ONLY valid JSON:
{
  "thought": "Step-by-step reasoning before scoring ...",
  "answer_relevance": {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "groundedness":     {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."}
}
"""

_SUPPORTING_JUDGE_PROMPT = """
You are a strict evaluator for a research-paper Q/A application.
Evaluate ONLY these four supporting dimensions from the payload:

  citation_quality    — Are citations specific, correctly attributed, and
                        well-formatted? Fail if citations are vague or absent
                        when a factual claim was made. NOTE: citations with
                        section="Paper Metadata" are valid and correct for
                        answers derived from paper metadata fields (authors,
                        title, year, venue, categories).
  retrieval_relevance — Did the tool calls (retrieve_paper_chunks, find_evidence)
                        retrieve content relevant to the question? Judge from
                        the tool_timeline and citations. Mark not_applicable if
                        no retrieval tools were called.
  tool_choice_quality — Were the right MCP tools called in a sensible sequence?
                        Fail if unnecessary tools were called or critical ones
                        were skipped.
  artifact_match      — Did generated files / images match what the user
                        requested? Mark not_applicable if no artifact was
                        requested.

Rules:
- Reason step-by-step in the "thought" field BEFORE assigning scores.
- Do not use outside knowledge — judge only from the supplied payload.
- score is a float from 0.0 (worst) to 1.0 (best).
- note must be one concise sentence explaining the score.

Return ONLY valid JSON:
{
  "thought": "Step-by-step reasoning before scoring ...",
  "citation_quality":    {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "retrieval_relevance": {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "tool_choice_quality": {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "artifact_match":      {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."}
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def metric(status: str, score: float, note: str) -> dict[str, Any]:
    return {"status": status, "score": max(0.0, min(1.0, float(score))), "note": note}


def metric_score(item: dict[str, Any] | None) -> float:
    if not item:
        return 0.0
    try:
        return float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def tracking_score(tracking: dict[str, Any] | None) -> float:
    if not tracking:
        return 0.0
    scores = [
        metric_score(item)
        for name in METRICS
        if (item := tracking.get(name)) and item.get("status") != "not_applicable"
    ]
    return sum(scores) / len(scores) if scores else 0.0


def should_repair(tracking: dict[str, Any] | None) -> bool:
    if not tracking:
        return False
    if tracking.get("repair_recommended"):
        return True
    return any(
        tracking.get(name, {}).get("status") == "fail"
        for name in ("answer_relevance", "groundedness", "citation_quality")
    )


def _tool_counts(tool_timeline: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in tool_timeline:
        tool = event.get("tool")
        if event.get("kind") == "tool" and event.get("status") == "completed" and tool:
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def _safe_metric(raw: Any, fallback_note: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return metric("not_applicable", 0.0, fallback_note)
    status = str(raw.get("status", "fail"))
    if status not in {"pass", "fail", "not_applicable"}:
        status = "fail"
    return metric(status, raw.get("score", 0.0), str(raw.get("note", fallback_note))[:240])


def _call_judge(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Single judge call — returns raw parsed JSON or empty dict on error."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content or "{}")


# ── Local fallback (judge disabled or unavailable) ────────────────────────────

def _local_tracking(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    tool_timeline: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    """Heuristic-only evaluation used when the LLM judge is off or fails.

    Scores are deliberately 0.0 to signal these are not real judge scores.
    """
    has_answer = bool((answer or "").strip())
    has_citations = bool(citations)
    requested_artifact = any(
        token in question.lower()
        for token in ("download", "pdf", "markdown", ".md", "image", "graphic", "slide", "presentation")
    )
    has_assets = bool(assets)
    has_retrieval = any(
        e.get("tool") in ("retrieve_paper_chunks", "find_evidence")
        for e in tool_timeline
        if e.get("status") == "completed"
    )

    tracking = {
        "answer_relevance": metric(
            "pass" if has_answer else "fail", 0.0,
            "Answer text produced (heuristic — LLM judge disabled)." if has_answer
            else "No answer text produced.",
        ),
        "groundedness": metric(
            "pass" if has_citations else "not_applicable", 0.0,
            "Citations present (heuristic)." if has_citations
            else "No citations — cannot assess groundedness locally.",
        ),
        "citation_quality": metric(
            "pass" if has_citations else "not_applicable", 0.0,
            "Citation objects returned (heuristic)." if has_citations
            else "No citations returned.",
        ),
        "retrieval_relevance": metric(
            "pass" if has_retrieval else "not_applicable", 0.0,
            "Retrieval tools were called (heuristic)." if has_retrieval
            else "No retrieval tools called.",
        ),
        "tool_choice_quality": metric(
            "pass" if tool_timeline else "fail", 0.0,
            "Tool timeline captured (heuristic)." if tool_timeline
            else "No tool timeline captured.",
        ),
        "artifact_match": metric(
            "pass" if has_assets else "fail" if requested_artifact else "not_applicable",
            0.0,
            "Artifact generated (heuristic)." if has_assets
            else "Artifact was requested but not generated." if requested_artifact
            else "No artifact requested.",
        ),
        "overall_status": "needs_review" if error else "passed",
        "repair_recommended": False,
        "repair_reason": error or "",
        "tool_counts": _tool_counts(tool_timeline),
        "phoenix": phoenix_status(),
        "source": "local_fallback",
    }
    if error:
        tracking["evaluation_error"] = error
    return tracking


# ── Artifact-only evaluation (continuation requests) ─────────────────────────

def evaluate_artifact_request(
    assets: list[dict[str, Any]],
    tool_timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Lightweight tracking for pure artifact-creation continuations.

    When the user says 'Make this a PDF' or 'Turn it into an image', they are
    asking to reformat the previous answer — no new content Q&A is happening.
    Running the full LLM judge on the status-confirmation message produces
    misleading answer_relevance failures. Instead, mark content metrics as
    not_applicable and only evaluate artifact_match.
    """
    has_assets = bool(assets)
    return {
        "answer_relevance":    metric("not_applicable", 1.0, "Pure artifact request — no content answer to evaluate."),
        "groundedness":        metric("not_applicable", 1.0, "No new content claims — transforming previous answer."),
        "citation_quality":    metric("not_applicable", 1.0, "No new citations — using previous answer's context."),
        "retrieval_relevance": metric("not_applicable", 1.0, "No retrieval needed — transforming previous answer."),
        "tool_choice_quality": metric(
            "pass" if has_assets else "fail",
            1.0 if has_assets else 0.0,
            "Artifact created successfully from previous answer." if has_assets
            else "Artifact creation failed.",
        ),
        "artifact_match": metric(
            "pass" if has_assets else "fail",
            1.0 if has_assets else 0.0,
            "Requested file generated from previous answer." if has_assets
            else "File generation failed.",
        ),
        "overall_status": "passed" if has_assets else "needs_review",
        "repair_recommended": False,
        "repair_reason": "",
        "tool_counts": _tool_counts(tool_timeline),
        "phoenix": phoenix_status(),
        "source": "artifact_only",
    }


# ── Main evaluation entry-point ───────────────────────────────────────────────

def evaluate_qa_response(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    tool_timeline: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    generated_image: dict[str, Any] | None,
    evidence_bundle: dict[str, Any] | None,
    paper_metadata: dict[str, Any] | None,
    repair_attempted: bool = False,
) -> dict[str, Any]:
    if not QA_JUDGE_ENABLED:
        tracking = _local_tracking(question, answer, citations, tool_timeline, assets)
        tracking["source"] = "disabled"
        tracking["overall_status"] = "passed"
        return tracking

    # Shared payload slices for both judges
    base_payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "citations": citations[:6],
        "tool_timeline": tool_timeline[-12:],
        "paper_metadata": paper_metadata,
        "repair_attempted": repair_attempted,
    }
    quality_payload = {**base_payload}
    supporting_payload = {
        **base_payload,
        "assets": assets,
        "generated_image": generated_image,
        "evidence_bundle": evidence_bundle,
    }

    quality_raw: dict[str, Any] = {}
    supporting_raw: dict[str, Any] = {}

    try:
        with trace_span("judge_eval", {
            "qa.question": question[:200],
            "qa.repair_attempted": repair_attempted,
        }) as span:
            # ── Call 1: quality (answer_relevance + groundedness) ─────────────
            quality_raw = _call_judge(_QUALITY_JUDGE_PROMPT, quality_payload)

            # ── Call 2: supporting (citations + retrieval + tools + artifacts) ─
            supporting_raw = _call_judge(_SUPPORTING_JUDGE_PROMPT, supporting_payload)

            # ── Merge results ─────────────────────────────────────────────────
            tracking: dict[str, Any] = {
                "answer_relevance":    _safe_metric(quality_raw.get("answer_relevance"),    "Quality judge did not return this metric."),
                "groundedness":        _safe_metric(quality_raw.get("groundedness"),        "Quality judge did not return this metric."),
                "citation_quality":    _safe_metric(supporting_raw.get("citation_quality"), "Supporting judge did not return this metric."),
                "retrieval_relevance": _safe_metric(supporting_raw.get("retrieval_relevance"), "Supporting judge did not return this metric."),
                "tool_choice_quality": _safe_metric(supporting_raw.get("tool_choice_quality"), "Supporting judge did not return this metric."),
                "artifact_match":      _safe_metric(supporting_raw.get("artifact_match"),   "Supporting judge did not return this metric."),
            }

            failed_core = any(
                tracking[name]["status"] == "fail"
                for name in ("answer_relevance", "groundedness", "citation_quality")
            )
            failed_any = any(
                item["status"] == "fail"
                for item in tracking.values()
                if isinstance(item, dict)
            )

            tracking.update({
                "overall_status": "needs_review" if failed_any else "passed",
                "repair_recommended": failed_core,
                "repair_reason": quality_raw.get("thought", "") or supporting_raw.get("thought", ""),
                "tool_counts": _tool_counts(tool_timeline),
                "phoenix": phoenix_status(),
                "source": "llm_judge",
                # Expose CoT thoughts for the UI
                "quality_thought":    (quality_raw.get("thought") or ""),
                "supporting_thought": (supporting_raw.get("thought") or ""),
            })

            # ── Set Phoenix span attributes so metrics are queryable in UI ────
            if span is not None:
                for name in METRICS:
                    m = tracking.get(name, {})
                    if isinstance(m, dict):
                        span.set_attribute(f"eval.{name}.score",  m.get("score", 0.0))
                        span.set_attribute(f"eval.{name}.status", m.get("status", "unknown"))
                span.set_attribute("eval.overall_status", tracking["overall_status"])
                span.set_attribute("eval.overall_score",  tracking_score(tracking))

    except Exception as exc:
        return _local_tracking(question, answer, citations, tool_timeline, assets, error=str(exc))

    return tracking
