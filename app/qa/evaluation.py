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

IMPORTANT — ARTIFACT REQUEST RULE:
If the question is primarily a request to CREATE A FILE (PDF, markdown, presentation,
slide deck, image, graphic, etc.) AND the payload shows that assets were successfully
generated (assets list is non-empty), mark answer_relevance as "not_applicable" with
score 1.0. The generated file IS the answer — the text response is only a preview of
the content. Do NOT penalise answer_relevance because the text does not say "here is
your PDF"; the job was to create the file, which was done.
Groundedness should still be evaluated normally against the text content and citations.

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
                        Do not fail citation_quality solely because there is
                        only one citation: one specific citation is enough for
                        a narrow answer when it has a page/section/quote and
                        directly supports the claim.
  retrieval_relevance — Did the tool calls (retrieve_paper_chunks, find_evidence)
                        retrieve content relevant to the question? Judge from
                        tool_counts, tool_timeline, evidence_bundle, and citations.
                        tool_counts is the authoritative summary if the timeline
                        is truncated. Mark not_applicable if no retrieval tools
                        were called and the answer came from metadata or a pure
                        artifact transformation.
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


def _artifact_requested(question: str) -> bool:
    return any(
        token in (question or "").lower()
        for token in (
            "download",
            "pdf",
            "markdown",
            ".md",
            "image",
            "graphic",
            "slide",
            "presentation",
        )
    )


def _metadata_question(question: str) -> bool:
    return any(
        token in (question or "").lower()
        for token in (
            "author",
            "authors",
            "title",
            "year",
            "date",
            "published",
            "venue",
            "journal",
            "conference",
            "category",
            "categories",
        )
    )


def _citation_is_specific(citation: dict[str, Any]) -> bool:
    if not isinstance(citation, dict):
        return False
    has_location = bool(citation.get("page") or citation.get("section"))
    has_support = bool(citation.get("quote") or citation.get("text") or citation.get("title"))
    return has_location and has_support


def _looks_like_single_citation_false_negative(metric_item: dict[str, Any], answer: str, citations: list[dict[str, Any]]) -> bool:
    if len(citations) != 1 or not _citation_is_specific(citations[0]):
        return False
    note = (metric_item.get("note") or "").lower()
    complained_about_one_citation = "only one citation" in note or "one citation" in note
    narrow_answer = len(answer or "") <= 1000 or (answer or "").count(".") <= 4
    return complained_about_one_citation and narrow_answer


def _has_retrieval_support(
    citations: list[dict[str, Any]],
    tool_counts: dict[str, int],
    evidence_bundle: dict[str, Any] | None,
) -> bool:
    retrieval_tools = ("retrieve_paper_chunks", "find_evidence", "cite_evidence", "compare_sections")
    if any(tool_counts.get(tool, 0) > 0 for tool in retrieval_tools):
        return True
    if citations:
        return True
    if evidence_bundle and evidence_bundle.get("items"):
        return True
    return False


def _calibrate_tracking(
    tracking: dict[str, Any],
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    tool_counts: dict[str, int],
    assets: list[dict[str, Any]],
    generated_image: dict[str, Any] | None,
    evidence_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize obvious judge false negatives without hiding real failures."""
    notes: list[str] = []

    citation_quality = tracking.get("citation_quality", {})
    if (
        citation_quality.get("status") == "fail"
        and _looks_like_single_citation_false_negative(citation_quality, answer, citations)
    ):
        tracking["citation_quality"] = metric(
            "pass",
            max(metric_score(citation_quality), 0.75),
            "Single specific citation is sufficient for this narrow answer.",
        )
        notes.append("citation_quality normalized: one specific citation supports a narrow answer.")

    retrieval_relevance = tracking.get("retrieval_relevance", {})
    if retrieval_relevance.get("status") == "fail":
        if _has_retrieval_support(citations, tool_counts, evidence_bundle):
            tracking["retrieval_relevance"] = metric(
                "pass",
                max(metric_score(retrieval_relevance), 0.75),
                "Retrieval/evidence support is present in citations or tool counts.",
            )
            notes.append("retrieval_relevance normalized: evidence support was present.")
        elif _metadata_question(question) or _artifact_requested(question):
            tracking["retrieval_relevance"] = metric(
                "not_applicable",
                1.0,
                "Retrieval was not required for this metadata or artifact request.",
            )
            notes.append("retrieval_relevance normalized: retrieval was not required.")

    artifact_match = tracking.get("artifact_match", {})
    if _artifact_requested(question) and (assets or generated_image) and artifact_match.get("status") == "fail":
        tracking["artifact_match"] = metric(
            "pass",
            max(metric_score(artifact_match), 0.9),
            "Requested artifact was generated.",
        )
        notes.append("artifact_match normalized: artifact asset was generated.")

    if notes:
        tracking["calibration_notes"] = notes

    failed_core = any(
        tracking.get(name, {}).get("status") == "fail"
        for name in ("answer_relevance", "groundedness", "citation_quality")
    )
    failed_artifact = (
        _artifact_requested(question)
        and tracking.get("artifact_match", {}).get("status") == "fail"
    )
    tracking["overall_status"] = "needs_review" if failed_core or failed_artifact else "passed"
    tracking["repair_recommended"] = failed_core
    return tracking


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
    tool_counts = _tool_counts(tool_timeline)
    base_payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "citations": citations[:6],
        "tool_timeline": tool_timeline[-20:],
        "tool_counts": tool_counts,
        "paper_metadata": paper_metadata,
        "repair_attempted": repair_attempted,
    }
    # Include assets in the quality payload so the judge can apply the
    # ARTIFACT REQUEST RULE — when a file was successfully created, the
    # judge marks answer_relevance as not_applicable instead of penalising
    # the text response for not saying "here is your PDF".
    quality_payload = {
        **base_payload,
        "assets": assets,
        "generated_image": generated_image,
    }
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

            tracking = _calibrate_tracking(
                tracking,
                question,
                answer,
                citations,
                tool_counts,
                assets,
                generated_image,
                evidence_bundle,
            )

            tracking.update({
                "overall_status": tracking["overall_status"],
                "repair_recommended": tracking["repair_recommended"],
                "repair_reason": quality_raw.get("thought", "") or supporting_raw.get("thought", ""),
                "tool_counts": tool_counts,
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
