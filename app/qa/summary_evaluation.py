"""
LLM-as-judge evaluation for the Analyze Paper summary pipeline.

Two independent judge calls (same pattern as Q&A evaluation):
  - Grounding judge : faithfulness + specificity
  - Coverage judge  : completeness + section_accuracy

Results are attached to the interrupt payload so the user sees quality
scores in the approval modal before deciding to approve / reject / revise.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL, QA_JUDGE_ENABLED

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

SUMMARY_METRICS = ("faithfulness", "specificity", "completeness", "section_accuracy")

_CONTENT_SECTIONS = (
    "overview", "problem_addressed", "main_contribution", "method",
    "datasets_experiments", "results", "limitations", "why_it_matters",
)

# ── Judge prompts ──────────────────────────────────────────────────────────────

_GROUNDING_JUDGE_PROMPT = """
You are an evaluator for an AI-generated research paper summary.
The payload contains: the paper abstract, a sample of source chunks from the PDF,
and the generated summary. Evaluate ONLY these two dimensions:

  faithfulness — Are the factual claims in the summary consistent with the
                 provided abstract and source_chunks?

  IMPORTANT SCORING RULES FOR FAITHFULNESS:
  - The source_chunks are a SUBSET of the full paper. Claims not found in the
    chunks may still be grounded in parts of the paper not included here.
  - PASS (score ≥ 0.7): All specific claims (exact numbers, named methods,
    named datasets, specific results) appear in the abstract or source_chunks,
    OR are reasonable high-level characterisations that do not add invented detail.
  - BORDERLINE PASS (0.5–0.69): Some specific details are absent from the
    provided context but are plausible given the paper's topic. No claims
    actively contradict the abstract or chunks.
  - FAIL (score < 0.5): The summary contains fabricated specifics — exact
    percentages, named systems, or experimental results — that are NOT present
    anywhere in the abstract or source_chunks AND are suspiciously precise,
    strongly suggesting hallucination rather than truncated context.
  Only FAIL when you are confident a claim was invented, not merely when context
  is incomplete.

  specificity — Does the summary include CONCRETE details from the paper
                (specific accuracy numbers, dataset names, algorithm names,
                comparison baselines)? Fail only if sections are entirely vague
                generalities that could apply to any research paper.

Reason step-by-step in "thought" BEFORE scoring.
score is 0.0–1.0. note must be one concise sentence.

Return ONLY valid JSON:
{
  "thought": "...",
  "faithfulness": {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "specificity":  {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."}
}
"""

_COVERAGE_JUDGE_PROMPT = """
You are a strict evaluator for an AI-generated research paper summary.
The summary must cover 8 named sections. Evaluate ONLY these two dimensions:

  completeness    — Are all 8 sections (overview, problem_addressed,
                    main_contribution, method, datasets_experiments, results,
                    limitations, why_it_matters) substantively filled with at
                    least 3 meaningful sentences? Fail if any section is blank,
                    "N/A", very short, or says "Not found in retrieved sections."
  section_accuracy — Does each section answer ITS intended question?
                    (e.g. "method" describes the technical approach, "results"
                    gives quantitative outcomes, "limitations" covers failure
                    cases). Fail if sections contain content that belongs elsewhere.

Reason step-by-step in "thought" BEFORE scoring.
score is 0.0–1.0. note must be one concise sentence.

Return ONLY valid JSON:
{
  "thought": "...",
  "completeness":     {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."},
  "section_accuracy": {"status": "pass"|"fail"|"not_applicable", "score": 0.0, "note": "..."}
}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _metric(status: str, score: float, note: str) -> dict[str, Any]:
    return {"status": status, "score": max(0.0, min(1.0, float(score))), "note": note}


def _safe_metric(raw: Any, fallback_note: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _metric("not_applicable", 0.0, fallback_note)
    status = str(raw.get("status", "fail"))
    if status not in {"pass", "fail", "not_applicable"}:
        status = "fail"
    return _metric(status, raw.get("score", 0.0), str(raw.get("note", fallback_note))[:240])


def _call_judge(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def _overall_status(metrics: dict[str, Any]) -> str:
    statuses = [v.get("status") for v in metrics.values() if isinstance(v, dict)]
    if any(s == "fail" for s in statuses):
        return "needs_review"
    return "passed"


# ── Heuristic fallback ─────────────────────────────────────────────────────────

_PLACEHOLDER_PHRASES = {"n/a", "not found in retrieved sections", "summary generation failed", "error during generation"}


def _heuristic_evaluation(summary: dict[str, Any]) -> dict[str, Any]:
    """Fast fallback used when the LLM judge is disabled or fails."""
    empty_count = sum(
        1 for s in _CONTENT_SECTIONS
        if not summary.get(s) or summary.get(s, "").lower().strip() in _PLACEHOLDER_PHRASES
    )
    completeness_ok = empty_count == 0
    return {
        "faithfulness":     _metric("not_applicable", 0.0, "Judge disabled — faithfulness not evaluated."),
        "specificity":      _metric("not_applicable", 0.0, "Judge disabled — specificity not evaluated."),
        "completeness":     _metric(
            "pass" if completeness_ok else "fail",
            1.0 if completeness_ok else 0.0,
            "All sections filled (heuristic)." if completeness_ok
            else f"{empty_count} section(s) are empty or placeholder (heuristic).",
        ),
        "section_accuracy": _metric("not_applicable", 0.0, "Judge disabled — section accuracy not evaluated."),
        "overall_status": "passed" if completeness_ok else "needs_review",
        "grounding_thought": "",
        "coverage_thought": "",
        "source": "heuristic",
    }


# ── Main entry-point ───────────────────────────────────────────────────────────

def evaluate_summary(
    summary: dict[str, Any],
    source_chunks: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run two independent LLM judges on the generated summary.

    Returns a tracking dict that is injected into the interrupt payload so
    the user sees quality scores in the approval modal.
    """
    if not QA_JUDGE_ENABLED:
        result = _heuristic_evaluation(summary)
        result["source"] = "disabled"
        return result

    # Give the judge more context than the original 400-char limit.
    # The summary was generated from up to 12,000 chars of chunks; the judge
    # needs enough of the same material to make a fair faithfulness call.
    # Also include the paper abstract — many high-level claims come from there.
    truncated_chunks = [
        {
            "section": c.get("section", ""),
            "page": c.get("page"),
            "text": (c.get("text", ""))[:700],
        }
        for c in (source_chunks or [])[:16]
    ]

    summary_content = {k: summary.get(k, "") for k in _CONTENT_SECTIONS}

    grounding_payload = {
        "paper_title":    metadata.get("title", ""),
        "paper_abstract": metadata.get("abstract", ""),   # abstract grounds high-level claims
        "summary":        summary_content,
        "source_chunks":  truncated_chunks,
    }
    coverage_payload = {
        "paper_title": metadata.get("title", ""),
        "summary":     summary_content,
    }

    try:
        grounding_raw = _call_judge(_GROUNDING_JUDGE_PROMPT, grounding_payload)
        coverage_raw  = _call_judge(_COVERAGE_JUDGE_PROMPT, coverage_payload)

        metrics = {
            "faithfulness":     _safe_metric(grounding_raw.get("faithfulness"),     "Grounding judge did not return this metric."),
            "specificity":      _safe_metric(grounding_raw.get("specificity"),      "Grounding judge did not return this metric."),
            "completeness":     _safe_metric(coverage_raw.get("completeness"),      "Coverage judge did not return this metric."),
            "section_accuracy": _safe_metric(coverage_raw.get("section_accuracy"), "Coverage judge did not return this metric."),
        }
        return {
            **metrics,
            "overall_status":    _overall_status(metrics),
            "grounding_thought": grounding_raw.get("thought", ""),
            "coverage_thought":  coverage_raw.get("thought", ""),
            "source": "llm_judge",
        }
    except Exception as exc:
        result = _heuristic_evaluation(summary)
        result["evaluation_error"] = str(exc)
        return result
