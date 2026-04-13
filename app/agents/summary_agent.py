"""
Summary agent — generates structured summary from retrieved chunks.
Uses ONLY retrieved chunks — never invents details.
Follows a fixed 10-section template.

Phase 3 — Week 10: Multi-Agent Workflows. Single responsibility: produce Summary.
"""
import json
from typing import Any

from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts import SUMMARY_SYSTEM_PROMPT
from app.schemas import Summary
from app.mcp_client import get_mcp_client
from app.qa.summary_evaluation import evaluate_summary

# 90-second timeout — gpt-4o with a large context can take 30-60 s
client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)

# Max characters of chunk context to send — gpt-4o handles 128k tokens;
# 24k chars gives ~6k tokens of context, well within limits.
_MAX_CONTEXT_CHARS = 24_000

# Targeted semantic queries for each summary section.
# These run as separate retrieve_paper_chunks calls so even sections buried
# deep in a long PDF (datasets, results, limitations) get their own chunks.
_SECTION_QUERIES: dict[str, str] = {
    "overview":            "overview introduction abstract background motivation",
    "problem_addressed":   "problem challenge gap limitation existing work prior methods",
    "main_contribution":   "contribution novelty proposed key insight innovation",
    "method":              "methodology method approach architecture algorithm framework design",
    "datasets_experiments":"dataset experiment benchmark evaluation setup training testing data",
    "results":             "results performance accuracy findings improvement comparison baseline",
    "limitations":         "limitations future work drawbacks constraints failure cases scope",
    "why_it_matters":      "impact significance applications real-world deployment broader impact",
}

_SUMMARY_CONTENT_KEYS = (
    "overview",
    "problem_addressed",
    "main_contribution",
    "method",
    "datasets_experiments",
    "results",
    "limitations",
    "why_it_matters",
)

_SUMMARY_METRIC_KEYS = ("faithfulness", "specificity", "completeness", "section_accuracy")

_SUMMARY_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "conservative",
        "name": "Conservative Agent",
        "temperature": 0.0,
        "instruction": (
            "Generate the most source-faithful summary possible. Prefer cautious wording, "
            "avoid unsupported specifics, and say when a detail is not found in the retrieved text."
        ),
    },
    {
        "id": "technical",
        "name": "Technical Agent",
        "temperature": 0.2,
        "instruction": (
            "Generate a technical research summary. Emphasize methods, datasets, experimental "
            "setup, results, limitations, and concrete details that appear in the retrieved text."
        ),
    },
    {
        "id": "teaching",
        "name": "Teaching Agent",
        "temperature": 0.2,
        "instruction": (
            "Generate a clear class-friendly summary. Explain the paper in accessible language "
            "while preserving the required structure and using only retrieved text."
        ),
    },
)


def _fallback_summary_data(confidence_note: str) -> dict[str, str]:
    return {
        "overview": "Summary generation failed.",
        "problem_addressed": "N/A",
        "main_contribution": "N/A",
        "method": "N/A",
        "datasets_experiments": "N/A",
        "results": "N/A",
        "limitations": "N/A",
        "why_it_matters": "N/A",
        "confidence_note": confidence_note,
    }


def _normalise_summary_data(raw: dict[str, Any], confidence_note: str) -> dict[str, str]:
    fallback = _fallback_summary_data(confidence_note)
    data: dict[str, str] = {}
    for key in _SUMMARY_CONTENT_KEYS:
        value = raw.get(key)
        data[key] = str(value).strip() if value else fallback[key]
    data["confidence_note"] = str(raw.get("confidence_note") or confidence_note).strip()
    return data


def _make_summary(arxiv_id: str, title: str, summary_data: dict[str, Any]) -> Summary:
    return Summary(
        arxiv_id=arxiv_id,
        title=title,
        **{key: str(summary_data.get(key, "")) for key in (*_SUMMARY_CONTENT_KEYS, "confidence_note")},
    )


def _build_summary_user_content(
    *,
    arxiv_id: str,
    metadata: dict[str, Any],
    context: str,
    revision_note: str = "",
) -> str:
    content = f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}"
    if revision_note:
        content += (
            "\n\nRevision requested by the human reviewer. Apply this feedback while still using "
            f"only the retrieved paper text:\n{revision_note}"
        )
    return content


def _candidate_system_prompt(candidate: dict[str, Any]) -> str:
    return (
        f"{SUMMARY_SYSTEM_PROMPT.strip()}\n\n"
        f"Competitive summary role: {candidate['name']}.\n"
        f"Role-specific instruction: {candidate['instruction']}\n"
        "You are one independent candidate in a best-answer-wins competition."
    )


def _generate_candidate_summary(
    *,
    candidate: dict[str, Any],
    arxiv_id: str,
    metadata: dict[str, Any],
    context: str,
    confidence_note: str,
    revision_note: str,
) -> Summary:
    print(f"[summary_agent] Generating {candidate['name']} candidate...")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _candidate_system_prompt(candidate)},
            {"role": "user", "content": _build_summary_user_content(
                arxiv_id=arxiv_id,
                metadata=metadata,
                context=context,
                revision_note=revision_note,
            )},
        ],
        response_format={"type": "json_object"},
        temperature=float(candidate.get("temperature", 0.2)),
    )
    raw = json.loads(response.choices[0].message.content or "{}")
    print(
        f"[summary_agent] {candidate['name']} generated "
        f"({len(response.choices[0].message.content or '')} chars)"
    )
    return _make_summary(
        arxiv_id=arxiv_id,
        title=metadata.get("title") or arxiv_id,
        summary_data=_normalise_summary_data(raw, confidence_note),
    )


def _metric_score(evaluation: dict[str, Any], key: str) -> float:
    metric = evaluation.get(key, {}) if isinstance(evaluation, dict) else {}
    try:
        return max(0.0, min(1.0, float(metric.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _candidate_average_score(candidate: dict[str, Any]) -> float:
    evaluation = candidate.get("evaluation", {})
    scores = [_metric_score(evaluation, key) for key in _SUMMARY_METRIC_KEYS]
    return sum(scores) / len(scores)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    evaluation = candidate.get("evaluation", {})
    faithfulness_pass = evaluation.get("faithfulness", {}).get("status") == "pass"
    order = int(candidate.get("order", 999))
    return (
        1.0 if faithfulness_pass else 0.0,
        _candidate_average_score(candidate),
        _metric_score(evaluation, "completeness"),
        _metric_score(evaluation, "specificity"),
        -order,
    )


def select_summary_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best candidate deterministically so the competition is testable."""
    if not candidates:
        raise ValueError("select_summary_candidate requires at least one candidate")
    return max(candidates, key=_candidate_sort_key)


def build_summary_competition(
    *,
    candidates: list[dict[str, Any]],
    winner: dict[str, Any],
    repair_status: str,
) -> dict[str, Any]:
    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)
    scorecards = []
    for index, candidate in enumerate(ranked, start=1):
        evaluation = candidate.get("evaluation", {})
        scorecards.append({
            "rank": index,
            "id": candidate.get("id"),
            "name": candidate.get("name"),
            "selected": candidate.get("id") == winner.get("id"),
            "overall_status": evaluation.get("overall_status", "needs_review"),
            "score": round(_candidate_average_score(candidate), 3),
            "faithfulness": evaluation.get("faithfulness", {}).get("status", "not_applicable"),
            "specificity": evaluation.get("specificity", {}).get("score", 0.0),
            "completeness": evaluation.get("completeness", {}).get("score", 0.0),
            "section_accuracy": evaluation.get("section_accuracy", {}).get("score", 0.0),
        })

    winner_score = round(_candidate_average_score(winner), 3)
    return {
        "mode": "competitive_best_answer_wins",
        "winner_id": winner.get("id"),
        "winner_name": winner.get("name"),
        "winner_score": winner_score,
        "repair_status": repair_status,
        "scorecards": scorecards,
        "selection_rationale": (
            f"{winner.get('name')} selected with score {winner_score:.2f}. "
            "Faithfulness pass is prioritized before average judge score."
        ),
    }


def run_summary_agent(state: dict) -> dict:
    """Generate a structured summary from paper chunks via MCP retrieval."""
    mcp = get_mcp_client()
    arxiv_id = state.get("selected_arxiv_id")
    if not arxiv_id:
        return {"error": "summary_agent: selected_arxiv_id is missing from state"}

    # ── Pass 1: section-label retrieval (fast, works when chunker labels match) ──
    sections_to_cover = ["introduction", "method", "results", "conclusion", "limitations"]
    all_chunks = []
    for section in sections_to_cover:
        chunk_result = mcp.call_tool("get_paper_section", {
            "arxiv_id": arxiv_id,
            "section_hint": section,
        })
        all_chunks.extend(chunk_result.get("chunks", [])[:2])

    # ── Pass 2: section-targeted semantic retrieval ───────────────────────────
    # One dedicated retrieve_paper_chunks call per summary section so that
    # unlabeled or deeply-buried sections (datasets, results, limitations in a
    # 177-page paper) still surface relevant chunks via embedding similarity.
    metadata = mcp.call_tool("get_paper_metadata", {"arxiv_id": arxiv_id})
    # MCP may return an error dict or None — normalise to empty dict
    if not isinstance(metadata, dict) or "error" in metadata:
        metadata = {}

    for section_key, query in _SECTION_QUERIES.items():
        targeted_chunks = mcp.call_tool("retrieve_paper_chunks", {
            "arxiv_id": arxiv_id,
            "question": query,
            "k": 3,
        })
        if isinstance(targeted_chunks, list):
            # Tag each chunk with which summary section it was fetched for
            for c in targeted_chunks:
                c["_for_section"] = section_key
            all_chunks.extend(targeted_chunks)
        print(f"[summary_agent] Section '{section_key}': "
              f"{len(targeted_chunks) if isinstance(targeted_chunks, list) else 0} chunks")

    # ── Pass 3: title-relevance retrieval (anchors the overview) ─────────────
    title_chunks = mcp.call_tool("retrieve_paper_chunks", {
        "arxiv_id": arxiv_id,
        "question": metadata.get("title", arxiv_id),
        "k": 3,
    })
    if isinstance(title_chunks, list):
        all_chunks.extend(title_chunks)

    # Deduplicate by chunk_id (keep first occurrence — earlier passes are more reliable)
    seen: set[str] = set()
    unique_chunks = []
    for c in all_chunks:
        cid = c.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(c)

    print(f"[summary_agent] {len(unique_chunks)} unique chunks collected across all retrieval passes")

    # Build context string — label each chunk with both its structural section
    # (from chunker metadata) and the summary section it was fetched for.
    context_parts = []
    for c in unique_chunks:
        structural = c.get("section", "UNKNOWN").upper()
        for_section = c.get("_for_section", "").upper()
        label = f"[{structural}]" if not for_section else f"[{structural} → for:{for_section}]"
        context_parts.append(f"{label}\n{c.get('text', '')}")

    context = "\n\n".join(context_parts)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[...context truncated for length...]"

    print(f"[summary_agent] Sending {len(context):,} chars of context to {OPENAI_MODEL}")

    if not context.strip():
        # Fallback: use abstract if no chunks available
        abstract = metadata.get("abstract", "Abstract not available.")
        context = f"[ABSTRACT]\n{abstract}"
        confidence_note = "Based on abstract only — PDF analysis unavailable"
    else:
        confidence_note = "Based on full PDF analysis"

    revision_note = state.get("revision_note", "")

    # ── Competitive summary generation ──────────────────────────────────────
    # Three independent summary candidates use the same source evidence but
    # different instructions. The judge then picks the strongest candidate.
    candidates: list[dict[str, Any]] = []
    print(f"[summary_agent] Generating competitive summaries with {OPENAI_MODEL}...")
    for order, candidate in enumerate(_SUMMARY_CANDIDATES):
        try:
            candidate_summary = _generate_candidate_summary(
                candidate=candidate,
                arxiv_id=arxiv_id,
                metadata=metadata,
                context=context,
                confidence_note=confidence_note,
                revision_note=revision_note,
            )
        except Exception as e:
            print(f"[summary_agent] {candidate['name']} LLM call failed: {type(e).__name__}: {e}")
            candidate_summary = _make_summary(
                arxiv_id=arxiv_id,
                title=metadata.get("title") or arxiv_id,
                summary_data=_fallback_summary_data("Error during generation"),
            )

        print(f"[summary_agent] Judging {candidate['name']}...")
        evaluation = evaluate_summary(candidate_summary.model_dump(), unique_chunks, metadata)
        print(
            f"[summary_agent] {candidate['name']} evaluation: {evaluation.get('overall_status')} "
            f"(faithfulness={evaluation.get('faithfulness', {}).get('status', '?')}, "
            f"score={round(sum(_metric_score(evaluation, key) for key in _SUMMARY_METRIC_KEYS) / len(_SUMMARY_METRIC_KEYS), 3)})"
        )
        candidates.append({
            "id": candidate["id"],
            "name": candidate["name"],
            "order": order,
            "summary": candidate_summary.model_dump(),
            "evaluation": evaluation,
        })

    winner = select_summary_candidate(candidates)
    summary = _make_summary(
        arxiv_id=arxiv_id,
        title=metadata.get("title") or arxiv_id,
        summary_data=winner["summary"],
    )
    evaluation = winner["evaluation"]
    repair_status = "not_needed"
    print(
        f"[summary_agent] Selected {winner['name']} summary "
        f"(score={_candidate_average_score(winner):.3f}, "
        f"faithfulness={evaluation.get('faithfulness', {}).get('status', '?')})"
    )

    # ── Auto-repair on faithfulness failure ───────────────────────────────────
    # If the winning candidate is not grounded, repair only the selected winner.
    if evaluation.get("faithfulness", {}).get("status") == "fail":
        print(f"[summary_agent] Winning candidate failed faithfulness — running repair pass...")
        repair_user_content = (
            f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}\n\n"
            "REVISION REQUIRED — the selected competitive summary contained claims that could "
            "not be traced to the source text above. Regenerate the summary using "
            "ONLY information explicitly stated in the text provided. Apply these rules:\n"
            "1. Do NOT include exact numbers, dataset names, system names, or results "
            "unless they appear verbatim in the text above.\n"
            "2. If you are unsure whether a specific detail is in the text, describe "
            "the concept generally rather than speculating.\n"
            "3. Every sentence must be directly traceable to a passage above.\n"
            "4. It is better to be slightly vague than to state an unverified specific."
        )
        try:
            repair_response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,   # deterministic for repair
            )
            repaired_data = json.loads(repair_response.choices[0].message.content)
            repaired_data["confidence_note"] = confidence_note
            summary = _make_summary(
                arxiv_id=arxiv_id,
                title=metadata.get("title") or arxiv_id,
                summary_data=_normalise_summary_data(repaired_data, confidence_note),
            )
            evaluation = evaluate_summary(summary.model_dump(), unique_chunks, metadata)
            winner["summary"] = summary.model_dump()
            winner["evaluation"] = evaluation
            repair_status = (
                "repaired"
                if evaluation.get("faithfulness", {}).get("status") == "pass"
                else "repair_attempted"
            )
            print(f"[summary_agent] Post-repair evaluation: {evaluation.get('overall_status')} "
                  f"(faithfulness={evaluation.get('faithfulness', {}).get('status', '?')})")
        except Exception as exc:
            print(f"[summary_agent] Repair pass failed: {exc}")
            repair_status = "repair_failed"

    competition = build_summary_competition(
        candidates=candidates,
        winner=winner,
        repair_status=repair_status,
    )

    return {
        "draft_summary": summary.model_dump(),
        "summary_evaluation": evaluation,
        "summary_candidates": candidates,
        "summary_competition": competition,
    }
