"""
Summary agent — generates structured summary from retrieved chunks.
Uses ONLY retrieved chunks — never invents details.
Follows a fixed 10-section template.

Phase 3 — Week 10: Multi-Agent Workflows. Single responsibility: produce Summary.
"""
import json
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

    try:
        print(f"[summary_agent] Calling {OPENAI_MODEL} for summary...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}"
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        summary_data = json.loads(response.choices[0].message.content)
        print(f"[summary_agent] Summary generated ({len(response.choices[0].message.content)} chars)")
    except Exception as e:
        print(f"[summary_agent] LLM call failed: {type(e).__name__}: {e}")
        summary_data = {
            "overview": "Summary generation failed.",
            "problem_addressed": "N/A",
            "main_contribution": "N/A",
            "method": "N/A",
            "datasets_experiments": "N/A",
            "results": "N/A",
            "limitations": "N/A",
            "why_it_matters": "N/A",
            "confidence_note": "Error during generation",
        }

    summary_data["confidence_note"] = summary_data.get("confidence_note", confidence_note)

    revision_note = state.get("revision_note", "")
    if revision_note:
        # Append revision context and regenerate
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}\n\n"
                        f"Revision requested: {revision_note}"
                    )},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            summary_data = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[summary_agent] revision LLM call failed: {e}")

    summary = Summary(
        arxiv_id=arxiv_id,
        title=metadata.get("title") or arxiv_id,
        **{k: v for k, v in summary_data.items() if k in Summary.model_fields and v is not None},
    )

    # Run the LLM-as-judge on the final summary (after any revision pass)
    print(f"[summary_agent] Running quality evaluation...")
    evaluation = evaluate_summary(summary.model_dump(), unique_chunks, metadata)
    print(f"[summary_agent] Evaluation: {evaluation.get('overall_status')} "
          f"(faithfulness={evaluation.get('faithfulness', {}).get('status', '?')}, "
          f"completeness={evaluation.get('completeness', {}).get('status', '?')})")

    # ── Auto-repair on faithfulness failure ───────────────────────────────────
    # If the judge finds that claims are not grounded in the source text, do one
    # repair pass with a maximally conservative prompt before showing the user.
    if evaluation.get("faithfulness", {}).get("status") == "fail":
        print(f"[summary_agent] Faithfulness failed — running repair pass...")
        repair_user_content = (
            f"Paper title: {metadata.get('title', arxiv_id)}\n\n{context}\n\n"
            "REVISION REQUIRED — your previous summary contained claims that could "
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
            summary = Summary(
                arxiv_id=arxiv_id,
                title=metadata.get("title") or arxiv_id,
                **{k: v for k, v in repaired_data.items() if k in Summary.model_fields and v is not None},
            )
            evaluation = evaluate_summary(summary.model_dump(), unique_chunks, metadata)
            print(f"[summary_agent] Post-repair evaluation: {evaluation.get('overall_status')} "
                  f"(faithfulness={evaluation.get('faithfulness', {}).get('status', '?')})")
        except Exception as exc:
            print(f"[summary_agent] Repair pass failed: {exc}")

    return {"draft_summary": summary.model_dump(), "summary_evaluation": evaluation}
