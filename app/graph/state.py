"""
GraphState TypedDict — shared state across all LangGraph nodes.
Phase 3 — Week 10: Multi-Agent Workflows. Single state object flows through all agents.
"""
from typing import TypedDict, Optional


class GraphState(TypedDict, total=False):
    # User inputs
    session_id: str
    user_query: str
    selected_arxiv_id: Optional[str]

    # Manager outputs
    intent: str          # "discover" | "analyze_paper" | "ask_question" | "save_or_review"
    user_preferences: dict

    # Search inputs
    max_results: int
    year_from: Optional[int]
    required_categories: Optional[list[str]]

    # Discovery flow outputs
    raw_search_results: list
    filter_report: dict
    filtered_results: list
    ranked_results: list

    # Analysis flow outputs
    pdf_path: str
    extracted_text: str
    chunks: list
    paper_indexed: bool
    draft_summary: Optional[dict]
    final_summary: Optional[dict]
    summary_evaluation: Optional[dict]   # LLM-as-judge scores for the summary

    # Q&A flow outputs
    question: str
    retrieved_chunks: list
    final_answer: str
    answer_citations: list[str]

    # Human-in-loop
    approval_status: str   # "pending" | "approved" | "rejected" | "revised"
    revision_note: str

    # Memory outputs
    saved_to_library: bool
    weights_updated: bool

    # Error handling
    error: Optional[str]
