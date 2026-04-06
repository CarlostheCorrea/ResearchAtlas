"""
Conditional edge functions for the LangGraph manager routing.
Phase 3 — Week 10: Multi-Agent Workflows. All routing logic in one place.
"""
from app.graph.state import GraphState


def route_by_intent(state: GraphState) -> str:
    """Conditional edge function — routes manager output to correct sub-pipeline."""
    intent = state.get("intent", "discover")
    if intent == "discover":
        return "search_agent"
    elif intent == "analyze_paper":
        return "human_gate_before_download"
    elif intent == "ask_question":
        # If paper not yet indexed, must analyze first
        if not state.get("paper_indexed"):
            return "human_gate_before_download"
        return "retrieval_agent"
    elif intent == "save_or_review":
        return "memory_agent"
    else:
        return "__end__"


def route_after_filter(state: GraphState) -> str:
    """After pre-filter, check if any papers passed."""
    if not state.get("filtered_results"):
        return "end_no_results"
    return "ranking_agent"


def route_after_approval(state: GraphState) -> str:
    """After human review gate, route based on decision."""
    status = state.get("approval_status", "pending")
    if status == "approved":
        return "memory_agent"
    elif status == "rejected":
        return "__end__"
    elif status == "revised":
        return "summary_agent"   # regenerate with revision note
    else:
        return "__end__"
