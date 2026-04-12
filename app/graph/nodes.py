"""
Thin LangGraph node wrappers that call agent methods.
Phase 3 — Week 10: Multi-Agent Workflows.
Phase 3 — Week 11: Human-in-the-Loop — interrupt() pause points live here.
"""
from langgraph.types import interrupt
from app.graph.state import GraphState
from app.agents.manager import run_manager
from app.agents.search_agent import run_search_agent
from app.agents.pre_filter import pre_filter_node
from app.agents.ranking_agent import run_ranking_agent
from app.agents.ingestion_agent import run_ingestion_agent
from app.agents.summary_agent import run_summary_agent
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.qa_agent import run_qa_agent
from app.agents.memory_agent import run_memory_agent
import app.database as db


def run_manager_node(state: GraphState) -> dict:
    """Manager node — classifies intent and loads user preferences."""
    result = run_manager(state)

    # If the LLM didn't extract an arxiv_id (e.g. message was just "analyze"),
    # fall back to whatever was already in the state.
    if not result.get("selected_arxiv_id") and state.get("selected_arxiv_id"):
        result["selected_arxiv_id"] = state["selected_arxiv_id"]

    # Similarly, preserve the intent if it was already set explicitly.
    if state.get("intent") and state["intent"] != "discover":
        result["intent"] = state["intent"]

    # Load user preferences from SQLite for this session
    prefs_list = db.get_all_preferences()
    dismissed = db.get_dismissed_ids()
    saved_ids = list(db.get_saved_arxiv_ids())

    result["user_preferences"] = {
        "topics": [{"topic": p["topic"], "weight": p["weight"]} for p in prefs_list],
        "dismissed": [{"arxiv_id": aid} for aid in dismissed],
        "saved": [{"arxiv_id": aid} for aid in saved_ids],
    }
    return result


def run_search_node(state: GraphState) -> dict:
    return run_search_agent(state)


def run_pre_filter_node(state: GraphState) -> dict:
    return pre_filter_node(state)


def run_ranking_node(state: GraphState) -> dict:
    return run_ranking_agent(state)


def human_gate_before_download(state: GraphState) -> dict:
    """
    Pause Point 1 (Week 11): Before downloading and deeply analyzing a paper.
    LangGraph pauses here and waits for the frontend to resume with a decision.
    """
    paper = None
    arxiv_id = state.get("selected_arxiv_id")
    ranked = state.get("ranked_results", [])
    if arxiv_id and ranked:
        for r in ranked:
            p = r.get("paper", {}) if isinstance(r, dict) else {}
            if p.get("arxiv_id") == arxiv_id:
                paper = p
                break

    result = interrupt({
        "pause_point": "before_download",
        "message": "Would you like to download and analyze this paper in depth?",
        "paper": paper,
        "estimated_time": "30-60 seconds",
        "warning": "This will download the full PDF and process it.",
    })

    return {"approval_status": result.get("decision", "rejected")}


def run_ingestion_node(state: GraphState) -> dict:
    return run_ingestion_agent(state)


def run_summary_node(state: GraphState) -> dict:
    return run_summary_agent(state)


def human_gate_before_save(state: GraphState) -> dict:
    """
    Pause Point 2 (Week 11): Before saving summary to research library.
    User sees the full draft summary and decides to approve, reject, or revise.
    The interrupt payload now includes summary_evaluation so the frontend can
    display quality scores (faithfulness, specificity, completeness, section_accuracy)
    alongside the approval modal.
    """
    result = interrupt({
        "pause_point": "before_save",
        "message": "Would you like to save this summary to your research library?",
        "draft_summary": state.get("draft_summary"),
        "summary_evaluation": state.get("summary_evaluation"),
        "options": ["approve", "reject", "revise"],
    })

    return {
        "approval_status": result.get("decision", "rejected"),
        "revision_note": result.get("revision_note", ""),
    }


def run_retrieval_node(state: GraphState) -> dict:
    return run_retrieval_agent(state)


def run_qa_node(state: GraphState) -> dict:
    return run_qa_agent(state)


def run_memory_node(state: GraphState) -> dict:
    return run_memory_agent(state)


def end_no_results_node(state: GraphState) -> dict:
    return {"error": "No papers matched your search after filtering. Try broader keywords."}
