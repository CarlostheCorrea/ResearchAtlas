"""
Assembles the full LangGraph StateGraph.
Pattern: Hierarchical orchestrator with sequential sub-pipelines.
Two interrupt() pause points in the analysis flow.
SqliteSaver checkpointer for session persistence (Week 11).

Phase 3 — Week 10: Multi-Agent Workflows.
Phase 3 — Week 11: Human-in-the-Loop & Memory.
"""
import sqlite3
from langgraph.graph import StateGraph, END
from app.graph.state import GraphState

# SqliteSaver moved to langgraph-checkpoint-sqlite in langgraph >= 0.3
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver
    except ModuleNotFoundError:
        from langgraph.checkpoint.memory import MemorySaver as SqliteSaver  # fallback
from app.graph import nodes
from app.graph.router import route_by_intent, route_after_filter, route_after_approval
from app.config import DB_PATH
import os


def build_graph() -> StateGraph:
    """Build the StateGraph without compiling (useful for testing)."""
    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("manager",                    nodes.run_manager_node)
    builder.add_node("search_agent",               nodes.run_search_node)
    builder.add_node("pre_filter",                 nodes.run_pre_filter_node)
    builder.add_node("ranking_agent",              nodes.run_ranking_node)
    builder.add_node("human_gate_before_download", nodes.human_gate_before_download)
    builder.add_node("ingestion_agent",            nodes.run_ingestion_node)
    builder.add_node("summary_agent",              nodes.run_summary_node)
    builder.add_node("human_gate_before_save",     nodes.human_gate_before_save)
    builder.add_node("retrieval_agent",            nodes.run_retrieval_node)
    builder.add_node("qa_agent",                   nodes.run_qa_node)
    builder.add_node("memory_agent",               nodes.run_memory_node)
    builder.add_node("end_no_results",             nodes.end_no_results_node)

    # Entry point
    builder.set_entry_point("manager")

    # Manager routes by intent
    builder.add_conditional_edges("manager", route_by_intent, {
        "search_agent":               "search_agent",
        "human_gate_before_download": "human_gate_before_download",
        "retrieval_agent":            "retrieval_agent",
        "memory_agent":               "memory_agent",
        "__end__":                    END,
    })

    # Discovery sub-pipeline: search → pre_filter → (ranking | end_no_results)
    builder.add_edge("search_agent", "pre_filter")
    builder.add_conditional_edges("pre_filter", route_after_filter, {
        "ranking_agent":  "ranking_agent",
        "end_no_results": "end_no_results",
    })
    builder.add_edge("ranking_agent", END)

    # Analysis sub-pipeline (two interrupt() pause points)
    builder.add_edge("human_gate_before_download", "ingestion_agent")
    builder.add_edge("ingestion_agent",            "summary_agent")
    builder.add_edge("summary_agent",              "human_gate_before_save")
    builder.add_conditional_edges("human_gate_before_save", route_after_approval, {
        "memory_agent":  "memory_agent",
        "summary_agent": "summary_agent",
        "__end__":       END,
    })

    # Q&A sub-pipeline
    builder.add_edge("retrieval_agent", "qa_agent")
    builder.add_edge("qa_agent",        END)

    # Terminal nodes
    builder.add_edge("memory_agent",    END)
    builder.add_edge("end_no_results",  END)

    return builder


def compile_graph():
    """
    Compile graph with SqliteSaver checkpointer for Week 11 session persistence.
    Interrupts before both human gate nodes so the graph pauses and waits.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # SqliteSaver takes a connection; MemorySaver (fallback) takes no args
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except TypeError:
        checkpointer = SqliteSaver()
    # Do NOT use interrupt_before — we rely on interrupt() inside the gate nodes.
    # interrupt_before would pause before the node runs, so interrupt() never fires
    # and the payload is always empty, causing an infinite modal loop.
    return build_graph().compile(
        checkpointer=checkpointer,
    )


# Module-level compiled graph singleton
_compiled_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_graph()
    return _compiled_graph
