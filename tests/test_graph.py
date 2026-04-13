"""
Tests for the LangGraph graph structure — routing and state management.
These tests verify graph construction without running LLM calls.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DB_PATH", "/tmp/test_graph.db")
os.environ.setdefault("VECTORSTORE_DIR", "/tmp/test_vectorstore_graph")


class TestRouter:
    def test_route_discover(self):
        from app.graph.router import route_by_intent
        state = {"intent": "discover"}
        assert route_by_intent(state) == "search_agent"

    def test_route_analyze_paper(self):
        from app.graph.router import route_by_intent
        state = {"intent": "analyze_paper"}
        assert route_by_intent(state) == "human_gate_before_download"

    def test_route_ask_question_not_indexed(self):
        from app.graph.router import route_by_intent
        state = {"intent": "ask_question", "paper_indexed": False}
        assert route_by_intent(state) == "human_gate_before_download"

    def test_route_ask_question_indexed(self):
        from app.graph.router import route_by_intent
        state = {"intent": "ask_question", "paper_indexed": True}
        assert route_by_intent(state) == "retrieval_agent"

    def test_route_save_or_review(self):
        from app.graph.router import route_by_intent
        state = {"intent": "save_or_review"}
        assert route_by_intent(state) == "memory_agent"

    def test_route_unknown_intent(self):
        from app.graph.router import route_by_intent
        state = {"intent": "unknown_xyz"}
        assert route_by_intent(state) == "__end__"

    def test_route_after_filter_no_results(self):
        from app.graph.router import route_after_filter
        state = {"filtered_results": []}
        assert route_after_filter(state) == "end_no_results"

    def test_route_after_filter_with_results(self):
        from app.graph.router import route_after_filter
        state = {"filtered_results": [{"arxiv_id": "test"}]}
        assert route_after_filter(state) == "ranking_agent"

    def test_route_after_approval_approved(self):
        from app.graph.router import route_after_approval
        assert route_after_approval({"approval_status": "approved"}) == "memory_agent"

    def test_route_after_approval_rejected(self):
        from app.graph.router import route_after_approval
        assert route_after_approval({"approval_status": "rejected"}) == "__end__"

    def test_route_after_approval_revised(self):
        from app.graph.router import route_after_approval
        assert route_after_approval({"approval_status": "revised"}) == "summary_agent"


class TestGraphBuild:
    def test_graph_builds_without_error(self):
        """StateGraph can be constructed without LLM credentials."""
        from app.graph.build_graph import build_graph
        builder = build_graph()
        assert builder is not None

    def test_graph_state_schema(self):
        """GraphState TypedDict has all required fields."""
        from app.graph.state import GraphState
        # Check required keys are present as annotations
        annotations = GraphState.__annotations__
        for key in ["session_id", "user_query", "intent", "ranked_results",
                    "draft_summary", "summary_candidates", "summary_competition",
                    "final_answer", "approval_status"]:
            assert key in annotations, f"Missing key: {key}"
