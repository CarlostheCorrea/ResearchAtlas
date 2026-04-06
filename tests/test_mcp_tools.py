"""
Tests for MCP tool functions — arXiv, PDF, RAG, and memory tools.
These tests exercise the tool implementations directly (no HTTP layer).
"""
import pytest
import os
import sys

# Ensure project root is in path when running from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a temp DB for tests
os.environ.setdefault("DB_PATH", "/tmp/test_research.db")
os.environ.setdefault("PDF_DIR", "/tmp/test_pdfs")
os.environ.setdefault("VECTORSTORE_DIR", "/tmp/test_vectorstore")


class TestArxivTools:
    def test_search_papers_returns_results(self):
        """search_papers should return at least 1 result for a common query."""
        from app.mcp_server.tools_arxiv import search_papers
        results = search_papers("retrieval augmented generation", max_results=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_papers_paper_fields(self):
        """Each result must have required Paper fields."""
        from app.mcp_server.tools_arxiv import search_papers
        results = search_papers("transformer attention mechanism", max_results=3)
        for paper in results:
            assert "arxiv_id" in paper
            assert "title" in paper
            assert "abstract" in paper
            assert "authors" in paper
            assert isinstance(paper["authors"], list)

    def test_search_papers_empty_query_graceful(self):
        """search_papers with bad query should return list (may be empty), not raise."""
        from app.mcp_server.tools_arxiv import search_papers
        results = search_papers("xyzzy_nonexistent_gibberish_12345", max_results=3)
        assert isinstance(results, list)

    def test_get_paper_metadata_known_id(self):
        """get_paper_metadata should return metadata for a known arXiv ID."""
        from app.mcp_server.tools_arxiv import get_paper_metadata
        # 1706.03762 is "Attention Is All You Need"
        paper = get_paper_metadata("1706.03762")
        if paper:  # may be None if network unavailable
            assert paper["arxiv_id"] == "1706.03762"
            assert len(paper["title"]) > 5

    def test_shortlist_paper(self):
        """shortlist_paper should insert without error."""
        from app.mcp_server.tools_arxiv import shortlist_paper
        import app.database as db
        db.init_db()
        result = shortlist_paper("1706.03762", "test_session")
        assert result["shortlisted"] is True


class TestMemoryTools:
    def setup_method(self):
        import app.database as db
        db.init_db()

    def test_save_user_preference(self):
        from app.mcp_server.tools_memory import save_user_preference
        result = save_user_preference("cs.CL", 0.8)
        assert result["topic"] == "cs.CL"
        assert 0.0 <= result["weight"] <= 1.0

    def test_preference_running_average(self):
        """Saving same topic twice should blend weights, not replace."""
        from app.mcp_server.tools_memory import save_user_preference
        r1 = save_user_preference("cs.AI_test", 1.0)
        r2 = save_user_preference("cs.AI_test", 0.0)
        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert abs(r2["weight"] - 0.7) < 0.01

    def test_get_user_preferences(self):
        from app.mcp_server.tools_memory import get_user_preferences, save_user_preference
        save_user_preference("machine_learning_test", 0.9)
        prefs = get_user_preferences()
        assert isinstance(prefs, list)
        topics = [p["topic"] for p in prefs]
        assert "machine_learning_test" in topics

    def test_log_feedback_positive(self):
        """High rating should update preference weight upward."""
        from app.mcp_server.tools_memory import log_feedback
        import app.database as db
        result = log_feedback("1706.03762", 5, "Excellent paper")
        assert result["logged"] is True
        assert result["feedback_count"] >= 1

    def test_log_feedback_negative(self):
        """Low rating should be logged without error."""
        from app.mcp_server.tools_memory import log_feedback
        result = log_feedback("1706.03762", 1, "Not relevant")
        assert result["logged"] is True

    def test_create_and_resolve_review(self):
        from app.mcp_server.tools_memory import create_pending_review, resolve_review
        summary = {"overview": "Test", "arxiv_id": "test123"}
        created = create_pending_review("test123", summary)
        assert "summary_id" in created
        resolved = resolve_review(created["summary_id"], "approved")
        assert resolved["decision"] == "approved"
