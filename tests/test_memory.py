"""
Tests for SQLite database operations and memory persistence.
"""
import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a fresh temp DB for each test session
_tmp_db = tempfile.mktemp(suffix=".db")
os.environ["DB_PATH"] = _tmp_db


class TestDatabase:
    def setup_method(self):
        import app.database as db
        db.init_db()

    def teardown_method(self):
        # Clean up any state between tests
        pass

    def test_init_db_creates_tables(self):
        """init_db should create all required tables."""
        import sqlite3
        import app.database as db
        conn = sqlite3.connect(_tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        for table in ["saved_papers", "user_preferences", "paper_feedback",
                      "pending_reviews", "papers_cache", "pdf_cache", "chunks"]:
            assert table in tables, f"Table '{table}' not created"

    def test_upsert_saved_paper(self):
        import app.database as db
        count = db.upsert_saved_paper("test_arxiv_001", "Test Title", {"overview": "Test"})
        assert count >= 1
        papers = db.list_saved_papers()
        ids = [p["arxiv_id"] for p in papers]
        assert "test_arxiv_001" in ids

    def test_upsert_saved_paper_idempotent(self):
        """Upserting the same paper twice should not duplicate."""
        import app.database as db
        db.upsert_saved_paper("dup_test_001", "Title A", {})
        db.upsert_saved_paper("dup_test_001", "Title B", {})
        papers = db.list_saved_papers()
        matches = [p for p in papers if p["arxiv_id"] == "dup_test_001"]
        assert len(matches) == 1
        assert matches[0]["title"] == "Title B"

    def test_delete_saved_paper(self):
        import app.database as db
        db.upsert_saved_paper("delete_test_001", "Delete Me", {})
        deleted = db.delete_saved_paper("delete_test_001")
        assert deleted is True
        ids = [p["arxiv_id"] for p in db.list_saved_papers()]
        assert "delete_test_001" not in ids

    def test_get_saved_arxiv_ids(self):
        import app.database as db
        db.upsert_saved_paper("set_test_001", "Set Test", {})
        ids = db.get_saved_arxiv_ids()
        assert "set_test_001" in ids

    def test_upsert_preference(self):
        import app.database as db
        result = db.upsert_preference("cs.LG", 0.7, "explicit")
        assert result["topic"] == "cs.LG"
        assert 0.0 <= result["weight"] <= 1.0
        assert result["is_new"] in (True, False)

    def test_preference_running_average(self):
        import app.database as db
        db.upsert_preference("running_avg_test", 1.0, "explicit")
        r2 = db.upsert_preference("running_avg_test", 0.0, "explicit")
        assert abs(r2["weight"] - 0.7) < 0.01

    def test_insert_and_retrieve_feedback(self):
        import app.database as db
        db.insert_feedback("feedback_test_id", 4, "Good paper")
        history = db.get_feedback_history(10)
        matches = [f for f in history if f["arxiv_id"] == "feedback_test_id"]
        assert len(matches) >= 1
        assert matches[0]["rating"] == 4

    def test_dismissed_ids_from_low_rating(self):
        import app.database as db
        db.insert_feedback("dismiss_test_id", 1, "Bad")
        db.insert_feedback("dismiss_test_id", 2, "Also bad")
        dismissed = db.get_dismissed_ids()
        assert "dismiss_test_id" in dismissed

    def test_chunk_storage_and_retrieval(self):
        import app.database as db
        chunks = [
            {"chunk_id": "chunk_test_001", "arxiv_id": "chunk_paper_001",
             "text": "This is test chunk text.", "section": "Introduction",
             "page": 1, "word_count": 5},
        ]
        db.insert_chunks(chunks)
        retrieved = db.get_chunks_for_paper("chunk_paper_001")
        assert len(retrieved) >= 1
        assert retrieved[0]["chunk_id"] == "chunk_test_001"

    def test_get_chunks_by_section(self):
        import app.database as db
        chunks = [
            {"chunk_id": "sec_chunk_001", "arxiv_id": "sec_paper_001",
             "text": "Methods chunk text here.", "section": "Methodology",
             "page": 2, "word_count": 4},
        ]
        db.insert_chunks(chunks)
        results = db.get_chunks_by_section("sec_paper_001", "method")
        assert len(results) >= 1

    def test_mark_and_check_indexed(self):
        import app.database as db
        assert db.is_paper_indexed("not_indexed_id") is False
        db.mark_paper_indexed("indexed_id_test", 42)
        assert db.is_paper_indexed("indexed_id_test") is True

    def test_create_and_resolve_pending_review(self):
        import app.database as db
        result = db.create_pending_review("review_test_id", {"overview": "Test summary"})
        assert "summary_id" in result
        resolved = db.resolve_review(result["summary_id"], "approved")
        assert resolved["decision"] == "approved"
