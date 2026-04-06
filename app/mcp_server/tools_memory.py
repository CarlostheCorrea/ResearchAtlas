"""
Memory tools for the MCP server — research library, preferences, feedback.
Phase 3 — Week 11: Human-in-the-Loop & Memory. All write-backs go through here.
Agents call these tools; they never import sqlite3 directly.
"""
import app.database as db
from app.mcp_server.tools_arxiv import get_paper_metadata


def save_to_library(arxiv_id: str, title: str, summary: dict, notes: str = "") -> dict:
    """Permanently save a paper + summary to the research library after human approval."""
    count = db.upsert_saved_paper(arxiv_id, title, summary, notes)

    # Auto-update preferences for this paper's categories
    paper = get_paper_metadata(arxiv_id)
    if paper:
        for cat in paper.get("categories", []):
            db.upsert_preference(cat, 0.6, source="save")

    return {"saved": True, "arxiv_id": arxiv_id, "library_count": count}


def list_saved_papers() -> list[dict]:
    """Return all saved papers ordered by save date."""
    return db.list_saved_papers()


def save_user_preference(topic: str, weight: float) -> dict:
    """Record or update interest in a topic (running average: 0.7*old + 0.3*new)."""
    return db.upsert_preference(topic, weight, source="explicit")


def get_user_preferences() -> list[dict]:
    """Return all user preferences ordered by weight."""
    return db.get_all_preferences()


def log_feedback(arxiv_id: str, rating: int, comment: str = "") -> dict:
    """Record feedback for a paper and update topic preferences accordingly."""
    count = db.insert_feedback(arxiv_id, rating, comment)

    paper = get_paper_metadata(arxiv_id)
    if paper:
        categories = paper.get("categories", [])
        if rating >= 4:
            for cat in categories:
                db.upsert_preference(cat, 0.8, source="feedback_positive")
        elif rating <= 2:
            for cat in categories:
                db.upsert_preference(cat, 0.1, source="feedback_negative")

    return {"logged": True, "feedback_count": count}


def get_feedback_history(limit: int = 50) -> list[dict]:
    """Return recent feedback entries."""
    return db.get_feedback_history(limit)


def create_pending_review(arxiv_id: str, draft_summary: dict) -> dict:
    """Create a pending review entry for human approval."""
    return db.create_pending_review(arxiv_id, draft_summary)


def resolve_review(summary_id: str, decision: str, revision_note: str = "") -> dict:
    """Resolve a pending review — approve, reject, or request revision."""
    result = db.resolve_review(summary_id, decision, revision_note)
    return result
