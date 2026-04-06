"""
Library API routes — manage saved papers, preferences, and feedback.
Phase 3 — Week 11: Human-in-the-Loop & Memory.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import glob as _glob
import app.database as db
from app.mcp_server.tools_memory import log_feedback as _log_feedback
from app.rag.vectorstore import delete_collection, delete_all_collections
from app.config import PDF_DIR

router = APIRouter()


class FeedbackRequest(BaseModel):
    rating: int
    comment: Optional[str] = ""


class RatingRequest(BaseModel):
    rating: int


@router.get("/api/library")
async def get_library():
    """Return all saved papers from SQLite."""
    return db.list_saved_papers()


@router.delete("/api/library")
async def clear_library():
    """Delete ALL papers from the research library and clear all RAG memory."""
    count = db.delete_all_saved_papers()
    # Clear ChromaDB collections
    chroma_deleted = delete_all_collections()
    # Clear SQLite RAG tables
    db.delete_all_rag_data()
    # Delete all downloaded PDFs
    pdf_deleted = 0
    for pdf_path in _glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        try:
            os.remove(pdf_path)
            pdf_deleted += 1
        except OSError:
            pass
    return {"deleted": True, "count": count, "chroma_collections_deleted": chroma_deleted, "pdfs_deleted": pdf_deleted}


@router.delete("/api/library/{arxiv_id}")
async def remove_from_library(arxiv_id: str):
    """Remove a single paper from the research library and clear its RAG memory."""
    deleted = db.delete_saved_paper(arxiv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found in library")
    # Clear ChromaDB collection for this paper
    delete_collection(arxiv_id)
    # Clear SQLite RAG rows for this paper
    db.delete_rag_data_for_paper(arxiv_id)
    # Delete the downloaded PDF if it exists
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    pdf_deleted = False
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            pdf_deleted = True
        except OSError:
            pass
    return {"deleted": True, "arxiv_id": arxiv_id, "rag_cleared": True, "pdf_deleted": pdf_deleted}


@router.post("/api/library/{arxiv_id}/rating")
async def save_rating(arxiv_id: str, body: RatingRequest):
    """Persist a star rating on the saved_papers row so it survives page changes."""
    if not 1 <= body.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    db.set_paper_rating(arxiv_id, body.rating)
    return {"saved": True, "arxiv_id": arxiv_id, "rating": body.rating}


@router.post("/api/library/{arxiv_id}/feedback")
async def submit_feedback(arxiv_id: str, body: FeedbackRequest):
    """Rate a saved paper 1-5. Updates user preference weights."""
    if not 1 <= body.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    # Persist to saved_papers row so stars survive tab switches
    db.set_paper_rating(arxiv_id, body.rating)
    result = _log_feedback(arxiv_id, body.rating, body.comment or "")
    return result


@router.get("/api/preferences")
async def get_preferences():
    """Return current user preference weights."""
    return db.get_all_preferences()


@router.delete("/api/preferences")
async def clear_preferences():
    """Wipe all user preference weights."""
    count = db.clear_all_preferences()
    return {"deleted": True, "count": count}
