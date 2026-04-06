"""
Pydantic models shared across API + agents.
Phase 3 — Week 8: MCP Foundations (data contracts for all agent communication).
"""
from pydantic import BaseModel
from typing import Optional


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str          # ISO date string e.g. "2024-03-15"
    pdf_url: str
    categories: list[str]   # e.g. ["cs.CL", "cs.AI"]


class RankedPaper(BaseModel):
    paper: Paper
    relevance_score: float      # 0-100, computed by ranking agent
    recency_score: float        # 0-100, based on published date
    preference_score: float     # 0-100, based on user topic preferences
    composite_score: float      # weighted final score
    score_breakdown: dict       # {"relevance": 40, "recency": 30, "preference": 30}


class PaperChunk(BaseModel):
    chunk_id: str               # f"{arxiv_id}_chunk_{n}"
    arxiv_id: str
    text: str
    section: str                # "Introduction", "Methods", "Results", etc.
    page: int
    word_count: int


class Summary(BaseModel):
    arxiv_id: str
    title: str
    overview: str
    problem_addressed: str
    main_contribution: str
    method: str
    datasets_experiments: str
    results: str
    limitations: str
    why_it_matters: str
    confidence_note: str        # "Based on full PDF" or "Based on abstract only"


class SearchRequest(BaseModel):
    query: str
    max_results: int = 20       # fetch more than needed — pre-filter will cut this down
    year_from: Optional[int] = None
    categories: Optional[list[str]] = None


class ChatRequest(BaseModel):
    message: str
    arxiv_id: Optional[str] = None  # if set, Q&A mode on this paper
    session_id: str


class ReviewDecision(BaseModel):
    session_id: str
    decision: str               # "approve", "reject", "revise"
    revision_note: Optional[str] = None
