"""
RAG tools for the MCP server — chunk indexing and semantic retrieval.
Phase 3 — Week 9: Custom MCP server. Agents call these instead of importing chromadb directly.
"""
import app.database as db
from app.rag.embeddings import embed_texts, embed_query
from app.rag.vectorstore import get_or_create_collection, index_chunks, query_collection, collection_count


def index_paper(arxiv_id: str) -> dict:
    """Embed all chunks for a paper and store in ChromaDB. Skips if already indexed."""
    if collection_count(arxiv_id) > 0:
        existing = db.get_chunks_for_paper(arxiv_id)
        return {"arxiv_id": arxiv_id, "chunks_indexed": len(existing), "already_indexed": True}

    chunks = db.get_chunks_for_paper(arxiv_id)
    if not chunks:
        return {"arxiv_id": arxiv_id, "chunks_indexed": 0, "already_indexed": False,
                "error": "No chunks found in database — run chunk_paper first"}

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, batch_size=32)
    indexed = index_chunks(arxiv_id, chunks, embeddings)
    db.mark_paper_indexed(arxiv_id, indexed)
    return {"arxiv_id": arxiv_id, "chunks_indexed": indexed, "already_indexed": False}


def retrieve_paper_chunks(arxiv_id: str, question: str, k: int = 5) -> list[dict]:
    """Semantic search over a paper's indexed chunks. Returns top-k relevant chunks."""
    from app.rag.retriever import retrieve_chunks
    return retrieve_chunks(arxiv_id, question, k=k)


def get_paper_section(arxiv_id: str, section_hint: str) -> dict:
    """Return all chunks from a specific section of a paper."""
    chunks = db.get_chunks_by_section(arxiv_id, section_hint)
    return {"arxiv_id": arxiv_id, "section_hint": section_hint, "chunks": chunks}
