"""
Top-k semantic search over a paper's indexed chunks.
Phase 3 — Week 9: MCP Server (called by retrieve_paper_chunks MCP tool).
"""
from app.rag.embeddings import embed_query
from app.rag.vectorstore import query_collection


def retrieve_chunks(arxiv_id: str, question: str, k: int = 5) -> list[dict]:
    """
    Embed the question and retrieve top-k relevant chunks from ChromaDB.
    Chunks are sorted by relevance (lowest distance = most relevant).
    Returns list of PaperChunk-compatible dicts.
    """
    if not question or not arxiv_id:
        return []
    query_embedding = embed_query(question)
    raw = query_collection(arxiv_id, query_embedding, k=k)

    # Build chunk_id from arxiv_id + sequential counter for deduplication
    chunks = []
    for i, item in enumerate(raw):
        chunks.append({
            "chunk_id": f"{arxiv_id}_retrieved_{i:03d}",
            "arxiv_id": item["arxiv_id"],
            "text": item["text"],
            "section": item["section"],
            "page": item["page"],
            "word_count": len(item["text"].split()),
        })

    return chunks
