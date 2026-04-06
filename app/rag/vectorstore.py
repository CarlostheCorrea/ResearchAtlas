"""
ChromaDB collection management — one collection per paper (paper_{arxiv_id}).
Phase 3 — Week 9: MCP Server. Persistent vector store for semantic search.
"""
import os
import shutil
import chromadb
from app.config import VECTORSTORE_DIR

_client = None  # module-level singleton


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB persistent client."""
    global _client
    if _client is None:
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return _client


def _reset_client():
    """Close and discard the singleton so the next call recreates it fresh."""
    global _client
    _client = None


def get_or_create_collection(arxiv_id: str):
    """Get or create the ChromaDB collection for a paper."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=f"paper_{arxiv_id}",
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(arxiv_id: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Add chunks to ChromaDB collection for the given paper.
    Returns number of chunks indexed.
    """
    collection = get_or_create_collection(arxiv_id)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"section": c["section"], "page": c["page"], "arxiv_id": arxiv_id}
        for c in chunks
    ]

    # ChromaDB upsert in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    return len(ids)


def query_collection(arxiv_id: str, query_embedding: list[float], k: int = 5) -> list[dict]:
    """
    Query the collection for the top-k most similar chunks.
    Returns list of dicts with text, metadata, and distance.
    """
    collection = get_or_create_collection(arxiv_id)
    count = collection.count()
    if count == 0:
        return []

    actual_k = min(k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist <= 1.5:  # filter out chunks too dissimilar to be useful
            chunks.append({
                "text": doc,
                "section": meta.get("section", "Unknown"),
                "page": meta.get("page", 1),
                "arxiv_id": meta.get("arxiv_id", arxiv_id),
                "distance": dist,
            })

    return chunks


def collection_count(arxiv_id: str) -> int:
    """Return number of chunks indexed for this paper."""
    collection = get_or_create_collection(arxiv_id)
    return collection.count()


def delete_collection(arxiv_id: str) -> bool:
    """Delete the ChromaDB collection for a paper. Returns True if it existed."""
    client = get_chroma_client()
    try:
        client.delete_collection(f"paper_{arxiv_id}")
        return True
    except Exception:
        return False


def delete_all_collections() -> int:
    """
    Wipe the entire ChromaDB vectorstore from disk and reset the client.
    This is the only reliable way to remove all segment folders in ChromaDB >= 0.6,
    where delete_collection() removes registry entries but leaves UUID dirs on disk.
    Returns the number of paper_ collections that were registered before deletion.
    """
    client = get_chroma_client()
    # Count how many paper collections existed
    try:
        collections = client.list_collections()
        count = sum(
            1 for col in collections
            if (col if isinstance(col, str) else getattr(col, "name", str(col))).startswith("paper_")
        )
    except Exception:
        count = 0

    # Reset singleton so the client releases its file handles
    _reset_client()

    # Delete the entire vectorstore directory (removes all UUID segment folders)
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    # Recreate the empty directory so the next get_chroma_client() works cleanly
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    return count
