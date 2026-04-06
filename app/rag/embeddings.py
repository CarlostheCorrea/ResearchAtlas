"""
sentence-transformers embedding wrapper.
Phase 3 — Week 9: MCP Server. Uses all-MiniLM-L6-v2 (free, local, ~80MB).
Model downloads once and is cached at ~/.cache/torch/ for subsequent runs.
"""
from typing import Optional
from app.config import EMBEDDING_MODEL

_model = None  # module-level singleton — load once, reuse across requests


def get_embedding_model():
    """Lazy-load the sentence transformer model (downloads ~80MB on first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] Loading model '{EMBEDDING_MODEL}' (may download ~80MB on first run)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[embeddings] Model loaded.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of strings. Returns a list of float vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if not text:
        # Return a zero vector matching all-MiniLM-L6-v2's output dimension (384)
        return [0.0] * 384
    return embed_texts([text])[0]
