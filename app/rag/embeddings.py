"""
OpenAI embeddings wrapper.
Phase 3 — Week 9: MCP Server. Uses the configured OpenAI embedding model.
"""
from openai import OpenAI

from app.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)


def _normalize_inputs(texts: list[str]) -> list[str]:
    # OpenAI embeddings reject empty strings, so normalize blank items to a space.
    return [text if (text or "").strip() else " " for text in texts]


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of strings. Returns a list of float vectors."""
    if not texts:
        return []

    normalized = _normalize_inputs(texts)
    embeddings: list[list[float]] = []
    for start in range(0, len(normalized), batch_size):
        batch = normalized[start:start + batch_size]
        params = {"model": EMBEDDING_MODEL, "input": batch}
        if EMBEDDING_DIMENSIONS is not None:
            params["dimensions"] = EMBEDDING_DIMENSIONS

        response = client.embeddings.create(**params)
        embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if not text or not text.strip():
        return []
    return embed_texts([text])[0]
