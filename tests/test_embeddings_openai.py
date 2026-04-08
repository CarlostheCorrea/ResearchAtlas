"""
Tests for the OpenAI-backed embedding wrapper and vectorstore compatibility checks.
These tests mock remote calls and run without network access.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeEmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        batch = kwargs["input"]
        data = [
            _FakeEmbeddingItem(index=i, embedding=[float(len(text)), float(i)])
            for i, text in enumerate(batch)
        ]
        return _FakeEmbeddingResponse(data)


class _FakeCollection:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeClient:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_collection(self, _name):
        return _FakeCollection(self._metadata)


def test_embed_texts_uses_openai_client(monkeypatch):
    import app.rag.embeddings as embeddings

    fake_api = _FakeEmbeddingsAPI()
    monkeypatch.setattr(embeddings.client, "embeddings", fake_api)

    result = embeddings.embed_texts(["alpha", "", "gamma"], batch_size=2)

    assert len(result) == 3
    assert fake_api.calls[0]["model"] == embeddings.EMBEDDING_MODEL
    assert fake_api.calls[0]["input"] == ["alpha", " "]
    assert fake_api.calls[1]["input"] == ["gamma"]


def test_embed_query_blank_returns_empty_vector():
    import app.rag.embeddings as embeddings

    assert embeddings.embed_query("") == []
    assert embeddings.embed_query("   ") == []


def test_collection_compatibility_matches_expected_metadata(monkeypatch):
    import app.rag.vectorstore as vectorstore

    monkeypatch.setattr(
        vectorstore,
        "get_chroma_client",
        lambda: _FakeClient(vectorstore.expected_collection_metadata()),
    )

    assert vectorstore.is_collection_compatible("paper123") is True


def test_collection_compatibility_rejects_old_metadata(monkeypatch):
    import app.rag.vectorstore as vectorstore

    monkeypatch.setattr(
        vectorstore,
        "get_chroma_client",
        lambda: _FakeClient({"hnsw:space": "cosine"}),
    )

    assert vectorstore.is_collection_compatible("paper123") is False
