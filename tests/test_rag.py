"""
Tests for RAG pipeline — text cleaning, chunking, and embeddings.
These tests use only local computation (no API calls, no network).
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DB_PATH", "/tmp/test_research.db")
os.environ.setdefault("VECTORSTORE_DIR", "/tmp/test_vectorstore_rag")


SAMPLE_TEXT = """
Abstract
This paper presents a novel approach to natural language processing.
We demonstrate state-of-the-art results on multiple benchmarks.

1. Introduction
Large language models have revolutionized the field of NLP.
Previous work has shown that transformer architectures can achieve remarkable performance.
In this paper, we propose a new method for improving efficiency.

2. Methods
Our approach uses attention mechanisms combined with sparse representations.
We train on a large corpus of text using self-supervised learning.
The model architecture consists of 12 layers with 768 hidden dimensions.

3. Results
We achieve 95.3% accuracy on the GLUE benchmark.
Our method outperforms previous baselines by 3.2 points.
The model runs 2x faster than the baseline at inference time.

4. Conclusion
We have presented a new method for efficient language model training.
Future work will explore scaling to larger model sizes.
""".strip()


class TestCleaner:
    def test_clean_removes_extra_newlines(self):
        from app.rag.cleaner import clean_pdf_text
        text = "Hello\n\n\n\n\nWorld"
        result = clean_pdf_text(text)
        assert "\n\n\n" not in result["cleaned_text"]

    def test_clean_fixes_hyphenated_breaks(self):
        from app.rag.cleaner import clean_pdf_text
        text = "This is a hy-\nphenated word in text."
        result = clean_pdf_text(text)
        assert "hyphenated" in result["cleaned_text"]

    def test_clean_removes_page_numbers(self):
        from app.rag.cleaner import clean_pdf_text
        text = "Some text\n42\nMore text\n\n100\nEnd"
        result = clean_pdf_text(text)
        cleaned = result["cleaned_text"]
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        assert '42' not in lines
        assert '100' not in lines

    def test_clean_reports_removed_chars(self):
        from app.rag.cleaner import clean_pdf_text
        text = "A   B   C  \n\n\n\nD"
        result = clean_pdf_text(text)
        assert "removed_chars" in result
        assert isinstance(result["removed_chars"], int)


class TestChunker:
    def test_chunk_basic(self):
        from app.rag.chunker import chunk_text
        result = chunk_text("test123", SAMPLE_TEXT)
        assert result["arxiv_id"] == "test123"
        assert result["chunk_count"] >= 1
        assert len(result["chunks"]) == result["chunk_count"]

    def test_chunk_fields(self):
        from app.rag.chunker import chunk_text
        result = chunk_text("test456", SAMPLE_TEXT)
        for chunk in result["chunks"]:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "section" in chunk
            assert "page" in chunk
            assert "word_count" in chunk
            assert chunk["word_count"] > 0

    def test_chunk_id_format(self):
        from app.rag.chunker import chunk_text
        result = chunk_text("arxiv_id_789", SAMPLE_TEXT)
        for i, chunk in enumerate(result["chunks"]):
            assert chunk["chunk_id"].startswith("arxiv_id_789_chunk_")

    def test_chunk_empty_text(self):
        from app.rag.chunker import chunk_text
        result = chunk_text("empty_test", "")
        assert result["chunk_count"] == 0
        assert result["chunks"] == []

    def test_chunk_overlap(self):
        """With overlap, adjacent chunks should share some words."""
        from app.rag.chunker import chunk_text
        # Create a text long enough for multiple chunks
        long_text = " ".join([f"word{i}" for i in range(2000)])
        result = chunk_text("overlap_test", long_text)
        if len(result["chunks"]) >= 2:
            words0 = set(result["chunks"][0]["text"].split())
            words1 = set(result["chunks"][1]["text"].split())
            assert len(words0 & words1) > 0, "Adjacent chunks should overlap"


class TestPreFilter:
    def test_pre_filter_removes_already_saved(self):
        from app.agents.pre_filter import pre_filter_papers
        from app.schemas import Paper

        paper = Paper(
            arxiv_id="saved_id",
            title="Test Paper",
            authors=["Author One"],
            abstract="This paper demonstrates something about machine learning and neural networks.",
            published="2024-01-01",
            pdf_url="https://arxiv.org/pdf/saved_id",
            categories=["cs.LG"],
        )
        filtered, report = pre_filter_papers(
            papers=[paper],
            user_query="machine learning",
            user_preferences={},
            saved_arxiv_ids={"saved_id"},
        )
        assert len(filtered) == 0
        assert report["dropped"] == 1

    def test_pre_filter_passes_relevant_paper(self):
        from app.agents.pre_filter import pre_filter_papers
        from app.schemas import Paper

        paper = Paper(
            arxiv_id="new_id_xyz",
            title="Deep Learning for Natural Language Processing",
            authors=["Author Two"],
            abstract="We present a deep learning approach to NLP using transformers and attention mechanisms.",
            published="2024-01-01",
            pdf_url="https://arxiv.org/pdf/new_id_xyz",
            categories=["cs.CL"],
        )
        filtered, report = pre_filter_papers(
            papers=[paper],
            user_query="deep learning NLP",
            user_preferences={},
            saved_arxiv_ids=set(),
        )
        assert len(filtered) == 1
        assert report["passed"] == 1

    def test_pre_filter_drops_no_keyword_match(self):
        from app.agents.pre_filter import pre_filter_papers
        from app.schemas import Paper

        paper = Paper(
            arxiv_id="unrelated_id",
            title="Quantum Mechanics and Field Theory",
            authors=["Physicist"],
            abstract="We study quantum field theory and its applications to particle physics.",
            published="2024-01-01",
            pdf_url="https://arxiv.org/pdf/unrelated_id",
            categories=["physics.hep-th"],
        )
        filtered, report = pre_filter_papers(
            papers=[paper],
            user_query="machine learning neural networks",
            user_preferences={},
            saved_arxiv_ids=set(),
        )
        assert len(filtered) == 0

    def test_pre_filter_report_structure(self):
        from app.agents.pre_filter import pre_filter_papers
        _, report = pre_filter_papers([], "test", {}, set())
        assert "total_fetched" in report
        assert "passed" in report
        assert "dropped" in report
        assert "pass_rate" in report
