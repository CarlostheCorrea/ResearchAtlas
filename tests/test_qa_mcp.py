"""
Focused tests for the Q/A MCP server and its core tools.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import AnyUrl, TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
_URL = TypeAdapter(AnyUrl)


class TestQATools:
    def test_find_evidence_formats_quote_level_results(self, monkeypatch, tmp_path):
        env = os.environ.copy()
        env["DB_PATH"] = str(tmp_path / "research.db")
        env["PDF_DIR"] = str(tmp_path / "pdfs")
        env["VECTORSTORE_DIR"] = str(tmp_path / "vectorstore")
        env["QA_ASSETS_DIR"] = str(tmp_path / "qa_assets")

        from app.qa import mcp_server

        monkeypatch.setattr(
            mcp_server,
            "retrieve_paper_chunks",
            lambda arxiv_id, question, k=5: [
                {
                    "text": "The model improves accuracy by 12 percent over the baseline on the benchmark dataset.",
                    "section": "Results",
                    "page": 7,
                    "arxiv_id": arxiv_id,
                },
                {
                    "text": "A second supporting excerpt appears in the discussion section.",
                    "section": "Discussion",
                    "page": 9,
                    "arxiv_id": arxiv_id,
                },
            ],
        )

        result = json.loads(mcp_server.find_evidence("1234.5678", "What evidence supports the main claim?", 2))
        assert result["arxiv_id"] == "1234.5678"
        assert len(result["evidence"]) == 2
        assert result["evidence"][0]["page"] == 7
        assert result["evidence"][0]["section"] == "Results"
        assert "accuracy" in result["evidence"][0]["quote"].lower()

    def test_evidence_items_include_source_chunk_context(self):
        from app.qa import mcp_server

        evidence = mcp_server._evidence_from_chunks(
            "1234.5678",
            "What evidence supports the main claim?",
            [
                {
                    "text": "The model improves accuracy by 12 percent over the baseline on the benchmark dataset.",
                    "section": "Results",
                    "page": 7,
                    "arxiv_id": "1234.5678",
                }
            ],
            max_items=1,
        )

        assert "accuracy" in evidence[0]["quote"].lower()
        assert evidence[0]["source_chunk"].startswith("The model improves accuracy")

    def test_export_tools_write_files(self, tmp_path, monkeypatch):
        from app.qa import mcp_server
        from app.qa import assets as qa_assets

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        citations = json.dumps([{"section": "Results", "page": 7, "quote": "Improves accuracy by 12 percent."}])
        md_asset = json.loads(
            mcp_server.create_md("sess-1", "Test Answer", "What happened?", "It improved.", citations)
        )
        pdf_asset = json.loads(
            mcp_server.create_pdf("sess-1", "Test Answer", "What happened?", "It improved.", citations)
        )

        assert Path(md_asset["path"]).exists()
        assert md_asset["filename"] == "test-answer-qa-response.md"
        assert Path(md_asset["path"]).read_text(encoding="utf-8").startswith("# Test Answer")
        assert Path(pdf_asset["path"]).exists()
        assert pdf_asset["filename"] == "test-answer-qa-response.pdf"
        assert Path(pdf_asset["path"]).stat().st_size > 0

    def test_export_tools_do_not_include_question_answer_scaffolding(self, tmp_path, monkeypatch):
        import fitz
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        md_asset = json.loads(
            mcp_server.create_md("sess-2", "Clean Export", "Question text", "Only the answer body.", "[]")
        )
        pdf_asset = json.loads(
            mcp_server.create_pdf("sess-2", "Clean Export", "Question text", "Only the answer body.", "[]")
        )

        md_text = Path(md_asset["path"]).read_text(encoding="utf-8")
        assert "## Question" not in md_text
        assert "## Answer" not in md_text
        assert "## Response" in md_text

        pdf_text = fitz.open(pdf_asset["path"])[0].get_text()
        assert "Question " not in pdf_text
        assert "Answer " not in pdf_text

    def test_markdown_export_uses_relevant_filename_and_headers(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        asset = json.loads(
            mcp_server.create_md(
                "sess-headers",
                "Experimental Search for Quantum Gravity",
                "Create a well formatted markdown summary with headers.",
                "This is the body.",
                "[]",
            )
        )

        text = Path(asset["path"]).read_text(encoding="utf-8")
        assert asset["filename"] == "experimental-search-for-quantum-gravity-summary.md"
        assert "# Experimental Search for Quantum Gravity" in text
        assert "## Summary" in text

    def test_markdown_export_respects_requested_list_format(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        asset = json.loads(
            mcp_server.create_md(
                "sess-style",
                "Experimental Search for Quantum Gravity",
                "Make this a list format with headers.",
                "First key point. Second key point.",
                "[]",
            )
        )

        text = Path(asset["path"]).read_text(encoding="utf-8")
        assert "## Response" in text
        assert "1. First key point." in text
        assert "2. Second key point." in text

    def test_markdown_export_uses_key_findings_heading_for_findings_request(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        asset = json.loads(
            mcp_server.create_md(
                "sess-findings",
                "Random Forest Variance Estimation",
                "Write me a list of the key findings made in the research.",
                "The key findings of the research are as follows: 1. First finding. 2. Second finding.",
                "[]",
            )
        )

        text = Path(asset["path"]).read_text(encoding="utf-8")
        assert "## Key Findings" in text
        assert "1. First finding." in text
        assert "2. Second finding." in text

    def test_pdf_export_wraps_long_titles_without_truncation(self, tmp_path, monkeypatch):
        import fitz
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        long_title = "A Comparison of Resampling and Recursive Partitioning Methods in Random Forest Forecasting"
        asset = json.loads(
            mcp_server.create_pdf(
                "sess-long-title",
                long_title,
                "Create a formatted summary.",
                "This is the exported answer body.",
                "[]",
            )
        )

        pdf_text = fitz.open(asset["path"])[0].get_text()
        normalized = " ".join(pdf_text.split())
        assert "A Comparison of Resampling and Recursive Partitioning Methods in Random Forest Forecasting" in normalized
        assert "Summary" in pdf_text
        assert "This is the exported answer body." in pdf_text

    def test_pdf_export_respects_requested_list_format(self, tmp_path, monkeypatch):
        import fitz
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        asset = json.loads(
            mcp_server.create_pdf(
                "sess-list-pdf",
                "List Export",
                "Make this a list format.",
                "First point. Second point.",
                "[]",
            )
        )

        pdf_text = fitz.open(asset["path"])[0].get_text()
        assert "List Export" in pdf_text
        assert "Response" in pdf_text
        assert "1. First point." in pdf_text
        assert "2. Second point." in pdf_text

    def test_pdf_export_uses_key_findings_heading_for_findings_request(self, tmp_path, monkeypatch):
        import fitz
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        asset = json.loads(
            mcp_server.create_pdf(
                "sess-findings-pdf",
                "Random Forest Variance Estimation",
                "Write me a list of the key findings made in the research. Make it a PDF file with red text.",
                "The key findings of the research are as follows: 1. First finding. 2. Second finding.",
                "[]",
            )
        )

        pdf_text = fitz.open(asset["path"])[0].get_text()
        assert "Key Findings" in pdf_text
        assert "1. First finding." in pdf_text
        assert "2. Second finding." in pdf_text

    def test_repeated_markdown_exports_get_unique_filenames(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        first = json.loads(
            mcp_server.create_md("sess-repeat", "Experimental Search for Quantum Gravity", "Create a markdown summary.", "First version.", "[]")
        )
        second = json.loads(
            mcp_server.create_md("sess-repeat", "Experimental Search for Quantum Gravity", "Create a markdown summary.", "Second version.", "[]")
        )

        assert first["filename"] == "experimental-search-for-quantum-gravity-summary.md"
        assert second["filename"] == "experimental-search-for-quantum-gravity-summary-2.md"
        assert first["url"] != second["url"]
        assert "Second version." in Path(second["path"]).read_text(encoding="utf-8")

    def test_clear_all_assets_removes_session_folders(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))
        session_dir = qa_assets.ensure_session_dir("sess-clear")
        (session_dir / "file.txt").write_text("x", encoding="utf-8")

        removed = qa_assets.clear_all_assets()

        assert removed == 1
        assert not any(Path(qa_assets.QA_ASSETS_DIR).iterdir())

    def test_create_graphic_uses_minimal_openai_request(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        captured = {}

        class _Image:
            b64_json = base64.b64encode(b"png-bytes").decode("utf-8")
            revised_prompt = "revised"
            url = None

        class _Response:
            data = [_Image()]

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return _Response()

        monkeypatch.setattr(mcp_server.client.images, "generate", fake_generate)
        monkeypatch.setattr(mcp_server, "_validate_image_text", lambda image_bytes: [])
        asset = json.loads(mcp_server.create_graphic("sess-3", "draw a workflow", "Graphic"))

        assert Path(asset["path"]).exists()
        assert captured["model"] == mcp_server.OPENAI_IMAGE_MODEL
        assert captured["size"] == "1024x1024"
        assert captured["prompt"].startswith("draw a workflow")
        assert "spelled correctly" in captured["prompt"]

    def test_create_presentation_writes_html_asset(self, tmp_path, monkeypatch):
        from app.qa import assets as qa_assets
        from app.qa import mcp_server

        monkeypatch.setattr(qa_assets, "QA_ASSETS_DIR", str(tmp_path / "qa_assets"))

        class _Message:
            content = json.dumps({
                "deck_title": "Test Paper Presentation",
                "slides": [
                    {
                        "title": "Main Idea",
                        "bullets": ["The paper introduces a test idea.", "The result is useful for class discussion."],
                        "speaker_notes": "Use this slide to explain the core contribution.",
                    }
                ],
            })

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        monkeypatch.setattr(mcp_server.client.chat.completions, "create", lambda **kwargs: _Response())

        asset = json.loads(
            mcp_server.create_presentation(
                "sess-presentation",
                "Test Paper",
                "Make a one-slide presentation with speaker notes.",
                "The paper introduces a test idea.",
                json.dumps([{"section": "Abstract", "page": 1, "quote": "The paper introduces a test idea."}]),
                slide_count=1,
            )
        )

        html = Path(asset["path"]).read_text(encoding="utf-8")
        assert asset["kind"] == "presentation"
        assert asset["label"] == "Download Presentation"
        assert asset["filename"] == "test-paper-presentation.html"
        assert asset["slide_count"] == 1
        assert "<!doctype html>" in html
        assert "Test Paper Presentation" in html
        assert "Speaker Notes" in html
        assert "Abstract, p. 1" in html


class TestQAOrchestratorHelpers:
    def test_requested_download_tools_respect_requested_format(self):
        from app.qa.orchestrator import _requested_download_tools

        assert _requested_download_tools("Write this as a MD file download.") == ["create_md"]
        assert _requested_download_tools("Export this answer as PDF.") == ["create_pdf"]
        assert _requested_download_tools("Make this downloadable.") == ["create_md"]
        assert _requested_download_tools("Give me both markdown and pdf.") == ["create_md", "create_pdf"]
        assert _requested_download_tools("Make a 3-slide class presentation.") == ["create_presentation"]

    def test_content_artifact_requests_still_need_grounding(self):
        from app.qa.orchestrator import _artifact_content_query, _content_artifact_needs_grounding

        question = "Make me a MD file of the limitations of rougher texture and higher contrast in the generated images."

        assert _content_artifact_needs_grounding(question) is True
        assert _artifact_content_query(question) == "limitations of rougher texture and higher contrast in the generated images"

    def test_presentation_options_parse_slide_count_and_audience(self):
        from app.qa.orchestrator import _presentation_options

        assert _presentation_options("Make a one-slide presentation for a general audience.") == {
            "slide_count": 1,
            "audience": "general",
            "include_speaker_notes": True,
        }
        assert _presentation_options("Make 3 slides for a technical audience with no speaker notes.") == {
            "slide_count": 3,
            "audience": "technical",
            "include_speaker_notes": False,
        }

    def test_needs_evidence_only_for_evidence_requests(self):
        from app.qa.orchestrator import _needs_evidence

        assert _needs_evidence("What evidence supports the main claim?") is True
        assert _needs_evidence("Give me a 2 sentence summary.") is False

    def test_continuation_detection_for_previous_answer_transforms(self):
        from app.qa.orchestrator import _is_continuation_request, _needs_graphic

        assert _is_continuation_request("Now make your previous answer a PDF.") is True
        assert _is_continuation_request("Make that an image.") is True
        assert _is_continuation_request("Now make that a presentation.") is True
        assert _is_continuation_request("Make a MD file of the previous question.") is True
        assert _needs_graphic("Make that an image.") is True
        assert _is_continuation_request("What are the key findings?") is False
        assert _is_continuation_request("What was the previous question?") is False

    def test_latest_context_answer_prefers_most_recent_answer(self):
        from app.qa.orchestrator import _latest_context_answer

        context = {
            "recent_turns": [
                {"question": "old", "answer": "Old answer", "citations": []},
                {"question": "new", "answer": "New answer", "citations": [{"page": 2}]},
            ]
        }

        latest = _latest_context_answer(context)
        assert latest["answer"] == "New answer"
        assert latest["citations"] == [{"page": 2}]

    def test_memory_query_answers_last_question_from_recent_turns(self):
        from app.qa.orchestrator import _answer_memory_query

        context = {
            "recent_turns": [
                {"question": "Who is the author?", "answer": "The author is Ada.", "citations": []},
            ]
        }

        assert _answer_memory_query("What was the last question?", context) == 'Your last question was: "Who is the author?"'

    def test_memory_query_handles_empty_thread(self):
        from app.qa.orchestrator import _answer_memory_query

        assert _answer_memory_query("What was the last question?", {"recent_turns": []}) == "I do not have a previous question in this Q/A thread yet."

    def test_planner_tool_catalog_excludes_asset_tools(self):
        from app.qa.orchestrator import _planner_tool_catalog

        catalog = [
            {"name": "retrieve_paper_chunks", "description": "retrieve", "input_schema": {}},
            {"name": "create_md", "description": "md", "input_schema": {}},
            {"name": "create_pdf", "description": "pdf", "input_schema": {}},
            {"name": "create_graphic", "description": "graphic", "input_schema": {}},
            {"name": "create_presentation", "description": "presentation", "input_schema": {}},
            {"name": "find_evidence", "description": "evidence", "input_schema": {}},
        ]

        planned = _planner_tool_catalog(catalog)
        assert [tool["name"] for tool in planned] == ["retrieve_paper_chunks", "find_evidence"]

    def test_tool_error_message_prefers_explicit_error_fields(self):
        from app.qa.orchestrator import _tool_error_message

        class _Raw:
            isError = True

        assert _tool_error_message(_Raw(), {"error": "bad request"}) == "bad request"
        assert _tool_error_message(_Raw(), {"text": "tool failed"}) == "tool failed"

    def test_synthesis_trace_for_graphic_request_does_not_claim_missing_capability(self):
        from app.qa.orchestrator import _synthesis_trace

        trace = _synthesis_trace(
            "Make that an image.",
            "The previous answer explains the paper workflow.",
            [],
            [{"tool": "retrieve_paper_chunks", "result": {"chunks": []}}],
            graphic_requested=True,
            requested_download_tools=[],
        )

        assert "generate a visual" in trace
        assert "not provide" not in trace.lower()
        assert "not found" not in trace.lower()

    def test_synthesis_trace_counts_supporting_citations(self):
        from app.qa.orchestrator import _synthesis_trace

        trace = _synthesis_trace(
            "What evidence supports this?",
            "The paper supports the claim.",
            [{"page": 2}, {"page": 5}],
            [{"tool": "find_evidence", "result": {"evidence": []}}],
            graphic_requested=False,
            requested_download_tools=[],
        )

        assert trace == "The final answer is grounded in 2 supporting citations from the paper."

    def test_graphic_source_failed_only_blocks_core_failures(self):
        from app.qa.orchestrator import _graphic_source_failed

        artifact_only_failure = {
            "overall_status": "needs_review",
            "answer_relevance": {"status": "pass", "score": 0.9},
            "groundedness": {"status": "pass", "score": 0.9},
            "citation_quality": {"status": "pass", "score": 0.9},
            "artifact_match": {"status": "fail", "score": 0.0},
            "repair_recommended": False,
        }
        core_failure = {
            "overall_status": "needs_review",
            "answer_relevance": {"status": "pass", "score": 0.9},
            "groundedness": {"status": "fail", "score": 0.2},
            "citation_quality": {"status": "pass", "score": 0.9},
            "repair_recommended": True,
        }
        high_score_citation_gap = {
            "overall_status": "needs_review",
            "answer_relevance": {"status": "pass", "score": 0.95},
            "groundedness": {"status": "fail", "score": 0.78},
            "citation_quality": {"status": "pass", "score": 0.92},
            "tool_choice_quality": {"status": "pass", "score": 0.95},
            "artifact_match": {"status": "not_applicable", "score": 1.0},
            "repair_recommended": True,
        }

        assert _graphic_source_failed(artifact_only_failure) is False
        assert _graphic_source_failed(core_failure) is True
        assert _graphic_source_failed(high_score_citation_gap) is False

    def test_graphic_source_question_removes_artifact_requirement(self):
        from app.qa.orchestrator import _graphic_source_question

        assert _graphic_source_question("Create an image of the methodology.") == (
            "What methodology details from the paper should be used as source content?"
        )


class TestQAEvaluationHelpers:
    def test_should_repair_when_core_metric_fails(self):
        from app.qa.evaluation import should_repair

        tracking = {
            "answer_relevance": {"status": "pass", "score": 0.9},
            "groundedness": {"status": "fail", "score": 0.2},
            "citation_quality": {"status": "pass", "score": 0.8},
            "repair_recommended": False,
        }

        assert should_repair(tracking) is True

    def test_tracking_score_ignores_not_applicable_metrics(self):
        from app.qa.evaluation import tracking_score

        tracking = {
            "answer_relevance": {"status": "pass", "score": 1.0},
            "groundedness": {"status": "pass", "score": 0.8},
            "citation_quality": {"status": "not_applicable", "score": 0.0},
            "tool_choice_quality": {"status": "pass", "score": 0.6},
            "artifact_match": {"status": "not_applicable", "score": 0.0},
        }

        assert tracking_score(tracking) == (1.0 + 0.8 + 0.6) / 3

    def test_local_tracking_marks_requested_missing_artifact_as_fail(self):
        from app.qa.evaluation import _local_tracking

        tracking = _local_tracking(
            "Make this a PDF download.",
            "Answer text.",
            [],
            [{"kind": "tool", "tool": "retrieve_paper_chunks", "status": "completed"}],
            [],
        )

        assert tracking["artifact_match"]["status"] == "fail"
        assert tracking["tool_counts"]["retrieve_paper_chunks"] == 1

    def test_tracking_calibration_handles_specific_single_citation(self):
        from app.qa.evaluation import _calibrate_tracking

        tracking = {
            "answer_relevance": {"status": "pass", "score": 0.9, "note": "Relevant."},
            "groundedness": {"status": "pass", "score": 0.8, "note": "Grounded."},
            "citation_quality": {
                "status": "fail",
                "score": 0.3,
                "note": "Only one citation was provided.",
            },
            "retrieval_relevance": {
                "status": "fail",
                "score": 0.0,
                "note": "No retrieval tools were used.",
            },
            "tool_choice_quality": {"status": "fail", "score": 0.4, "note": "Weak tool choice."},
            "artifact_match": {"status": "not_applicable", "score": 1.0, "note": "No artifact."},
        }
        citations = [{"section": "Methods", "page": 5, "quote": "The study used a controlled experiment."}]

        calibrated = _calibrate_tracking(
            tracking,
            "What method did the paper use?",
            "The paper used a controlled experimental method.",
            citations,
            {"find_evidence": 1},
            [],
            None,
            {"items": citations},
        )

        assert calibrated["citation_quality"]["status"] == "pass"
        assert calibrated["retrieval_relevance"]["status"] == "pass"
        assert calibrated["overall_status"] == "passed"
        assert calibrated["repair_recommended"] is False
        assert calibrated["calibration_notes"]

    def test_tracking_calibration_scores_content_artifact_relevance(self):
        from app.qa.evaluation import _calibrate_tracking

        tracking = {
            "answer_relevance": {"status": "not_applicable", "score": 1.0, "note": "Artifact created."},
            "groundedness": {"status": "pass", "score": 0.8, "note": "Grounded."},
            "citation_quality": {"status": "pass", "score": 0.8, "note": "Cited."},
            "retrieval_relevance": {"status": "pass", "score": 0.8, "note": "Relevant."},
            "tool_choice_quality": {"status": "pass", "score": 0.8, "note": "Good."},
            "artifact_match": {"status": "pass", "score": 1.0, "note": "File created."},
        }

        calibrated = _calibrate_tracking(
            tracking,
            "Make me a MD file of the limitations.",
            "The paper identifies limitations in the generated images.",
            [{"section": "Limitations", "page": 4, "quote": "The generated images have limitations."}],
            {"find_evidence": 1, "create_md": 1},
            [{"kind": "markdown"}],
            None,
            {"items": [{"page": 4}]},
        )

        assert calibrated["answer_relevance"]["status"] == "pass"
        assert calibrated["overall_status"] == "passed"


class TestQAMcpHostDecoding:
    def test_decode_tool_result_unwraps_fastmcp_result_wrapper(self):
        from app.qa.mcp_host import decode_tool_result

        class _Text:
            type = "text"

            def __init__(self, text):
                self.text = text

        class _Result:
            def __init__(self):
                self.structuredContent = {
                    "result": json.dumps({
                        "kind": "markdown",
                        "label": "Download Markdown",
                        "url": "/qa-assets/sess1/answer.md",
                    })
                }
                self.content = [_Text(self.structuredContent["result"])]

        decoded = decode_tool_result(_Result())
        assert decoded["kind"] == "markdown"
        assert decoded["url"] == "/qa-assets/sess1/answer.md"


class TestQAMcpServerStdio:
    def test_stdio_server_lists_tools_and_creates_markdown(self, tmp_path):
        self._seed_metadata(tmp_path)
        asyncio.run(self._exercise_stdio(tmp_path))

    def _seed_metadata(self, tmp_path: Path) -> None:
        env = os.environ.copy()
        env.update({
            "DB_PATH": str(tmp_path / "research.db"),
            "PDF_DIR": str(tmp_path / "pdfs"),
            "VECTORSTORE_DIR": str(tmp_path / "vectorstore"),
            "QA_ASSETS_DIR": str(tmp_path / "qa_assets"),
        })

        seed_script = """
import app.database as db
db.init_db()
db.cache_paper_metadata({
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani"],
    "abstract": "Cached abstract for stdio MCP test.",
    "published": "2017-06-12",
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "categories": ["cs.CL"],
})
"""
        subprocess.run([sys.executable, "-c", seed_script], cwd=ROOT, env=env, check=True)

    async def _exercise_stdio(self, tmp_path: Path) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = os.environ.copy()
        env.update({
            "DB_PATH": str(tmp_path / "research.db"),
            "PDF_DIR": str(tmp_path / "pdfs"),
            "VECTORSTORE_DIR": str(tmp_path / "vectorstore"),
            "QA_ASSETS_DIR": str(tmp_path / "qa_assets"),
        })

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.qa.mcp_server"],
            cwd=ROOT,
            env=env,
        )

        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}
                assert {"find_evidence", "create_md", "create_pdf", "create_graphic", "create_presentation", "compare_sections", "cite_evidence"}.issubset(tool_names)

                templates = await session.list_resource_templates()
                template_names = {tpl.uriTemplate for tpl in templates.resourceTemplates}
                assert "researchatlas://paper/{arxiv_id}/metadata" in template_names

                metadata = await session.read_resource(
                    _URL.validate_python("researchatlas://paper/1706.03762/metadata")
                )
                payload = json.loads(metadata.contents[0].text)
                assert payload["title"] == "Attention Is All You Need"

                created = await session.call_tool(
                    "create_md",
                    {
                        "session_id": "qa-stdio",
                        "title": "Downloadable Answer",
                        "question": "What is the contribution?",
                        "answer": "A transformer architecture.",
                        "citations_json": "[]",
                    },
                )
                text_payload = json.loads(created.content[0].text)
                assert Path(text_payload["path"]).exists()
