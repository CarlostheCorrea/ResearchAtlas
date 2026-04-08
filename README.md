# ArXiv Research Assistant

An AI-powered research assistant that autonomously searches arXiv, downloads and analyzes full PDFs, answers grounded questions about paper content, and maintains a persistent research library. The project now uses two distinct runtime patterns: a button-driven LangGraph analysis flow for summaries, and an MCP-driven Q/A flow where the app acts as an MCP host and dynamically invokes paper tools.

---

## Why this can't be replicated by prompting

- **Live arXiv API**: The agent queries the arXiv export API autonomously — not copy-pasted text
- **Full PDF processing**: Downloads PDFs from disk, extracts and chunks text — not uploaded files
- **Persistent library**: Research library grows across sessions; preferences change agent behavior over time
- **Human approval gates**: `interrupt()` pause points halt mid-execution and wait — chat sessions can't do this
- **Split interaction model**: Summary generation is a fixed, reliable analysis flow, while Q/A uses dynamic MCP tool selection at runtime
- **Pre-filter gate**: Pure Python filter runs before any LLM call — zero cost, sub-10ms for 50 papers

---

## Architecture

```
Overall pattern: Hybrid research assistant
├── Discovery flow:   Manager → Search Worker → Pre-filter → Ranking Worker → END
├── Analysis flow:    [interrupt] → Ingestion → Chunker → Embedder → Summary → [interrupt] → Memory
└── Q&A flow:         FastAPI host → MCP stdio server → dynamic tool calls → grounded answer

Ports:
  :8000  FastAPI app + static frontend + Q/A host
  :8001  Legacy internal tool server used by the existing analysis pipeline

Storage:
  data/research.db      SQLite (library, preferences, feedback, chunks)
  data/pdfs/            Downloaded PDFs
  data/vectorstore/     ChromaDB embeddings
  data/qa_assets/       Per-session Markdown, PDF, and image outputs from Q/A
```

---

## Week 8 — MCP as the USB for AI

The original project introduced an MCP-style tool boundary so agents could reach arXiv, PDFs, ChromaDB, and SQLite without importing those systems directly in agent code.

This means: swap arXiv for Semantic Scholar tomorrow and only `tools_arxiv.py` changes. Agent code stays identical. Local PDF files are accessed via the `download_pdf` MCP tool, not direct filesystem calls.

---

## Week 9 — The MCP Server

The repo now has two different tool layers:

- `app/mcp_server/server.py` — the original internal FastAPI tool dispatcher used by the existing LangGraph analysis flow
- `app/qa/mcp_server.py` — a real MCP SDK server used by the Q/A experience over `stdio`

The Q/A MCP server is where the MCP-specific work now lives. The app itself acts as the MCP host via `app/qa/mcp_host.py` and `app/qa/orchestrator.py`.

**Q/A MCP tools** (`app/qa/mcp_server.py`)
- `ensure_paper_context` — prepare a paper for grounded Q/A by downloading, extracting, chunking, and indexing it if needed
- `retrieve_paper_chunks` — semantic retrieval for general paper questions
- `find_evidence` — return quote-level evidence with page and section metadata for highlighting
- `cite_evidence` — produce exact citations for a claim or answer
- `compare_sections` — compare two named sections inside the current paper
- `create_md` — generate a downloadable Markdown answer
- `create_pdf` — generate a downloadable PDF answer
- `create_graphic` — generate an image/graphic using the OpenAI image API

**Q/A MCP resources**
- `researchatlas://paper/{arxiv_id}/metadata`
- `researchatlas://paper/{arxiv_id}/abstract`

The original internal FastAPI tool server remains in place for the existing summary pipeline, with 4 tool groups:

**Discovery tools** (`tools_arxiv.py`)
- `search_papers` — queries `https://export.arxiv.org/api/query`, no key required
- `get_paper_metadata` — fetches metadata with SQLite cache
- `get_paper_abstract` — returns abstract for a paper ID
- `shortlist_paper` — adds paper to user shortlist

**PDF tools** (`tools_pdf.py`)
- `download_pdf` — downloads from `https://arxiv.org/pdf/{id}`, skips re-downloads
- `extract_pdf_text` — PyMuPDF primary, pdfplumber fallback
- `clean_pdf_text` — removes noise, fixes hyphenation, normalizes whitespace
- `chunk_paper` — 800-word target chunks with 120-word overlap, section detection

**RAG tools** (`tools_rag.py`)
- `index_paper` — embeds chunks with the configured OpenAI embedding model, stores in ChromaDB
- `retrieve_paper_chunks` — semantic search, filters distance > 1.5
- `get_paper_section` — returns all chunks from a named section

**Memory tools** (`tools_memory.py`)
- `save_to_library` — UPSERT to SQLite after human approval
- `save_user_preference` — running average weight update (0.7 × old + 0.3 × new)
- `log_feedback` — ratings 4–5 boost topic weights, 1–2 reduce them
- `create_pending_review` / `resolve_review` — approval workflow

This lets the project show both patterns clearly:
- a deterministic analysis workflow for summaries
- a dynamic MCP-driven Q/A workflow for tool discovery and tool use

---

## Week 10 — Multi-Agent Pattern

**Hybrid orchestration**: Discovery and summary analysis still use the existing LangGraph orchestrator. Q/A is now its own MCP-backed runtime because it benefits from dynamic tool choice, evidence gathering, asset generation, and visible tool traces.

| Agent | Responsibility |
|-------|---------------|
| Manager | Classify intent, extract parameters, load user preferences |
| Search Worker | Call `search_papers` MCP tool |
| Pre-filter | Pure Python filter — zero LLM calls |
| Ranking Worker | Score relevance (1 GPT-4o call for all papers at once) |
| Ingestion Agent | download → extract → clean → chunk → index via MCP |
| Summary Agent | Retrieve section chunks, generate 10-section summary |
| Retrieval Agent | Semantic search for question answering |
| QA Agent | Answer from chunks only — refuses general knowledge |
| Memory Agent | Write to library after human approval |

**Why split summary and Q/A**: the summary experience is intentionally button-driven and reliable, while Q/A is the place where dynamic tool use is visible to the user. This keeps the summary flow stable and lets the Q/A tab demonstrate MCP more clearly.

---

## Week 11 — Human-in-the-Loop & Memory

**Two pause points using LangGraph `interrupt()`:**

1. **Before download** (`human_gate_before_download`): User approves before the PDF is downloaded and processed. Shows paper title and estimated time.
2. **Before save** (`human_gate_before_save`): User reviews the full 10-section summary before it's written to the library. Can approve, reject, or request revision.

**Session persistence**: `SqliteSaver` checkpointer auto-saves all graph state to SQLite after every node. Sessions survive server restarts.

**Preference learning**: Topic weights influence the `preference_score` in ranking. Topics from papers rated 4–5 stars receive weight 0.8; papers rated 1–2 receive weight 0.1.

---

## Pre-filter Gate

Pure Python, zero LLM calls, zero cost. Runs in under 10ms for 20 papers.

Filters applied in order (cheapest first):
1. Already in research library — never resurface saved papers
2. Previously dismissed — arxiv_ids with avg rating ≤ 2
3. Year cutoff — drops papers older than `year_from`
4. Category filter — if user specified required arXiv categories
5. Keyword relevance — at least 1 query keyword must appear in title or abstract
6. Abstract length — drops stubs under 50 characters

The filter report is returned with every search: `50 fetched → 12 passed`.

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
# 1. Clone and install
git clone <repo>
cd ResearchAtlas
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY

# 3. Configure the embedding model if needed
#    Default: text-embedding-3-small
#    Embeddings now use the OpenAI API instead of a local Hugging Face download.

# 4. Start the app and the legacy internal tool server
uvicorn app.main:app --reload --port 8000 &
uvicorn app.mcp_server.server:app --reload --port 8001 &

# 5. Open the frontend
open http://localhost:8000
```

Notes:
- The Q/A MCP server is launched internally over `stdio` by the app when Q/A requests run.
- `create_graphic` uses your configured OpenAI API key and image model.
- Generated Q/A artifacts are written to `data/qa_assets/`.
- Existing Chroma collections created with the old local embedding model are treated as stale and will be rebuilt automatically the next time a paper is prepared or analyzed.

---

## Demo Walkthrough

1. Open `http://localhost:8000`
2. Search: `"retrieval augmented generation"` — filter report appears, papers ranked with score bars
3. Click a paper → sidebar shows title, authors, abstract
4. Click **Analyze Paper** → approval modal: "Download and analyze?" → click Yes
5. Progress steps animate: Downloading → Extracting → Indexing → Generating summary
6. Second approval modal: review the 10-section summary → Approve & Save
7. Switch to **Ask a Question** and ask:
   - `"What evidence supports the main claim?"`
   - `"Give me a downloadable answer as markdown and pdf"`
   - `"Show an image of the workflow"`
   - `"Compare the methods and results sections"`
8. The Q/A tab now shows:
   - a visible MCP tool timeline
   - exact quote-level citations with page/section references
   - downloadable Markdown/PDF outputs when requested
   - a generated image when requested
   - a PDF viewer with evidence highlighting
9. Switch to **My Library** → rate the paper 4 stars
10. Search again — preferences now influence ranking scores

---

## API Cost Estimate

A full demo session (1 search + 1 paper analysis + 3 questions):
- Manager classification: ~300 tokens × 1 = $0.001
- Ranking (20 papers): ~2000 tokens × 1 = $0.006
- Summary generation: ~4000 tokens × 1 = $0.012
- Q&A synthesis and tool-planning (3 questions): ~1500 tokens × 3 = $0.014
- Optional image generation: variable, only when `create_graphic` is used
- **Typical text-only session: ~$0.03–0.05**

Embeddings use your configured OpenAI embedding model, so semantic indexing/retrieval now incurs API cost.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Key local test groups:

- `tests/test_qa_mcp.py` — Q/A MCP server tools, `stdio` MCP integration, exports
- `tests/test_graph.py` — existing LangGraph workflow behavior
- `tests/test_rag.py` — cleaning, chunking, retrieval-related helpers
- `tests/test_memory.py` — SQLite memory and preference persistence

The legacy `tests/test_mcp_tools.py` suite still contains live arXiv-dependent checks and may fail in fully sandboxed or offline environments.
