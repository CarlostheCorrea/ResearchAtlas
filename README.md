# ArXiv Research Assistant

An AI-powered research assistant that autonomously searches arXiv, downloads and analyzes full PDFs, answers questions grounded in paper content, and maintains a persistent research library — built with MCP, LangGraph multi-agent workflows, and human-in-the-loop approval gates.

---

## Why this can't be replicated by prompting

- **Live arXiv API**: The agent queries the arXiv export API autonomously — not copy-pasted text
- **Full PDF processing**: Downloads PDFs from disk, extracts and chunks text — not uploaded files
- **Persistent library**: Research library grows across sessions; preferences change agent behavior over time
- **Human approval gates**: `interrupt()` pause points halt mid-execution and wait — chat sessions can't do this
- **Multi-agent pipeline**: Manager routes to specialized workers in a defined graph sequence — not one LLM doing everything
- **Pre-filter gate**: Pure Python filter runs before any LLM call — zero cost, sub-10ms for 50 papers

---

## Architecture

```
Overall pattern: Hierarchical Orchestrator
├── Discovery flow:   Manager → Search Worker → Pre-filter → Ranking Worker → END
├── Analysis flow:    [interrupt] → Ingestion → Chunker → Embedder → Summary → [interrupt] → Memory
└── Q&A flow:         Manager → Retrieval Worker → QA Worker → END

Ports:
  :8000  FastAPI main API + static frontend
  :8001  MCP tool server (agents call this)

Storage:
  data/research.db      SQLite (library, preferences, feedback, chunks)
  data/pdfs/            Downloaded PDFs
  data/vectorstore/     ChromaDB embeddings
```

---

## Week 8 — MCP as the USB for AI

The MCP server (`app/mcp_server/`) connects all agents to data sources **without custom API integrations hardcoded into agent code**. Agents call MCP tools via HTTP — they never import `requests`, `sqlite3`, `fitz`, or `chromadb` directly.

This means: swap arXiv for Semantic Scholar tomorrow and only `tools_arxiv.py` changes. Agent code stays identical. Local PDF files are accessed via the `download_pdf` MCP tool, not direct filesystem calls.

---

## Week 9 — The MCP Server

Custom FastAPI MCP server at `app/mcp_server/server.py` with 4 tool groups:

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
- `index_paper` — embeds chunks with `all-MiniLM-L6-v2`, stores in ChromaDB
- `retrieve_paper_chunks` — semantic search, filters distance > 1.5
- `get_paper_section` — returns all chunks from a named section

**Memory tools** (`tools_memory.py`)
- `save_to_library` — UPSERT to SQLite after human approval
- `save_user_preference` — running average weight update (0.7 × old + 0.3 × new)
- `log_feedback` — ratings 4–5 boost topic weights, 1–2 reduce them
- `create_pending_review` / `resolve_review` — approval workflow

Test: `GET http://localhost:8001/tools` returns all 18 registered tools.

---

## Week 10 — Multi-Agent Pattern

**Hierarchical Orchestrator**: The Manager Agent classifies intent and routes to one of three sequential sub-pipelines. It never calls arXiv, reads a PDF, or writes to SQLite.

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

**Why not swarm/collaborative**: Each worker has exactly one job and doesn't know about other workers. The manager holds all routing logic. This makes debugging deterministic and testing isolated.

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

# 3. First run downloads the embedding model (~80MB)
#    This is the all-MiniLM-L6-v2 sentence-transformers model.
#    It caches at ~/.cache/torch/ and is never re-downloaded.

# 4. Start both servers
uvicorn app.main:app --reload --port 8000 &
uvicorn app.mcp_server.server:app --reload --port 8001 &

# 5. Open the frontend
open http://localhost:8000
```

---

## Demo Walkthrough

1. Open `http://localhost:8000`
2. Search: `"retrieval augmented generation"` — filter report appears, papers ranked with score bars
3. Click a paper → sidebar shows title, authors, abstract
4. Click **Analyze Paper** → approval modal: "Download and analyze?" → click Yes
5. Progress steps animate: Downloading → Extracting → Indexing → Generating summary
6. Second approval modal: review the 10-section summary → Approve & Save
7. Switch to **Ask a Question** tab → ask "What dataset did they use?"
8. Answer appears with section citations like `[Methods]` `[Experiments]`
9. Switch to **My Library** → rate the paper 4 stars
10. Search again — preferences now influence ranking scores

---

## API Cost Estimate

A full demo session (1 search + 1 paper analysis + 3 questions):
- Manager classification: ~300 tokens × 1 = $0.001
- Ranking (20 papers): ~2000 tokens × 1 = $0.006
- Summary generation: ~4000 tokens × 1 = $0.012
- Q&A (3 questions): ~1500 tokens × 3 = $0.014
- **Total: ~$0.03–0.05 per session**

Embedding model (`all-MiniLM-L6-v2`) runs locally — zero cost after first download.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: MCP tools (arXiv, memory), RAG pipeline (cleaning, chunking), graph routing, and SQLite persistence.
