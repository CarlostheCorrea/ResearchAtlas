# ResearchAtlas

An AI-powered research assistant that searches arXiv, downloads and analyzes full PDFs, answers grounded questions about paper content, and maintains a persistent research library.

Built with **FastAPI · LangGraph · MCP · ChromaDB · OpenAI · Arize Phoenix**.

------------------------------------------------------------------------

## What it does

| Feature | Description |
|------------------------------------|------------------------------------|
| **Paper search** | Queries the live arXiv API, pre-filters results, and ranks them by relevance |
| **PDF analysis** | Downloads PDFs, extracts and chunks text, generates a 10-section summary |
| **Grounded Q&A** | MCP-driven question answering with tool timelines, citations, and evidence viewer |
| **Downloadable assets** | Generates Markdown, PDF, and image outputs on request |
| **LLM-as-judge** | Two independent judge calls score every answer on 6 quality metrics |
| **Auto-repair** | If the judge finds failures, the orchestrator re-runs evidence gathering |
| **Arize Phoenix** | Optional observability — traces every tool call and eval score in the Phoenix UI |
| **Research library** | Persistent library with star ratings that influence future ranking |

------------------------------------------------------------------------

## Quick Start

### 1 — Prerequisites

-   **Python 3.11+**
-   **OpenAI API key** (needs access to `gpt-4o` and `gpt-image-1`)
-   Git

### 2 — Clone and install

``` bash
git clone https://github.com/CarlostheCorrea/ResearchAtlas.git
cd ResearchAtlas
pip install -r requirements.txt
```

### 3 — Configure environment

``` bash
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=
```

Everything else has a working default. See `.env.example` for the full list of options.

### 4 — Create data directories

``` bash
mkdir -p data/pdfs data/vectorstore data/qa_assets
```

### 5 — Start the servers

Start everything from one terminal with the included script:

``` bash
bash start.sh
```

This starts all three servers in the background and prints their URLs:

- **App** → `http://localhost:8000` — FastAPI app + frontend
- **Tools** → `http://localhost:8001` — Internal tool server
- **Phoenix** → `http://localhost:6006` — Arize Phoenix UI *(optional)*

Press `Ctrl+C` once to stop all servers cleanly.

> **Make it executable (optional):** Run `chmod +x start.sh` once, then you can use `./start.sh` instead of `bash start.sh`.

> **Start without Phoenix:** Comment out the `python -m phoenix.server.main serve &` line in `start.sh` before running.

> **Prefer separate terminals?** You can run each server manually instead:
> ```bash
> # Terminal 1
> uvicorn app.main:app --reload --port 8000
> # Terminal 2
> uvicorn app.mcp_server.server:app --reload --port 8001
> # Terminal 3 (optional)
> python -m phoenix.server.main serve
> ```

> The Q&A MCP server (`app/qa/mcp_server.py`) launches **automatically** over `stdio` when Q&A requests run — you do not need to start it manually.

### 6 — Open the app

``` bash
open http://localhost:8000
```

------------------------------------------------------------------------

## Optional: Arize Phoenix Tracing

Phoenix gives you a live UI showing every LLM call, tool span, and eval score from the judge.

**Terminal 3 — Phoenix collector**

``` bash
python -m phoenix.server.main serve
```

Then open **http://localhost:6006** to see the Phoenix dashboard.

When Phoenix is running, the **Tracking** tab in the Q&A panel shows a "View traces in Phoenix ↗" link after each answer.

To disable the LLM judge (faster, no extra API calls):

```
QA_JUDGE_ENABLED=false
```

------------------------------------------------------------------------

## Demo Walkthrough

1.  Open **http://localhost:8000**
2.  Search for a topic — e.g. `"retrieval augmented generation"` — and choose a year and category filter
3.  Papers are ranked by relevance. Click one to open it.
4.  Click **Analyze Paper** → approve the download → watch the progress steps
5.  Approve the 10-section summary to save it to your library
6.  Switch to the **Ask a Question** tab and try:
    -   `"What is the main contribution of this paper?"`
    -   `"Find me evidence for the main claim"`
    -   `"Make me an image of the data pipeline"`
    -   `"Give me a downloadable markdown summary"`
7.  Watch the **Tools** side panel for the live MCP tool timeline
8.  Switch to **Logs** to see the timestamped event stream
9.  Switch to **Tracking** to see the LLM-as-judge scores for the answer
10. Rate the paper in **My Library** — ratings shift future ranking weights

------------------------------------------------------------------------

## Project Structure

```
ResearchAtlas/
├── app/
│   ├── main.py                  # FastAPI app, startup, Phoenix init
│   ├── config.py                # All env-var settings
│   ├── observability.py         # Phoenix/OpenTelemetry tracing (optional)
│   ├── prompts.py               # All LLM system prompts
│   ├── schemas.py               # Pydantic request/response models
│   ├── database.py              # SQLite helpers
│   ├── agents/                  # LangGraph agents (manager, search, summary, memory)
│   ├── graph/                   # LangGraph graph definition and state
│   ├── mcp_server/              # Internal FastAPI tool server (port 8001)
│   │   ├── tools_arxiv.py       # arXiv search and metadata tools
│   │   ├── tools_pdf.py         # PDF download, extract, chunk
│   │   ├── tools_rag.py         # ChromaDB embed and retrieve
│   │   └── tools_memory.py      # Library and preference tools
│   ├── qa/
│   │   ├── mcp_server.py        # Real MCP SDK server (stdio) — Q&A tools
│   │   ├── mcp_host.py          # MCP client that spawns the server
│   │   ├── orchestrator.py      # Q&A planning loop, synthesis, repair
│   │   ├── evaluation.py        # LLM-as-judge (two independent calls)
│   │   └── assets.py            # Per-session file management
│   └── rag/
│       ├── embeddings.py        # OpenAI text-embedding-3-small
│       └── vectorstore.py       # ChromaDB collection management
├── api/
│   ├── routes_qa.py             # POST /api/qa, GET /api/qa/status/{id}
│   ├── routes_search.py         # POST /api/search
│   ├── routes_library.py        # Library CRUD
│   ├── routes_review.py         # Human approval endpoints
│   └── routes_chat.py           # Summary chat endpoints
├── frontend/
│   ├── index.html               # Single-page app shell
│   ├── app.js                   # All UI logic (no build step)
│   ├── styles.css               # Dual-theme CSS (light/dark)
│   └── workflow_diagram.html    # Downloadable architecture diagram
├── tests/
├── .env.example                 # All config vars with comments
├── requirements.txt
└── pyproject.toml
```

------------------------------------------------------------------------

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser  (localhost:8000)                                       │
│  Static frontend — no build step, served by FastAPI             │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP polling every 1.5 s
┌───────────────────────▼─────────────────────────────────────────┐
│  FastAPI  :8000                                                  │
│  ├── Search/Library/Review routes                                │
│  └── Q&A routes  →  run_qa_orchestrator (background task)       │
└───────────┬──────────────────────────┬──────────────────────────┘
            │                          │
   LangGraph graph                     │  stdio
   (analysis pipeline)                 ▼
            │              ┌──────────────────────┐
            │              │  MCP Server (stdio)  │
            │              │  app/qa/mcp_server.py│
            │              │  ├── ensure_context  │
            │              │  ├── retrieve_chunks │
            │              │  ├── find_evidence   │
            │              │  ├── create_graphic  │
            │              │  └── create_md/pdf   │
            │              └──────────────────────┘
            │
   Internal tool server
   app.mcp_server  :8001
   ├── tools_arxiv.py
   ├── tools_pdf.py
   ├── tools_rag.py
   └── tools_memory.py

Storage:
  data/research.db       SQLite  — library, preferences, graph state
  data/pdfs/             Downloaded PDFs
  data/vectorstore/      ChromaDB embeddings
  data/qa_assets/        Per-session Q&A outputs (MD, PDF, PNG)
```

------------------------------------------------------------------------

## Q&A Pipeline (detailed)

Every question goes through this pipeline:

```
Question
  │
  ▼
Planner (GPT-4o) ──► MCP tool call ──► tool result
  │  ▲___________________________|
  │  (loop up to QA_MAX_TOOL_STEPS)
  ▼
Synthesize answer from all tool results
  │
  ▼
LLM-as-judge — two independent calls:
  ├── Quality judge:    answer_relevance + groundedness
  └── Supporting judge: citation_quality + retrieval_relevance
                        + tool_choice_quality + artifact_match
  │
  ▼
Repair? (if core metrics fail)
  ├── Yes → force find_evidence → re-synthesize → re-judge → compare scores
  └── No  → return answer with tracking
  │
  ▼
Return to frontend: answer + citations + assets + tracking scores
```

------------------------------------------------------------------------

## Technology Stack

| Layer               | Technology                                         |
|---------------------|----------------------------------------------------|
| LLM                 | OpenAI GPT-4o                                      |
| Image generation    | OpenAI gpt-image-1                                 |
| Embeddings          | OpenAI text-embedding-3-small                      |
| Agent orchestration | LangGraph (analysis) + custom MCP host (Q&A)       |
| MCP protocol        | `mcp` Python SDK (FastMCP + stdio transport)       |
| Observability       | Arize Phoenix + OpenInference + OpenTelemetry      |
| Vector store        | ChromaDB                                           |
| Database            | SQLite via Python `sqlite3`                        |
| PDF processing      | PyMuPDF (primary) + pdfplumber (fallback)          |
| Backend             | FastAPI + Uvicorn                                  |
| Frontend            | Vanilla JS + CSS custom properties (no build step) |

------------------------------------------------------------------------

## Environment Variables

See `.env.example` for the full list. Most important:

| Variable | Default | Description |
|------------------------|------------------------|------------------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o` | Model for planning, synthesis, and judge |
| `OPENAI_IMAGE_MODEL` | `gpt-image-1` | Model for image generation |
| `QA_JUDGE_ENABLED` | `true` | Enable/disable LLM-as-judge scoring |
| `QA_REPAIR_ENABLED` | `true` | Enable/disable auto-repair on judge failures |
| `PHOENIX_PROJECT_NAME` | `researchatlas` | Project name shown in the Phoenix UI |
| `QA_MAX_TOOL_STEPS` | `4` | Max MCP tool calls per Q&A session |

------------------------------------------------------------------------

## Running Tests

``` bash
python -m pytest tests/ -v
```

| Test file | What it covers |
|------------------------------------|------------------------------------|
| `tests/test_qa_mcp.py` | Q&A MCP server tools, stdio integration, asset exports |
| `tests/test_graph.py` | LangGraph analysis workflow |
| `tests/test_rag.py` | Chunking, cleaning, retrieval helpers |
| `tests/test_memory.py` | SQLite memory and preference persistence |

> `tests/test_mcp_tools.py` makes live arXiv API calls and may fail in offline environments.

------------------------------------------------------------------------

## Cost Estimate

A typical session (1 search + 1 paper analysis + 3 Q&A questions):

| Operation                       | Approx. cost       |
|---------------------------------|--------------------|
| Manager classification          | \~\$0.001          |
| Ranking 20 papers               | \~\$0.006          |
| Summary generation              | \~\$0.012          |
| Q&A synthesis × 3               | \~\$0.014          |
| LLM-as-judge × 3 (2 calls each) | \~\$0.012          |
| Image generation (if requested) | \~\$0.04 per image |
| **Total (text only)**           | **\~\$0.05**       |

------------------------------------------------------------------------

## Troubleshooting

**App starts but Q&A always times out** The Q&A MCP server spawns a subprocess using `sys.executable`. Make sure the same virtualenv that runs the main app also has all dependencies installed.

**"Could not load MCP tool list"** The background MCP subprocess couldn't start. Check terminal 1 logs for a Python error. Often caused by a missing `OPENAI_API_KEY` in the environment.

**Chroma collection rebuilt on every start** This happens when the embedding model changes. Delete `data/vectorstore/` to start fresh — collections are rebuilt automatically on the next paper analysis.

**Phoenix not receiving traces** Make sure `python -m phoenix.server.main serve` is running *before* you start the FastAPI server. The Phoenix status appears in the Tracking tab after each Q&A answer.

**"Request timed out" on image generation** Image generation + vision validation can take up to 90 seconds. The frontend polls for up to 180 seconds. If you still hit timeouts, set `QA_JUDGE_ENABLED=false` to skip the validation step.
