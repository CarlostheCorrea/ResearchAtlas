"""
FastAPI app entry point — mounts all routers and serves the frontend.
Phase 3 — Week 8: MCP Foundations. The main API coordinates agents and serves the UI.

Run: uvicorn app.main:app --reload --port 8000
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import app.database as db
from app.config import PDF_DIR, QA_ASSET_MAX_AGE_HOURS, QA_ASSETS_DIR, VECTORSTORE_DIR
from app.observability import setup_phoenix_tracing

PHOENIX_BOOT_STATUS = setup_phoenix_tracing()

from api.routes_search import router as search_router
from api.routes_chat import router as chat_router
from api.routes_qa import router as qa_router
from api.routes_review import router as review_router
from api.routes_library import router as library_router
from app.qa.assets import purge_old_assets

app = FastAPI(
    title="ArXiv Research Assistant",
    description="AI-powered research assistant with MCP, LangGraph, and multi-agent workflows",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup():
    phoenix = setup_phoenix_tracing()
    db.init_db()
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    os.makedirs(QA_ASSETS_DIR, exist_ok=True)
    purge_old_assets(QA_ASSET_MAX_AGE_HOURS)
    print("[main] Database initialized, data directories ready.")
    print(f"[main] Phoenix tracing: {phoenix['status']} - {phoenix['message']}")


# Register API routers
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(qa_router)
app.include_router(review_router)
app.include_router(library_router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "arxiv-research-assistant"}


# Serve frontend as static files at root
# This must come AFTER API routes to avoid catching /api/* paths
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(QA_ASSETS_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
app.mount("/qa-assets", StaticFiles(directory=QA_ASSETS_DIR), name="qa-assets")
app.mount("/paper-pdfs", StaticFiles(directory=PDF_DIR), name="paper-pdfs")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
