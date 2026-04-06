"""
Ingestion agent — downloads PDF, extracts text, cleans, chunks, embeds.
Calls MCP tools: download_pdf → extract_pdf_text → clean_pdf_text → chunk_paper → index_paper.
All file I/O goes through MCP tools — agent never imports fitz or chromadb directly.

Phase 3 — Week 9: MCP Foundations. Agent as MCP client pattern.
"""
from app.mcp_client import get_mcp_client


def run_ingestion_agent(state: dict) -> dict:
    """Download, process, and index a paper via MCP tools."""
    mcp = get_mcp_client()
    arxiv_id = state["selected_arxiv_id"]
    print(f"[ingestion] Starting pipeline for {arxiv_id}")

    # Download PDF
    print(f"[ingestion] Step 1/5 — download_pdf")
    download_result = mcp.call_tool("download_pdf", {"arxiv_id": arxiv_id})
    if download_result.get("error"):
        return {"error": f"PDF download failed: {download_result['error']}"}
    print(f"[ingestion] Step 1/5 done — {download_result.get('size_bytes', 0):,} bytes")

    # Extract text
    print(f"[ingestion] Step 2/5 — extract_pdf_text (may take up to 60s for large PDFs)")
    extract_result = mcp.call_tool("extract_pdf_text", {"arxiv_id": arxiv_id})
    if not extract_result.get("text"):
        err = extract_result.get("error", "empty result")
        print(f"[ingestion] Step 2/5 FAILED — {err}")
        return {"error": f"PDF text extraction failed: {err}"}
    print(f"[ingestion] Step 2/5 done — {extract_result.get('char_count', 0):,} chars via {extract_result.get('method')}")

    # Clean text
    print(f"[ingestion] Step 3/5 — clean_pdf_text")
    clean_result = mcp.call_tool("clean_pdf_text", {
        "arxiv_id": arxiv_id,
        "raw_text": extract_result["text"],
    })
    print(f"[ingestion] Step 3/5 done")

    # Chunk
    print(f"[ingestion] Step 4/5 — chunk_paper")
    chunk_result = mcp.call_tool("chunk_paper", {
        "arxiv_id": arxiv_id,
        "cleaned_text": clean_result["cleaned_text"],
    })
    print(f"[ingestion] Step 4/5 done — {chunk_result.get('chunk_count', 0)} chunks")

    # Index into ChromaDB
    print(f"[ingestion] Step 5/5 — index_paper (embedding chunks...)")
    index_result = mcp.call_tool("index_paper", {"arxiv_id": arxiv_id})
    print(f"[ingestion] Step 5/5 done — {index_result.get('chunks_indexed', 0)} chunks indexed")

    return {
        "pdf_path": download_result["path"],
        "extracted_text": extract_result["text"],
        "chunks": chunk_result.get("chunks", []),
        "paper_indexed": index_result.get("chunks_indexed", 0) > 0,
    }
