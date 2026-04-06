"""
SQLite connection, table creation, and all DB operations.
Phase 3 — Week 11: Memory & Persistence. All research library, preferences,
feedback, and session state live here.
"""
import sqlite3
import json
import os
from typing import Optional
from app.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory for dict-like access."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        -- Research library
        CREATE TABLE IF NOT EXISTS saved_papers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id     TEXT UNIQUE NOT NULL,
            title        TEXT,
            summary_json TEXT,
            notes        TEXT DEFAULT '',
            rating       INTEGER,
            saved_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_saved_arxiv ON saved_papers(arxiv_id);

        -- User topic preferences (drives ranking agent weighting)
        CREATE TABLE IF NOT EXISTS user_preferences (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            topic      TEXT UNIQUE NOT NULL,
            weight     REAL DEFAULT 0.5,
            source     TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Feedback on papers
        CREATE TABLE IF NOT EXISTS paper_feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id   TEXT NOT NULL,
            rating     INTEGER NOT NULL,
            comment    TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_arxiv ON paper_feedback(arxiv_id);

        -- Summaries awaiting human approval
        CREATE TABLE IF NOT EXISTS pending_reviews (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id      TEXT NOT NULL,
            summary_json  TEXT NOT NULL,
            status        TEXT DEFAULT 'pending',
            revision_note TEXT DEFAULT '',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at   TIMESTAMP
        );

        -- Paper metadata cache
        CREATE TABLE IF NOT EXISTS papers_cache (
            arxiv_id         TEXT PRIMARY KEY,
            title            TEXT,
            authors_json     TEXT,
            abstract         TEXT,
            published        TEXT,
            pdf_url          TEXT,
            categories_json  TEXT,
            fetched_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- PDF download tracking
        CREATE TABLE IF NOT EXISTS pdf_cache (
            arxiv_id      TEXT PRIMARY KEY,
            local_path    TEXT,
            size_bytes    INTEGER,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Chunk storage (also in ChromaDB but SQLite for structured queries)
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            arxiv_id   TEXT NOT NULL,
            text       TEXT NOT NULL,
            section    TEXT,
            page       INTEGER,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_arxiv ON chunks(arxiv_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);

        -- Shortlisted papers
        CREATE TABLE IF NOT EXISTS shortlist (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id       TEXT NOT NULL,
            session_id     TEXT,
            shortlisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexed papers tracking
        CREATE TABLE IF NOT EXISTS indexed_papers (
            arxiv_id    TEXT PRIMARY KEY,
            chunk_count INTEGER,
            indexed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Extracted text cache
        CREATE TABLE IF NOT EXISTS extracted_text (
            arxiv_id   TEXT PRIMARY KEY,
            text       TEXT,
            char_count INTEGER,
            page_count INTEGER,
            method     TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


# ── Saved papers ──────────────────────────────────────────────────────────────

def upsert_saved_paper(arxiv_id: str, title: str, summary_dict: dict, notes: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO saved_papers (arxiv_id, title, summary_json, notes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            title = excluded.title,
            summary_json = excluded.summary_json,
            notes = excluded.notes,
            saved_at = CURRENT_TIMESTAMP
    """, (arxiv_id, title, json.dumps(summary_dict), notes))
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM saved_papers").fetchone()[0]
    conn.close()
    return count


def list_saved_papers() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM saved_papers ORDER BY saved_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_paper_rating(arxiv_id: str, rating: int) -> bool:
    """Persist star rating directly on the saved_papers row."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE saved_papers SET rating = ? WHERE arxiv_id = ?", (rating, arxiv_id))
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_all_saved_papers() -> int:
    """Delete every paper from the library. Returns count deleted."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_papers")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


def clear_all_preferences() -> int:
    """Wipe all user preference weights. Returns count deleted."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_preferences")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


def delete_saved_paper(arxiv_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_papers WHERE arxiv_id = ?", (arxiv_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_saved_arxiv_ids() -> set[str]:
    conn = get_connection()
    rows = conn.execute("SELECT arxiv_id FROM saved_papers").fetchall()
    conn.close()
    return {r[0] for r in rows}


# ── User preferences ──────────────────────────────────────────────────────────

def upsert_preference(topic: str, weight: float, source: str = "explicit") -> dict:
    conn = get_connection()
    cur = conn.cursor()
    existing = cur.execute(
        "SELECT weight FROM user_preferences WHERE topic = ?", (topic,)
    ).fetchone()

    if existing:
        new_weight = 0.7 * existing[0] + 0.3 * weight
        cur.execute("""
            UPDATE user_preferences SET weight = ?, source = ?, updated_at = CURRENT_TIMESTAMP
            WHERE topic = ?
        """, (new_weight, source, topic))
        is_new = False
    else:
        new_weight = weight
        cur.execute(
            "INSERT INTO user_preferences (topic, weight, source) VALUES (?, ?, ?)",
            (topic, weight, source)
        )
        is_new = True

    conn.commit()
    conn.close()
    return {"topic": topic, "weight": new_weight, "is_new": is_new}


def get_all_preferences() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT topic, weight, source, updated_at FROM user_preferences ORDER BY weight DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Feedback ──────────────────────────────────────────────────────────────────

def insert_feedback(arxiv_id: str, rating: int, comment: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO paper_feedback (arxiv_id, rating, comment) VALUES (?, ?, ?)",
        (arxiv_id, rating, comment)
    )
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM paper_feedback").fetchone()[0]
    conn.close()
    return count


def get_feedback_history(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM paper_feedback ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dismissed_ids() -> list[str]:
    """Return arxiv_ids with avg rating <= 2."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT arxiv_id FROM paper_feedback
        GROUP BY arxiv_id HAVING AVG(rating) <= 2
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── Pending reviews ───────────────────────────────────────────────────────────

def create_pending_review(arxiv_id: str, summary_dict: dict) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pending_reviews (arxiv_id, summary_json) VALUES (?, ?)",
        (arxiv_id, json.dumps(summary_dict))
    )
    conn.commit()
    summary_id = str(cur.lastrowid)
    conn.close()
    return {"summary_id": summary_id, "status": "pending"}


def resolve_review(summary_id: str, decision: str, revision_note: str = "") -> dict:
    conn = get_connection()
    conn.execute("""
        UPDATE pending_reviews
        SET status = ?, revision_note = ?, resolved_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (decision, revision_note, summary_id))
    conn.commit()
    conn.close()
    return {"summary_id": summary_id, "decision": decision}


# ── Papers cache ──────────────────────────────────────────────────────────────

def cache_paper_metadata(paper_dict: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO papers_cache
        (arxiv_id, title, authors_json, abstract, published, pdf_url, categories_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        paper_dict["arxiv_id"],
        paper_dict["title"],
        json.dumps(paper_dict["authors"]),
        paper_dict["abstract"],
        paper_dict["published"],
        paper_dict["pdf_url"],
        json.dumps(paper_dict["categories"]),
    ))
    conn.commit()
    conn.close()


def get_cached_paper(arxiv_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM papers_cache WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["authors"] = json.loads(d.pop("authors_json", "[]"))
    d["categories"] = json.loads(d.pop("categories_json", "[]"))
    return d


# ── PDF cache ─────────────────────────────────────────────────────────────────

def cache_pdf(arxiv_id: str, path: str, size_bytes: int) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO pdf_cache (arxiv_id, local_path, size_bytes)
        VALUES (?, ?, ?)
    """, (arxiv_id, path, size_bytes))
    conn.commit()
    conn.close()


# ── Chunks ────────────────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]) -> None:
    conn = get_connection()
    conn.executemany("""
        INSERT OR REPLACE INTO chunks (chunk_id, arxiv_id, text, section, page, word_count)
        VALUES (:chunk_id, :arxiv_id, :text, :section, :page, :word_count)
    """, chunks)
    conn.commit()
    conn.close()


def get_chunks_for_paper(arxiv_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY chunk_id", (arxiv_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_chunks_by_section(arxiv_id: str, section_hint: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE arxiv_id = ? AND LOWER(section) LIKE ?",
        (arxiv_id, f"%{section_hint.lower()}%")
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Indexed papers ────────────────────────────────────────────────────────────

def mark_paper_indexed(arxiv_id: str, chunk_count: int) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO indexed_papers (arxiv_id, chunk_count)
        VALUES (?, ?)
    """, (arxiv_id, chunk_count))
    conn.commit()
    conn.close()


def is_paper_indexed(arxiv_id: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM indexed_papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    conn.close()
    return row is not None


# ── Shortlist ──────────────────────────────────────────────────────────────────

def shortlist_paper(arxiv_id: str, session_id: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO shortlist (arxiv_id, session_id) VALUES (?, ?)",
        (arxiv_id, session_id)
    )
    conn.commit()
    conn.close()


# ── Extracted text ────────────────────────────────────────────────────────────

def cache_extracted_text(arxiv_id: str, text: str, char_count: int, page_count: int, method: str) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO extracted_text (arxiv_id, text, char_count, page_count, method)
        VALUES (?, ?, ?, ?, ?)
    """, (arxiv_id, text, char_count, page_count, method))
    conn.commit()
    conn.close()


# ── RAG cleanup ───────────────────────────────────────────────────────────────

def delete_rag_data_for_paper(arxiv_id: str) -> dict:
    """Delete all RAG-related SQLite rows for a single paper."""
    conn = get_connection()
    cur = conn.cursor()
    counts = {}
    for table in ("chunks", "indexed_papers", "pdf_cache", "extracted_text"):
        cur.execute(f"DELETE FROM {table} WHERE arxiv_id = ?", (arxiv_id,))
        counts[table] = cur.rowcount
    conn.commit()
    conn.close()
    return counts


def delete_all_rag_data() -> dict:
    """Delete all RAG-related SQLite rows across every paper."""
    conn = get_connection()
    cur = conn.cursor()
    counts = {}
    for table in ("chunks", "indexed_papers", "pdf_cache", "extracted_text"):
        cur.execute(f"DELETE FROM {table}")
        counts[table] = cur.rowcount
    conn.commit()
    conn.close()
    return counts
