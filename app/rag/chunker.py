"""
Splits cleaned PDF text into overlapping chunks with section metadata.
Phase 3 — Week 9: MCP Server (called by chunk_paper MCP tool).
Target: 800 words per chunk, 120-word overlap, section tracking.
"""
import re
from typing import Optional

KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "approach", "model",
    "experiments", "experimental", "evaluation",
    "results", "discussion", "conclusion", "conclusions",
    "limitations", "future work", "references", "appendix",
    "acknowledgements", "acknowledgments",
]

TARGET_WORDS = 800
OVERLAP_WORDS = 120


def detect_section(line: str) -> Optional[str]:
    """Return section name if line looks like a section header, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None

    # ALL CAPS header
    if stripped.isupper() and len(stripped) > 2:
        return stripped.title()

    # Numbered section: "1. Introduction", "2.1 Methods"
    m = re.match(r'^\d+(\.\d+)*\s+([A-Z][A-Za-z ]+)$', stripped)
    if m:
        return m.group(2).strip()

    # Title Case line that matches a known section keyword
    lower = stripped.lower()
    for kw in KNOWN_SECTIONS:
        if lower == kw or lower.startswith(kw + " ") or lower.endswith(" " + kw):
            return stripped.title()

    return None


def chunk_text(arxiv_id: str, cleaned_text: str) -> dict:
    """
    Split text into overlapping chunks with section metadata.
    Returns: { arxiv_id, chunk_count, chunks: list[dict] }
    """
    words = cleaned_text.split()
    total_words = len(words)

    if total_words == 0:
        return {"arxiv_id": arxiv_id, "chunk_count": 0, "chunks": []}

    # Build a per-word section map by scanning lines
    lines = cleaned_text.split('\n')
    current_section = "Abstract"
    word_index = 0
    section_map: list[str] = []  # section name for each word index

    for line in lines:
        line_words = line.split()
        detected = detect_section(line)
        if detected:
            current_section = detected
        for _ in line_words:
            section_map.append(current_section)
            word_index += 1

    # Pad section_map if needed
    while len(section_map) < total_words:
        section_map.append(current_section)

    # Estimate chars per page for page numbering (~3000 chars/page typical)
    chars_per_page = max(3000, len(cleaned_text) // max(1, len(cleaned_text) // 3000 + 1))

    chunks = []
    chunk_n = 0
    start = 0

    while start < total_words:
        end = min(start + TARGET_WORDS, total_words)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        section = section_map[start] if start < len(section_map) else "Unknown"

        # Estimate page from character position
        char_pos = len(" ".join(words[:start]))
        page = max(1, char_pos // chars_per_page + 1)

        chunks.append({
            "chunk_id": f"{arxiv_id}_chunk_{chunk_n:03d}",
            "arxiv_id": arxiv_id,
            "text": chunk_text_str,
            "section": section,
            "page": page,
            "word_count": len(chunk_words),
        })

        chunk_n += 1
        # Advance by (TARGET_WORDS - OVERLAP_WORDS) to create overlap
        start += TARGET_WORDS - OVERLAP_WORDS
        if start >= total_words:
            break

    return {
        "arxiv_id": arxiv_id,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
