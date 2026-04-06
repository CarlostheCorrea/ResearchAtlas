"""
Text cleaning for extracted PDF content.
Phase 3 — Week 9: MCP Server (called by clean_pdf_text MCP tool).
Pure function — deterministic, no side effects, fast.
"""
import re
from collections import Counter


def clean_pdf_text(raw_text: str) -> dict:
    """
    Remove noise from extracted PDF text.
    Steps: page numbers, repeated headers/footers, hyphenated line breaks,
    arXiv watermarks, collapsed newlines, whitespace normalization.
    Returns: { cleaned_text, removed_chars }
    """
    original_len = len(raw_text)
    text = raw_text

    # 1. Remove page numbers: lines that are only digits (with optional whitespace)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 2. Remove arXiv watermarks: lines starting with "arXiv:"
    text = re.sub(r'^arXiv:.*$', '', text, flags=re.MULTILINE)

    # 3. Remove repeated headers/footers: lines appearing 3+ times
    lines = text.split('\n')
    line_counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in line_counts.items() if count >= 3 and len(line) > 5}
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
        text = '\n'.join(lines)

    # 4. Fix hyphenated line breaks: "hyphen-\nated" → "hyphenated"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # 5. Collapse multiple newlines: 3+ consecutive → 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Normalize whitespace: multiple spaces → single space (preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 7. Strip leading/trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    cleaned_len = len(text)
    return {
        "cleaned_text": text.strip(),
        "removed_chars": original_len - cleaned_len,
    }
