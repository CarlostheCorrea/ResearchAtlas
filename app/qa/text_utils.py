"""Text cleanup helpers for user-facing Q/A content."""
from __future__ import annotations

import re


_INLINE_CITATION_MARKERS = (
    re.compile(r'(?is)(?:```(?:json)?\s*)?["\']?citations?["\']?\s*:\s*\[\s*\{?\s*"section"\s*:'),
    re.compile(r'(?is)(?:```(?:json)?\s*)?["\']?references?["\']?\s*:\s*\[\s*\{?\s*"section"\s*:'),
    re.compile(r'(?is)(?:```(?:json)?\s*)?["\']?sources?["\']?\s*:\s*\[\s*\{?\s*"section"\s*:'),
    re.compile(r'(?is)(?:```(?:json)?\s*)?\[\s*\{?\s*"section"\s*:'),
    re.compile(r'(?is)(?:```(?:json)?\s*)?\{\s*"section"\s*:\s*".+?"\s*,\s*"page"\s*:'),
)


def strip_inline_citation_metadata(text: str) -> str:
    """Remove serialized citation objects from answer prose.

    The synthesis model should return citations in a separate JSON field, but it
    can occasionally append objects like [{"section": "Paper Metadata", ...}]
    directly to the answer. Those are internal grounding metadata, not
    user-facing prose, so trim them before rendering or exporting.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    cutoff: int | None = None
    for pattern in _INLINE_CITATION_MARKERS:
        match = pattern.search(cleaned)
        if match:
            cutoff = match.start() if cutoff is None else min(cutoff, match.start())

    if cutoff is not None:
        cleaned = cleaned[:cutoff]

    return cleaned.rstrip(" \t\r\n,;:")
