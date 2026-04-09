"""
Per-session asset storage for Q/A outputs.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from app.config import QA_ASSETS_DIR


def ensure_assets_root() -> Path:
    root = Path(QA_ASSETS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_session_dir(session_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id)
    session_dir = ensure_assets_root() / safe
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def build_asset_url(session_id: str, filename: str) -> str:
    return f"/qa-assets/{session_id}/{filename}"


def _unique_filename(session_dir: Path, filename: str) -> str:
    candidate = session_dir / filename
    if not candidate.exists():
        return filename

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        next_name = f"{stem}-{counter}{suffix}"
        if not (session_dir / next_name).exists():
            return next_name
        counter += 1


def record_asset(session_id: str, filename: str, kind: str, label: str) -> dict:
    session_dir = ensure_session_dir(session_id)
    filename = _unique_filename(session_dir, filename)
    path = session_dir / filename
    return {
        "kind": kind,
        "label": label,
        "filename": filename,
        "path": str(path),
        "url": build_asset_url(session_id, filename),
    }


def purge_old_assets(max_age_hours: int = 24) -> None:
    root = ensure_assets_root()
    cutoff = time.time() - (max_age_hours * 3600)
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink(missing_ok=True)
                for nested_dir in sorted(child.rglob("*"), reverse=True):
                    if nested_dir.is_dir():
                        nested_dir.rmdir()
                child.rmdir()
        except OSError:
            continue


def clear_all_assets() -> int:
    root = ensure_assets_root()
    removed = 0
    for child in list(root.iterdir()):
        try:
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink(missing_ok=True)
                for nested_dir in sorted(child.rglob("*"), reverse=True):
                    if nested_dir.is_dir():
                        nested_dir.rmdir()
                child.rmdir()
                removed += 1
            elif child.is_file():
                child.unlink(missing_ok=True)
        except OSError:
            continue
    return removed
