"""
Optional Phoenix/OpenInference tracing hooks.

The app should keep running if Phoenix is not installed or not listening locally.
"""
from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any

from app.config import PHOENIX_PROJECT_NAME

_tracer = None
_setup_done = False
_phoenix_status: dict[str, Any] = {
    "enabled": False,
    "project_name": PHOENIX_PROJECT_NAME,
    "status": "not_started",
    "message": "Phoenix tracing has not been initialized.",
}


def setup_phoenix_tracing() -> dict[str, Any]:
    """Initialize Phoenix tracing if the optional packages are available."""
    global _tracer, _phoenix_status, _setup_done

    if _setup_done:
        return dict(_phoenix_status)

    try:
        from phoenix.otel import register
        from opentelemetry import trace

        # Suppress the verbose Phoenix banner that prints to stdout/stderr on startup
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            register(project_name=PHOENIX_PROJECT_NAME, auto_instrument=True)
        _tracer = trace.get_tracer("researchatlas.qa")
        _phoenix_status = {
            "enabled": True,
            "project_name": PHOENIX_PROJECT_NAME,
            "status": "ready",
            "message": "Phoenix tracing is configured. OpenAI calls and Q/A spans will be exported when a Phoenix collector is available.",
        }
    except Exception as exc:
        _tracer = None
        _phoenix_status = {
            "enabled": False,
            "project_name": PHOENIX_PROJECT_NAME,
            "status": "disabled",
            "message": f"Phoenix tracing unavailable: {exc}",
        }

    _setup_done = True
    return dict(_phoenix_status)


def phoenix_status() -> dict[str, Any]:
    return dict(_phoenix_status)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Create an OpenTelemetry span when tracing is active, otherwise no-op."""
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield span
