"""Distributed tracing with contextvars-based span collection."""

import functools
import time
from contextvars import ContextVar
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_spans_var: ContextVar[list[dict[str, Any]]] = ContextVar("trace_spans", default=[])


def set_trace_context(trace_id: str) -> None:
    """Initialize a new trace context."""
    _trace_id_var.set(trace_id)
    _spans_var.set([])


def get_trace_id() -> str | None:
    """Get the current trace ID."""
    return _trace_id_var.get()


def get_trace_spans() -> list[dict[str, Any]]:
    """Get all recorded spans for the current trace."""
    return _spans_var.get()


def clear_trace_context() -> None:
    """Clear the trace context."""
    _trace_id_var.set(None)
    _spans_var.set([])


def get_trace_summary() -> dict[str, Any]:
    """Get a summary of the current trace.

    Returns:
        Dict with trace_id, spans, and total_ms.
    """
    spans = get_trace_spans()
    total_ms = sum(s.get("duration_ms", 0) for s in spans)
    return {
        "trace_id": get_trace_id(),
        "spans": spans,
        "total_ms": round(total_ms, 2),
    }


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(component: str):
    """Decorator that records a span for the decorated function.

    Args:
        component: Name of the component (e.g. 'retrieval', 'generation').

    Usage:
        @traced("retrieval")
        def search(query):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                span = {
                    "component": component,
                    "function": func.__name__,
                    "duration_ms": duration_ms,
                    "trace_id": get_trace_id(),
                }
                try:
                    spans = _spans_var.get()
                    spans.append(span)
                except LookupError:
                    pass

                logger.debug(
                    "Span complete: %s.%s (%.2fms)",
                    component,
                    func.__name__,
                    duration_ms,
                    extra={"component": component, "duration_ms": duration_ms},
                )

        return wrapper

    return decorator