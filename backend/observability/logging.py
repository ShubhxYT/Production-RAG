"""Structured logging configuration with JSON support and request context."""

import logging
import logging.handlers
import sys
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter

# ---------------------------------------------------------------------------
# Request context (propagated via contextvars)
# ---------------------------------------------------------------------------

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_context(request_id: str) -> None:
    """Set the request ID for the current context."""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Get the current request ID, or None if not set."""
    return _request_id_var.get()


def clear_request_context() -> None:
    """Clear the request context."""
    _request_id_var.set(None)


# ---------------------------------------------------------------------------
# Context filter - injects request_id into every log record
# ---------------------------------------------------------------------------


class ContextFilter(logging.Filter):
    """Injects request_id from contextvars into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or ""  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------

_NOISY_LOGGERS = [
    "uvicorn.access",
    "sqlalchemy.engine",
    "httpcore",
    "httpx",
    "urllib3",
    "sentence_transformers",
]


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
) -> None:
    """Configure the root logger with structured or text formatting.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Format type - 'json' for structured JSON, 'text' for human-readable.
        log_file: Optional file path for a rotating log file.
    """
    root = logging.getLogger()

    # Clear existing handlers
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Context filter for request_id
    ctx_filter = ContextFilter()

    if fmt == "json":
        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s "
            "[request_id=%(request_id)s]",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Stderr handler
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ctx_filter)
    root.addHandler(stream_handler)

    # Optional rotating file handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ctx_filter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)
