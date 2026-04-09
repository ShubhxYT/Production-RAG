"""In-memory log handler for live log viewing in the Streamlit debug UI."""

import logging
from collections import deque
from datetime import datetime
from typing import Deque

MAX_LOG_RECORDS = 500


class LogRecord:
    """Lightweight serialisable log record."""

    __slots__ = ("timestamp", "level", "logger", "message", "request_id")

    def __init__(self, record: logging.LogRecord) -> None:
        self.timestamp = datetime.fromtimestamp(record.created).strftime(
            "%H:%M:%S.%f"
        )[:-3]
        self.level = record.levelname
        self.logger = record.name
        self.message = record.getMessage()
        self.request_id: str = getattr(record, "request_id", "") or ""

    def to_dict(self) -> dict:
        return {
            "time": self.timestamp,
            "level": self.level,
            "logger": self.logger,
            "message": self.message,
            "request_id": self.request_id,
        }


class InMemoryLogHandler(logging.Handler):
    """Logging handler that stores records in a fixed-size deque."""

    def __init__(self, maxlen: int = MAX_LOG_RECORDS) -> None:
        super().__init__()
        self._buffer: Deque[LogRecord] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(LogRecord(record))
        except Exception:
            self.handleError(record)

    def get_records(self) -> list[dict]:
        return [r.to_dict() for r in self._buffer]

    def clear(self) -> None:
        self._buffer.clear()


# --------------------------------------------------------------------------- #
# Module-level singleton — attach once and reuse everywhere
# --------------------------------------------------------------------------- #

_handler: InMemoryLogHandler | None = None


def get_log_handler() -> InMemoryLogHandler:
    """Return (and lazily attach) the singleton in-memory log handler."""
    global _handler
    if _handler is None:
        _handler = InMemoryLogHandler()
        _handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        _handler.setFormatter(formatter)
        logging.getLogger().addHandler(_handler)
    return _handler
