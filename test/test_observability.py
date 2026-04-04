"""Tests for structured logging and observability."""

import io
import json
import logging

from observability.logging import (
    ContextFilter,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_id,
    set_request_context,
)


class TestRequestContext:
    """Tests for request context propagation."""

    def test_default_request_id_is_none(self):
        clear_request_context()
        assert get_request_id() is None

    def test_set_and_get_request_id(self):
        set_request_context("test-id-123")
        assert get_request_id() == "test-id-123"
        clear_request_context()

    def test_clear_request_context(self):
        set_request_context("test-id")
        clear_request_context()
        assert get_request_id() is None


class TestContextFilter:
    """Tests for the ContextFilter log filter."""

    def test_filter_injects_request_id(self):
        set_request_context("req-abc")
        filt = ContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=None, exc_info=None,
        )
        filt.filter(record)
        assert record.request_id == "req-abc"  # type: ignore[attr-defined]
        clear_request_context()

    def test_filter_empty_when_no_context(self):
        clear_request_context()
        filt = ContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=None, exc_info=None,
        )
        filt.filter(record)
        assert record.request_id == ""  # type: ignore[attr-defined]


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_json_format_produces_valid_json(self):
        configure_logging(level="DEBUG", fmt="json")
        log = get_logger("test.json_format")

        # Capture output
        handler = logging.StreamHandler(stream := io.StringIO())
        handler.setFormatter(log.root.handlers[0].formatter)
        handler.addFilter(ContextFilter())
        log.addHandler(handler)

        set_request_context("json-test-id")
        log.info("test message", extra={"component": "test"})
        clear_request_context()

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert "timestamp" in data

        log.removeHandler(handler)

    def test_text_format_produces_readable_output(self):
        configure_logging(level="DEBUG", fmt="text")
        log = get_logger("test.text_format")

        handler = logging.StreamHandler(stream := io.StringIO())
        handler.setFormatter(log.root.handlers[0].formatter)
        handler.addFilter(ContextFilter())
        log.addHandler(handler)

        set_request_context("text-test-id")
        log.info("readable message")
        clear_request_context()

        output = stream.getvalue()
        assert "readable message" in output
        assert "request_id=text-test-id" in output

        log.removeHandler(handler)

    def test_get_logger_returns_named_logger(self):
        log = get_logger("my.module")
        assert log.name == "my.module"
        assert isinstance(log, logging.Logger)
