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


from observability.tracing import (
    clear_trace_context,
    get_trace_id,
    get_trace_spans,
    get_trace_summary,
    set_trace_context,
    traced,
)


class TestTraceContext:
    """Tests for trace context management."""

    def test_set_and_get_trace_id(self):
        set_trace_context("trace-123")
        assert get_trace_id() == "trace-123"
        clear_trace_context()

    def test_clear_trace_context(self):
        set_trace_context("trace-abc")
        clear_trace_context()
        assert get_trace_id() is None
        assert get_trace_spans() == []


class TestTracedDecorator:
    """Tests for the @traced decorator."""

    def test_traced_records_span(self):
        set_trace_context("trace-span-test")

        @traced("test_component")
        def sample_function():
            return 42

        result = sample_function()

        assert result == 42
        spans = get_trace_spans()
        assert len(spans) == 1
        assert spans[0]["component"] == "test_component"
        assert spans[0]["function"] == "sample_function"
        assert spans[0]["duration_ms"] >= 0
        assert spans[0]["trace_id"] == "trace-span-test"
        clear_trace_context()

    def test_traced_accumulates_spans(self):
        set_trace_context("trace-multi")

        @traced("comp_a")
        def func_a():
            pass

        @traced("comp_b")
        def func_b():
            pass

        func_a()
        func_b()

        spans = get_trace_spans()
        assert len(spans) == 2
        assert spans[0]["component"] == "comp_a"
        assert spans[1]["component"] == "comp_b"
        clear_trace_context()

    def test_get_trace_summary(self):
        set_trace_context("trace-summary")

        @traced("summarized")
        def work():
            pass

        work()

        summary = get_trace_summary()
        assert summary["trace_id"] == "trace-summary"
        assert len(summary["spans"]) == 1
        assert summary["total_ms"] >= 0
        clear_trace_context()


class TestMiddleware:
    """Tests for the request context middleware."""

    def test_request_id_header_returned(self):
        from fastapi.testclient import TestClient

        from api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        # Verify it's a UUID-like string
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID format: 8-4-4-4-12
