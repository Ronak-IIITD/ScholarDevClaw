from __future__ import annotations

import pytest
import json
import logging
import io

from scholardevclaw.utils.structured_logging import (
    get_trace_id,
    set_trace_id,
    get_request_id,
    set_request_id,
    generate_trace_id,
    generate_request_id,
    StructuredFormatter,
    StructuredLogger,
    setup_structured_logging,
    LogContext,
    get_logger,
)


class TestTraceContext:
    def test_set_and_get_trace_id(self):
        set_trace_id("test-trace-123")
        assert get_trace_id() == "test-trace-123"

        set_trace_id(None)
        assert get_trace_id() is None

    def test_set_and_get_request_id(self):
        set_request_id("req-456")
        assert get_request_id() == "req-456"

        set_request_id(None)
        assert get_request_id() is None

    def test_generate_trace_id(self):
        trace_id = generate_trace_id()
        assert len(trace_id) == 16
        assert trace_id.isalnum()

    def test_generate_request_id(self):
        request_id = generate_request_id()
        assert len(request_id) == 8
        assert request_id.isalnum()


class TestStructuredFormatter:
    def test_formats_as_json(self):
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_includes_trace_id(self):
        set_trace_id("trace-123")

        formatter = StructuredFormatter(include_trace=True)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data.get("trace_id") == "trace-123"

        set_trace_id(None)

    def test_excludes_trace_when_disabled(self):
        set_trace_id("trace-123")

        formatter = StructuredFormatter(include_trace=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "trace_id" not in data

        set_trace_id(None)


class TestStructuredLogger:
    def test_info_log(self, caplog):
        caplog.set_level(logging.INFO)

        logger = get_logger("test_logger")
        logger.info("Test info message")

        assert "Test info message" in caplog.text

    def test_error_log(self, caplog):
        caplog.set_level(logging.ERROR)

        logger = get_logger("test_logger")
        logger.error("Test error message")

        assert "Test error message" in caplog.text

    def test_with_extra_data(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("test_extra")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        structured = StructuredLogger("test_extra")
        structured._logger = logger

        structured.info("Test", extra={"key": "value"})

        stream.seek(0)
        output = stream.read()
        data = json.loads(output)

        assert data["message"] == "Test"


class TestLogContext:
    def test_context_logs_start_and_end(self, caplog):
        caplog.set_level(logging.INFO)

        logger = get_logger("context_test")

        with LogContext("test_operation", logger=logger):
            pass

        logs = caplog.text
        assert "Completed: test_operation" in logs

    def test_context_sets_trace_id(self):
        set_trace_id(None)

        with LogContext("test"):
            assert get_trace_id() is not None

        assert get_trace_id() is None

    def test_context_preserves_existing_trace_id(self):
        set_trace_id("existing-trace")

        with LogContext("test"):
            assert get_trace_id() == "existing-trace"

        assert get_trace_id() == "existing-trace"

        set_trace_id(None)

    def test_context_logs_error_on_exception(self, caplog):
        caplog.set_level(logging.ERROR)

        logger = get_logger("error_test")

        with pytest.raises(ValueError):
            with LogContext("failing_op", logger=logger):
                raise ValueError("Test error")

        assert "Failed: failing_op" in caplog.text


class TestSetupStructuredLogging:
    def test_creates_logger_with_handlers(self):
        logger = setup_structured_logging(
            name="setup_test",
            level=logging.INFO,
            json_output=False,
        )

        assert logger is not None
        assert isinstance(logger, StructuredLogger)

    def test_json_output_mode(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        root = logging.getLogger("json_test")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        logger = StructuredLogger("json_test")
        logger._logger = root
        logger.info("JSON test")

        stream.seek(0)
        output = stream.read()
        data = json.loads(output)

        assert data["message"] == "JSON test"
        assert data["level"] == "INFO"
