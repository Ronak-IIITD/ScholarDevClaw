import time
import threading

import pytest

from scholardevclaw.utils.tracing import (
    Span,
    TraceContext,
    start_trace,
    get_current_trace,
    get_trace_id,
    get_request_id,
    end_trace,
    span,
    Tracer,
    extract_trace_headers,
    inject_trace_headers,
    trace_context,
)


class TestSpan:
    def test_span_creation(self):
        span = Span(
            span_id="span123",
            name="test_span",
            start_time=time.time(),
        )
        assert span.span_id == "span123"
        assert span.name == "test_span"
        assert span.status == "ok"

    def test_span_duration(self):
        start = time.time()
        span = Span(span_id="span1", name="test", start_time=start)
        assert span.duration_ms() is None

        span.end_time = start + 0.1
        assert span.duration_ms() == pytest.approx(100, rel=0.1)

    def test_span_add_event(self):
        span = Span(span_id="span1", name="test", start_time=time.time())
        span.add_event("checkpoint", {"key": "value"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["key"] == "value"

    def test_span_set_attribute(self):
        span = Span(span_id="span1", name="test", start_time=time.time())
        span.set_attribute("user_id", "123")
        span.set_attribute("count", 42)

        assert span.attributes["user_id"] == "123"
        assert span.attributes["count"] == 42

    def test_span_set_status(self):
        span = Span(span_id="span1", name="test", start_time=time.time())
        span.set_status("error")
        assert span.status == "error"

    def test_span_to_dict(self):
        span = Span(
            span_id="span1",
            name="test",
            start_time=100.0,
            end_time=100.5,
            parent_id="parent1",
        )
        span.set_attribute("key", "value")

        result = span.to_dict()
        assert result["span_id"] == "span1"
        assert result["name"] == "test"
        assert result["duration_ms"] == 500.0
        assert result["parent_id"] == "parent1"
        assert result["attributes"]["key"] == "value"


class TestTraceContext:
    def test_context_creation(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        assert ctx.trace_id == "trace1"
        assert ctx.request_id == "req1"

    def test_start_span(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        span = ctx.start_span("operation")

        assert span.name == "operation"
        assert len(ctx.spans) == 1
        assert ctx.current_span == span

    def test_start_span_with_parent(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        parent = ctx.start_span("parent")
        child = ctx.start_span("child")

        assert child.parent_id == parent.span_id

    def test_end_span(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        span = ctx.start_span("operation")
        assert span.end_time is None

        ctx.end_span(span)
        assert span.end_time is not None

    def test_get_span(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        span = ctx.start_span("operation")

        found = ctx.get_span(span.span_id)
        assert found == span

        not_found = ctx.get_span("nonexistent")
        assert not_found is None

    def test_baggage(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        ctx.set_baggage("user", "alice")
        ctx.set_baggage("tenant", "org1")

        assert ctx.get_baggage("user") == "alice"
        assert ctx.get_baggage("tenant") == "org1"
        assert ctx.get_baggage("missing") is None

    def test_to_dict(self):
        ctx = TraceContext(
            trace_id="trace1",
            request_id="req1",
            user_id="user1",
            session_id="sess1",
        )
        ctx.start_span("op1")

        result = ctx.to_dict()
        assert result["trace_id"] == "trace1"
        assert result["request_id"] == "req1"
        assert result["user_id"] == "user1"
        assert len(result["spans"]) == 1

    def test_total_duration(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        span1 = ctx.start_span("op1")
        time.sleep(0.01)
        ctx.end_span(span1)

        duration = ctx.total_duration_ms()
        assert duration is not None
        assert duration >= 10

    def test_thread_safety(self):
        ctx = TraceContext(trace_id="trace1", request_id="req1")
        results = []

        def worker():
            span = ctx.start_span(f"op-{threading.current_thread().name}")
            time.sleep(0.001)
            ctx.end_span(span)
            results.append(span.span_id)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 10


class TestGlobalFunctions:
    def test_start_trace(self):
        ctx = start_trace()
        assert ctx is not None
        assert ctx.trace_id is not None
        assert ctx.request_id is not None
        end_trace()

    def test_start_trace_with_ids(self):
        ctx = start_trace(
            trace_id="custom-trace",
            request_id="custom-req",
            user_id="user1",
        )
        assert ctx.trace_id == "custom-trace"
        assert ctx.request_id == "custom-req"
        assert ctx.user_id == "user1"
        end_trace()

    def test_get_current_trace(self):
        assert get_current_trace() is None

        ctx = start_trace()
        assert get_current_trace() == ctx
        end_trace()

        assert get_current_trace() is None

    def test_get_trace_id(self):
        assert get_trace_id() is None

        ctx = start_trace(trace_id="my-trace")
        assert get_trace_id() == "my-trace"
        end_trace()

    def test_get_request_id(self):
        assert get_request_id() is None

        ctx = start_trace(request_id="my-req")
        assert get_request_id() == "my-req"
        end_trace()

    def test_end_trace_closes_spans(self):
        ctx = start_trace()
        span = ctx.start_span("op")
        assert span.end_time is None

        result = end_trace()
        assert span.end_time is not None
        assert result is not None


class TestSpanDecorator:
    def test_span_decorator(self):
        start_trace()

        @span("my_operation")
        def my_function():
            return 42

        result = my_function()
        assert result == 42

        ctx = get_current_trace()
        assert len(ctx.spans) == 1
        assert ctx.spans[0].name == "my_operation"
        assert ctx.spans[0].status == "ok"

        end_trace()

    def test_span_decorator_with_attributes(self):
        start_trace()

        @span("op", attributes={"key": "value"})
        def my_function():
            return "done"

        my_function()

        ctx = get_current_trace()
        assert ctx.spans[0].attributes["key"] == "value"

        end_trace()

    def test_span_decorator_on_error(self):
        start_trace()

        @span("failing_op")
        def failing_function():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            failing_function()

        ctx = get_current_trace()
        assert ctx.spans[0].status == "error"
        assert "error.type" in ctx.spans[0].attributes

        end_trace()

    def test_span_decorator_without_trace(self):
        @span("op")
        def my_function():
            return 42

        result = my_function()
        assert result == 42


class TestTracer:
    def test_tracer_start_trace(self):
        tracer = Tracer("test-service")
        ctx = tracer.start_trace(trace_id="trace1")

        assert ctx.trace_id == "trace1"
        assert ctx.get_baggage("service") == "test-service"

        tracer.end_trace()

    def test_tracer_export_callback(self):
        exported = []

        def callback(trace_data):
            exported.append(trace_data)

        tracer = Tracer()
        tracer.set_export_callback(callback)

        ctx = tracer.start_trace(trace_id="trace1")
        tracer.end_trace()

        assert len(exported) == 1
        assert exported[0]["trace_id"] == "trace1"

    def test_tracer_get_exported(self):
        tracer = Tracer()

        ctx = tracer.start_trace(trace_id="trace1")
        tracer.end_trace()

        ctx = tracer.start_trace(trace_id="trace2")
        tracer.end_trace()

        exported = tracer.get_exported_traces()
        assert len(exported) == 2

    def test_tracer_clear(self):
        tracer = Tracer()

        ctx = tracer.start_trace()
        tracer.end_trace()

        assert len(tracer.get_exported_traces()) == 1
        tracer.clear_exported()
        assert len(tracer.get_exported_traces()) == 0


class TestHeaderExtraction:
    def test_extract_trace_headers(self):
        headers = {
            "X-Trace-ID": "trace123",
            "X-Request-ID": "req456",
            "X-User-ID": "user1",
        }

        result = extract_trace_headers(headers)
        assert result["trace_id"] == "trace123"
        assert result["request_id"] == "req456"
        assert result["user_id"] == "user1"

    def test_extract_lowercase_headers(self):
        headers = {
            "x-trace-id": "trace123",
            "x-request-id": "req456",
        }

        result = extract_trace_headers(headers)
        assert result["trace_id"] == "trace123"
        assert result["request_id"] == "req456"

    def test_extract_missing_headers(self):
        headers = {}
        result = extract_trace_headers(headers)
        assert result["trace_id"] is None
        assert result["request_id"] is None


class TestHeaderInjection:
    def test_inject_trace_headers(self):
        start_trace(
            trace_id="trace123",
            request_id="req456",
            user_id="user1",
        )

        headers = inject_trace_headers()
        assert headers["X-Trace-ID"] == "trace123"
        assert headers["X-Request-ID"] == "req456"
        assert headers["X-User-ID"] == "user1"

        end_trace()

    def test_inject_without_trace(self):
        headers = inject_trace_headers()
        assert headers == {}

    def test_inject_optional_fields(self):
        start_trace(
            trace_id="trace123",
            request_id="req456",
            session_id="sess1",
            parent_trace_id="parent1",
        )

        headers = inject_trace_headers()
        assert headers["X-Session-ID"] == "sess1"
        assert headers["X-Parent-Trace-ID"] == "parent1"

        end_trace()
