from __future__ import annotations

import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from contextvars import ContextVar
from functools import wraps


trace_context: ContextVar["TraceContext | None"] = ContextVar("trace_context", default=None)


@dataclass
class Span:
    span_id: str
    name: str
    start_time: float
    end_time: float | None = None
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "parent_id": self.parent_id,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


@dataclass
class TraceContext:
    trace_id: str
    request_id: str
    user_id: str | None = None
    session_id: str | None = None
    parent_trace_id: str | None = None
    spans: list[Span] = field(default_factory=list)
    current_span: Span | None = None
    baggage: dict[str, str] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def start_span(self, name: str, parent_id: str | None = None) -> Span:
        span_id = str(uuid.uuid4())[:16]
        actual_parent = parent_id or (self.current_span.span_id if self.current_span else None)

        span = Span(
            span_id=span_id,
            name=name,
            start_time=time.time(),
            parent_id=actual_parent,
        )

        with self._lock:
            self.spans.append(span)
            self.current_span = span

        return span

    def end_span(self, span: Span) -> None:
        span.end_time = time.time()
        with self._lock:
            if self.current_span == span:
                parent_id = span.parent_id
                self.current_span = next(
                    (s for s in reversed(self.spans) if s.span_id == parent_id),
                    None,
                )

    def get_span(self, span_id: str) -> Span | None:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def set_baggage(self, key: str, value: str) -> None:
        self.baggage[key] = value

    def get_baggage(self, key: str) -> str | None:
        return self.baggage.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "parent_trace_id": self.parent_trace_id,
            "baggage": self.baggage,
            "spans": [s.to_dict() for s in self.spans],
        }

    def total_duration_ms(self) -> float | None:
        if not self.spans:
            return None
        first = min(s.start_time for s in self.spans)
        last = max(s.end_time or time.time() for s in self.spans)
        return (last - first) * 1000


def start_trace(
    trace_id: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    parent_trace_id: str | None = None,
) -> TraceContext:
    ctx = TraceContext(
        trace_id=trace_id or str(uuid.uuid4()),
        request_id=request_id or str(uuid.uuid4())[:8],
        user_id=user_id,
        session_id=session_id,
        parent_trace_id=parent_trace_id,
    )
    trace_context.set(ctx)
    return ctx


def get_current_trace() -> TraceContext | None:
    return trace_context.get()


def get_trace_id() -> str | None:
    ctx = get_current_trace()
    return ctx.trace_id if ctx else None


def get_request_id() -> str | None:
    ctx = get_current_trace()
    return ctx.request_id if ctx else None


def end_trace() -> dict[str, Any] | None:
    ctx = get_current_trace()
    if ctx:
        for span in ctx.spans:
            if span.end_time is None:
                span.end_time = time.time()
        trace_context.set(None)
        return ctx.to_dict()
    return None


def span(name: str, attributes: dict[str, Any] | None = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = get_current_trace()
            if ctx is None:
                return func(*args, **kwargs)

            current_span = ctx.start_span(name)
            if attributes:
                for k, v in attributes.items():
                    current_span.set_attribute(k, v)

            try:
                result = func(*args, **kwargs)
                current_span.set_status("ok")
                return result
            except Exception as e:
                current_span.set_status("error")
                current_span.set_attribute("error.type", type(e).__name__)
                current_span.set_attribute("error.message", str(e))
                raise
            finally:
                ctx.end_span(current_span)

        return wrapper

    return decorator


class Tracer:
    def __init__(self, service_name: str = "scholardevclaw"):
        self.service_name = service_name
        self._spans_exported: list[dict[str, Any]] = []
        self._export_callback: Callable[[dict[str, Any]], None] | None = None

    def set_export_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._export_callback = callback

    def start_trace(self, **kwargs) -> TraceContext:
        ctx = start_trace(**kwargs)
        ctx.set_baggage("service", self.service_name)
        return ctx

    def end_trace(self) -> dict[str, Any] | None:
        result = end_trace()
        if result and self._export_callback:
            self._export_callback(result)
        if result:
            self._spans_exported.append(result)
        return result

    def get_exported_traces(self) -> list[dict[str, Any]]:
        return list(self._spans_exported)

    def clear_exported(self) -> None:
        self._spans_exported.clear()


def extract_trace_headers(headers: dict[str, str]) -> dict[str, str | None]:
    return {
        "trace_id": headers.get("X-Trace-ID") or headers.get("x-trace-id"),
        "request_id": headers.get("X-Request-ID") or headers.get("x-request-id"),
        "parent_trace_id": headers.get("X-Parent-Trace-ID") or headers.get("x-parent-trace-id"),
        "user_id": headers.get("X-User-ID") or headers.get("x-user-id"),
        "session_id": headers.get("X-Session-ID") or headers.get("x-session-id"),
    }


def inject_trace_headers() -> dict[str, str]:
    ctx = get_current_trace()
    if ctx is None:
        return {}

    headers = {
        "X-Trace-ID": ctx.trace_id,
        "X-Request-ID": ctx.request_id,
    }
    if ctx.user_id:
        headers["X-User-ID"] = ctx.user_id
    if ctx.session_id:
        headers["X-Session-ID"] = ctx.session_id
    if ctx.parent_trace_id:
        headers["X-Parent-Trace-ID"] = ctx.parent_trace_id

    return headers


class TraceMiddleware:
    def __init__(self, app, tracer: Tracer | None = None):
        self.app = app
        self.tracer = tracer or Tracer()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        headers = {k.decode(): v.decode() for k, v in headers.items()}

        extracted = extract_trace_headers(headers)

        ctx = self.tracer.start_trace(
            trace_id=extracted.get("trace_id"),
            request_id=extracted.get("request_id"),
            user_id=extracted.get("user_id"),
            session_id=extracted.get("session_id"),
            parent_trace_id=extracted.get("parent_trace_id"),
        )

        span = ctx.start_span(
            name=f"{scope['method']} {scope['path']}",
        )
        span.set_attribute("http.method", scope["method"])
        span.set_attribute("http.path", scope["path"])
        span.set_attribute("http.scheme", scope.get("scheme", "http"))

        async def send_with_trace(message):
            if message["type"] == "http.response.start":
                status = message.get("status", 200)
                span.set_attribute("http.status_code", status)
                span.set_status("ok" if 200 <= status < 400 else "error")
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        except Exception as e:
            span.set_status("error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            ctx.end_span(span)
            self.tracer.end_trace()
