from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable
from pathlib import Path


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def get_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(trace_id: str | None) -> None:
    _trace_id.set(trace_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_user_id() -> str | None:
    return _user_id.get()


def set_user_id(user_id: str | None) -> None:
    _user_id.set(user_id)


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def generate_request_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class StructuredLogRecord:
    timestamp: str
    level: str
    logger: str
    message: str
    trace_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    duration_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def __init__(self, include_trace: bool = True):
        super().__init__()
        self.include_trace = include_trace

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_trace:
            trace_id = get_trace_id()
            request_id = get_request_id()
            user_id = get_user_id()

            if trace_id:
                log_data["trace_id"] = trace_id
            if request_id:
                log_data["request_id"] = request_id
            if user_id:
                log_data["user_id"] = user_id

        if hasattr(record, "extra_data"):
            log_data["extra"] = getattr(record, "extra_data", None)  # type: ignore

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = getattr(record, "duration_ms", None)  # type: ignore

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter with trace ID support."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        trace_id = get_trace_id()
        request_id = get_request_id()

        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        parts = [
            datetime.now().strftime("%H:%M:%S"),
            f"{color}{record.levelname:8}{reset}",
        ]

        if trace_id:
            parts.append(f"[{trace_id}]")
        if request_id:
            parts.append(f"({request_id})")

        parts.append(record.getMessage())

        if hasattr(record, "duration_ms"):
            duration = getattr(record, "duration_ms", 0)  # type: ignore
            parts.append(f"({duration:.1f}ms)")

        return " ".join(parts)


class StructuredLogger:
    """Logger with structured logging support."""

    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

    def _log(
        self,
        level: int,
        message: str,
        extra: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        log_extra: dict[str, Any] = {}

        if extra:
            log_extra["data"] = extra
        if duration_ms is not None:
            log_extra["duration_ms"] = duration_ms

        self._logger.log(level, message, extra=log_extra if log_extra else None)

    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, extra=kwargs if kwargs else None)

    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, extra=kwargs if kwargs else None)

    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, extra=kwargs if kwargs else None)

    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, extra=kwargs if kwargs else None)

    def critical(self, message: str, **kwargs) -> None:
        self._log(logging.CRITICAL, message, extra=kwargs if kwargs else None)

    def with_duration(self, message: str, duration_ms: float, **kwargs) -> None:
        self._log(logging.INFO, message, extra=kwargs if kwargs else None, duration_ms=duration_ms)

    def bind(self, **context) -> "BoundLogger":
        return BoundLogger(self, context)


class BoundLogger:
    """Logger with bound context."""

    def __init__(self, logger: StructuredLogger, context: dict[str, Any]):
        self._logger = logger
        self._context = context

    def info(self, message: str, **kwargs) -> None:
        self._logger.info(message, **{**self._context, **kwargs})

    def error(self, message: str, **kwargs) -> None:
        self._logger.error(message, **{**self._context, **kwargs})

    def warning(self, message: str, **kwargs) -> None:
        self._logger.warning(message, **{**self._context, **kwargs})

    def debug(self, message: str, **kwargs) -> None:
        self._logger.debug(message, **{**self._context, **kwargs})


def setup_structured_logging(
    name: str = "scholardevclaw",
    level: int = logging.INFO,
    json_output: bool = False,
    log_file: Path | None = None,
) -> StructuredLogger:
    """Setup structured logging."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    if json_output:
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return StructuredLogger(name, level)


class LogContext:
    """Context manager for request logging."""

    def __init__(
        self,
        operation: str,
        logger: StructuredLogger | None = None,
        **context,
    ):
        self.operation = operation
        self.logger = logger or get_logger()
        self.context = context
        self.start_time: float | None = None
        self._had_trace_id = get_trace_id() is not None

    def __enter__(self):
        if not self._had_trace_id:
            set_trace_id(generate_trace_id())

        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting: {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000 if self.start_time else 0

        if exc_type:
            self.logger.error(
                f"Failed: {self.operation}",
                error=str(exc_val),
                duration_ms=duration_ms,
                **self.context,
            )
        else:
            self.logger.with_duration(
                f"Completed: {self.operation}",
                duration_ms,
                **self.context,
            )

        if not self._had_trace_id:
            set_trace_id(None)

        return False


def get_logger(name: str = "scholardevclaw") -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(name)


def log_function(logger: StructuredLogger | None = None):
    """Decorator to log function entry/exit."""

    def decorator(func: Callable) -> Callable:
        _logger = logger or get_logger()

        def wrapper(*args, **kwargs):
            with LogContext(func.__name__, logger=_logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator
