from __future__ import annotations

import logging
import traceback
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "scholardevclaw",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class ErrorContext:
    """Context manager for error handling with detailed traces."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = datetime.now()

    def __enter__(self):
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.debug(f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.2f}s\n"
                f"  Error: {exc_type.__name__}: {exc_val}\n"
                f"  Trace: {traceback.format_exc(limit=5)}"
            )

        return False


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_backoff: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_backoff**attempt)
        return min(delay, self.max_delay)


def retry_with_backoff(
    func: Callable,
    config: RetryConfig | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Retry function with exponential backoff."""
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = config.get_delay(attempt)
                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                import time

                time.sleep(delay)
            else:
                if logger:
                    logger.error(f"All {config.max_attempts} attempts failed: {e}")

    if last_exception:
        raise last_exception


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = datetime.now()
        self.result: Any = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"[TIMER] {self.operation}: {duration:.2f}s")
        else:
            self.logger.warning(f"[TIMER] {self.operation}: failed after {duration:.2f}s")

        return False

    def set_result(self, result: Any):
        self.result = result
        return result


def format_error_response(error: Exception, include_trace: bool = False) -> dict[str, Any]:
    """Format error for API response."""
    response = {
        "error": type(error).__name__,
        "message": str(error),
        "timestamp": datetime.now().isoformat(),
    }

    if include_trace:
        response["trace"] = traceback.format_exc()

    return response


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    logger: logging.Logger | None = None,
    **kwargs,
) -> Any:
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Execution failed: {e}")
        return default
