"""
Unified retry utility with exponential backoff and jitter.

Modeled after OpenClaw's retry contract: bounded attempts, policy hooks,
and protocol-aware retry-after extraction for rate-limited APIs.

Usage::

    from scholardevclaw.utils.retry import retry

    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def flaky_call(): ...

    # Or as a context wrapper:
    result = retry(lambda: httpx.get(url), max_attempts=3)
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Configurable retry policy with hooks.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (including the first).
    base_delay : float
        Initial backoff in seconds.
    max_delay : float
        Maximum backoff cap in seconds.
    exponential_base : float
        Multiplier for exponential growth.
    jitter : bool
        Add random jitter to prevent thundering herd.
    should_retry : callable
        ``(exc: Exception, attempt: int) -> bool`` — return ``False`` to abort early.
    retry_after : callable
        ``(exc: Exception) -> float | None`` — extract ``Retry-After`` or custom delay.
    on_retry : callable
        ``(exc: Exception, attempt: int, delay: float) -> None`` — hook for logging.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        should_retry: Callable[[Exception, int], bool] | None = None,
        retry_after: Callable[[Exception], float | None] | None = None,
        on_retry: Callable[[Exception, int, float], None] | None = None,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.base_delay = max(0.0, base_delay)
        self.max_delay = max(self.base_delay, max_delay)
        self.exponential_base = max(1.0, exponential_base)
        self.jitter = jitter
        self._should_retry = should_retry or _default_should_retry
        self._retry_after = retry_after
        self._on_retry = on_retry

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``func`` with retry semantics."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_attempts:
                    break
                if not self._should_retry(exc, attempt):
                    raise
                delay = self._compute_delay(exc, attempt)
                if self._on_retry:
                    self._on_retry(exc, attempt, delay)
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _compute_delay(self, exc: Exception, attempt: int) -> float:
        # Check for explicit retry-after hint
        if self._retry_after:
            explicit = self._retry_after(exc)
            if explicit is not None:
                return min(explicit, self.max_delay)

        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return min(delay, self.max_delay)


# ---------------------------------------------------------------------------
# Default predicates
# ---------------------------------------------------------------------------

# Common transient HTTP status codes that warrant retry
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


def _default_should_retry(exc: Exception, attempt: int) -> bool:
    """Retry on network errors and transient HTTP status codes."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP
    if isinstance(exc, (httpx.RequestError, OSError, ConnectionError, TimeoutError)):
        return True
    return False


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header from HTTPStatusError."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# Convenience decorator / wrapper
# ---------------------------------------------------------------------------


def retry(
    func: Callable[..., T] | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    should_retry: Callable[[Exception, int], bool] | None = None,
    retry_after: Callable[[Exception], float | None] | None = None,
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> T | Callable[..., T]:
    """Retry wrapper — usable as decorator or direct call.

    Examples::

        @retry(max_attempts=3)
        def fetch(): ...

        result = retry(lambda: httpx.get(url), max_attempts=3)
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        should_retry=should_retry,
        retry_after=retry_after,
        on_retry=on_retry,
    )

    if func is not None:
        return policy.execute(func)

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return policy.execute(fn, *args, **kwargs)

        return wrapper

    return decorator  # type: ignore[return-value]
