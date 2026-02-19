from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Any
from functools import wraps
from collections import defaultdict


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    window_seconds: int = 60


@dataclass
class RateLimitState:
    tokens: float
    last_update: float
    request_count: int = 0
    blocked_count: int = 0


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: float | None = None
    limit: int = 0


class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(
        self,
        rate: float,
        burst_size: int,
    ):
        self.rate = rate
        self.burst_size = burst_size
        self._buckets: dict[str, RateLimitState] = {}
        self._lock = threading.RLock()

    def _get_state(self, key: str) -> RateLimitState:
        if key not in self._buckets:
            self._buckets[key] = RateLimitState(
                tokens=float(self.burst_size),
                last_update=time.time(),
            )
        return self._buckets[key]

    def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        with self._lock:
            now = time.time()
            state = self._get_state(key)

            elapsed = now - state.last_update
            state.tokens = min(self.burst_size, state.tokens + elapsed * self.rate)
            state.last_update = now

            if state.tokens >= tokens:
                state.tokens -= tokens
                state.request_count += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=int(state.tokens),
                    reset_at=now + (self.burst_size - state.tokens) / self.rate,
                    limit=self.burst_size,
                )

            state.blocked_count += 1
            retry_after = (tokens - state.tokens) / self.rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=retry_after,
                limit=self.burst_size,
            )

    def get_stats(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key not in self._buckets:
                return None
            state = self._buckets[key]
            return {
                "tokens": state.tokens,
                "request_count": state.request_count,
                "blocked_count": state.blocked_count,
            }

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def clear_all(self) -> None:
        with self._lock:
            self._buckets.clear()


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""

    def __init__(
        self,
        window_seconds: int,
        max_requests: int,
    ):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def _cleanup(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._windows[key] = [ts for ts in self._windows[key] if ts > cutoff]

    def check(self, key: str) -> RateLimitResult:
        with self._lock:
            now = time.time()
            self._cleanup(key, now)

            current_count = len(self._windows[key])
            remaining = max(0, self.max_requests - current_count)
            reset_at = now + self.window_seconds

            if self._windows[key]:
                reset_at = self._windows[key][0] + self.window_seconds

            if current_count < self.max_requests:
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_at=reset_at,
                    limit=self.max_requests,
                )

            retry_after = reset_at - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=max(0, retry_after),
                limit=self.max_requests,
            )

    def record(self, key: str) -> None:
        with self._lock:
            now = time.time()
            self._cleanup(key, now)
            self._windows[key].append(now)

    def consume(self, key: str) -> RateLimitResult:
        result = self.check(key)
        if result.allowed:
            self.record(key)
            result.remaining = max(0, result.remaining - 1)
        return result

    def get_count(self, key: str) -> int:
        with self._lock:
            self._cleanup(key, time.time())
            return len(self._windows[key])


class RateLimiter:
    """Unified rate limiter with multiple algorithms."""

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()

        self._minute_limiter = SlidingWindowCounter(
            window_seconds=60,
            max_requests=self.config.requests_per_minute,
        )

        self._hour_limiter = SlidingWindowCounter(
            window_seconds=3600,
            max_requests=self.config.requests_per_hour,
        )

        self._burst_limiter = TokenBucket(
            rate=self.config.requests_per_minute / 60.0,
            burst_size=self.config.burst_size,
        )

    def check(self, key: str) -> RateLimitResult:
        burst_result = self._burst_limiter.consume(key)
        if not burst_result.allowed:
            return burst_result

        minute_result = self._minute_limiter.check(key)
        if not minute_result.allowed:
            return minute_result

        hour_result = self._hour_limiter.check(key)
        if not hour_result.allowed:
            return hour_result

        self._minute_limiter.record(key)
        self._hour_limiter.record(key)

        return RateLimitResult(
            allowed=True,
            remaining=min(burst_result.remaining, minute_result.remaining, hour_result.remaining),
            reset_at=max(burst_result.reset_at, minute_result.reset_at, hour_result.reset_at),
            limit=self.config.requests_per_minute,
        )

    def get_stats(self, key: str) -> dict[str, Any]:
        return {
            "burst": self._burst_limiter.get_stats(key),
            "minute_count": self._minute_limiter.get_count(key),
            "hour_count": self._hour_limiter.get_count(key),
        }

    def reset(self, key: str) -> None:
        self._burst_limiter.reset(key)
        if key in self._minute_limiter._windows:
            del self._minute_limiter._windows[key]
        if key in self._hour_limiter._windows:
            del self._hour_limiter._windows[key]


def rate_limit(
    limiter: RateLimiter,
    key_func: Callable[..., str] | None = None,
):
    """Decorator for rate limiting functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"default:{func.__name__}"

            result = limiter.check(key)
            if not result.allowed:
                raise RateLimitExceeded(
                    retry_after=result.retry_after or 60,
                    limit=result.limit,
                    remaining=result.remaining,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: float,
        limit: int,
        remaining: int,
    ):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.")


default_rate_limiter = RateLimiter()
