from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_failure_message: str | None
    opened_at: float | None
    total_calls: int
    total_failures: int
    total_successes: int


class CircuitBreaker(Generic[T]):
    """Circuit breaker for external service calls."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_failure_message: str | None = None
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        if self._state == CircuitState.OPEN:
            if self._opened_at and time.time() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def call(self, func: Callable[[], T]) -> T:
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open. "
                    f"Last failure: {self._last_failure_message or 'unknown'}"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is in half-open state with max calls reached"
                    )
                self._half_open_calls += 1

        self._total_calls += 1

        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._success_count += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.half_open_max_calls:
                    self._reset()

    def _on_failure(self, error: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.time()
            self._last_failure_message = str(error)

            if self._state == CircuitState.HALF_OPEN:
                self._trip()
            elif self._failure_count >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.time()

    def _reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None

    def force_open(self) -> None:
        with self._lock:
            self._trip()

    def force_close(self) -> None:
        with self._lock:
            self._reset()

    def get_stats(self) -> CircuitStats:
        with self._lock:
            return CircuitStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_failure_message=self._last_failure_message,
                opened_at=self._opened_at,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_successes=self._total_successes,
            )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        with self._lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, CircuitStats]:
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def get_health(self) -> dict[str, Any]:
        stats = self.get_all_stats()
        open_circuits = [name for name, s in stats.items() if s.state == CircuitState.OPEN]

        return {
            "healthy": len(open_circuits) == 0,
            "total_circuits": len(stats),
            "open_circuits": open_circuits,
            "stats": {
                name: {
                    "state": s.state.value,
                    "failure_count": s.failure_count,
                    "total_calls": s.total_calls,
                }
                for name, s in stats.items()
            },
        }


circuit_registry = CircuitBreakerRegistry()


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
):
    """Decorator to wrap function with circuit breaker."""

    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        breaker = circuit_registry.get_or_create(
            name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

        def wrapper(*args, **kwargs):
            return breaker.call(lambda: func(*args, **kwargs))

        return wrapper

    return decorator
