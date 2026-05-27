"""Tests for utils/circuit_breaker.py"""

import time

import pytest

from scholardevclaw.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    circuit_registry,
    with_circuit_breaker,
)


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_allows_calls_when_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        def success():
            return "ok"

        assert cb.call(success) == "ok"

    def test_trips_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        def failing():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing)

        assert cb.state == CircuitState.OPEN

    def test_blocks_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=2)

        def failing():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing)

        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "ok")

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.01)

        def failing():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing)

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovers_in_half_open(self):
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.01, half_open_max_calls=2
        )

        def failing():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing)

        time.sleep(0.02)

        def success():
            return "recovered"

        cb.call(success)
        cb.call(success)
        assert cb.state == CircuitState.CLOSED

    def test_opens_again_on_half_open_failure(self):
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.01, half_open_max_calls=3
        )

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("err")))

        time.sleep(0.02)

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("half_open_fail")))

        assert cb.state == CircuitState.OPEN

    def test_force_open(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.force_open()
        assert cb.state == CircuitState.OPEN

    def test_force_close(self):
        cb = CircuitBreaker("test", failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("err")))

        cb.force_close()
        assert cb.state == CircuitState.CLOSED
        assert cb.call(lambda: "ok") == "ok"

    def test_get_stats(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.call(lambda: "ok")

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("err")))

        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED
        assert stats.total_calls == 2
        assert stats.total_successes == 1
        assert stats.total_failures == 1
        assert stats.last_failure_message == "err"

    def test_half_open_max_calls_limit(self):
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1
        )

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("err")))

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        result = cb.call(lambda: "ok")
        assert result == "ok"

        assert cb.state == CircuitState.CLOSED

    def test_half_open_rejects_excess_on_failure(self):
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=2
        )

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("err")))

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("half_err")))

        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("half_err2")))

        assert cb.state == CircuitState.OPEN

    def test_state_property_transitions(self):
        cb = CircuitBreaker("test", failure_threshold=100)
        assert cb.state == CircuitState.CLOSED
        cb.force_open()
        assert cb.state == CircuitState.OPEN

    def test_circuit_open_error_message(self):
        err = CircuitOpenError("test message")
        assert "test message" in str(err)


class TestCircuitBreakerRegistry:
    def test_get_or_create(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("svc1", failure_threshold=5)
        cb2 = reg.get_or_create("svc1", failure_threshold=10)
        assert cb1 is cb2
        assert cb1.failure_threshold == 5

    def test_get_missing(self):
        reg = CircuitBreakerRegistry()
        assert reg.get("nonexistent") is None

    def test_get_health_healthy(self):
        reg = CircuitBreakerRegistry()
        reg.get_or_create("svc1")
        health = reg.get_health()
        assert health["healthy"] is True

    def test_get_health_unhealthy(self):
        reg = CircuitBreakerRegistry()

        def failing():
            raise ValueError("err")

        cb = reg.get_or_create("bad_svc", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing)

        health = reg.get_health()
        assert health["healthy"] is False
        assert "bad_svc" in health["open_circuits"]


class TestWithCircuitBreaker:
    def test_decorator_caches_result(self):
        call_count = 0

        @with_circuit_breaker("decorated_func", failure_threshold=5)
        def my_func():
            nonlocal call_count
            call_count += 1
            return "result"

        assert my_func() == "result"
        assert my_func() == "result"
        assert call_count == 2

    def test_decorator_opens_on_failures(self):
        @with_circuit_breaker("failing_func", failure_threshold=2, recovery_timeout=0.1)
        def fails():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                fails()

        with pytest.raises(CircuitOpenError):
            fails()

    def test_registry_module_instance(self):
        assert isinstance(circuit_registry, CircuitBreakerRegistry)


class TestCircuitState:
    def test_enum_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"
