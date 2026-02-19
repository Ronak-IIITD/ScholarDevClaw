from __future__ import annotations

import pytest
import time

from scholardevclaw.utils.rate_limit import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    SlidingWindowCounter,
    RateLimitExceeded,
    rate_limit,
)


class TestTokenBucket:
    def test_allows_within_burst(self):
        bucket = TokenBucket(rate=1.0, burst_size=5)

        for _ in range(5):
            result = bucket.consume("test_key")
            assert result.allowed is True

    def test_blocks_over_burst(self):
        bucket = TokenBucket(rate=1.0, burst_size=3)

        for _ in range(3):
            bucket.consume("test_key")

        result = bucket.consume("test_key")
        assert result.allowed is False
        assert result.retry_after is not None

    def test_refills_over_time(self):
        bucket = TokenBucket(rate=10.0, burst_size=2)

        bucket.consume("test_key")
        bucket.consume("test_key")

        result = bucket.consume("test_key")
        assert result.allowed is False

        time.sleep(0.2)

        result = bucket.consume("test_key")
        assert result.allowed is True

    def test_different_keys_independent(self):
        bucket = TokenBucket(rate=1.0, burst_size=2)

        result1 = bucket.consume("key1")
        result2 = bucket.consume("key2")

        assert result1.allowed is True
        assert result2.allowed is True

    def test_reset_clears_state(self):
        bucket = TokenBucket(rate=1.0, burst_size=2)

        bucket.consume("test_key")
        bucket.consume("test_key")

        bucket.reset("test_key")

        result = bucket.consume("test_key")
        assert result.allowed is True

    def test_get_stats(self):
        bucket = TokenBucket(rate=1.0, burst_size=5)

        bucket.consume("test_key")

        stats = bucket.get_stats("test_key")
        assert stats is not None
        assert stats["request_count"] == 1


class TestSlidingWindowCounter:
    def test_allows_within_limit(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)

        for _ in range(5):
            result = counter.consume("test_key")
            assert result.allowed is True

    def test_blocks_over_limit(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=3)

        for _ in range(3):
            counter.consume("test_key")

        result = counter.consume("test_key")
        assert result.allowed is False

    def test_remaining_decreases(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)

        result = counter.consume("test_key")
        assert result.remaining == 4

        result = counter.consume("test_key")
        assert result.remaining == 3

    def test_different_keys_independent(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=2)

        counter.consume("key1")
        counter.consume("key1")

        result = counter.consume("key2")
        assert result.allowed is True

    def test_get_count(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)

        counter.consume("test_key")
        counter.consume("test_key")

        assert counter.get_count("test_key") == 2


class TestRateLimiter:
    def test_allows_normal_requests(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=5,
            )
        )

        for _ in range(5):
            result = limiter.check("test_key")
            assert result.allowed is True

    def test_blocks_on_burst(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=3,
            )
        )

        for _ in range(3):
            limiter.check("test_key")

        result = limiter.check("test_key")
        assert result.allowed is False

    def test_returns_remaining(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=5,
            )
        )

        result = limiter.check("test_key")
        assert result.remaining >= 0

    def test_get_stats(self):
        limiter = RateLimiter()

        limiter.check("test_key")

        stats = limiter.get_stats("test_key")
        assert "burst" in stats
        assert "minute_count" in stats
        assert "hour_count" in stats

    def test_reset(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=2,
                requests_per_hour=10,
                burst_size=2,
            )
        )

        limiter.check("test_key")
        limiter.check("test_key")

        result = limiter.check("test_key")
        assert result.allowed is False

        limiter.reset("test_key")

        result = limiter.check("test_key")
        assert result.allowed is True


class TestRateLimitDecorator:
    def test_allows_within_limit(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=5,
            )
        )

        @rate_limit(limiter, key_func=lambda: "test")
        def my_function():
            return "success"

        assert my_function() == "success"

    def test_blocks_over_limit(self):
        limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=2,
            )
        )

        @rate_limit(limiter, key_func=lambda: "test")
        def my_function():
            return "success"

        my_function()
        my_function()

        with pytest.raises(RateLimitExceeded) as exc_info:
            my_function()

        assert exc_info.value.retry_after > 0

    def test_rate_limit_exceeded_attributes(self):
        error = RateLimitExceeded(
            retry_after=30.0,
            limit=10,
            remaining=0,
        )

        assert error.retry_after == 30.0
        assert error.limit == 10
        assert error.remaining == 0
        assert "Rate limit exceeded" in str(error)
