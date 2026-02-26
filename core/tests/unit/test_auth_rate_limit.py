"""Tests for rate limiting (rate_limit.py)."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from scholardevclaw.auth.rate_limit import (
    KeyUsageStats,
    RateLimitConfig,
    RateLimiter,
    UsageRecord,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def limiter(temp_dir):
    return RateLimiter(str(temp_dir))


class TestRateLimitConfig:
    def test_defaults(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_minute == 60
        assert cfg.requests_per_hour == 1000
        assert cfg.requests_per_day == 10000
        assert cfg.burst_size == 10

    def test_custom_values(self):
        cfg = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=500,
            burst_size=5,
        )
        assert cfg.requests_per_minute == 10
        assert cfg.requests_per_hour == 100
        assert cfg.requests_per_day == 500
        assert cfg.burst_size == 5


class TestUsageRecord:
    def test_creation(self):
        rec = UsageRecord(timestamp=time.time(), provider="openai", endpoint="/v1/chat")
        assert rec.provider == "openai"
        assert rec.endpoint == "/v1/chat"

    def test_defaults(self):
        rec = UsageRecord(timestamp=time.time())
        assert rec.provider is None
        assert rec.endpoint is None


class TestKeyUsageStats:
    def test_creation(self):
        stats = KeyUsageStats(key_id="key_abc")
        assert stats.key_id == "key_abc"
        assert stats.total_requests == 0
        assert stats.is_rate_limited is False

    def test_to_dict(self):
        stats = KeyUsageStats(
            key_id="key_abc",
            total_requests=42,
            requests_last_minute=5,
            requests_last_hour=20,
            requests_last_day=42,
            first_used="2025-01-01T00:00:00",
            last_used="2025-01-02T00:00:00",
            is_rate_limited=False,
        )
        d = stats.to_dict()
        assert d["key_id"] == "key_abc"
        assert d["total_requests"] == 42
        assert d["requests_last_minute"] == 5


class TestRateLimiter:
    def test_init_creates_dir(self, temp_dir):
        sub = temp_dir / "nested" / "limiter"
        limiter = RateLimiter(str(sub))
        assert sub.exists()

    def test_default_limit(self, limiter):
        cfg = limiter.get_limit("any_key")
        assert cfg.requests_per_minute == 60
        assert cfg.requests_per_hour == 1000

    def test_set_limit(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=5)
        limiter.set_limit("key_1", cfg)
        result = limiter.get_limit("key_1")
        assert result.requests_per_minute == 5

    def test_set_limit_does_not_affect_others(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=5)
        limiter.set_limit("key_1", cfg)
        default = limiter.get_limit("key_2")
        assert default.requests_per_minute == 60

    def test_check_rate_limit_allowed(self, limiter):
        allowed, reason = limiter.check_rate_limit("key_1")
        assert allowed is True
        assert reason == "OK"

    def test_record_usage_increments(self, limiter):
        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True
        stats = limiter.get_usage_stats("key_1")
        assert stats.total_requests == 2
        assert stats.requests_last_minute == 2

    def test_rate_limit_per_minute(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=3, requests_per_hour=1000, requests_per_day=10000)
        limiter.set_limit("key_1", cfg)

        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True

        # 4th should be rejected
        allowed, reason = limiter.check_rate_limit("key_1")
        assert allowed is False
        assert "requests/minute" in reason

    def test_rate_limit_per_hour(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=100, requests_per_hour=3, requests_per_day=10000)
        limiter.set_limit("key_1", cfg)

        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True

        allowed, reason = limiter.check_rate_limit("key_1")
        assert allowed is False
        assert "requests/hour" in reason

    def test_rate_limit_per_day(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=100, requests_per_hour=100, requests_per_day=3)
        limiter.set_limit("key_1", cfg)

        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True

        allowed, reason = limiter.check_rate_limit("key_1")
        assert allowed is False
        assert "requests/day" in reason

    def test_record_usage_returns_false_when_limited(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=2, requests_per_hour=1000, requests_per_day=10000)
        limiter.set_limit("key_1", cfg)

        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is True
        assert limiter.record_usage("key_1") is False

    def test_get_usage_stats_empty(self, limiter):
        stats = limiter.get_usage_stats("nonexistent")
        assert stats.total_requests == 0
        assert stats.requests_last_minute == 0
        assert stats.is_rate_limited is False
        assert stats.first_used is None
        assert stats.last_used is None

    def test_get_usage_stats_with_data(self, limiter):
        limiter.record_usage("key_1", provider="anthropic")
        limiter.record_usage("key_1", provider="anthropic")

        stats = limiter.get_usage_stats("key_1")
        assert stats.total_requests == 2
        assert stats.requests_last_minute == 2
        assert stats.first_used is not None
        assert stats.last_used is not None

    def test_get_all_usage(self, limiter):
        limiter.record_usage("key_1")
        limiter.record_usage("key_2")

        all_stats = limiter.get_all_usage()
        assert "key_1" in all_stats
        assert "key_2" in all_stats
        assert all_stats["key_1"].total_requests == 1
        assert all_stats["key_2"].total_requests == 1

    def test_reset_usage(self, limiter):
        limiter.record_usage("key_1")
        limiter.record_usage("key_1")
        limiter.reset_usage("key_1")

        stats = limiter.get_usage_stats("key_1")
        assert stats.total_requests == 0
        assert stats.requests_last_minute == 0

    def test_reset_all(self, limiter):
        limiter.record_usage("key_1")
        limiter.record_usage("key_2")
        limiter.reset_all()

        assert limiter.get_all_usage() == {}

    def test_persistence(self, temp_dir):
        """Total counts persist across instances, but sliding windows reset."""
        limiter1 = RateLimiter(str(temp_dir))
        limiter1.record_usage("key_1")
        limiter1.record_usage("key_1")
        limiter1.record_usage("key_1")

        # New instance should load persisted total counts
        limiter2 = RateLimiter(str(temp_dir))
        stats = limiter2.get_usage_stats("key_1")
        assert stats.total_requests == 3

        # But sliding window resets — minute/hour/day counters are 0
        assert stats.requests_last_minute == 0

    def test_persistence_file_created(self, limiter, temp_dir):
        limiter.record_usage("key_1")
        assert (temp_dir / "usage.json").exists()

    def test_persistence_file_format(self, limiter, temp_dir):
        limiter.record_usage("key_1")
        data = json.loads((temp_dir / "usage.json").read_text())
        assert "key_1" in data
        assert data["key_1"]["total_requests"] == 1

    def test_corrupted_usage_file(self, temp_dir):
        """Should handle corrupted usage file gracefully."""
        usage_file = temp_dir / "usage.json"
        usage_file.write_text("{{invalid json")
        limiter = RateLimiter(str(temp_dir))
        # Should not crash
        stats = limiter.get_usage_stats("key_1")
        assert stats.total_requests == 0

    def test_multiple_keys_independent(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=2, requests_per_hour=1000, requests_per_day=10000)
        limiter.set_limit("key_1", cfg)
        limiter.set_limit("key_2", cfg)

        limiter.record_usage("key_1")
        limiter.record_usage("key_1")
        # key_1 is now limited
        assert limiter.record_usage("key_1") is False
        # key_2 should still be allowed
        assert limiter.record_usage("key_2") is True

    def test_is_rate_limited_flag(self, limiter):
        cfg = RateLimitConfig(requests_per_minute=1, requests_per_hour=1000, requests_per_day=10000)
        limiter.set_limit("key_1", cfg)
        limiter.record_usage("key_1")

        stats = limiter.get_usage_stats("key_1")
        assert stats.is_rate_limited is True

    def test_reset_clears_persistence(self, temp_dir):
        limiter = RateLimiter(str(temp_dir))
        limiter.record_usage("key_1")
        limiter.reset_all()
        assert not (temp_dir / "usage.json").exists()

    def test_first_used_tracked(self, limiter):
        limiter.record_usage("key_1")
        stats = limiter.get_usage_stats("key_1")
        assert stats.first_used is not None

    def test_last_used_tracked(self, limiter):
        limiter.record_usage("key_1")
        stats = limiter.get_usage_stats("key_1")
        assert stats.last_used is not None


class TestRateLimiterWithStore:
    """Integration tests: rate limiting + AuthStore."""

    def test_rate_limit_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir))
        key = store.add_api_key("sk-test-123456789", "test", AuthProvider.CUSTOM)

        cfg = RateLimitConfig(requests_per_minute=2, requests_per_hour=1000, requests_per_day=10000)
        store.set_rate_limit(key.id, cfg)

        k1, msg1 = store.get_api_key_with_rate_check()
        assert k1 is not None
        assert msg1 == "OK"

        k2, msg2 = store.get_api_key_with_rate_check()
        assert k2 is not None

        k3, msg3 = store.get_api_key_with_rate_check()
        assert k3 is None
        assert "Rate limit exceeded" in msg3

    def test_get_key_usage_from_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir))
        key = store.add_api_key("sk-test-123456789", "test", AuthProvider.CUSTOM)

        store.get_api_key_with_rate_check()
        usage = store.get_key_usage(key.id)
        assert usage["total_requests"] >= 1

    def test_get_all_key_usage(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-1", "key1", AuthProvider.CUSTOM)
        store.add_api_key("sk-test-2", "key2", AuthProvider.OPENAI)

        store.get_api_key_with_rate_check()
        usage = store.get_key_usage()
        # At least one key should have usage data
        assert isinstance(usage, dict)

    def test_rate_limit_disabled(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir), enable_rate_limit=False)
        assert store.set_rate_limit("any_key", RateLimitConfig()) is False
        assert store.get_key_usage() == {}
