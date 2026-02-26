"""Tests for API usage analytics."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scholardevclaw.auth.analytics import (
    UsageRecord,
    DailyUsage,
    UsageAnalytics,
    UsageTracker,
    UsageDashboard,
    PROVIDER_PRICING,
)


class TestUsageRecord:
    def test_creation(self):
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider="anthropic",
            endpoint="/v1/messages",
            tokens_used=1000,
            cost_usd=0.01,
            latency_ms=500,
        )
        assert record.provider == "anthropic"
        assert record.tokens_used == 1000

    def test_to_dict_roundtrip(self):
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider="anthropic",
            endpoint="/v1/messages",
            tokens_used=1000,
        )
        data = record.to_dict()
        restored = UsageRecord.from_dict(data)
        assert restored.provider == record.provider


class TestDailyUsage:
    def test_creation(self):
        usage = DailyUsage(date="2024-01-01")
        assert usage.date == "2024-01-01"
        assert usage.total_requests == 0


class TestUsageTracker:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_record_usage(self, store_dir):
        tracker = UsageTracker(store_dir)
        record = tracker.record_usage(
            provider="anthropic",
            endpoint="/v1/messages",
            tokens_used=1000,
            input_tokens=500,
            output_tokens=500,
        )
        assert record.provider == "anthropic"
        assert record.cost_usd is not None

    def test_record_usage_no_tokens(self, store_dir):
        tracker = UsageTracker(store_dir)
        record = tracker.record_usage(provider="github", endpoint="/repos")
        assert record.tokens_used is None

    def test_get_usage(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1000)
        tracker.record_usage(provider="openai", endpoint="/v1/chat/completions", tokens_used=500)

        usage = tracker.get_usage(days=7)
        assert len(usage) == 2

    def test_get_usage_filtered(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1000)
        tracker.record_usage(provider="openai", endpoint="/v1/chat/completions", tokens_used=500)

        usage = tracker.get_usage(provider="anthropic")
        assert len(usage) == 1
        assert usage[0].provider == "anthropic"

    def test_get_analytics(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1000)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=2000)
        tracker.record_usage(provider="openai", endpoint="/v1/chat/completions", tokens_used=500)

        analytics = tracker.get_analytics(days=7)
        assert analytics.total_requests == 3
        assert analytics.total_tokens > 0

    def test_get_cost_alerts_exceeded(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1_000_000)

        alerts = tracker.get_cost_alerts(monthly_budget=10.0)
        assert len(alerts) > 0
        assert alerts[0]["type"] == "budget_exceeded"

    def test_cleanup_old_records(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1000)

        # Manually add old record
        old_record = UsageRecord(
            timestamp=(datetime.now() - timedelta(days=100)).isoformat(),
            provider="openai",
            endpoint="/v1/chat/completions",
            tokens_used=500,
        )
        with open(tracker.usage_file, "a") as f:
            f.write(json.dumps(old_record.to_dict()) + "\n")

        kept = tracker.cleanup_old_records(days=90)
        assert kept == 1


class TestUsageDashboard:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_get_summary(self, store_dir):
        tracker = UsageTracker(store_dir)
        tracker.record_usage(provider="anthropic", endpoint="/v1/messages", tokens_used=1000)

        dashboard = UsageDashboard(store_dir)
        summary = dashboard.get_summary()
        assert "last_7_days" in summary
        assert "last_30_days" in summary
        assert "providers" in summary


class TestProviderPricing:
    def test_anthropic_pricing(self):
        assert "anthropic" in PROVIDER_PRICING
        assert "input_per_million" in PROVIDER_PRICING["anthropic"]

    def test_openai_pricing(self):
        assert "openai" in PROVIDER_PRICING
        assert "gpt4o_input_per_million" in PROVIDER_PRICING["openai"]

    def test_github_pricing(self):
        assert "github" in PROVIDER_PRICING
        assert PROVIDER_PRICING["github"]["free"] == 0
