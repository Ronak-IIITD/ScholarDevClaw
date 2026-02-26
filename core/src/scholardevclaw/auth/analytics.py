"""API usage analytics and cost tracking.

Tracks API usage, costs, rate limits, and provides analytics dashboards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class UsageRecord:
    """A single API usage record."""

    timestamp: str
    provider: str
    endpoint: str
    tokens_used: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        return cls(
            timestamp=data["timestamp"],
            provider=data["provider"],
            endpoint=data["endpoint"],
            tokens_used=data.get("tokens_used"),
            cost_usd=data.get("cost_usd"),
            latency_ms=data.get("latency_ms"),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


@dataclass
class DailyUsage:
    """Aggregated daily usage."""

    date: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    provider_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_errors": self.total_errors,
            "avg_latency_ms": self.avg_latency_ms,
            "provider_breakdown": self.provider_breakdown,
        }


@dataclass
class UsageAnalytics:
    """Usage analytics for a time period."""

    start_date: str
    end_date: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_errors: int = 0
    unique_keys: int = 0
    daily_breakdown: list[DailyUsage] = field(default_factory=list)
    provider_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_endpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_errors": self.total_errors,
            "unique_keys": self.unique_keys,
            "daily_breakdown": [d.to_dict() for d in self.daily_breakdown],
            "provider_breakdown": self.provider_breakdown,
            "top_endpoints": self.top_endpoints,
        }


# Provider pricing (approximate, can be updated)
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "anthropic": {
        "input_per_million": 15.0,  # Claude 4 Opus input
        "output_per_million": 75.0,
    },
    "openai": {
        "gpt4o_input_per_million": 5.0,
        "gpt4o_output_per_million": 15.0,
        "gpt4o_mini_input_per_million": 0.15,
        "gpt4o_mini_output_per_million": 0.6,
    },
    "google": {
        "gemini_pro_input_per_million": 1.25,
        "gemini_pro_output_per_million": 5.0,
    },
    "github": {
        "free": 0,
    },
}


class UsageTracker:
    """Track API usage with cost estimation."""

    USAGE_FILE = "usage.jsonl"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.store_dir / self.USAGE_FILE

    def record_usage(
        self,
        provider: str,
        endpoint: str,
        tokens_used: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> UsageRecord:
        """Record an API usage event."""
        cost_usd = self._estimate_cost(provider, tokens_used, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            endpoint=endpoint,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )

        with open(self.usage_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def _estimate_cost(
        self,
        provider: str,
        total_tokens: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """Estimate cost based on provider pricing."""
        if provider not in PROVIDER_PRICING:
            return None

        pricing = PROVIDER_PRICING[provider]

        if provider == "anthropic":
            input_tok = input_tokens or (total_tokens // 2 if total_tokens else 0)
            output_tok = output_tokens or (total_tokens // 2 if total_tokens else 0)
            cost = (input_tok / 1_000_000) * pricing.get("input_per_million", 0)
            cost += (output_tok / 1_000_000) * pricing.get("output_per_million", 0)
            return round(cost, 6)

        elif provider == "openai":
            input_tok = input_tokens or (total_tokens // 2 if total_tokens else 0)
            output_tok = output_tokens or (total_tokens // 2 if total_tokens else 0)
            # Use GPT-4o pricing as default
            cost = (input_tok / 1_000_000) * pricing.get("gpt4o_input_per_million", 0)
            cost += (output_tok / 1_000_000) * pricing.get("gpt4o_output_per_million", 0)
            return round(cost, 6)

        elif provider == "google":
            input_tok = input_tokens or (total_tokens // 2 if total_tokens else 0)
            output_tok = output_tokens or (total_tokens // 2 if total_tokens else 0)
            cost = (input_tok / 1_000_000) * pricing.get("gemini_pro_input_per_million", 0)
            cost += (output_tok / 1_000_000) * pricing.get("gemini_pro_output_per_million", 0)
            return round(cost, 6)

        return None

    def get_usage(
        self, days: int = 30, provider: str | None = None, key_id: str | None = None
    ) -> list[UsageRecord]:
        """Get usage records for a time period."""
        if not self.usage_file.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        records = []

        for line in self.usage_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                record = UsageRecord(
                    timestamp=data["timestamp"],
                    provider=data["provider"],
                    endpoint=data["endpoint"],
                    tokens_used=data.get("tokens_used"),
                    cost_usd=data.get("cost_usd"),
                    latency_ms=data.get("latency_ms"),
                    status=data.get("status", "success"),
                    error=data.get("error"),
                )

                if datetime.fromisoformat(record.timestamp) < cutoff:
                    continue
                if provider and record.provider != provider:
                    continue

                records.append(record)
            except (json.JSONDecodeError, KeyError):
                continue

        return records

    def get_analytics(self, days: int = 30) -> UsageAnalytics:
        """Get aggregated analytics for a time period."""
        records = self.get_usage(days)

        if not records:
            return UsageAnalytics(
                start_date=(datetime.now() - timedelta(days=days)).isoformat()[:10],
                end_date=datetime.now().isoformat()[:10],
            )

        daily_map: dict[str, DailyUsage] = {}
        provider_map: dict[str, dict[str, Any]] = {}
        endpoint_counts: dict[str, int] = {}
        unique_keys = set()

        for record in records:
            date = record.timestamp[:10]

            # Daily breakdown
            if date not in daily_map:
                daily_map[date] = DailyUsage(date=date)
            daily = daily_map[date]
            daily.total_requests += 1
            if record.tokens_used:
                daily.total_tokens += record.tokens_used
            if record.cost_usd:
                daily.total_cost_usd += record.cost_usd
            if record.status == "error":
                daily.total_errors += 1
            if record.latency_ms:
                daily.avg_latency_ms = (
                    daily.avg_latency_ms * (daily.total_requests - 1) + record.latency_ms
                ) / daily.total_requests

            # Provider breakdown
            if record.provider not in provider_map:
                provider_map[record.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "errors": 0,
                }
            pb = provider_map[record.provider]
            pb["requests"] += 1
            pb["tokens"] = pb.get("tokens", 0) + (record.tokens_used or 0)
            pb["cost"] = pb.get("cost", 0.0) + (record.cost_usd or 0)
            if record.status == "error":
                pb["errors"] = pb.get("errors", 0) + 1

            # Endpoint counts
            endpoint_counts[record.endpoint] = endpoint_counts.get(record.endpoint, 0) + 1

            # Provider breakdown per day
            if record.provider not in daily.provider_breakdown:
                daily.provider_breakdown[record.provider] = {"requests": 0, "cost": 0.0}
            daily.provider_breakdown[record.provider]["requests"] += 1
            daily.provider_breakdown[record.provider]["cost"] = daily.provider_breakdown[
                record.provider
            ].get("cost", 0.0) + (record.cost_usd or 0)

        # Top endpoints
        top_endpoints = sorted(
            [{"endpoint": k, "requests": v} for k, v in endpoint_counts.items()],
            key=lambda x: x["requests"],
            reverse=True,
        )[:10]

        return UsageAnalytics(
            start_date=(datetime.now() - timedelta(days=days)).isoformat()[:10],
            end_date=datetime.now().isoformat()[:10],
            total_requests=len(records),
            total_tokens=sum(r.tokens_used or 0 for r in records),
            total_cost_usd=sum(r.cost_usd or 0 for r in records),
            total_errors=sum(1 for r in records if r.status == "error"),
            unique_keys=len(unique_keys),
            daily_breakdown=sorted(daily_map.values(), key=lambda x: x.date),
            provider_breakdown=provider_map,
            top_endpoints=top_endpoints,
        )

    def get_cost_alerts(self, monthly_budget: float = 100.0) -> list[dict[str, Any]]:
        """Check if usage exceeds budget thresholds."""
        analytics = self.get_analytics(days=30)
        alerts = []

        if analytics.total_cost_usd > monthly_budget:
            alerts.append(
                {
                    "type": "budget_exceeded",
                    "message": f"Monthly cost ${analytics.total_cost_usd:.2f} exceeds budget ${monthly_budget}",
                    "severity": "critical",
                }
            )
        elif analytics.total_cost_usd > monthly_budget * 0.8:
            alerts.append(
                {
                    "type": "budget_warning",
                    "message": f"Monthly cost ${analytics.total_cost_usd:.2f} is at 80% of budget ${monthly_budget}",
                    "severity": "warning",
                }
            )

        # Check for error rate
        if analytics.total_requests > 0:
            error_rate = analytics.total_errors / analytics.total_requests
            if error_rate > 0.1:
                alerts.append(
                    {
                        "type": "high_error_rate",
                        "message": f"Error rate {error_rate * 100:.1f}% exceeds 10%",
                        "severity": "warning",
                    }
                )

        return alerts

    def cleanup_old_records(self, days: int = 90) -> int:
        """Remove usage records older than specified days."""
        if not self.usage_file.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=days)
        kept = []

        for line in self.usage_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if datetime.fromisoformat(data["timestamp"]) >= cutoff:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                continue

        self.usage_file.write_text("\n".join(kept) + "\n")
        return len(kept)


class UsageDashboard:
    """Generate usage dashboard data."""

    def __init__(self, store_dir: str | Path):
        self.tracker = UsageTracker(store_dir)

    def get_summary(self) -> dict[str, Any]:
        """Get summary for dashboard."""
        analytics_7d = self.tracker.get_analytics(days=7)
        analytics_30d = self.tracker.get_analytics(days=30)
        alerts = self.tracker.get_cost_alerts()

        return {
            "last_7_days": {
                "requests": analytics_7d.total_requests,
                "cost": analytics_7d.total_cost_usd,
                "tokens": analytics_7d.total_tokens,
                "errors": analytics_7d.total_errors,
            },
            "last_30_days": {
                "requests": analytics_30d.total_requests,
                "cost": analytics_30d.total_cost_usd,
                "tokens": analytics_30d.total_tokens,
                "errors": analytics_30d.total_errors,
            },
            "providers": analytics_30d.provider_breakdown,
            "alerts": alerts,
            "daily_trend": [d.to_dict() for d in analytics_30d.daily_breakdown[-14:]],
        }
