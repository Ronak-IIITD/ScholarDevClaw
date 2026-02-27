"""Per-key rate limiting and usage tracking.

Tracks API call counts per key, enforces configurable limits, and
persists usage stats to disk so they survive restarts.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a key."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10


@dataclass
class UsageRecord:
    """Single usage record for tracking."""

    timestamp: float
    provider: str | None = None
    endpoint: str | None = None


@dataclass
class KeyUsageStats:
    """Aggregated usage statistics for a key."""

    key_id: str
    total_requests: int = 0
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    first_used: str | None = None
    last_used: str | None = None
    is_rate_limited: bool = False
    limit_resets_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "total_requests": self.total_requests,
            "requests_last_minute": self.requests_last_minute,
            "requests_last_hour": self.requests_last_hour,
            "requests_last_day": self.requests_last_day,
            "first_used": self.first_used,
            "last_used": self.last_used,
            "is_rate_limited": self.is_rate_limited,
            "limit_resets_at": self.limit_resets_at,
        }


class RateLimiter:
    """Per-key rate limiter with sliding window tracking."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.store_dir, 0o700)
        self.usage_file = self.store_dir / "usage.json"
        self._configs: dict[str, RateLimitConfig] = {}
        # In-memory sliding window: key_id -> list of timestamps
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._total_counts: dict[str, int] = defaultdict(int)
        self._first_used: dict[str, float] = {}
        self._load_persisted()

    def _load_persisted(self) -> None:
        """Load persisted usage data."""
        if not self.usage_file.exists():
            return
        try:
            with open(self.usage_file) as f:
                data = json.load(f)
            for key_id, info in data.items():
                self._total_counts[key_id] = info.get("total_requests", 0)
                if info.get("first_used"):
                    self._first_used[key_id] = info["first_used"]
                # Sliding windows are not persisted (they reset on restart)
                # This is intentional — rate limits reset on restart
        except (json.JSONDecodeError, KeyError):
            pass

    def _persist(self) -> None:
        """Persist usage data to disk."""
        data: dict[str, Any] = {}
        all_keys = set(self._total_counts.keys()) | set(self._first_used.keys())
        for key_id in all_keys:
            data[key_id] = {
                "total_requests": self._total_counts.get(key_id, 0),
                "first_used": self._first_used.get(key_id),
            }
        with open(self.usage_file, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(self.usage_file, 0o600)

    def set_limit(self, key_id: str, config: RateLimitConfig) -> None:
        """Set rate limit config for a key."""
        self._configs[key_id] = config

    def get_limit(self, key_id: str) -> RateLimitConfig:
        """Get rate limit config for a key (returns default if not set)."""
        return self._configs.get(key_id, RateLimitConfig())

    def check_rate_limit(self, key_id: str) -> tuple[bool, str]:
        """Check if a request is allowed under rate limits.

        Returns (allowed, reason). If not allowed, reason explains why.
        """
        now = time.time()
        config = self.get_limit(key_id)
        window = self._windows[key_id]

        # Prune entries older than 24h
        cutoff_day = now - 86400
        self._windows[key_id] = [t for t in window if t > cutoff_day]
        window = self._windows[key_id]

        # Check per-minute
        minute_ago = now - 60
        minute_count = sum(1 for t in window if t > minute_ago)
        if minute_count >= config.requests_per_minute:
            return (
                False,
                f"Rate limit exceeded: {minute_count}/{config.requests_per_minute} requests/minute",
            )

        # Check per-hour
        hour_ago = now - 3600
        hour_count = sum(1 for t in window if t > hour_ago)
        if hour_count >= config.requests_per_hour:
            return (
                False,
                f"Rate limit exceeded: {hour_count}/{config.requests_per_hour} requests/hour",
            )

        # Check per-day
        day_count = len(window)
        if day_count >= config.requests_per_day:
            return False, f"Rate limit exceeded: {day_count}/{config.requests_per_day} requests/day"

        return True, "OK"

    def record_usage(self, key_id: str, provider: str | None = None) -> bool:
        """Record a usage event. Returns True if allowed, False if rate-limited."""
        allowed, _ = self.check_rate_limit(key_id)
        if not allowed:
            return False

        now = time.time()
        self._windows[key_id].append(now)
        self._total_counts[key_id] += 1

        if key_id not in self._first_used:
            self._first_used[key_id] = now

        self._persist()
        return True

    def get_usage_stats(self, key_id: str) -> KeyUsageStats:
        """Get usage statistics for a key."""
        now = time.time()
        window = self._windows.get(key_id, [])

        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        minute_count = sum(1 for t in window if t > minute_ago)
        hour_count = sum(1 for t in window if t > hour_ago)
        day_count = sum(1 for t in window if t > day_ago)

        config = self.get_limit(key_id)
        is_limited = (
            minute_count >= config.requests_per_minute
            or hour_count >= config.requests_per_hour
            or day_count >= config.requests_per_day
        )

        first_used_ts = self._first_used.get(key_id)
        last_used = None
        if window:
            last_used = datetime.fromtimestamp(window[-1]).isoformat()

        return KeyUsageStats(
            key_id=key_id,
            total_requests=self._total_counts.get(key_id, 0),
            requests_last_minute=minute_count,
            requests_last_hour=hour_count,
            requests_last_day=day_count,
            first_used=datetime.fromtimestamp(first_used_ts).isoformat() if first_used_ts else None,
            last_used=last_used,
            is_rate_limited=is_limited,
        )

    def get_all_usage(self) -> dict[str, KeyUsageStats]:
        """Get usage stats for all tracked keys."""
        all_keys = set(self._total_counts.keys()) | set(self._windows.keys())
        return {key_id: self.get_usage_stats(key_id) for key_id in all_keys}

    def reset_usage(self, key_id: str) -> None:
        """Reset usage tracking for a key."""
        self._windows.pop(key_id, None)
        self._total_counts.pop(key_id, None)
        self._first_used.pop(key_id, None)
        self._persist()

    def reset_all(self) -> None:
        """Reset all usage tracking."""
        self._windows.clear()
        self._total_counts.clear()
        self._first_used.clear()
        if self.usage_file.exists():
            self.usage_file.unlink()
