from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import psutil


@dataclass
class HealthStatus:
    name: str
    healthy: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SystemHealth:
    overall_healthy: bool
    checks: list[HealthStatus]
    uptime_seconds: float
    version: str
    python_version: str
    platform: str


class HealthChecker:
    """Comprehensive health check system for ScholarDevClaw."""

    def __init__(self, start_time: float | None = None):
        self.start_time = start_time or time.time()
        self._checks: dict[str, Callable[[], HealthStatus]] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        self.register_check("python_version", self._check_python_version)
        self.register_check("memory", self._check_memory)
        self.register_check("disk", self._check_disk)
        self.register_check("environment", self._check_environment)
        self.register_check("filesystem", self._check_filesystem)

    def register_check(self, name: str, check_func: Callable[[], HealthStatus]) -> None:
        self._checks[name] = check_func

    def run_check(self, name: str) -> HealthStatus:
        if name not in self._checks:
            return HealthStatus(
                name=name,
                healthy=False,
                message=f"Unknown check: {name}",
            )

        try:
            return self._checks[name]()
        except Exception as e:
            return HealthStatus(
                name=name,
                healthy=False,
                message=f"Check failed: {e}",
                details={"error": str(e), "traceback": traceback.format_exc()},
            )

    def run_all_checks(self) -> SystemHealth:
        results = [self.run_check(name) for name in self._checks]
        all_healthy = all(r.healthy for r in results)

        return SystemHealth(
            overall_healthy=all_healthy,
            checks=results,
            uptime_seconds=time.time() - self.start_time,
            version=self._get_version(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=sys.platform,
        )

    def run_quick_check(self) -> bool:
        critical_checks = ["memory", "disk", "filesystem"]
        return all(self.run_check(name).healthy for name in critical_checks)

    def _get_version(self) -> str:
        try:
            from scholardevclaw import __version__

            return __version__
        except Exception:
            return "unknown"

    def _check_python_version(self) -> HealthStatus:
        version = sys.version_info
        if version >= (3, 10):
            return HealthStatus(
                name="python_version",
                healthy=True,
                message=f"Python {version.major}.{version.minor}.{version.micro}",
            )
        return HealthStatus(
            name="python_version",
            healthy=False,
            message=f"Python {version.major}.{version.minor} not supported. Need 3.10+",
        )

    def _check_memory(self) -> HealthStatus:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        percent_used = memory.percent

        healthy = percent_used < 90
        message = f"{available_gb:.1f}GB available of {total_gb:.1f}GB ({percent_used}% used)"

        if percent_used > 80:
            message += " [WARNING: High memory usage]"

        return HealthStatus(
            name="memory",
            healthy=healthy,
            message=message,
            details={
                "available_gb": round(available_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_used": percent_used,
            },
        )

    def _check_disk(self, path: str = "/") -> HealthStatus:
        try:
            usage = psutil.disk_usage(path)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            percent_used = usage.percent

            healthy = percent_used < 95
            message = f"{free_gb:.1f}GB free of {total_gb:.1f}GB ({percent_used}% used)"

            if percent_used > 85:
                message += " [WARNING: Low disk space]"

            return HealthStatus(
                name="disk",
                healthy=healthy,
                message=message,
                details={
                    "path": path,
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "percent_used": percent_used,
                },
            )
        except Exception as e:
            return HealthStatus(
                name="disk",
                healthy=False,
                message=f"Cannot check disk: {e}",
            )

    def _check_environment(self) -> HealthStatus:
        missing: list[str] = []
        present: list[str] = []

        env_vars = ["GITHUB_TOKEN", "ANTHROPIC_API_KEY", "CONVEX_URL"]

        for var in env_vars:
            if os.environ.get(var):
                present.append(var)
            else:
                missing.append(var)

        healthy = True
        message = f"{len(present)} configured, {len(missing)} optional"

        return HealthStatus(
            name="environment",
            healthy=healthy,
            message=message,
            details={
                "present": present,
                "missing": missing,
            },
        )

    def _check_filesystem(self) -> HealthStatus:
        issues: list[str] = []

        home = Path.home()
        if not home.exists():
            issues.append(f"Home directory not accessible: {home}")

        cache_dir = home / ".scholardevclaw" / "cache"
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create cache directory: {e}")

        temp_dir = Path("/tmp")
        if not temp_dir.exists():
            issues.append("Temp directory not accessible")

        healthy = len(issues) == 0
        message = "Filesystem accessible" if healthy else "; ".join(issues)

        return HealthStatus(
            name="filesystem",
            healthy=healthy,
            message=message,
            details={"issues": issues},
        )


class LivenessProbe:
    """Simple liveness probe for health endpoints."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = time.time()

    def heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < self.timeout_seconds

    def check(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat).isoformat(),
            "seconds_since_heartbeat": time.time() - self.last_heartbeat,
        }


class ReadinessProbe:
    """Readiness probe to check if service can handle requests."""

    def __init__(self):
        self._ready = True
        self._reasons: list[str] = []

    def set_ready(self, ready: bool, reason: str | None = None) -> None:
        self._ready = ready
        if reason and not ready:
            self._reasons.append(reason)
        elif ready:
            self._reasons.clear()

    def is_ready(self) -> bool:
        return self._ready

    def check(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "reasons": self._reasons,
        }


health_checker = HealthChecker()
liveness_probe = LivenessProbe()
readiness_probe = ReadinessProbe()
