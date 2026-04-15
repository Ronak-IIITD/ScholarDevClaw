from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import docker  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    docker = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

SANDBOX_IMAGE_ENV_VAR = "SCHOLARDEVCLAW_SANDBOX_IMAGE"
DEFAULT_SANDBOX_IMAGE = "scholardevclaw-sandbox:latest"


@dataclass(slots=True)
class ExecutionReport:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    peak_memory_mb: float
    tests_passed: int
    tests_failed: int
    tests_errors: int
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errors": self.tests_errors,
            "success": self.success,
        }


class SandboxRunner:
    def __init__(self, timeout_seconds: int = 300, memory_limit_mb: int = 4096) -> None:
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit_mb
        self.image_name = (
            os.getenv(SANDBOX_IMAGE_ENV_VAR, DEFAULT_SANDBOX_IMAGE).strip() or DEFAULT_SANDBOX_IMAGE
        )
        self.client: Any | None = None
        self._client_error: str | None = None

        if docker is None:
            self._client_error = (
                "docker SDK is not installed. Install with: pip install -e '.[execution]'"
            )
            return

        try:
            self.client = docker.from_env()
        except Exception as exc:  # pragma: no cover - depends on docker daemon/runtime
            self._client_error = f"Failed to initialize Docker client: {exc}"
            logger.warning(self._client_error)

    def run_tests(self, project_dir: Path) -> ExecutionReport:
        start = time.perf_counter()
        project_dir = project_dir.expanduser().resolve()

        if not project_dir.exists() or not project_dir.is_dir():
            return self._failed_report(
                stderr=f"Project directory not found: {project_dir}",
                duration_seconds=time.perf_counter() - start,
            )

        if self.client is None:
            return self._failed_report(
                stderr=self._client_error or "Docker client is unavailable",
                duration_seconds=time.perf_counter() - start,
            )

        container: Any | None = None
        try:
            container = self.client.containers.run(
                image=self.image_name,
                command=("pytest tests/ -v --json-report --json-report-file=/tmp/report.json"),
                volumes={str(project_dir): {"bind": "/workspace", "mode": "ro"}},
                working_dir="/workspace",
                mem_limit=f"{self.memory_limit}m",
                network_disabled=True,
                remove=False,
                detach=True,
            )

            if container is None:
                return self._failed_report(
                    stderr="Sandbox execution failed: container startup returned None",
                    duration_seconds=time.perf_counter() - start,
                )

            assert container is not None
            wait_result = container.wait(timeout=self.timeout)
            exit_code = (
                int(wait_result.get("StatusCode", 1)) if isinstance(wait_result, dict) else 1
            )
            stdout, stderr = self._collect_logs(container)

            report = self._extract_json_report(container)
            passed, failed, errors = self._parse_test_summary(report, stdout, stderr)
            peak_memory_mb = self._extract_peak_memory_mb(container)
            duration_seconds = time.perf_counter() - start

            return ExecutionReport(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration_seconds,
                peak_memory_mb=peak_memory_mb,
                tests_passed=passed,
                tests_failed=failed,
                tests_errors=errors,
                success=(exit_code == 0 and failed == 0 and errors == 0),
            )
        except Exception as exc:  # pragma: no cover - runtime-dependent docker failures
            logger.warning("Sandbox execution failed: %s", exc)
            return self._failed_report(
                stderr=f"Sandbox execution failed: {exc}",
                duration_seconds=time.perf_counter() - start,
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception as exc:  # pragma: no cover - cleanup best-effort
                    logger.debug("Failed to remove container: %s", exc)

    def _collect_logs(self, container: Any) -> tuple[str, str]:
        try:
            stdout_bytes = container.logs(stdout=True, stderr=False)
            stderr_bytes = container.logs(stdout=False, stderr=True)
            stdout = (
                stdout_bytes.decode("utf-8", errors="replace")
                if isinstance(stdout_bytes, (bytes, bytearray))
                else str(stdout_bytes)
            )
            stderr = (
                stderr_bytes.decode("utf-8", errors="replace")
                if isinstance(stderr_bytes, (bytes, bytearray))
                else str(stderr_bytes)
            )
            return stdout, stderr
        except Exception as exc:  # pragma: no cover - runtime-dependent docker failures
            logger.debug("Failed to read container logs: %s", exc)
            return "", f"Failed to collect logs: {exc}"

    def _extract_json_report(self, container: Any) -> dict[str, Any]:
        try:
            archive_stream, _ = container.get_archive("/tmp/report.json")
            buffer = io.BytesIO()
            for chunk in archive_stream:
                if isinstance(chunk, (bytes, bytearray)):
                    buffer.write(chunk)
            buffer.seek(0)

            with tarfile.open(fileobj=buffer, mode="r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.endswith("report.json"):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    payload = extracted.read().decode("utf-8", errors="replace")
                    loaded = json.loads(payload)
                    if isinstance(loaded, dict):
                        return loaded
                    return {}
            return {}
        except Exception as exc:  # pragma: no cover - runtime-dependent docker failures
            logger.debug("Failed to extract pytest JSON report: %s", exc)
            return {}

    def _parse_test_summary(
        self,
        report: dict[str, Any],
        stdout: str,
        stderr: str,
    ) -> tuple[int, int, int]:
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        if isinstance(summary, dict):
            passed = self._as_int(summary.get("passed", 0))
            failed = self._as_int(summary.get("failed", 0))
            errors = self._as_int(summary.get("errors", summary.get("error", 0)))
            if passed or failed or errors:
                return passed, failed, errors

        combined = f"{stdout}\n{stderr}"
        passed = self._extract_count(combined, r"(\d+)\s+passed")
        failed = self._extract_count(combined, r"(\d+)\s+failed")
        errors = self._extract_count(combined, r"(\d+)\s+errors?")
        return passed, failed, errors

    def _extract_peak_memory_mb(self, container: Any) -> float:
        try:
            stats = container.stats(stream=False)
        except Exception:  # pragma: no cover - runtime-dependent docker failures
            return 0.0

        if not isinstance(stats, dict):
            return 0.0
        memory_stats = stats.get("memory_stats", {})
        if not isinstance(memory_stats, dict):
            return 0.0

        max_usage = memory_stats.get("max_usage", memory_stats.get("usage"))
        if not isinstance(max_usage, (int, float)):
            return 0.0
        return float(max_usage) / (1024.0 * 1024.0)

    def _extract_count(self, text: str, pattern: str) -> int:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            return 0
        return self._as_int(match.group(1))

    def _as_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _failed_report(self, stderr: str, duration_seconds: float) -> ExecutionReport:
        return ExecutionReport(
            exit_code=1,
            stdout="",
            stderr=stderr,
            duration_seconds=duration_seconds,
            peak_memory_mb=0.0,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            success=False,
        )
