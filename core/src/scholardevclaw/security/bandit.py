from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .types import SecurityFinding, SecurityScanResult, SecurityTool, Severity

# Allowlists for validating config values before passing to subprocess
_VALID_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_VALID_SEVERITY_LEVELS = {"low", "medium", "high"}
_VALID_EXCLUDE_PATTERN = re.compile(r"^[a-zA-Z0-9_.*/?-]+$")


class BanditScanner:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.exclude_patterns = self.config.get("exclude", ["tests/", "test_", ".venv/", "venv/"])
        # Validate confidence and severity against allowlist
        confidence = self.config.get("confidence", "low")
        self.confidence_level = confidence if confidence in _VALID_CONFIDENCE_LEVELS else "low"
        severity = self.config.get("severity", "low")
        self.severity_level = severity if severity in _VALID_SEVERITY_LEVELS else "low"

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def scan(self, repo_path: str | Path) -> SecurityScanResult:
        start_time = time.time()
        path = Path(repo_path)

        if not path.exists():
            return SecurityScanResult(
                tool=SecurityTool.BANDIT,
                passed=False,
                error=f"Path does not exist: {repo_path}",
            )

        try:
            cmd = [
                "bandit",
                "-r",
                str(path),
                "-f",
                "json",
                "-ll",  # Show low severity
                "--confidence-level",
                self.confidence_level,
                "--severity-level",
                self.severity_level,
            ]

            for pattern in self.exclude_patterns:
                if not _VALID_EXCLUDE_PATTERN.match(pattern):
                    continue  # skip unsafe patterns
                cmd.extend(["--exclude", pattern])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            scan_time = time.time() - start_time

            if result.returncode in (0, 1):
                return self._parse_results(result.stdout, scan_time)
            else:
                return SecurityScanResult(
                    tool=SecurityTool.BANDIT,
                    passed=False,
                    error=f"Bandit scan failed: {result.stderr}",
                    scan_time_seconds=scan_time,
                )

        except subprocess.TimeoutExpired:
            return SecurityScanResult(
                tool=SecurityTool.BANDIT,
                passed=False,
                error="Bandit scan timed out after 300 seconds",
                scan_time_seconds=time.time() - start_time,
            )
        except FileNotFoundError:
            return SecurityScanResult(
                tool=SecurityTool.BANDIT,
                passed=False,
                error="Bandit not found. Install with: pip install bandit",
            )
        except Exception as e:
            return SecurityScanResult(
                tool=SecurityTool.BANDIT,
                passed=False,
                error=f"Bandit scan error: {str(e)}",
                scan_time_seconds=time.time() - start_time,
            )

    def _parse_results(self, output: str, scan_time: float) -> SecurityScanResult:
        findings: list[SecurityFinding] = []

        try:
            data = json.loads(output) if output.strip() else {"results": []}
        except json.JSONDecodeError:
            return SecurityScanResult(
                tool=SecurityTool.BANDIT,
                passed=True,
                scan_time_seconds=scan_time,
            )

        for item in data.get("results", []):
            finding = SecurityFinding(
                tool=SecurityTool.BANDIT,
                rule_id=item.get("test_id", ""),
                severity=self._map_severity(item.get("issue_severity", "LOW")),
                message=item.get("issue_text", ""),
                file_path=item.get("filename", ""),
                line_number=item.get("line_number"),
                code_snippet=item.get("code"),
                cwe_id=item.get("issue_cwe", {}).get("cwe_id")
                if isinstance(item.get("issue_cwe"), dict)
                else None,
                confidence=item.get("issue_confidence"),
            )
            findings.append(finding)

        passed = all(f.severity != Severity.HIGH for f in findings)

        return SecurityScanResult(
            tool=SecurityTool.BANDIT,
            passed=passed,
            findings=findings,
            scan_time_seconds=scan_time,
        )

    def _map_severity(self, severity: str) -> Severity:
        mapping = {
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
        }
        return mapping.get(severity.upper(), Severity.LOW)


def run_bandit_scan(
    repo_path: str,
    config: dict[str, Any] | None = None,
) -> SecurityScanResult:
    scanner = BanditScanner(config)
    return scanner.scan(repo_path)
