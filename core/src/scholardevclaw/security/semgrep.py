from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .types import SecurityFinding, SecurityScanResult, SecurityTool, Severity

# Allowlists for validating config values before passing to subprocess
_VALID_SEMGREP_RULE_PATTERN = re.compile(r"^[a-zA-Z0-9_./:=-]+$")
_VALID_EXCLUDE_PATTERN = re.compile(r"^[a-zA-Z0-9_.*/?-]+$")


class SemgrepScanner:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.exclude_patterns = self.config.get(
            "exclude",
            [
                "*.min.js",
                "*.min.css",
                "node_modules/",
                ".venv/",
                "venv/",
                "dist/",
                "build/",
            ],
        )
        self.rules = self.config.get("rules", ["auto"])  # "auto" or specific rule paths
        self.quiet = self.config.get("quiet", True)

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
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
                tool=SecurityTool.SEMGREP,
                passed=False,
                error=f"Path does not exist: {repo_path}",
            )

        try:
            cmd = [
                "semgrep",
                "--json",
                "--no-git-hooks",
            ]

            if self.quiet:
                cmd.append("--quiet")

            for pattern in self.exclude_patterns:
                if not _VALID_EXCLUDE_PATTERN.match(pattern):
                    continue  # skip unsafe patterns
                cmd.extend(["--exclude", pattern])

            for rule in self.rules:
                if not _VALID_SEMGREP_RULE_PATTERN.match(rule):
                    continue  # skip unsafe rule identifiers
                cmd.extend(["--config", rule])

            cmd.append(str(path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            scan_time = time.time() - start_time

            if result.returncode in (0, 1, 2):
                return self._parse_results(result.stdout, scan_time)
            else:
                return SecurityScanResult(
                    tool=SecurityTool.SEMGREP,
                    passed=False,
                    error=f"Semgrep scan failed: {result.stderr}",
                    scan_time_seconds=scan_time,
                )

        except subprocess.TimeoutExpired:
            return SecurityScanResult(
                tool=SecurityTool.SEMGREP,
                passed=False,
                error="Semgrep scan timed out after 600 seconds",
                scan_time_seconds=time.time() - start_time,
            )
        except FileNotFoundError:
            return SecurityScanResult(
                tool=SecurityTool.SEMGREP,
                passed=False,
                error="Semgrep not found. Install with: pip install semgrep",
            )
        except Exception as e:
            return SecurityScanResult(
                tool=SecurityTool.SEMGREP,
                passed=False,
                error=f"Semgrep scan error: {str(e)}",
                scan_time_seconds=time.time() - start_time,
            )

    def _parse_results(self, output: str, scan_time: float) -> SecurityScanResult:
        findings: list[SecurityFinding] = []

        try:
            data = json.loads(output) if output.strip() else {"results": []}
        except json.JSONDecodeError:
            return SecurityScanResult(
                tool=SecurityTool.SEMGREP,
                passed=True,
                scan_time_seconds=scan_time,
            )

        for item in data.get("results", []):
            extra = item.get("extra", {})
            metadata = extra.get("metadata", {})

            finding = SecurityFinding(
                tool=SecurityTool.SEMGREP,
                rule_id=item.get("check_id", ""),
                severity=self._map_severity(extra.get("severity", "INFO")),
                message=extra.get("message", ""),
                file_path=item.get("path", ""),
                line_number=item.get("start", {}).get("line"),
                code_snippet=extra.get("lines"),
                cwe_id=metadata.get("cwe"),
                confidence=extra.get("confidence"),
            )
            findings.append(finding)

        passed = all(f.severity != Severity.HIGH for f in findings)

        return SecurityScanResult(
            tool=SecurityTool.SEMGREP,
            passed=passed,
            findings=findings,
            scan_time_seconds=scan_time,
        )

    def _map_severity(self, severity: str) -> Severity:
        mapping = {
            "ERROR": Severity.HIGH,
            "WARNING": Severity.MEDIUM,
            "INFO": Severity.INFO,
        }
        return mapping.get(severity.upper(), Severity.LOW)


def run_semgrep_scan(
    repo_path: str,
    config: dict[str, Any] | None = None,
) -> SecurityScanResult:
    scanner = SemgrepScanner(config)
    return scanner.scan(repo_path)
