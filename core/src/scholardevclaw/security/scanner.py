from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .bandit import BanditScanner, run_bandit_scan
from .semgrep import SemgrepScanner, run_semgrep_scan
from .types import SecurityReport, SecurityScanResult, SecurityTool


class SecurityScanner:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.run_bandit = self.config.get("run_bandit", True)
        self.run_semgrep = self.config.get("run_semgrep", True)
        self.fail_on_high = self.config.get("fail_on_high", True)
        self.bandit_config = self.config.get("bandit", {})
        self.semgrep_config = self.config.get("semgrep", {})

    def is_available(self) -> dict[str, bool]:
        bandit = BanditScanner(self.bandit_config)
        semgrep = SemgrepScanner(self.semgrep_config)

        return {
            "bandit": bandit.is_available(),
            "semgrep": semgrep.is_available(),
        }

    def scan(
        self,
        repo_path: str,
        tools: list[str] | None = None,
    ) -> SecurityReport:
        path = Path(repo_path)

        if not path.exists():
            return SecurityReport(
                repo_path=repo_path,
                passed=False,
                error=f"Path does not exist: {repo_path}",
            )

        scans: list[SecurityScanResult] = []
        tools_to_run = tools or []

        if not tools_to_run:
            if self.run_bandit:
                tools_to_run.append("bandit")
            if self.run_semgrep:
                tools_to_run.append("semgrep")

        for tool in tools_to_run:
            if tool == "bandit":
                scanner = BanditScanner(self.bandit_config)
                result = scanner.scan(path)
                scans.append(result)
            elif tool == "semgrep":
                scanner = SemgrepScanner(self.semgrep_config)
                result = scanner.scan(path)
                scans.append(result)

        report = SecurityReport(
            repo_path=repo_path,
            scans=scans,
        )

        if self.fail_on_high:
            report.passed = all(s.high_severity_count == 0 for s in scans)
        else:
            report.passed = all(s.passed for s in scans)

        return report

    def scan_python_only(self, repo_path: str) -> SecurityScanResult:
        scanner = BanditScanner(self.bandit_config)
        return scanner.scan(repo_path)

    def scan_multi_language(self, repo_path: str) -> SecurityScanResult:
        scanner = SemgrepScanner(self.semgrep_config)
        return scanner.scan(repo_path)


def run_security_scan(
    repo_path: str,
    config: dict[str, Any] | None = None,
) -> SecurityReport:
    scanner = SecurityScanner(config)
    return scanner.scan(repo_path)


def run_security_scan_tools(
    repo_path: str,
    tools: list[str],
    config: dict[str, Any] | None = None,
) -> SecurityReport:
    scanner = SecurityScanner(config)
    return scanner.scan(repo_path, tools)
