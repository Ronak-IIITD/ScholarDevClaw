from __future__ import annotations

from .bandit import BanditScanner, run_bandit_scan
from .scanner import SecurityScanner, run_security_scan, run_security_scan_tools
from .semgrep import SemgrepScanner, run_semgrep_scan
from .types import (
    SecurityFinding,
    SecurityReport,
    SecurityScanResult,
    SecurityTool,
    Severity,
)


def get_scanner(config: dict | None = None) -> SecurityScanner:
    return SecurityScanner(config)


__all__ = [
    "SecurityScanner",
    "BanditScanner",
    "SemgrepScanner",
    "run_security_scan",
    "run_security_scan_tools",
    "run_bandit_scan",
    "run_semgrep_scan",
    "SecurityReport",
    "SecurityScanResult",
    "SecurityFinding",
    "SecurityTool",
    "Severity",
]
