from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityTool(str, Enum):
    BANDIT = "bandit"
    SEMGREP = "semgrep"


@dataclass
class SecurityFinding:
    tool: SecurityTool
    rule_id: str
    severity: Severity
    message: str
    file_path: str
    line_number: int | None = None
    code_snippet: str | None = None
    cwe_id: str | None = None
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.value,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "cwe_id": self.cwe_id,
            "confidence": self.confidence,
        }


@dataclass
class SecurityScanResult:
    tool: SecurityTool
    passed: bool
    findings: list[SecurityFinding] = field(default_factory=list)
    scan_time_seconds: float = 0.0
    error: str | None = None

    @property
    def high_severity_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_severity_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_severity_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.value,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "scan_time_seconds": self.scan_time_seconds,
            "error": self.error,
            "summary": {
                "high": self.high_severity_count,
                "medium": self.medium_severity_count,
                "low": self.low_severity_count,
                "total": len(self.findings),
            },
        }


@dataclass
class SecurityReport:
    repo_path: str
    scans: list[SecurityScanResult] = field(default_factory=list)
    passed: bool = True
    error: str | None = None

    @property
    def total_findings(self) -> int:
        return sum(len(s.findings) for s in self.scans)

    @property
    def high_severity_count(self) -> int:
        return sum(s.high_severity_count for s in self.scans)

    @property
    def medium_severity_count(self) -> int:
        return sum(s.medium_severity_count for s in self.scans)

    @property
    def low_severity_count(self) -> int:
        return sum(s.low_severity_count for s in self.scans)

    @property
    def tools_run(self) -> list[str]:
        return [s.tool.value for s in self.scans]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "passed": self.passed,
            "error": self.error,
            "tools_run": self.tools_run,
            "scans": [s.to_dict() for s in self.scans],
            "summary": {
                "high": self.high_severity_count,
                "medium": self.medium_severity_count,
                "low": self.low_severity_count,
                "total": self.total_findings,
            },
        }
