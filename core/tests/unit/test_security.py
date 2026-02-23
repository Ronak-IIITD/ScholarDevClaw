import json

import pytest

from scholardevclaw.security.types import (
    SecurityFinding,
    SecurityReport,
    SecurityScanResult,
    SecurityTool,
    Severity,
)
from scholardevclaw.security.scanner import SecurityScanner


class TestSecurityTypes:
    def test_security_finding_to_dict(self):
        finding = SecurityFinding(
            tool=SecurityTool.BANDIT,
            rule_id="B101",
            severity=Severity.HIGH,
            message="Use of assert detected",
            file_path="model.py",
            line_number=10,
            code_snippet="assert x > 0",
            cwe_id="CWE-703",
            confidence="HIGH",
        )

        data = finding.to_dict()
        assert data["tool"] == "bandit"
        assert data["rule_id"] == "B101"
        assert data["severity"] == "high"
        assert data["file_path"] == "model.py"
        assert data["line_number"] == 10

    def test_security_scan_result_counts(self):
        result = SecurityScanResult(
            tool=SecurityTool.BANDIT,
            passed=False,
            findings=[
                SecurityFinding(
                    tool=SecurityTool.BANDIT,
                    rule_id="B101",
                    severity=Severity.HIGH,
                    message="Test",
                    file_path="test.py",
                ),
                SecurityFinding(
                    tool=SecurityTool.BANDIT,
                    rule_id="B102",
                    severity=Severity.MEDIUM,
                    message="Test",
                    file_path="test.py",
                ),
                SecurityFinding(
                    tool=SecurityTool.BANDIT,
                    rule_id="B103",
                    severity=Severity.LOW,
                    message="Test",
                    file_path="test.py",
                ),
            ],
        )

        assert result.high_severity_count == 1
        assert result.medium_severity_count == 1
        assert result.low_severity_count == 1

    def test_security_report_summary(self):
        report = SecurityReport(
            repo_path="/path/to/repo",
            scans=[
                SecurityScanResult(
                    tool=SecurityTool.BANDIT,
                    passed=False,
                    findings=[
                        SecurityFinding(
                            tool=SecurityTool.BANDIT,
                            rule_id="B101",
                            severity=Severity.HIGH,
                            message="Test",
                            file_path="test.py",
                        ),
                    ],
                ),
                SecurityScanResult(
                    tool=SecurityTool.SEMGREP,
                    passed=True,
                    findings=[
                        SecurityFinding(
                            tool=SecurityTool.SEMGREP,
                            rule_id="javascript.lang.dangerous.eval",
                            severity=Severity.MEDIUM,
                            message="Test",
                            file_path="app.js",
                        ),
                    ],
                ),
            ],
        )

        assert report.high_severity_count == 1
        assert report.medium_severity_count == 1
        assert report.low_severity_count == 0
        assert report.total_findings == 2
        assert report.tools_run == ["bandit", "semgrep"]


class TestSecurityScanner:
    def test_scanner_initialization(self):
        scanner = SecurityScanner()
        assert scanner.run_bandit is True
        assert scanner.run_semgrep is True

    def test_scanner_with_config(self):
        scanner = SecurityScanner(
            {
                "run_bandit": False,
                "run_semgrep": True,
                "fail_on_high": False,
            }
        )
        assert scanner.run_bandit is False
        assert scanner.run_semgrep is True
        assert scanner.fail_on_high is False

    def test_scanner_with_empty_repo_path(self):
        scanner = SecurityScanner()
        report = scanner.scan("/nonexistent/path")

        assert report.passed is False
        assert "does not exist" in report.error

    def test_scanner_with_specific_tools(self):
        scanner = SecurityScanner()

        assert scanner.scan.__name__ == "scan"


class TestSeverityEnum:
    def test_severity_values(self):
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_tool_values(self):
        assert SecurityTool.BANDIT.value == "bandit"
        assert SecurityTool.SEMGREP.value == "semgrep"


class TestSecurityScanResult:
    def test_passed_with_no_findings(self):
        result = SecurityScanResult(
            tool=SecurityTool.BANDIT,
            passed=True,
            findings=[],
            scan_time_seconds=1.5,
        )

        assert result.high_severity_count == 0
        assert result.medium_severity_count == 0
        assert result.low_severity_count == 0

    def test_to_dict(self):
        result = SecurityScanResult(
            tool=SecurityTool.SEMGREP,
            passed=True,
            findings=[],
            scan_time_seconds=2.0,
        )

        data = result.to_dict()
        assert data["tool"] == "semgrep"
        assert data["passed"] is True
        assert data["scan_time_seconds"] == 2.0
        assert data["summary"]["total"] == 0
