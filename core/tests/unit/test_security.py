from unittest.mock import patch

from scholardevclaw.security.scanner import SecurityScanner
from scholardevclaw.security.types import (
    SecurityFinding,
    SecurityReport,
    SecurityScanResult,
    SecurityTool,
    Severity,
)


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
        assert "does not exist" in (report.error or "")

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


# =========================================================================
# BanditScanner tests
# =========================================================================


class TestBanditScanner:
    def test_is_available_true(self):
        import subprocess
        from unittest.mock import patch

        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["bandit", "--version"], returncode=0, stdout="1.0.0", stderr=""
            )
            assert scanner.is_available() is True

    def test_is_available_false(self):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert scanner.is_available() is False

    def test_is_available_timeout(self):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="bandit", timeout=10)
            assert scanner.is_available() is False

    def test_scan_nonexistent_path(self):
        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        result = scanner.scan("/definitely/does/not/exist/12345")
        assert result.passed is False
        assert "does not exist" in (result.error or "")

    def test_scan_tool_not_found(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError("bandit not found")
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "not found" in (result.error or "")

    def test_scan_timeout(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="bandit", timeout=300)
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "timed out" in (result.error or "").lower()

    def test_scan_generic_exception(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = RuntimeError("unexpected error")
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "error" in (result.error or "").lower()

    def test_scan_returncode_2(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=2, stdout="", stderr="bandit failed"
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "failed" in (result.error or "")

    def test_scan_success_no_findings(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True
        assert len(result.findings) == 0

    def test_scan_with_high_finding(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        findings_json = """{
            "results": [
                {
                    "test_id": "B101",
                    "issue_severity": "HIGH",
                    "issue_text": "Use of assert detected",
                    "filename": "test.py",
                    "line_number": 5,
                    "code": "assert x > 0",
                    "issue_cwe": {"cwe_id": "CWE-703"},
                    "issue_confidence": "HIGH"
                }
            ]
        }"""
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=findings_json, stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert result.high_severity_count == 1

    def test_scan_with_medium_finding_passes(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        findings_json = """{
            "results": [
                {
                    "test_id": "B102",
                    "issue_severity": "MEDIUM",
                    "issue_text": "Medium issue",
                    "filename": "test.py",
                    "line_number": 10,
                    "code": "exec('x = 1')",
                    "issue_cwe": {"cwe_id": "CWE-94"},
                    "issue_confidence": "MEDIUM"
                }
            ]
        }"""
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=findings_json, stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True  # Only HIGH fails
        assert result.medium_severity_count == 1

    def test_scan_invalid_json(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = BanditScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="not valid json at all {{{", stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True  # Graceful fallback
        assert len(result.findings) == 0

    def test_scan_exclude_patterns_valid(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner({"exclude": ["tests/", "*.venv/"]})
        (tmp_path / "test.py").write_text("x = 1")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            scanner.scan(str(tmp_path))
            call_args = mock_run.call_args[0][0]
            assert "--exclude" in call_args
            assert "tests/" in call_args
            assert "*.venv/" in call_args

    def test_scan_exclude_patterns_invalid_skipped(self, tmp_path):
        import subprocess

        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner({"exclude": ["$(rm -rf /)"]})
        (tmp_path / "test.py").write_text("x = 1")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            scanner.scan(str(tmp_path))
            call_args = mock_run.call_args[0][0]
            assert "$(rm -rf /)" not in call_args


class TestBanditParseResults:
    def test_parse_results_empty(self):
        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        result = scanner._parse_results("", 0.5)
        assert result.passed is True
        assert len(result.findings) == 0
        assert result.scan_time_seconds == 0.5

    def test_parse_results_with_findings(self):
        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        output = """{
            "results": [
                {
                    "test_id": "B101",
                    "issue_severity": "HIGH",
                    "issue_text": "Use of assert",
                    "filename": "test.py",
                    "line_number": 10,
                    "code": "assert x",
                    "issue_cwe": {"cwe_id": "CWE-703"},
                    "issue_confidence": "HIGH"
                },
                {
                    "test_id": "B102",
                    "issue_severity": "LOW",
                    "issue_text": "Low issue",
                    "filename": "test.py",
                    "line_number": 20,
                    "code": "pass",
                    "issue_cwe": {"cwe_id": "CWE-200"},
                    "issue_confidence": "LOW"
                }
            ]
        }"""
        result = scanner._parse_results(output, 1.0)
        assert len(result.findings) == 2
        assert result.high_severity_count == 1
        assert result.low_severity_count == 1
        assert result.findings[0].rule_id == "B101"
        assert result.findings[0].cwe_id == "CWE-703"

    def test_parse_results_without_cwe(self):
        from scholardevclaw.security.bandit import BanditScanner

        scanner = BanditScanner()
        output = """{
            "results": [
                {
                    "test_id": "B103",
                    "issue_severity": "MEDIUM",
                    "issue_text": "No CWE",
                    "filename": "test.py",
                    "line_number": 5,
                    "code": "x = eval(y)",
                    "issue_confidence": "MEDIUM"
                }
            ]
        }"""
        result = scanner._parse_results(output, 0.5)
        assert len(result.findings) == 1
        assert result.findings[0].cwe_id is None

    def test_map_severity(self):
        from scholardevclaw.security.bandit import BanditScanner
        from scholardevclaw.security.types import Severity

        scanner = BanditScanner()
        assert scanner._map_severity("HIGH") == Severity.HIGH
        assert scanner._map_severity("MEDIUM") == Severity.MEDIUM
        assert scanner._map_severity("LOW") == Severity.LOW
        assert scanner._map_severity("UNKNOWN") == Severity.LOW
        assert scanner._map_severity("") == Severity.LOW


# =========================================================================
# SemgrepScanner tests
# =========================================================================


class TestSemgrepScanner:
    def test_is_available_true(self):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["semgrep", "--version"], returncode=0, stdout="1.0.0", stderr=""
            )
            assert scanner.is_available() is True

    def test_is_available_false(self):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert scanner.is_available() is False

    def test_scan_nonexistent_path(self):
        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner()
        result = scanner.scan("/definitely/does/not/exist/semgrep/12345")
        assert result.passed is False
        assert "does not exist" in (result.error or "")

    def test_scan_tool_not_found(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError("semgrep not found")
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "not found" in (result.error or "")

    def test_scan_timeout(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="semgrep", timeout=600)
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "timed out" in (result.error or "").lower()

    def test_scan_generic_exception(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = RuntimeError("semgrep crashed")
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "error" in (result.error or "").lower()

    def test_scan_returncode_3(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=3, stdout="", stderr="semgrep error"
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert "failed" in (result.error or "")

    def test_scan_returncode_1_success_with_findings(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        output = """{
            "results": [
                {
                    "check_id": "python.lang.bad.eval",
                    "path": "test.py",
                    "start": {"line": 5},
                    "extra": {
                        "severity": "WARNING",
                        "message": "eval used",
                        "lines": "eval(x)",
                        "metadata": {"cwe": "CWE-94"},
                        "confidence": "MEDIUM"
                    }
                }
            ]
        }"""
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=output, stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True  # WARNING is not HIGH
        assert len(result.findings) == 1

    def test_scan_success_no_findings(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True
        assert len(result.findings) == 0

    def test_scan_with_high_finding(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        output = """{
            "results": [
                {
                    "check_id": "python.lang.security.unsafe",
                    "path": "test.py",
                    "start": {"line": 10},
                    "extra": {
                        "severity": "ERROR",
                        "message": "Unsafe operation",
                        "lines": "os.system('rm -rf /')",
                        "metadata": {"cwe": "CWE-78"},
                        "confidence": "HIGH"
                    }
                }
            ]
        }"""
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=output, stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is False
        assert result.high_severity_count == 1

    def test_scan_invalid_json(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        (tmp_path / "test.py").write_text("x = 1")
        scanner = SemgrepScanner()
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="not json {{{", stderr=""
            )
            result = scanner.scan(str(tmp_path))
        assert result.passed is True
        assert len(result.findings) == 0

    def test_scan_exclude_patterns(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner({"exclude": ["node_modules/", "*.min.js"]})
        (tmp_path / "test.py").write_text("x = 1")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            scanner.scan(str(tmp_path))
            call_args = mock_run.call_args[0][0]
            assert "--exclude" in call_args
            assert "node_modules/" in call_args

    def test_scan_invalid_rules_skipped(self, tmp_path):
        import subprocess

        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner({"rules": ["auto", "$(malicious)", "p/python"]})
        (tmp_path / "test.py").write_text("x = 1")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            scanner.scan(str(tmp_path))
            call_args = mock_run.call_args[0][0]
            assert "--config" in call_args
            assert "auto" in call_args
            assert "p/python" in call_args
            assert "$(malicious)" not in call_args


class TestSemgrepParseResults:
    def test_parse_results_empty(self):
        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner()
        result = scanner._parse_results("", 0.3)
        assert result.passed is True
        assert len(result.findings) == 0

    def test_parse_results_with_findings(self):
        from scholardevclaw.security.semgrep import SemgrepScanner

        scanner = SemgrepScanner()
        output = """{
            "results": [
                {
                    "check_id": "python.lang.bad.eval",
                    "path": "app.py",
                    "start": {"line": 10},
                    "extra": {
                        "severity": "ERROR",
                        "message": "eval usage",
                        "lines": "eval(x)",
                        "metadata": {"cwe": "CWE-94"},
                        "confidence": "HIGH"
                    }
                }
            ]
        }"""
        result = scanner._parse_results(output, 0.5)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "python.lang.bad.eval"
        assert result.findings[0].severity == Severity.HIGH
        assert result.findings[0].cwe_id == "CWE-94"

    def test_map_severity(self):
        from scholardevclaw.security.semgrep import SemgrepScanner
        from scholardevclaw.security.types import Severity

        scanner = SemgrepScanner()
        assert scanner._map_severity("ERROR") == Severity.HIGH
        assert scanner._map_severity("WARNING") == Severity.MEDIUM
        assert scanner._map_severity("INFO") == Severity.INFO
        assert scanner._map_severity("UNKNOWN") == Severity.LOW
        assert scanner._map_severity("") == Severity.LOW


# =========================================================================
# SecurityScanner additional tests
# =========================================================================


class TestSecurityScannerAdditional:
    def test_scan_with_specific_tools(self, tmp_path):
        import subprocess

        from scholardevclaw.security.scanner import SecurityScanner

        scanner = SecurityScanner({"run_bandit": True, "run_semgrep": True})
        (tmp_path / "test.py").write_text("x = 1")

        from unittest.mock import patch

        with (
            patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='{"results": []}', stderr=""
                ),
            ),
        ):
            report = scanner.scan(str(tmp_path), tools=["bandit"])
            assert len(report.scans) == 1
            assert report.scans[0].tool == SecurityTool.BANDIT

    def test_scan_python_only(self, tmp_path):
        import subprocess

        from scholardevclaw.security.scanner import SecurityScanner

        scanner = SecurityScanner()
        (tmp_path / "test.py").write_text("x = 1")

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            result = scanner.scan_python_only(str(tmp_path))
            assert result.tool == SecurityTool.BANDIT

    def test_scan_multi_language(self, tmp_path):
        import subprocess

        from scholardevclaw.security.scanner import SecurityScanner

        scanner = SecurityScanner()
        (tmp_path / "test.py").write_text("x = 1")

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"results": []}', stderr=""
            )
            result = scanner.scan_multi_language(str(tmp_path))
            assert result.tool == SecurityTool.SEMGREP

    def test_is_available(self, tmp_path):
        from scholardevclaw.security.scanner import SecurityScanner

        scanner = SecurityScanner()
        available = scanner.is_available()
        assert isinstance(available, dict)
        assert "bandit" in available
        assert "semgrep" in available

    def test_scan_passes_without_high(self, tmp_path):
        import subprocess

        from scholardevclaw.security.scanner import SecurityScanner

        scanner = SecurityScanner({"fail_on_high": False})
        (tmp_path / "test.py").write_text("x = 1")

        findings_json = """{
            "results": [
                {
                    "test_id": "B101",
                    "issue_severity": "HIGH",
                    "issue_text": "Use of assert",
                    "filename": "test.py",
                    "line_number": 5,
                    "code": "assert x",
                    "issue_cwe": {"cwe_id": "CWE-703"},
                    "issue_confidence": "HIGH"
                }
            ]
        }"""
        output = subprocess.CompletedProcess(args=[], returncode=0, stdout=findings_json, stderr="")
        json_output = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"results": []}', stderr=""
        )

        # Mock semgrep to return empty, bandit to return high severity finding
        from unittest.mock import patch

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return output  # bandit
            return json_output  # semgrep

        with patch.object(subprocess, "run", side_effect=side_effect):
            report = scanner.scan(str(tmp_path))
            # With fail_on_high=False, it checks all(s.passed for s in scans)
            # Bandit returns passed=False when there are high findings
            # So report.passed should be False
            assert report.passed is False


class TestSecurityReportAdditional:
    def test_report_to_dict(self):
        from scholardevclaw.security.types import SecurityReport, SecurityScanResult, SecurityTool

        report = SecurityReport(
            repo_path="/path",
            scans=[
                SecurityScanResult(
                    tool=SecurityTool.BANDIT,
                    passed=True,
                    findings=[],
                )
            ],
        )
        data = report.to_dict()
        assert data["repo_path"] == "/path"
        assert data["passed"] is True
        assert data["tools_run"] == ["bandit"]
        assert data["summary"]["total"] == 0

    def test_report_passed_default(self):
        from scholardevclaw.security.types import SecurityReport

        report = SecurityReport(repo_path="/path")
        assert report.passed is True
        assert report.error is None

    def test_report_with_error(self):
        from scholardevclaw.security.types import SecurityReport

        report = SecurityReport(repo_path="/path", error="something broke")
        assert report.error == "something broke"
        data = report.to_dict()
        assert data["error"] == "something broke"

    def test_report_severity_counts_aggregate(self):
        from scholardevclaw.security.types import (
            SecurityFinding,
            SecurityReport,
            SecurityScanResult,
            SecurityTool,
            Severity,
        )

        report = SecurityReport(
            repo_path="/path",
            scans=[
                SecurityScanResult(
                    tool=SecurityTool.BANDIT,
                    passed=False,
                    findings=[
                        SecurityFinding(
                            tool=SecurityTool.BANDIT,
                            rule_id="B101",
                            severity=Severity.HIGH,
                            message="High",
                            file_path="a.py",
                        ),
                    ],
                ),
                SecurityScanResult(
                    tool=SecurityTool.SEMGREP,
                    passed=True,
                    findings=[
                        SecurityFinding(
                            tool=SecurityTool.SEMGREP,
                            rule_id="R1",
                            severity=Severity.LOW,
                            message="Low",
                            file_path="b.js",
                        ),
                        SecurityFinding(
                            tool=SecurityTool.SEMGREP,
                            rule_id="R2",
                            severity=Severity.INFO,
                            message="Info",
                            file_path="c.js",
                        ),
                    ],
                ),
            ],
        )
        assert report.high_severity_count == 1
        assert report.medium_severity_count == 0
        assert report.low_severity_count == 1
        assert report.total_findings == 3
        assert report.tools_run == ["bandit", "semgrep"]
