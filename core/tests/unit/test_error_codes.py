import pytest

from scholardevclaw.utils.error_codes import (
    ErrorCategory,
    ErrorSeverity,
    ErrorCode,
    ErrorCodes,
    ERROR_CODE_MAP,
    AppException,
    get_error,
    create_error,
)


class TestErrorCategory:
    def test_all_categories_exist(self):
        expected = [
            "VALIDATION",
            "REPO",
            "RESEARCH",
            "MAPPING",
            "PATCH",
            "VALIDATION_RUN",
            "INTEGRATION",
            "NETWORK",
            "CONFIG",
            "INTERNAL",
            "PERMISSION",
            "RESOURCE",
            "TIMEOUT",
            "RATE_LIMIT",
            "PLUGIN",
            "WORKFLOW",
        ]
        for cat in expected:
            assert hasattr(ErrorCategory, cat)

    def test_category_values(self):
        assert ErrorCategory.VALIDATION.value == "VAL"
        assert ErrorCategory.REPO.value == "REP"
        assert ErrorCategory.NETWORK.value == "NET"


class TestErrorSeverity:
    def test_all_severities_exist(self):
        assert hasattr(ErrorSeverity, "INFO")
        assert hasattr(ErrorSeverity, "WARNING")
        assert hasattr(ErrorSeverity, "ERROR")
        assert hasattr(ErrorSeverity, "CRITICAL")

    def test_severity_values(self):
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.ERROR.value == "error"


class TestErrorCode:
    def test_error_code_creation(self):
        error = ErrorCode(
            code="TEST001",
            category=ErrorCategory.VALIDATION,
            message="Test error",
        )
        assert error.code == "TEST001"
        assert error.category == ErrorCategory.VALIDATION
        assert error.message == "Test error"

    def test_error_code_defaults(self):
        error = ErrorCode(
            code="TEST001",
            category=ErrorCategory.VALIDATION,
            message="Test",
        )
        assert error.severity == ErrorSeverity.ERROR
        assert error.http_status == 500
        assert error.remediation is None

    def test_error_code_str(self):
        error = ErrorCode(
            code="TEST001",
            category=ErrorCategory.VALIDATION,
            message="Test error",
        )
        assert str(error) == "[TEST001] Test error"

    def test_error_code_to_dict(self):
        error = ErrorCode(
            code="TEST001",
            category=ErrorCategory.VALIDATION,
            message="Test error",
            severity=ErrorSeverity.WARNING,
            http_status=400,
            remediation="Fix it",
        )
        result = error.to_dict()
        assert result["code"] == "TEST001"
        assert result["category"] == "VAL"
        assert result["message"] == "Test error"
        assert result["severity"] == "warning"
        assert result["http_status"] == 400
        assert result["remediation"] == "Fix it"


class TestErrorCodes:
    def test_validation_codes(self):
        assert ErrorCodes.VAL001.code == "VAL001"
        assert ErrorCodes.VAL001.category == ErrorCategory.VALIDATION
        assert ErrorCodes.VAL001.http_status == 400

    def test_repo_codes(self):
        assert ErrorCodes.REP001.code == "REP001"
        assert ErrorCodes.REP002.code == "REP002"
        assert ErrorCodes.REP003.code == "REP003"
        assert ErrorCodes.REP004.code == "REP004"

    def test_network_codes(self):
        assert ErrorCodes.NET001.code == "NET001"
        assert ErrorCodes.NET002.code == "NET002"

    def test_integration_codes(self):
        assert ErrorCodes.INT001.code == "INT001"
        assert ErrorCodes.INT002.code == "INT002"
        assert ErrorCodes.INT003.code == "INT003"

    def test_workflow_codes(self):
        assert ErrorCodes.WRK001.code == "WRK001"
        assert ErrorCodes.WRK002.code == "WRK002"
        assert ErrorCodes.WRK003.code == "WRK003"


class TestErrorCodeMap:
    def test_map_contains_all_codes(self):
        assert "VAL001" in ERROR_CODE_MAP
        assert "REP001" in ERROR_CODE_MAP
        assert "NET001" in ERROR_CODE_MAP

    def test_map_returns_correct_error(self):
        error = ERROR_CODE_MAP["VAL001"]
        assert error.code == "VAL001"

    def test_map_values_are_error_codes(self):
        for code, error in ERROR_CODE_MAP.items():
            assert isinstance(error, ErrorCode)
            assert error.code == code


class TestAppException:
    def test_exception_with_error_code(self):
        error = ErrorCodes.VAL001
        exc = AppException(error)
        assert exc.error_code == error

    def test_exception_with_string_code(self):
        exc = AppException("VAL001")
        assert exc.error_code == ErrorCodes.VAL001

    def test_exception_with_unknown_code(self):
        exc = AppException("UNKNOWN999")
        assert exc.error_code.code == "SYS001"

    def test_exception_with_details(self):
        exc = AppException("VAL001", details={"field": "name"})
        assert exc.details == {"field": "name"}

    def test_exception_with_cause(self):
        cause = ValueError("original error")
        exc = AppException("SYS001", cause=cause)
        assert exc.cause == cause

    def test_exception_to_dict(self):
        exc = AppException("VAL001", details={"field": "name"})
        result = exc.to_dict()
        assert result["code"] == "VAL001"
        assert result["details"]["field"] == "name"

    def test_exception_to_dict_with_cause(self):
        cause = ValueError("original error")
        exc = AppException("SYS001", cause=cause)
        result = exc.to_dict()
        assert "cause" in result


class TestGetError:
    def test_get_existing_error(self):
        error = get_error("VAL001")
        assert error is not None
        assert error.code == "VAL001"

    def test_get_nonexistent_error(self):
        error = get_error("UNKNOWN999")
        assert error is None


class TestCreateError:
    def test_create_custom_error(self):
        error = create_error(
            code="CUSTOM001",
            message="Custom error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            http_status=400,
            remediation="Fix it",
        )
        assert error.code == "CUSTOM001"
        assert error.message == "Custom error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.WARNING
        assert error.http_status == 400
        assert error.remediation == "Fix it"

    def test_create_error_defaults(self):
        error = create_error(
            code="CUSTOM002",
            message="Custom error",
            category=ErrorCategory.REPO,
        )
        assert error.severity == ErrorSeverity.ERROR
        assert error.http_status == 500
        assert error.remediation is None
