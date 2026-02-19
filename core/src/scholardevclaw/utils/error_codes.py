from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    VALIDATION = "VAL"
    REPO = "REP"
    RESEARCH = "RES"
    MAPPING = "MAP"
    PATCH = "PAT"
    VALIDATION_RUN = "VRU"
    INTEGRATION = "INT"
    NETWORK = "NET"
    CONFIG = "CFG"
    INTERNAL = "INT"
    PERMISSION = "PRM"
    RESOURCE = "RSC"
    TIMEOUT = "TIM"
    RATE_LIMIT = "RTL"
    PLUGIN = "PLG"
    WORKFLOW = "WRK"


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorCode:
    code: str
    category: ErrorCategory
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    http_status: int = 500
    remediation: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category.value,
            "message": self.message,
            "severity": self.severity.value,
            "http_status": self.http_status,
            "remediation": self.remediation,
            "details": self.details,
        }


class ErrorCodes:
    VAL001 = ErrorCode(
        code="VAL001",
        category=ErrorCategory.VALIDATION,
        message="Invalid input parameter",
        severity=ErrorSeverity.WARNING,
        http_status=400,
        remediation="Check the input parameter format and value",
    )
    VAL002 = ErrorCode(
        code="VAL002",
        category=ErrorCategory.VALIDATION,
        message="Required field missing",
        severity=ErrorSeverity.WARNING,
        http_status=400,
        remediation="Provide all required fields",
    )
    VAL003 = ErrorCode(
        code="VAL003",
        category=ErrorCategory.VALIDATION,
        message="Value out of range",
        severity=ErrorSeverity.WARNING,
        http_status=400,
        remediation="Ensure the value is within the allowed range",
    )

    REP001 = ErrorCode(
        code="REP001",
        category=ErrorCategory.REPO,
        message="Repository path not found",
        severity=ErrorSeverity.ERROR,
        http_status=404,
        remediation="Verify the repository path exists and is accessible",
    )
    REP002 = ErrorCode(
        code="REP002",
        category=ErrorCategory.REPO,
        message="Repository is not a valid Git repository",
        severity=ErrorSeverity.ERROR,
        http_status=400,
        remediation="Initialize a Git repository or provide a valid Git path",
    )
    REP003 = ErrorCode(
        code="REP003",
        category=ErrorCategory.REPO,
        message="Repository has uncommitted changes",
        severity=ErrorSeverity.WARNING,
        http_status=409,
        remediation="Commit or stash changes before proceeding",
    )
    REP004 = ErrorCode(
        code="REP004",
        category=ErrorCategory.REPO,
        message="Branch is protected",
        severity=ErrorSeverity.ERROR,
        http_status=403,
        remediation="Switch to a non-protected branch or create a feature branch",
    )

    RES001 = ErrorCode(
        code="RES001",
        category=ErrorCategory.RESEARCH,
        message="Failed to extract research specification",
        severity=ErrorSeverity.ERROR,
        http_status=422,
        remediation="Verify the PDF or arXiv source is valid and accessible",
    )
    RES002 = ErrorCode(
        code="RES002",
        category=ErrorCategory.RESEARCH,
        message="Research specification format invalid",
        severity=ErrorSeverity.ERROR,
        http_status=422,
        remediation="Ensure the research spec follows the expected schema",
    )
    RES003 = ErrorCode(
        code="RES003",
        category=ErrorCategory.RESEARCH,
        message="arXiv paper not found",
        severity=ErrorSeverity.ERROR,
        http_status=404,
        remediation="Verify the arXiv ID is correct",
    )

    MAP001 = ErrorCode(
        code="MAP001",
        category=ErrorCategory.MAPPING,
        message="No mapping targets found",
        severity=ErrorSeverity.WARNING,
        http_status=404,
        remediation="Check if the research spec matches the repository patterns",
    )
    MAP002 = ErrorCode(
        code="MAP002",
        category=ErrorCategory.MAPPING,
        message="Mapping confidence below threshold",
        severity=ErrorSeverity.WARNING,
        http_status=200,
        remediation="Review the mapping targets manually or adjust confidence threshold",
    )
    MAP003 = ErrorCode(
        code="MAP003",
        category=ErrorCategory.MAPPING,
        message="Multiple ambiguous mapping targets",
        severity=ErrorSeverity.WARNING,
        http_status=200,
        remediation="Specify target patterns more precisely",
    )

    PAT001 = ErrorCode(
        code="PAT001",
        category=ErrorCategory.PATCH,
        message="Patch generation failed",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Check the mapping targets and research spec compatibility",
    )
    PAT002 = ErrorCode(
        code="PAT002",
        category=ErrorCategory.PATCH,
        message="Patch contains invalid transformations",
        severity=ErrorSeverity.ERROR,
        http_status=422,
        remediation="Review the transformation syntax",
    )
    PAT003 = ErrorCode(
        code="PAT003",
        category=ErrorCategory.PATCH,
        message="Patch would create merge conflicts",
        severity=ErrorSeverity.WARNING,
        http_status=409,
        remediation="Resolve conflicts or regenerate the patch",
    )

    VRU001 = ErrorCode(
        code="VRU001",
        category=ErrorCategory.VALIDATION_RUN,
        message="Validation tests failed",
        severity=ErrorSeverity.ERROR,
        http_status=422,
        remediation="Review test output and fix generated code",
    )
    VRU002 = ErrorCode(
        code="VRU002",
        category=ErrorCategory.VALIDATION_RUN,
        message="Performance regression detected",
        severity=ErrorSeverity.WARNING,
        http_status=200,
        remediation="Review performance metrics and optimize generated code",
    )
    VRU003 = ErrorCode(
        code="VRU003",
        category=ErrorCategory.VALIDATION_RUN,
        message="Baseline metrics unavailable",
        severity=ErrorSeverity.WARNING,
        http_status=200,
        remediation="Run baseline validation first",
    )

    INT001 = ErrorCode(
        code="INT001",
        category=ErrorCategory.INTEGRATION,
        message="Integration failed",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Check integration logs for details",
    )
    INT002 = ErrorCode(
        code="INT002",
        category=ErrorCategory.INTEGRATION,
        message="Integration requires approval",
        severity=ErrorSeverity.INFO,
        http_status=202,
        remediation="Wait for approval or approve manually",
    )
    INT003 = ErrorCode(
        code="INT003",
        category=ErrorCategory.INTEGRATION,
        message="Integration timeout",
        severity=ErrorSeverity.ERROR,
        http_status=408,
        remediation="Increase timeout or check for hanging processes",
    )

    NET001 = ErrorCode(
        code="NET001",
        category=ErrorCategory.NETWORK,
        message="Network connection failed",
        severity=ErrorSeverity.ERROR,
        http_status=503,
        remediation="Check network connectivity and try again",
    )
    NET002 = ErrorCode(
        code="NET002",
        category=ErrorCategory.NETWORK,
        message="Request timeout",
        severity=ErrorSeverity.ERROR,
        http_status=408,
        remediation="Increase timeout or check server responsiveness",
    )

    CFG001 = ErrorCode(
        code="CFG001",
        category=ErrorCategory.CONFIG,
        message="Configuration file not found",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Create a configuration file or provide config via environment",
    )
    CFG002 = ErrorCode(
        code="CFG002",
        category=ErrorCategory.CONFIG,
        message="Invalid configuration value",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Check configuration format and values",
    )

    SYS001 = ErrorCode(
        code="SYS001",
        category=ErrorCategory.INTERNAL,
        message="Internal server error",
        severity=ErrorSeverity.CRITICAL,
        http_status=500,
        remediation="Contact support with error details",
    )
    SYS002 = ErrorCode(
        code="SYS002",
        category=ErrorCategory.INTERNAL,
        message="Unexpected exception",
        severity=ErrorSeverity.CRITICAL,
        http_status=500,
        remediation="Report this issue with the stack trace",
    )

    PRM001 = ErrorCode(
        code="PRM001",
        category=ErrorCategory.PERMISSION,
        message="Permission denied",
        severity=ErrorSeverity.ERROR,
        http_status=403,
        remediation="Check file/directory permissions",
    )
    PRM002 = ErrorCode(
        code="PRM002",
        category=ErrorCategory.PERMISSION,
        message="Authentication required",
        severity=ErrorSeverity.ERROR,
        http_status=401,
        remediation="Provide valid authentication credentials",
    )

    RSC001 = ErrorCode(
        code="RSC001",
        category=ErrorCategory.RESOURCE,
        message="Resource not found",
        severity=ErrorSeverity.ERROR,
        http_status=404,
        remediation="Verify the resource exists",
    )
    RSC002 = ErrorCode(
        code="RSC002",
        category=ErrorCategory.RESOURCE,
        message="Resource exhausted",
        severity=ErrorSeverity.ERROR,
        http_status=503,
        remediation="Free up resources or increase limits",
    )

    TIM001 = ErrorCode(
        code="TIM001",
        category=ErrorCategory.TIMEOUT,
        message="Operation timed out",
        severity=ErrorSeverity.ERROR,
        http_status=408,
        remediation="Increase timeout or optimize the operation",
    )

    RTL001 = ErrorCode(
        code="RTL001",
        category=ErrorCategory.RATE_LIMIT,
        message="Rate limit exceeded",
        severity=ErrorSeverity.WARNING,
        http_status=429,
        remediation="Wait before retrying or increase rate limit",
    )

    PLG001 = ErrorCode(
        code="PLG001",
        category=ErrorCategory.PLUGIN,
        message="Plugin not found",
        severity=ErrorSeverity.ERROR,
        http_status=404,
        remediation="Install the plugin or check plugin name",
    )
    PLG002 = ErrorCode(
        code="PLG002",
        category=ErrorCategory.PLUGIN,
        message="Plugin initialization failed",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Check plugin dependencies and configuration",
    )

    WRK001 = ErrorCode(
        code="WRK001",
        category=ErrorCategory.WORKFLOW,
        message="Workflow cycle detected",
        severity=ErrorSeverity.ERROR,
        http_status=400,
        remediation="Remove circular dependencies from workflow",
    )
    WRK002 = ErrorCode(
        code="WRK002",
        category=ErrorCategory.WORKFLOW,
        message="Workflow node failed",
        severity=ErrorSeverity.ERROR,
        http_status=500,
        remediation="Check node configuration and dependencies",
    )
    WRK003 = ErrorCode(
        code="WRK003",
        category=ErrorCategory.WORKFLOW,
        message="Workflow timeout",
        severity=ErrorSeverity.ERROR,
        http_status=408,
        remediation="Increase workflow timeout or optimize nodes",
    )


ERROR_CODE_MAP: dict[str, ErrorCode] = {
    "VAL001": ErrorCodes.VAL001,
    "VAL002": ErrorCodes.VAL002,
    "VAL003": ErrorCodes.VAL003,
    "REP001": ErrorCodes.REP001,
    "REP002": ErrorCodes.REP002,
    "REP003": ErrorCodes.REP003,
    "REP004": ErrorCodes.REP004,
    "RES001": ErrorCodes.RES001,
    "RES002": ErrorCodes.RES002,
    "RES003": ErrorCodes.RES003,
    "MAP001": ErrorCodes.MAP001,
    "MAP002": ErrorCodes.MAP002,
    "MAP003": ErrorCodes.MAP003,
    "PAT001": ErrorCodes.PAT001,
    "PAT002": ErrorCodes.PAT002,
    "PAT003": ErrorCodes.PAT003,
    "VRU001": ErrorCodes.VRU001,
    "VRU002": ErrorCodes.VRU002,
    "VRU003": ErrorCodes.VRU003,
    "INT001": ErrorCodes.INT001,
    "INT002": ErrorCodes.INT002,
    "INT003": ErrorCodes.INT003,
    "NET001": ErrorCodes.NET001,
    "NET002": ErrorCodes.NET002,
    "CFG001": ErrorCodes.CFG001,
    "CFG002": ErrorCodes.CFG002,
    "SYS001": ErrorCodes.SYS001,
    "SYS002": ErrorCodes.SYS002,
    "PRM001": ErrorCodes.PRM001,
    "PRM002": ErrorCodes.PRM002,
    "RSC001": ErrorCodes.RSC001,
    "RSC002": ErrorCodes.RSC002,
    "TIM001": ErrorCodes.TIM001,
    "RTL001": ErrorCodes.RTL001,
    "PLG001": ErrorCodes.PLG001,
    "PLG002": ErrorCodes.PLG002,
    "WRK001": ErrorCodes.WRK001,
    "WRK002": ErrorCodes.WRK002,
    "WRK003": ErrorCodes.WRK003,
}


class AppException(Exception):
    def __init__(
        self,
        error_code: ErrorCode | str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        if isinstance(error_code, str):
            error_code = ERROR_CODE_MAP.get(error_code, ErrorCodes.SYS001)

        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(str(error_code))

    def to_dict(self) -> dict[str, Any]:
        result = self.error_code.to_dict()
        result["details"] = {**self.details, **(result.get("details") or {})}
        if self.cause:
            result["cause"] = str(self.cause)
        return result


def get_error(code: str) -> ErrorCode | None:
    return ERROR_CODE_MAP.get(code)


def create_error(
    code: str,
    message: str,
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    http_status: int = 500,
    remediation: str | None = None,
) -> ErrorCode:
    return ErrorCode(
        code=code,
        category=category,
        message=message,
        severity=severity,
        http_status=http_status,
        remediation=remediation,
    )
