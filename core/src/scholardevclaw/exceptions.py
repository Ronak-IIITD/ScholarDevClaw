from __future__ import annotations

"""Shared exception hierarchy for ScholarDevClaw pipeline stages."""


class ScholarDevClawError(Exception):
    """Base class for domain-specific failures."""


class IngestionError(ScholarDevClawError):
    """Raised when paper ingestion or retrieval fails."""


class PaperFetchError(IngestionError):
    """Raised when a paper source cannot be fetched successfully."""


class PaperNotAccessibleError(PaperFetchError):
    """Raised when a paper exists but is not accessible for download."""


class PaperSourceResolutionError(PaperFetchError):
    """Raised when a paper source cannot be resolved into a supported type."""


class UnderstandingError(ScholarDevClawError):
    """Raised when paper comprehension fails."""


class PlanningError(ScholarDevClawError):
    """Raised when implementation planning fails."""


class GenerationError(ScholarDevClawError):
    """Raised when code generation fails."""


class ExecutionError(ScholarDevClawError):
    """Raised when execution or validation fails."""


class SandboxError(ExecutionError):
    """Raised for sandbox runtime or isolation failures."""


class KnowledgeBaseError(ScholarDevClawError):
    """Raised when knowledge-base storage or retrieval fails."""
