from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_value: Any = None


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class Validator:
    """Base validator class."""

    def validate(self, value: Any) -> ValidationResult:
        raise NotImplementedError


class RequiredValidator(Validator):
    """Validates that value is not None or empty."""

    def __init__(self, message: str = "This field is required"):
        self.message = message

    def validate(self, value: Any) -> ValidationResult:
        if value is None:
            return ValidationResult(valid=False, errors=[self.message])
        if isinstance(value, str) and not value.strip():
            return ValidationResult(valid=False, errors=[self.message])
        return ValidationResult(valid=True, sanitized_value=value)


class PathValidator(Validator):
    """Validates file/directory paths."""

    def __init__(
        self,
        must_exist: bool = True,
        must_be_dir: bool = False,
        must_be_file: bool = False,
        allow_absolute: bool = True,
        allow_relative: bool = True,
    ):
        self.must_exist = must_exist
        self.must_be_dir = must_be_dir
        self.must_be_file = must_be_file
        self.allow_absolute = allow_absolute
        self.allow_relative = allow_relative

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(valid=False, errors=["Path must be a string"])

        errors: list[str] = []
        warnings: list[str] = []

        path = Path(value).expanduser().resolve()

        if path.is_absolute() and not self.allow_absolute:
            errors.append("Absolute paths are not allowed")
        if not path.is_absolute() and not self.allow_relative:
            errors.append("Relative paths are not allowed")

        if self.must_exist and not path.exists():
            errors.append(f"Path does not exist: {value}")

        if self.must_be_dir and path.exists() and not path.is_dir():
            errors.append(f"Path is not a directory: {value}")

        if self.must_be_file and path.exists() and not path.is_file():
            errors.append(f"Path is not a file: {value}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=str(path),
        )


class StringValidator(Validator):
    """Validates string values."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        allowed_chars: str | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_chars = allowed_chars

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(valid=False, errors=["Value must be a string"])

        errors: list[str] = []

        if self.min_length is not None and len(value) < self.min_length:
            errors.append(f"String must be at least {self.min_length} characters")

        if self.max_length is not None and len(value) > self.max_length:
            errors.append(f"String must be at most {self.max_length} characters")

        if self.pattern and not self.pattern.match(value):
            errors.append(f"String does not match required pattern")

        if self.allowed_chars:
            invalid_chars = set(value) - set(self.allowed_chars)
            if invalid_chars:
                errors.append(f"Invalid characters: {invalid_chars}")

        sanitized = value.strip() if value else value

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=sanitized,
        )


class EnumValidator(Validator):
    """Validates against a set of allowed values."""

    def __init__(self, allowed: list[Any], case_insensitive: bool = False):
        self.allowed = allowed
        self.case_insensitive = case_insensitive

    def validate(self, value: Any) -> ValidationResult:
        if self.case_insensitive and isinstance(value, str):
            check_value = value.lower()
            allowed_check = [str(a).lower() for a in self.allowed]
        else:
            check_value = value
            allowed_check = self.allowed

        if check_value not in allowed_check:
            return ValidationResult(
                valid=False,
                errors=[f"Value must be one of: {self.allowed}"],
            )

        return ValidationResult(valid=True, sanitized_value=value)


class RangeValidator(Validator):
    """Validates numeric ranges."""

    def __init__(
        self,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        integer_only: bool = False,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only

    def validate(self, value: Any) -> ValidationResult:
        errors: list[str] = []

        if not isinstance(value, (int, float)):
            return ValidationResult(valid=False, errors=["Value must be a number"])

        if self.integer_only and not isinstance(value, int):
            errors.append("Value must be an integer")

        if self.min_value is not None and value < self.min_value:
            errors.append(f"Value must be at least {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            errors.append(f"Value must be at most {self.max_value}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=value,
        )


class CompositeValidator(Validator):
    """Combines multiple validators."""

    def __init__(self, validators: list[Validator], stop_on_first_error: bool = True):
        self.validators = validators
        self.stop_on_first_error = stop_on_first_error

    def validate(self, value: Any) -> ValidationResult:
        all_errors: list[str] = []
        all_warnings: list[str] = []
        sanitized = value

        for validator in self.validators:
            result = validator.validate(sanitized)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if result.valid:
                sanitized = result.sanitized_value
            elif self.stop_on_first_error:
                break

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_value=sanitized,
        )


class InputValidator:
    """Input validation for API/CLI inputs."""

    def __init__(self):
        self._field_validators: dict[str, Validator] = {}

    def field(self, name: str, validator: Validator) -> "InputValidator":
        self._field_validators[name] = validator
        return self

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        all_errors: list[str] = []
        all_warnings: list[str] = []
        sanitized: dict[str, Any] = {}

        for name, validator in self._field_validators.items():
            value = data.get(name)
            result = validator.validate(value)

            if not result.valid:
                all_errors.extend([f"{name}: {e}" for e in result.errors])
            else:
                sanitized[name] = result.sanitized_value

            all_warnings.extend([f"{name}: {w}" for w in result.warnings])

        for name in data:
            if name not in self._field_validators:
                all_warnings.append(f"Unknown field: {name}")

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_value=sanitized,
        )


def validate_repo_path(path: str) -> ValidationResult:
    """Validate repository path input."""
    return CompositeValidator(
        [
            RequiredValidator("Repository path is required"),
            PathValidator(must_exist=True, must_be_dir=True),
        ]
    ).validate(path)


def validate_spec_name(name: str) -> ValidationResult:
    """Validate spec name input."""
    return CompositeValidator(
        [
            RequiredValidator("Spec name is required"),
            StringValidator(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$"),
        ]
    ).validate(name)


def validate_output_dir(path: str | None) -> ValidationResult:
    """Validate output directory input."""
    if path is None:
        return ValidationResult(valid=True, sanitized_value=None)

    return PathValidator(must_exist=False, must_be_dir=True).validate(path)
