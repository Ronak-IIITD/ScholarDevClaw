from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSION = "1.0.0"


@dataclass(slots=True)
class CompatibilityReport:
    is_compatible: bool
    issues: list[str]
    warnings: list[str]
    notes: list[str]


def parse_semver(version: str) -> tuple[int, int, int] | None:
    raw = (version or "").strip()
    if not raw:
        return None

    parts = raw.split(".")
    if not 1 <= len(parts) <= 3:
        return None

    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None

    while len(numbers) < 3:
        numbers.append(0)

    major, minor, patch = numbers[:3]
    if major < 0 or minor < 0 or patch < 0:
        return None
    return major, minor, patch


def with_meta(payload: dict[str, Any], payload_type: str) -> dict[str, Any]:
    merged = dict(payload)
    merged["_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "payload_type": payload_type,
    }
    return merged


def evaluate_payload_compatibility(
    payload: dict[str, Any],
    *,
    expected_types: set[str] | None = None,
    supported_version: str = SUPPORTED_SCHEMA_VERSION,
) -> CompatibilityReport:
    issues: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    meta = payload.get("_meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return CompatibilityReport(
            is_compatible=False,
            issues=["Missing payload metadata (_meta)."],
            warnings=[],
            notes=["Regenerate payload with a schema-aware pipeline version."],
        )

    schema_version = str(meta.get("schema_version", "")).strip()
    payload_type = str(meta.get("payload_type", "")).strip()

    payload_semver = parse_semver(schema_version)
    supported_semver = parse_semver(supported_version)

    if payload_semver is None:
        issues.append(f"Invalid schema_version format: {schema_version or '<empty>'}")
    if supported_semver is None:
        issues.append(f"Invalid supported schema version config: {supported_version}")

    if payload_type:
        if expected_types and payload_type not in expected_types:
            issues.append(f"Unexpected payload_type: {payload_type}")
    else:
        issues.append("Missing payload_type in payload metadata.")

    if payload_semver is not None and supported_semver is not None:
        p_major, p_minor, p_patch = payload_semver
        s_major, s_minor, s_patch = supported_semver

        if p_major != s_major:
            issues.append(
                f"Incompatible schema major version: payload={schema_version} supported={supported_version}"
            )
            notes.append(
                "Use a matching major schema consumer or add an explicit migration adapter."
            )
        else:
            if p_minor > s_minor:
                warnings.append(
                    f"Payload minor version is newer ({schema_version}) than supported ({supported_version})."
                )
                notes.append(
                    "Proceed with caution; newer optional fields may be ignored."
                )
            elif p_minor < s_minor:
                notes.append(
                    f"Payload minor version is older ({schema_version}); compatibility fallback applied."
                )
            elif p_patch > s_patch:
                notes.append(
                    f"Payload patch version is newer ({schema_version}); expected to remain compatible."
                )

    return CompatibilityReport(
        is_compatible=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        notes=notes,
    )
