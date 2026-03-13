"""
security — Built-in security validator plugin.

Scans generated patches for dangerous code patterns (eval, exec, subprocess,
pickle, etc.) and reports security issues.

Hooks into AFTER_GENERATE and PATCH_CREATED to automatically validate
generated code.
"""

from __future__ import annotations

import re
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

PLUGIN_METADATA = {
    "name": "security",
    "version": "2.0.0",
    "description": "Security-focused validator for patches",
    "author": "ScholarDevClaw",
    "plugin_type": "validator",
}

_DANGEROUS_PATTERNS = [
    (r"eval\s*\(", "Use of eval()"),
    (r"exec\s*\(", "Use of exec()"),
    (r"__import__\s*\(", "Dynamic import"),
    (r"subprocess\.call\s*\(\s*\[", "Shell command execution"),
    (r"os\.system\s*\(", "os.system call"),
    (r"pickle\.load\s*\(", "Unpickling untrusted data"),
    (r"yaml\.load\s*\(", "YAML deserialization"),
    (r"shutil\.rmtree\s*\(", "Recursive directory deletion"),
    (r"os\.remove\s*\(", "File deletion"),
    (r"ctypes\.", "C-level memory access via ctypes"),
]


class SecurityValidator:
    """Scans code for dangerous patterns.

    Also registers hooks so that patches are automatically scanned during
    pipeline execution without needing explicit ``validate()`` calls.
    """

    HOOK_POINTS = [
        HookPoint.AFTER_GENERATE.value,
        HookPoint.PATCH_CREATED.value,
    ]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._last_results: list[dict[str, Any]] = []

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "security"

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.register(
            HookPoint.AFTER_GENERATE,
            self._on_after_generate,
            plugin_name=self.get_name(),
            priority=40,  # Before auto_lint (50).
        )
        registry.register(
            HookPoint.PATCH_CREATED,
            self._on_patch_created,
            plugin_name=self.get_name(),
            priority=40,
        )

    def teardown(self) -> None:
        self._last_results.clear()

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _on_after_generate(self, event: HookEvent) -> None:
        """Scan new_files and transformations after generation."""
        new_files = event.payload.get("new_files", [])
        transformations = event.payload.get("transformations", [])
        issues = self._scan_artifacts(new_files, transformations)
        if issues:
            event.payload["security_issues"] = issues

    def _on_patch_created(self, event: HookEvent) -> None:
        """Scan patch artifacts."""
        new_files = event.payload.get("new_files", [])
        transformations = event.payload.get("transformations", [])
        issues = self._scan_artifacts(new_files, transformations)
        if issues:
            event.payload["security_issues"] = issues

    # ------------------------------------------------------------------
    # Core validation logic
    # ------------------------------------------------------------------

    def validate(self, repo_path: str, patch_result: dict[str, Any]) -> dict[str, Any]:
        """Validate a patch result dict for security issues."""
        new_files = patch_result.get("new_files", [])
        transformations = patch_result.get("transformations", [])
        issues = self._scan_artifacts(new_files, transformations)

        result = {
            "passed": len(issues) == 0,
            "issues": issues,
            "validator": "security",
        }
        self._last_results.append(result)
        return result

    def get_validation_type(self) -> str:
        return "security"

    def _scan_artifacts(
        self,
        new_files: list[dict[str, Any]],
        transformations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Scan new files and transformations for dangerous patterns."""
        issues: list[dict[str, Any]] = []

        for nf in new_files:
            content = nf.get("content", "")
            file_path = nf.get("path", "")
            self._scan_content(content, file_path, issues)

        for tf in transformations:
            content = tf.get("modified", "")
            file_path = tf.get("file", "")
            self._scan_content(content, file_path, issues)

        return issues

    @staticmethod
    def _scan_content(content: str, file_path: str, issues: list[dict[str, Any]]) -> None:
        for pattern, description in _DANGEROUS_PATTERNS:
            if re.search(pattern, content):
                issues.append(
                    {
                        "type": "security",
                        "severity": "error",
                        "message": f"{description} in {file_path}",
                        "file": file_path,
                    }
                )


def get_plugin_instance() -> SecurityValidator:
    return SecurityValidator()
