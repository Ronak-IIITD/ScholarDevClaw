from __future__ import annotations

from pathlib import Path
from typing import Any

PLUGIN_METADATA = {
    "name": "security",
    "version": "1.0.0",
    "description": "Security-focused validator for patches",
    "author": "ScholarDevClaw",
    "plugin_type": "validator",
}


class SecurityValidator:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "security"

    def validate(self, repo_path: str, patch_result: dict[str, Any]) -> dict[str, Any]:
        issues = []

        new_files = patch_result.get("new_files", [])
        transformations = patch_result.get("transformations", [])

        dangerous_patterns = [
            (r"eval\s*\(", "Use of eval()"),
            (r"exec\s*\(", "Use of exec()"),
            (r"__import__\s*\(", "Dynamic import"),
            (r"subprocess\.call\s*\(\s*\[", "Shell command execution"),
            (r"os\.system\s*\(", "os.system call"),
            (r"pickle\.load\s*\(", "Unpickling untrusted data"),
            (r"yaml\.load\s*\(", "YAML deserialization"),
        ]

        for nf in new_files:
            content = nf.get("content", "")
            file_path = nf.get("path", "")

            for pattern, description in dangerous_patterns:
                import re

                if re.search(pattern, content):
                    issues.append(
                        {
                            "type": "security",
                            "severity": "error",
                            "message": f"{description} in {file_path}",
                            "file": file_path,
                        }
                    )

        for tf in transformations:
            content = tf.get("modified", "")
            file_path = tf.get("file", "")

            for pattern, description in dangerous_patterns:
                import re

                if re.search(pattern, content):
                    issues.append(
                        {
                            "type": "security",
                            "severity": "error",
                            "message": f"{description} in {file_path}",
                            "file": file_path,
                        }
                    )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "validator": "security",
        }

    def get_validation_type(self) -> str:
        return "security"


def get_plugin_instance():
    return SecurityValidator()
