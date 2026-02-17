from __future__ import annotations

from pathlib import Path
from typing import Any

PLUGIN_METADATA = {
    "name": "javalang",
    "version": "1.0.0",
    "description": "Java/Kotlin analyzer using tree-sitter-java",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class JavaLangAnalyzer:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "javalang"

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)

        java_files = list(path.rglob("*.java"))
        kotlin_files = list(path.rglob("*.kt"))

        frameworks = []

        if any(
            "spring" in f.read_text().lower()[:1000] for f in java_files if f.stat().st_size < 50000
        ):
            frameworks.append("spring")
        if any(
            "android" in f.read_text().lower()[:1000]
            for f in java_files
            if f.stat().st_size < 50000
        ):
            frameworks.append("android")
        if any("gradle" in f.name for f in (path.rglob("build.gradle*") or [])):
            frameworks.append("gradle")

        return {
            "languages": ["java", "kotlin"] if kotlin_files else ["java"],
            "frameworks": list(set(frameworks)),
            "file_counts": {
                "java": len(java_files),
                "kotlin": len(kotlin_files),
            },
        }

    def get_supported_languages(self) -> list[str]:
        return ["java", "kotlin"]


def get_plugin_instance():
    return JavaLangAnalyzer()
