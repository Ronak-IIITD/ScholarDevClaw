"""
javalang — Built-in Java/Kotlin analyzer plugin.

Detects Java/Kotlin files and common frameworks (Spring, Android, Gradle).

Hooks into AFTER_ANALYZE to enrich the analysis payload when Java/Kotlin
files are detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

PLUGIN_METADATA = {
    "name": "javalang",
    "version": "2.0.0",
    "description": "Java/Kotlin analyzer using tree-sitter-java",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class JavaLangAnalyzer:
    """Analyzes Java/Kotlin repositories."""

    HOOK_POINTS = [HookPoint.AFTER_ANALYZE.value]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "javalang"

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.register(
            HookPoint.AFTER_ANALYZE,
            self._on_after_analyze,
            plugin_name=self.get_name(),
            priority=80,
        )

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Hook callback
    # ------------------------------------------------------------------

    def _on_after_analyze(self, event: HookEvent) -> None:
        """Enrich analysis with Java/Kotlin data if detected."""
        languages = event.payload.get("languages", [])
        repo_path = event.metadata.get("repo_path", "")
        has_jvm = any(lang in languages for lang in ("java", "kotlin"))
        if not has_jvm and repo_path:
            path = Path(repo_path)
            if not any(path.rglob("*.java")) and not any(path.rglob("*.kt")):
                return
        if repo_path:
            java_info = self.analyze(repo_path)
            event.payload.setdefault("plugin_analysis", {})[self.get_name()] = java_info

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)
        java_files = list(path.rglob("*.java"))
        kotlin_files = list(path.rglob("*.kt"))
        frameworks: list[str] = []

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


def get_plugin_instance() -> JavaLangAnalyzer:
    return JavaLangAnalyzer()
