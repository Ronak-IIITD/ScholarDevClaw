"""
jsts — Built-in JavaScript/TypeScript analyzer plugin.

Detects JS/TS files and common frameworks (React, Vue, Angular, Next.js,
Express, NestJS) by reading ``package.json`` dependencies.

Hooks into AFTER_ANALYZE to enrich the analysis payload when JS/TS files
are detected.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

PLUGIN_METADATA = {
    "name": "jsts",
    "version": "2.0.0",
    "description": "JavaScript/TypeScript analyzer",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class JSTSAnalyzer:
    """Analyzes JavaScript/TypeScript repositories."""

    HOOK_POINTS = [HookPoint.AFTER_ANALYZE.value]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "jsts"

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
        """Enrich analysis with JS/TS data if detected."""
        languages = event.payload.get("languages", [])
        repo_path = event.metadata.get("repo_path", "")
        has_js = any(lang in languages for lang in ("javascript", "typescript"))
        if not has_js and repo_path:
            path = Path(repo_path)
            if not (path / "package.json").exists():
                return
        if repo_path:
            jsts_info = self.analyze(repo_path)
            event.payload.setdefault("plugin_analysis", {})[self.get_name()] = jsts_info

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)
        js_files = list(path.rglob("*.js"))
        ts_files = list(path.rglob("*.ts"))
        jsx_files = list(path.rglob("*.jsx"))
        tsx_files = list(path.rglob("*.tsx"))
        frameworks: list[str] = []

        package_json = path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    pkg = _json.load(f)
                    deps = pkg.get("dependencies", {})
                    if "react" in deps:
                        frameworks.append("react")
                    if "vue" in deps:
                        frameworks.append("vue")
                    if "angular" in deps:
                        frameworks.append("angular")
                    if "next" in deps:
                        frameworks.append("nextjs")
                    if "express" in deps:
                        frameworks.append("express")
                    if "nest" in deps:
                        frameworks.append("nestjs")
            except Exception:
                pass

        has_ts = len(ts_files) > 0 or len(tsx_files) > 0

        return {
            "languages": ["typescript"] if has_ts else ["javascript"],
            "frameworks": list(set(frameworks)),
            "file_counts": {
                "js": len(js_files),
                "ts": len(ts_files),
                "jsx": len(jsx_files),
                "tsx": len(tsx_files),
            },
        }

    def get_supported_languages(self) -> list[str]:
        return ["javascript", "typescript"]


def get_plugin_instance() -> JSTSAnalyzer:
    return JSTSAnalyzer()
