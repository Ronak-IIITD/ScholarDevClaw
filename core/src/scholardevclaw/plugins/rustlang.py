"""
rustlang — Built-in Rust analyzer plugin.

Detects Rust files, Cargo configuration, and common frameworks (wasm, tokio)
in a repository.

Hooks into AFTER_ANALYZE to enrich the analysis payload with Rust-specific
details when Rust files are detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

PLUGIN_METADATA = {
    "name": "rustlang",
    "version": "2.0.0",
    "description": "Rust analyzer using tree-sitter-rust",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class RustLangAnalyzer:
    """Analyzes Rust repositories for language and framework usage."""

    HOOK_POINTS = [HookPoint.AFTER_ANALYZE.value]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "rustlang"

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
        """Enrich analysis with Rust-specific data if Rust detected."""
        languages = event.payload.get("languages", [])
        repo_path = event.metadata.get("repo_path", "")
        if "rust" not in languages and repo_path:
            # Quick check: does the repo have Rust files?
            path = Path(repo_path)
            if not any(path.rglob("*.rs")):
                return
        if repo_path:
            rust_info = self.analyze(repo_path)
            event.payload.setdefault("plugin_analysis", {})[self.get_name()] = rust_info

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)
        rust_files = list(path.rglob("*.rs"))
        frameworks: list[str] = []

        cargo_toml = path / "Cargo.toml"
        if cargo_toml.exists():
            frameworks.append("cargo")

        has_wasm = any("wasm" in f.name for f in rust_files)
        if has_wasm:
            frameworks.append("wasm")

        has_async = any(
            "async fn" in f.read_text() for f in rust_files[:10] if f.stat().st_size < 50000
        )
        if has_async:
            frameworks.append("tokio")

        return {
            "languages": ["rust"],
            "frameworks": list(set(frameworks)),
            "file_counts": {
                "rust": len(rust_files),
            },
        }

    def get_supported_languages(self) -> list[str]:
        return ["rust"]


def get_plugin_instance() -> RustLangAnalyzer:
    return RustLangAnalyzer()
