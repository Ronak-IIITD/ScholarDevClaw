from __future__ import annotations

from pathlib import Path
from typing import Any

PLUGIN_METADATA = {
    "name": "rustlang",
    "version": "1.0.0",
    "description": "Rust analyzer using tree-sitter-rust",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class RustLangAnalyzer:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "rustlang"

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)

        rust_files = list(path.rglob("*.rs"))

        frameworks = []

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


def get_plugin_instance():
    return RustLangAnalyzer()
