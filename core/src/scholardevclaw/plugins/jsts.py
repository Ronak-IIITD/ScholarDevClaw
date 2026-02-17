from __future__ import annotations

from pathlib import Path
from typing import Any

PLUGIN_METADATA = {
    "name": "jsts",
    "version": "1.0.0",
    "description": "JavaScript/TypeScript analyzer",
    "author": "ScholarDevClaw",
    "plugin_type": "analyzer",
}


class JSTSAnalyzer:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "jsts"

    def analyze(self, repo_path: str) -> dict[str, Any]:
        path = Path(repo_path)

        js_files = list(path.rglob("*.js"))
        ts_files = list(path.rglob("*.ts"))
        jsx_files = list(path.rglob("*.jsx"))
        tsx_files = list(path.rglob("*.tsx"))

        frameworks = []

        package_json = path / "package.json"
        if package_json.exists():
            try:
                import json

                with open(package_json) as f:
                    pkg = json.load(f)
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


def get_plugin_instance():
    return JSTSAnalyzer()
