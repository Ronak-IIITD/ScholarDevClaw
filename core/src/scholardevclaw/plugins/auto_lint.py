"""
auto_lint — Built-in hook plugin that runs Ruff on generated patch files.

Hooks into AFTER_GENERATE and PATCH_CREATED to automatically lint and
optionally fix generated code before it is written or validated.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

logger = logging.getLogger(__name__)

PLUGIN_METADATA = {
    "name": "auto_lint",
    "version": "1.0.0",
    "description": "Auto-lint generated patches with Ruff",
    "author": "ScholarDevClaw",
    "plugin_type": "hook",
}


class AutoLintPlugin:
    """Lints generated patches using Ruff.

    Configuration keys (via ``plugin_state.json`` or ``set_plugin_config``):
        fix (bool): Automatically apply safe fixes (default ``True``).
        select (str): Comma-separated Ruff rule selectors (default ``"E,W,F,I"``).
        ignore (str): Comma-separated Ruff rules to ignore (default ``""``).
    """

    HOOK_POINTS = [
        HookPoint.AFTER_GENERATE.value,
        HookPoint.PATCH_CREATED.value,
    ]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._lint_results: list[dict[str, Any]] = []

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "auto_lint"

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.register(
            HookPoint.AFTER_GENERATE,
            self._on_after_generate,
            plugin_name=self.get_name(),
            priority=50,
        )
        registry.register(
            HookPoint.PATCH_CREATED,
            self._on_patch_created,
            plugin_name=self.get_name(),
            priority=50,
        )

    def teardown(self) -> None:
        self._lint_results.clear()

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _on_after_generate(self, event: HookEvent) -> None:
        """Lint new_files from the generation payload."""
        new_files = event.payload.get("new_files", [])
        if not new_files:
            return

        results = []
        for file_info in new_files:
            content = file_info.get("content", "")
            path = file_info.get("path", "generated.py")
            if not path.endswith(".py"):
                continue
            lint_result = self._lint_content(content, path)
            results.append(lint_result)
            if lint_result.get("fixed_content"):
                file_info["content"] = lint_result["fixed_content"]

        event.payload["auto_lint_results"] = results
        self._lint_results.extend(results)

    def _on_patch_created(self, event: HookEvent) -> None:
        """Lint transformation modified code."""
        transformations = event.payload.get("transformations", [])
        if not transformations:
            return

        results = []
        for tf in transformations:
            modified = tf.get("modified", "")
            path = tf.get("file", "transformed.py")
            if not path.endswith(".py"):
                continue
            lint_result = self._lint_content(modified, path)
            results.append(lint_result)
            if lint_result.get("fixed_content"):
                tf["modified"] = lint_result["fixed_content"]

        event.payload["auto_lint_transform_results"] = results
        self._lint_results.extend(results)

    # ------------------------------------------------------------------
    # Lint logic
    # ------------------------------------------------------------------

    def _lint_content(self, content: str, filename: str) -> dict[str, Any]:
        """Lint a Python code string using Ruff and return results."""
        apply_fix = self.config.get("fix", True)
        select = self.config.get("select", "E,W,F,I")
        ignore = self.config.get("ignore", "")

        result: dict[str, Any] = {
            "file": filename,
            "issues": [],
            "fixed": False,
            "fixed_content": None,
        }

        if not content.strip():
            return result

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Check for issues.
            cmd = ["ruff", "check", tmp_path, "--output-format=json"]
            if select:
                cmd.extend(["--select", select])
            if ignore:
                cmd.extend(["--ignore", ignore])

            check = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if check.stdout.strip():
                import json

                try:
                    issues = json.loads(check.stdout)
                    result["issues"] = [
                        {
                            "code": i.get("code", ""),
                            "message": i.get("message", ""),
                            "line": i.get("location", {}).get("row", 0),
                        }
                        for i in issues
                    ]
                except Exception:
                    pass

            # Apply fixes if configured.
            if apply_fix and result["issues"]:
                fix_cmd = ["ruff", "check", "--fix", tmp_path]
                if select:
                    fix_cmd.extend(["--select", select])
                if ignore:
                    fix_cmd.extend(["--ignore", ignore])

                subprocess.run(
                    fix_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                fixed_content = Path(tmp_path).read_text()
                if fixed_content != content:
                    result["fixed"] = True
                    result["fixed_content"] = fixed_content

            # Cleanup.
            Path(tmp_path).unlink(missing_ok=True)

        except FileNotFoundError:
            logger.debug("Ruff not found — auto_lint skipped for %s", filename)
        except Exception as exc:
            logger.debug("auto_lint failed for %s: %s", filename, exc)

        return result

    @property
    def lint_results(self) -> list[dict[str, Any]]:
        return list(self._lint_results)


def get_plugin_instance() -> AutoLintPlugin:
    return AutoLintPlugin()
