from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scholardevclaw.patch_generation.generator import PatchGenerator


LogCallback = Callable[[str], None]


@dataclass(slots=True)
class CriticResult:
    ok: bool
    title: str
    payload: dict[str, Any]
    logs: list[str]
    error: str | None = None


def _log(logs: list[str], message: str, log_callback: LogCallback | None = None) -> None:
    logs.append(message)
    if log_callback is not None:
        log_callback(message)


def run_critic(
    repo_path: str,
    spec_name: str | None = None,
    *,
    patch_result: dict[str, Any] | None = None,
    log_callback: LogCallback | None = None,
) -> CriticResult:
    from pathlib import Path

    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    _log(logs, f"Running critic analysis for: {repo_path}", log_callback)

    try:
        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        if patch_result is None and spec_name:
            _log(logs, "Analyzing repository for critic review...", log_callback)
            analyzer = TreeSitterAnalyzer(path)
            analysis = analyzer.analyze()

            extractor = ResearchExtractor()
            spec = extractor.get_spec(spec_name)
            if spec is None:
                raise ValueError(f"Unknown spec: {spec_name}")

            engine = MappingEngine(analysis.__dict__, spec)
            mapping = engine.map()

            generator = PatchGenerator(path)
            patch = generator.generate(
                {
                    "targets": [
                        {
                            "file": t.file,
                            "line": t.line,
                            "current_code": t.current_code,
                            "replacement_required": t.replacement_required,
                            "context": t.context,
                        }
                        for t in mapping.targets
                    ],
                    "strategy": mapping.strategy,
                    "confidence": mapping.confidence,
                }
            )

            patch_result = {
                "new_files": [{"path": f.path, "content": f.content} for f in patch.new_files],
                "transformations": [
                    {
                        "file": t.file,
                        "original": t.original,
                        "modified": t.modified,
                        "changes": t.changes,
                    }
                    for t in patch.transformations
                ],
            }

        if patch_result is None:
            raise ValueError("No patch result provided or generated")

        all_issues: list[dict[str, Any]] = []
        all_warnings: list[dict[str, Any]] = []
        checks_passed: list[str] = []

        new_files = patch_result.get("new_files", [])
        for nf in new_files:
            file_path = nf.get("path", "")
            content = nf.get("content", "")

            syntax_issues = _check_syntax(content, file_path)
            all_issues.extend(syntax_issues)

            import_issues = _check_imports(content, file_path, path)
            all_issues.extend(import_issues)

            antipattern_issues = _check_antipatterns(content, file_path)
            all_warnings.extend(antipattern_issues)

        transformations = patch_result.get("transformations", [])
        for tf in transformations:
            file_path = tf.get("file", "")
            modified = tf.get("modified", "")

            transform_issues = _check_transform_safety(modified, file_path)
            all_issues.extend(transform_issues)

        safety_checks = _check_safety(path, patch_result)
        all_issues.extend(safety_checks.get("issues", []))
        all_warnings.extend(safety_checks.get("warnings", []))

        if not all_issues:
            checks_passed.append("Syntax validation passed")
            checks_passed.append("Import validation passed")
            checks_passed.append("Transform safety checks passed")
            checks_passed.append("Security checks passed")

        payload = {
            "repo_path": str(path),
            "spec": spec_name,
            "issues": all_issues,
            "warnings": all_warnings,
            "checks_passed": checks_passed,
            "issue_count": len(all_issues),
            "warning_count": len(all_warnings),
            "summary": "pass" if not all_issues else "fail",
            "severity_counts": {
                "error": len([i for i in all_issues if i.get("severity") == "error"]),
                "warning": len([i for i in all_issues if i.get("severity") == "warning"]),
            },
        }

        if all_issues:
            _log(
                logs,
                f"Critic found {len(all_issues)} issues, {len(all_warnings)} warnings",
                log_callback,
            )
        else:
            _log(logs, "Critic passed: no issues found", log_callback)

        return CriticResult(
            ok=len(all_issues) == 0,
            title="Code Critic",
            payload=payload,
            logs=logs,
        )
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return CriticResult(
            ok=False,
            title="Code Critic",
            payload={},
            logs=logs,
            error=str(exc),
        )


def _check_syntax(content: str, file_path: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    if not content:
        return issues

    try:
        ast.parse(content)
    except SyntaxError as e:
        issues.append(
            {
                "type": "syntax_error",
                "severity": "error",
                "file": file_path,
                "message": f"Syntax error at line {e.lineno}: {e.msg}",
                "line": e.lineno,
            }
        )
    except ValueError as e:
        issues.append(
            {
                "type": "parse_error",
                "severity": "error",
                "file": file_path,
                "message": f"Parse error: {str(e)}",
            }
        )

    return issues


def _check_imports(content: str, file_path: str, repo_path: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    imports = re.findall(r"^import\s+(\S+)|^from\s+(\S+)\s+import", content, re.MULTILINE)

    for imp in imports:
        module = imp[0] or imp[1]

        if module.startswith("."):
            continue

        stdlib_modules = {
            "os",
            "sys",
            "re",
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "pathlib",
            "typing",
            "abc",
            "copy",
            "pickle",
            "shutil",
            "tempfile",
            "argparse",
            "logging",
            "traceback",
        }

        common_third_party = {
            "torch",
            "tensorflow",
            "numpy",
            "pandas",
            "sklearn",
            "transformers",
            "flax",
            "jax",
            "einops",
            "tqdm",
            "requests",
            "fastapi",
            "pydantic",
            "pytest",
            "scipy",
        }

        if (
            module.split(".")[0] not in stdlib_modules
            and module.split(".")[0] not in common_third_party
        ):
            if not (repo_path / module.replace(".", "/") + ".py").exists():
                if not (repo_path / module.split(".")[0]).exists():
                    issues.append(
                        {
                            "type": "missing_import",
                            "severity": "warning",
                            "file": file_path,
                            "message": f"Potential missing import: '{module}'",
                        }
                    )

    return issues


def _check_antipatterns(content: str, file_path: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    if not content:
        return warnings

    if re.search(r"for\s+\w+\s+in\s+range\(len\(", content):
        warnings.append(
            {
                "type": "antipattern",
                "severity": "warning",
                "file": file_path,
                "message": "Consider using enumerate() instead of range(len())",
            }
        )

    if re.search(r"==\s*True|==\s*False", content):
        warnings.append(
            {
                "type": "antipattern",
                "severity": "warning",
                "file": file_path,
                "message": "Use 'if x:' or 'if not x:' instead of '== True' or '== False'",
            }
        )

    if re.search(r"except\s*:", content):
        warnings.append(
            {
                "type": "antipattern",
                "severity": "warning",
                "file": file_path,
                "message": "Bare except clause - specify exception type",
            }
        )

    if re.search(r"global\s+\w+", content):
        warnings.append(
            {
                "type": "antipattern",
                "severity": "warning",
                "file": file_path,
                "message": "Use of 'global' - consider class attributes or dependency injection",
            }
        )

    if re.search(r"eval\(|exec\(", content):
        warnings.append(
            {
                "type": "security",
                "severity": "error",
                "file": file_path,
                "message": "Use of eval() or exec() - potential security risk",
            }
        )

    if "print(" in content and "logging" not in content:
        warnings.append(
            {
                "type": "style",
                "severity": "warning",
                "file": file_path,
                "message": "Consider using logging instead of print() for production code",
            }
        )

    return warnings


def _check_transform_safety(modified: str, file_path: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    if not modified:
        return issues

    if modified.count("(") != modified.count(")"):
        issues.append(
            {
                "type": "unbalanced_brackets",
                "severity": "error",
                "file": file_path,
                "message": "Unbalanced parentheses in transformation",
            }
        )

    if modified.count("[") != modified.count("]"):
        issues.append(
            {
                "type": "unbalanced_brackets",
                "severity": "error",
                "file": file_path,
                "message": "Unbalanced square brackets in transformation",
            }
        )

    if modified.count("{") != modified.count("}"):
        issues.append(
            {
                "type": "unbalanced_brackets",
                "severity": "error",
                "file": file_path,
                "message": "Unbalanced curly braces in transformation",
            }
        )

    return issues


def _check_safety(repo_path: Path, patch_result: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    protected_patterns = [
        r"__import__\s*\(",
        r"import\s+os\s+as\s+_",
        r"subprocess\.call\s*\(\s*\[",
    ]

    new_files = patch_result.get("new_files", [])
    for nf in new_files:
        content = nf.get("content", "")
        file_path = nf.get("path", "")

        for pattern in protected_patterns:
            if re.search(pattern, content):
                issues.append(
                    {
                        "type": "suspicious_import",
                        "severity": "error",
                        "file": file_path,
                        "message": f"Suspicious import pattern detected: {pattern}",
                    }
                )

        if content.count("TODO") > 3:
            warnings.append(
                {
                    "type": "incomplete_code",
                    "severity": "warning",
                    "file": file_path,
                    "message": f"Multiple TODO markers found ({content.count('TODO')})",
                }
            )

    return {"issues": issues, "warnings": warnings}
