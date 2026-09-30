"""nanogpt-opt — lean 4-command CLI over the frozen 10-spec surface.

Commands
    analyze   Scan a Python transformer repo; report applicable frozen specs.
    suggest   Map applicable specs onto concrete file/line targets.
    apply     Preview (default) or apply the patch; ``--revert`` restores.
    validate  Honest verification: compile / tests / benchmark, with explicit
              ``skipped`` states — a stage that did not run never reports pass.

Design rules:
- Additive entry point; the legacy ``scholardevclaw`` CLI is untouched.
- ``apply`` writes nothing without ``--write``; ``--revert`` restores the
  exact original bytes captured at apply time (no-clobber on later edits).
- No fabricated numbers: every stage reports pass / fail / skipped.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scholardevclaw.nano_analyze import analyze_repo
from scholardevclaw.nano_specs import list_supported_specs, load_spec

MANIFEST_DIR = ".nanogpt-opt"
MANIFEST_NAME = "manifest.json"
_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".nanogpt-opt",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
}
_TIMEOUT_TESTS = 300
_TIMEOUT_BENCH = 120


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _IGNORED_DIRS for part in rel_parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def _has_test_files(root: Path) -> bool:
    return any(p.name.startswith("test_") or p.name.endswith("_test.py") for p in _py_files(root))


def _bench_scripts(root: Path) -> list[Path]:
    return [
        p for p in _py_files(root) if p.name.startswith("benchmark") or p.name.startswith("bench")
    ]


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _spec_or_all(repo: Path, spec_arg: str | None) -> tuple[list[str], dict[str, Any]]:
    """Return (spec names to operate on, analysis payload)."""
    analysis = analyze_repo(repo)
    applicable = list(analysis.applicable_specs)
    supported = list_supported_specs()
    if spec_arg:
        if spec_arg not in supported:
            raise SystemExit(f"unknown spec '{spec_arg}' (supported: {', '.join(supported)})")
        specs = [spec_arg]
    else:
        specs = [name for name in supported if name in applicable]
    payload = {
        "repo": str(repo),
        "files_scanned": analysis.files_scanned,
        "models": analysis.models,
        "applicable_specs": specs,
        "supported_specs": supported,
    }
    return specs, payload


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        _err(f"not a directory: {repo}")
        return 2
    specs, payload = _spec_or_all(repo, args.spec)
    if args.json:
        _print_json(payload)
        return 0
    print(f"nanogpt-opt analyze: {repo}")
    print(f"  python files scanned : {payload['files_scanned']}")
    print(f"  models detected       : {', '.join(payload['models']) or 'none'}")
    print(f"  supported specs       : {len(payload['supported_specs'])}")
    print(f"  applicable specs      : {len(specs)}")
    for name in specs:
        spec = load_spec(name)
        algorithm = spec.get("algorithm", {}).get("name", name)
        print(f"    - {name} ({algorithm})")
    if not specs:
        print("    (none matched — try `nanogpt-opt suggest` after adding transformer code)")
    return 0


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------


def cmd_suggest(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        _err(f"not a directory: {repo}")
        return 2
    from scholardevclaw.application.pipeline import run_map

    specs, base = _spec_or_all(repo, args.spec)
    suggestions: list[dict[str, Any]] = []
    for name in specs:
        result = run_map(str(repo), name, use_cache=False)
        if not result.ok:
            suggestions.append({"spec": name, "ok": False, "error": result.error, "targets": []})
            continue
        targets = [
            {
                "file": t.get("file", ""),
                "line": t.get("line"),
                "context": t.get("context", {}),
            }
            for t in (result.payload.get("targets") or [])
        ]
        suggestions.append(
            {
                "spec": name,
                "ok": True,
                "algorithm": result.payload.get("algorithm", ""),
                "confidence": result.payload.get("confidence", 0),
                "strategy": result.payload.get("strategy", ""),
                "targets": targets,
            }
        )
    if args.json:
        _print_json({**base, "suggestions": suggestions})
        return 0
    print(f"nanogpt-opt suggest: {repo}")
    if not suggestions:
        print("  no applicable specs for this repository")
        return 0
    for item in suggestions:
        if not item["ok"]:
            print(f"  {item['spec']}: mapping failed ({item['error']})")
            continue
        print(
            f"  {item['spec']}: {len(item['targets'])} target(s), "
            f"confidence {item['confidence']}%, strategy {item['strategy']}"
        )
        for target in item["targets"]:
            context = target.get("context") or {}
            replacement = context.get("replacement", "?")
            print(f"    {target['file']}:{target['line']}  ->  {replacement}")
    return 0


# ---------------------------------------------------------------------------
# apply / revert
# ---------------------------------------------------------------------------


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_DIR / MANIFEST_NAME


def _unified_preview(path: str, original: str, modified: str) -> str:
    import difflib

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _apply_spec(repo: Path, spec: str) -> dict[str, Any]:
    from scholardevclaw.application.pipeline import run_generate

    result = run_generate(str(repo), spec)
    if not result.ok:
        return {"spec": spec, "ok": False, "error": result.error}
    return {
        "spec": spec,
        "ok": True,
        "algorithm": result.payload.get("algorithm", ""),
        "new_files": [
            {"path": f.get("path", ""), "content": f.get("content", "")}
            for f in (result.payload.get("new_files") or [])
        ],
        "transformations": list(result.payload.get("transformations") or []),
    }


def _spec_write_plan(
    repo: Path, item: dict[str, Any]
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Validate one spec's writes.

    Returns ``(conflicts, new_files_to_write, kept_existing)``. Created files
    are create-only: if a path already exists (our earlier output on an
    idempotent re-run, or a user's own file) it is kept untouched and never
    recorded for revert. Only transformation original-mismatches are true
    conflicts — those are the edits that would clobber existing content.
    """
    conflicts: list[str] = []
    for tr in item.get("transformations", []):
        target = (repo / tr["file"]).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            conflicts.append(f"{tr['file']} (escapes repository root)")
            continue
        if not target.exists():
            conflicts.append(f"{tr['file']} (missing)")
            continue
        if target.read_text() != tr.get("original", ""):
            conflicts.append(f"{tr['file']} (modified since generation)")
    writable_new: list[dict[str, str]] = []
    kept_existing: list[str] = []
    for f in item.get("new_files", []):
        target = (repo / f["path"]).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            conflicts.append(f"{f['path']} (escapes repository root)")
            continue
        if target.exists():
            kept_existing.append(f["path"])
            continue
        writable_new.append(f)
    return conflicts, writable_new, kept_existing


def _write_manifest(
    repo: Path,
    spec_names: list[str],
    created: list[dict[str, str]],
    transformations: list[dict[str, str]],
) -> Path:
    manifest_dir = repo / MANIFEST_DIR
    manifest_dir.mkdir(exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "specs": spec_names,
        "new_files": [f["path"] for f in created],
        "transformations": transformations,
    }
    path = _manifest_path(repo)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def cmd_apply(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        _err(f"not a directory: {repo}")
        return 2

    if args.revert:
        return _cmd_revert(repo, args.json)

    specs, base = _spec_or_all(repo, args.spec)

    if not args.write:
        # Dry run: generate per spec against the current file state. Previews
        # of different specs may overlap (nothing is written, no chaining).
        results = [_apply_spec(repo, name) for name in specs]
        created: list[dict[str, str]] = []
        transformations: list[dict[str, str]] = []
        for item in results:
            if not item.get("ok"):
                continue
            created.extend(item["new_files"])
            transformations.extend(item["transformations"])
        results_summary = [
            {
                "spec": item["spec"],
                "ok": item.get("ok", False),
                "error": item.get("error"),
                "algorithm": item.get("algorithm", ""),
                "new_file_count": len(item.get("new_files", [])),
                "transformation_count": len(item.get("transformations", [])),
            }
            for item in results
        ]
        if args.json:
            _print_json(
                {
                    **base,
                    "dry_run": True,
                    "results": results_summary,
                    "new_files": [f["path"] for f in created],
                    "changed_files": sorted({t["file"] for t in transformations}),
                    "diffs": {
                        tr["file"]: _unified_preview(
                            tr["file"], tr.get("original", ""), tr.get("modified", "")
                        )
                        for tr in transformations
                    },
                }
            )
            return 0
        print(f"nanogpt-opt apply (dry run): {repo}")
        for item in results:
            if not item.get("ok"):
                print(f"  {item['spec']}: generation failed ({item.get('error')})")
                continue
            print(
                f"  {item['spec']}: {len(item['new_files'])} new file(s), "
                f"{len(item['transformations'])} transformation(s)"
            )
        if not created and not transformations:
            print("  nothing to apply (repo already matches the selected specs, or no targets)")
            return 0
        print("  dry run — nothing written. Re-run with --write to apply.")
        for tr in transformations:
            print(_unified_preview(tr["file"], tr.get("original", ""), tr.get("modified", "")))
        if created:
            print("  new files that would be created:")
            for f in created:
                print(f"    + {f['path']}")
        return 0

    # --write: sequential generate -> conflict-check -> write per spec, so
    # later specs build on the files written by earlier ones (chained
    # originals; revert walks the chain backwards).
    applied_specs: list[str] = []
    created = []
    transformations = []
    results_summary = []
    for name in specs:
        item = _apply_spec(repo, name)  # reads the current disk state
        ok = bool(item.get("ok"))
        results_summary.append(
            {
                "spec": name,
                "ok": ok,
                "error": item.get("error"),
                "algorithm": item.get("algorithm", ""),
                "new_file_count": len(item.get("new_files", [])) if ok else 0,
                "transformation_count": len(item.get("transformations", [])) if ok else 0,
            }
        )
        if not ok:
            continue
        conflicts, writable_new, kept_existing = _spec_write_plan(repo, item)
        results_summary[-1]["new_file_count"] = len(writable_new)
        if kept_existing:
            results_summary[-1]["kept_existing_files"] = kept_existing
        if conflicts:
            # Persist what was already applied so it stays revertible.
            if transformations or created:
                _write_manifest(repo, applied_specs, created, transformations)
            if args.json:
                _print_json(
                    {
                        "error": f"conflict while applying spec '{name}'",
                        "conflicts": conflicts,
                        "partial": bool(transformations or created),
                    }
                )
            else:
                _err(f"conflict while applying spec '{name}':")
                for conflict in conflicts:
                    print(f"  - {conflict}", file=sys.stderr)
                if transformations or created:
                    print(
                        "partial apply — revert with: nanogpt-opt apply <repo> --revert",
                        file=sys.stderr,
                    )
            return 1
        for tr in item["transformations"]:
            (repo / tr["file"]).write_text(tr["modified"])
            transformations.append(tr)
        for f in writable_new:
            destination = repo / f["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f["content"])
            created.append(f)
        if item["transformations"] or writable_new:
            applied_specs.append(name)

    nothing_done = not transformations and not created
    manifest_path: Path | None = None
    if not nothing_done:
        manifest_path = _write_manifest(repo, applied_specs, created, transformations)
    if args.json:
        payload = {
            **base,
            "dry_run": False,
            "results": results_summary,
            "wrote_files": sorted({t["file"] for t in transformations}),
            "created_files": [f["path"] for f in created],
        }
        if manifest_path is not None:
            payload["manifest"] = str(manifest_path.relative_to(repo))
        _print_json(payload)
        return 0
    print(f"nanogpt-opt apply: {repo}")
    for entry in results_summary:
        if not entry["ok"]:
            print(f"  {entry['spec']}: generation failed ({entry.get('error')})")
        else:
            print(
                f"  {entry['spec']}: {entry['new_file_count']} new file(s), "
                f"{entry['transformation_count']} transformation(s)"
            )
    if nothing_done:
        print("  nothing to apply (repo already matches the selected specs, or no targets)")
        return 0
    print(f"  wrote {len(transformations)} file(s), created {len(created)} new file(s)")
    for rel in sorted({t["file"] for t in transformations}):
        print(f"    ~ {rel}")
    for f in created:
        print(f"    + {f['path']}")
    print(f"  manifest: {_manifest_path(repo).relative_to(repo)}")
    print("  revert with: nanogpt-opt apply <repo> --revert")
    return 0


def _cmd_revert(repo: Path, as_json: bool) -> int:
    manifest_file = _manifest_path(repo)
    if not manifest_file.exists():
        _err(f"no manifest at {manifest_file}")
        return 1
    manifest = json.loads(manifest_file.read_text())

    from scholardevclaw.patch_generation.generator import PatchGenerator

    # Chained originals: each transformation was generated against the state
    # left by the previous one, so revert walks the chain backwards.
    entries = [dict(t) for t in manifest.get("transformations", [])]
    restored = PatchGenerator.revert_transformations(repo, list(reversed(entries)))
    restored = list(dict.fromkeys(restored))  # dedupe files restored via the chain
    removed: list[str] = []
    kept: list[str] = []
    for rel in manifest.get("new_files", []):
        target = (repo / rel).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            kept.append(rel)
            continue
        if target.exists():
            target.unlink()
            removed.append(rel)
    chain_ok = len(restored) == len({t.get("file", "") for t in entries})
    if chain_ok and not kept:
        manifest_file.unlink(missing_ok=True)
        try:
            manifest_file.parent.rmdir()  # remove .nanogpt-opt/ if now empty
        except OSError:
            pass
    if as_json:
        _print_json({"restored": restored, "removed_new_files": removed, "kept": kept})
        return 0
    print(f"nanogpt-opt revert: {repo}")
    print(f"  restored: {len(restored)} file(s)")
    for rel in restored:
        print(f"    - {rel}")
    print(f"  removed new files: {len(removed)}")
    for rel in removed:
        print(f"    - {rel}")
    if kept:
        print("  kept (modified after apply):")
        for rel in kept:
            print(f"    - {rel}")
    if not restored and not removed:
        print("  nothing reverted")
    return 0


# ---------------------------------------------------------------------------
# validate — honest stage reporting (pass / fail / skipped)
# ---------------------------------------------------------------------------


def _stage_compile(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for path in _py_files(root):
        checked += 1
        try:
            ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(root)}:{exc.lineno}: {exc.msg}")
    return {
        "stage": "compile",
        "status": "pass" if not failures else "fail",
        "checked": checked,
        "failures": failures,
    }


def _stage_tests(root: Path) -> dict[str, Any]:
    if not _has_test_files(root):
        return {"stage": "tests", "status": "skipped", "reason": "no test files found"}
    try:
        import importlib.util

        if importlib.util.find_spec("pytest") is None:
            return {"stage": "tests", "status": "skipped", "reason": "pytest not installed"}
    except (ImportError, ValueError):
        return {"stage": "tests", "status": "skipped", "reason": "pytest not installed"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_TESTS,
        )
    except subprocess.TimeoutExpired:
        return {"stage": "tests", "status": "fail", "reason": f"timeout after {_TIMEOUT_TESTS}s"}
    except OSError as exc:
        return {"stage": "tests", "status": "fail", "reason": f"could not run pytest: {exc}"}
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
    return {
        "stage": "tests",
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "output_tail": tail,
    }


def _stage_benchmark(root: Path) -> dict[str, Any]:
    scripts = _bench_scripts(root)
    if not scripts:
        return {"stage": "benchmark", "status": "skipped", "reason": "no benchmark scripts found"}
    results: list[dict[str, Any]] = []
    for script in scripts:
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_BENCH,
            )
            results.append(
                {
                    "script": script.name,
                    "status": "pass" if proc.returncode == 0 else "fail",
                    "exit_code": proc.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            results.append({"script": script.name, "status": "fail", "reason": "timeout"})
        except OSError as exc:
            results.append({"script": script.name, "status": "fail", "reason": str(exc)})
    failed = [r for r in results if r["status"] == "fail"]
    return {
        "stage": "benchmark",
        "status": "fail" if failed else "pass",
        "results": results,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        _err(f"not a directory: {repo}")
        return 2
    stages = [_stage_compile(repo), _stage_tests(repo), _stage_benchmark(repo)]
    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for stage in stages:
        counts[stage["status"]] += 1
    overall = "fail" if counts["fail"] else "pass"
    payload = {
        "repo": str(repo),
        "overall": overall,
        "counts": counts,
        "stages": stages,
    }
    if args.json:
        _print_json(payload)
        return 0 if overall == "pass" else 1
    print(f"nanogpt-opt validate: {repo}")
    for stage in stages:
        line = f"  {stage['stage']:<10} {stage['status']}"
        if stage["status"] == "skipped":
            line += f" ({stage.get('reason', '')})"
        elif stage["stage"] == "compile":
            line += f" ({stage['checked']} files)"
        elif stage["stage"] == "benchmark":
            line += f" ({len(stage.get('results', []))} script(s))"
        print(line)
        if stage["status"] == "fail":
            for key in ("failures", "reason"):
                value = stage.get(key)
                if isinstance(value, list):
                    for item in value:
                        print(f"      {item}")
                elif value:
                    print(f"      {value}")
            if stage.get("output_tail"):
                for line_text in str(stage["output_tail"]).splitlines():
                    print(f"      {line_text}")
    print(
        f"  summary: {counts['pass']} passed, {counts['fail']} failed, "
        f"{counts['skipped']} skipped -> {overall}"
    )
    return 0 if overall == "pass" else 1


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanogpt-opt",
        description="Lean research-to-code patches for PyTorch transformer repos "
        "(10 frozen specs: nanoGPT scope).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Scan a repo; report applicable frozen specs")
    p_analyze.add_argument("repo", help="Repository path")
    p_analyze.add_argument("--spec", help="Only consider this spec")
    p_analyze.add_argument("--json", action="store_true", help="Machine-readable output")

    p_suggest = sub.add_parser("suggest", help="Map applicable specs to file/line targets")
    p_suggest.add_argument("repo", help="Repository path")
    p_suggest.add_argument("--spec", help="Only consider this spec")
    p_suggest.add_argument("--json", action="store_true", help="Machine-readable output")

    p_apply = sub.add_parser(
        "apply", help="Preview (default) or apply the patch; --revert restores originals"
    )
    p_apply.add_argument("repo", help="Repository path")
    p_apply.add_argument("--spec", help="Only apply this spec")
    p_apply.add_argument("--write", action="store_true", help="Actually modify the repository")
    p_apply.add_argument("--revert", action="store_true", help="Restore files from the manifest")
    p_apply.add_argument("--json", action="store_true", help="Machine-readable output")

    p_validate = sub.add_parser(
        "validate", help="compile + tests + benchmark with explicit pass/fail/skipped"
    )
    p_validate.add_argument("repo", help="Repository path")
    p_validate.add_argument("--json", action="store_true", help="Machine-readable output")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "analyze": cmd_analyze,
        "suggest": cmd_suggest,
        "apply": cmd_apply,
        "validate": cmd_validate,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
