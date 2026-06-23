from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
CORE_ROOT = BENCHMARK_ROOT.parent
REPO_ROOT = CORE_ROOT.parent
SRC_ROOT = CORE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MANIFEST_PATH = BENCHMARK_ROOT / "acceptance_cases.json"
DEFAULT_REPORT_PATH = BENCHMARK_ROOT / "acceptance_report.json"
DEFAULT_SUMMARY_PATH = BENCHMARK_ROOT / "acceptance_summary.md"


@dataclass(slots=True)
class AcceptanceThresholds:
    patch_apply_rate: float = 0.90
    test_pass_rate: float = 0.80
    human_acceptance_rate: float = 0.70


@dataclass(slots=True)
class AcceptanceCase:
    id: str
    repository: str
    spec: str
    commit: str
    local_path: Path | None = None
    repo_url: str | None = None
    test_command: list[str] | None = None
    run_validation: bool = False
    timeout_seconds: int = 180
    notes: str = ""


@dataclass(slots=True)
class AcceptanceResult:
    id: str
    repository: str
    spec: str
    commit: str
    mapping_ok: bool = False
    mapping_target_count: int = 0
    mapping_confidence: float = 0.0
    patch_generated: bool = False
    patch_applied: bool = False
    changed_files: list[str] = field(default_factory=list)
    compile_ok: bool = False
    tests_status: str = "not_configured"
    validation_status: str = "not_configured"
    human_review: str = "unreviewed"
    failure_stage: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    notes: str = ""


@dataclass(slots=True)
class AcceptanceReport:
    generated_at: str
    thresholds: AcceptanceThresholds
    total_cases: int
    patch_apply_rate: float
    test_pass_rate: float | None
    test_coverage_rate: float
    human_acceptance_rate: float | None
    human_review_coverage_rate: float
    gate_status: str
    failure_counts: dict[str, int]
    results: list[AcceptanceResult]


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[AcceptanceThresholds, list[AcceptanceCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold_payload = payload.get("thresholds", {})
    thresholds = AcceptanceThresholds(
        patch_apply_rate=float(threshold_payload.get("patch_apply_rate", 0.90)),
        test_pass_rate=float(threshold_payload.get("test_pass_rate", 0.80)),
        human_acceptance_rate=float(threshold_payload.get("human_acceptance_rate", 0.70)),
    )

    cases: list[AcceptanceCase] = []
    for item in payload.get("cases", []):
        local_path = item.get("local_path")
        cases.append(
            AcceptanceCase(
                id=str(item["id"]),
                repository=str(item["repository"]),
                spec=str(item["spec"]),
                commit=str(item["commit"]),
                local_path=(REPO_ROOT / str(local_path)).resolve() if local_path else None,
                repo_url=str(item["repo_url"]) if item.get("repo_url") else None,
                test_command=[str(part) for part in item["test_command"]]
                if item.get("test_command")
                else None,
                run_validation=bool(item.get("run_validation", False)),
                timeout_seconds=int(item.get("timeout_seconds", 180)),
                notes=str(item.get("notes", "")),
            )
        )
    return thresholds, cases


def _safe_destination(root: Path, relative_path: str) -> Path:
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Patch path escapes repository root: {relative_path}") from exc
    return destination


def _copy_local_repo(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Local repository is unavailable: {source}")
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache"),
    )


def _clone_repo(case: AcceptanceCase, destination: Path) -> None:
    if not case.repo_url:
        raise FileNotFoundError(
            f"No usable local_path or repo_url configured for acceptance case '{case.id}'."
        )
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", case.repo_url, str(destination)],
        check=True,
        capture_output=True,
        text=True,
        timeout=case.timeout_seconds,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", case.commit],
        check=True,
        capture_output=True,
        text=True,
        timeout=case.timeout_seconds,
    )


def prepare_repository(case: AcceptanceCase, destination: Path, *, allow_network: bool) -> None:
    if case.local_path and case.local_path.is_dir():
        _copy_local_repo(case.local_path, destination)
        return
    if not allow_network:
        raise FileNotFoundError(
            f"Local checkout missing for '{case.id}'. Re-run with --allow-network to clone "
            f"{case.repo_url or 'the configured repository'}."
        )
    _clone_repo(case, destination)


def apply_patch_payload(repo_path: Path, payload: dict[str, Any]) -> list[str]:
    changed: list[str] = []

    for new_file in payload.get("new_files", []):
        if not isinstance(new_file, dict):
            continue
        relative_path = str(new_file.get("path", "")).strip()
        if not relative_path:
            continue
        destination = _safe_destination(repo_path, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(new_file.get("content", "")), encoding="utf-8")
        changed.append(relative_path)

    for transformation in payload.get("transformations", []):
        if not isinstance(transformation, dict):
            continue
        relative_path = str(transformation.get("file", "")).strip()
        if not relative_path:
            continue
        destination = _safe_destination(repo_path, relative_path)
        if not destination.is_file():
            raise FileNotFoundError(f"Transformation target does not exist: {relative_path}")
        current = destination.read_text(encoding="utf-8")
        original = str(transformation.get("original", ""))
        if original and current != original:
            raise ValueError(f"Transformation original does not match checkout: {relative_path}")
        destination.write_text(str(transformation.get("modified", "")), encoding="utf-8")
        changed.append(relative_path)

    if not changed:
        raise ValueError("Patch generation produced no files or transformations.")
    return sorted(set(changed))


def compile_changed_python(repo_path: Path, changed_files: list[str]) -> None:
    for relative_path in changed_files:
        if not relative_path.endswith(".py"):
            continue
        path = _safe_destination(repo_path, relative_path)
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def _run_command(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _default_generate(repo_path: Path, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from scholardevclaw.application.pipeline import run_generate, run_map

    mapping = run_map(str(repo_path), spec, use_cache=False)
    if not mapping.ok:
        raise RuntimeError(mapping.error or "Mapping failed.")
    generated = run_generate(str(repo_path), spec)
    if not generated.ok:
        raise RuntimeError(generated.error or "Patch generation failed.")
    return mapping.payload, generated.payload


def _default_validate(repo_path: Path, payload: dict[str, Any]) -> bool:
    from scholardevclaw.application.pipeline import run_validate

    return bool(run_validate(str(repo_path), payload).ok)


def run_case(
    case: AcceptanceCase,
    *,
    allow_network: bool = False,
    human_reviews: dict[str, str] | None = None,
    generate: Callable[[Path, str], tuple[dict[str, Any], dict[str, Any]]] = _default_generate,
    validate: Callable[[Path, dict[str, Any]], bool] = _default_validate,
) -> AcceptanceResult:
    started = time.perf_counter()
    result = AcceptanceResult(
        id=case.id,
        repository=case.repository,
        spec=case.spec,
        commit=case.commit,
        notes=case.notes,
    )
    review = (human_reviews or {}).get(case.id, "unreviewed").strip().lower()
    if review in {"accept", "accepted"}:
        result.human_review = "accepted"
    elif review in {"reject", "rejected"}:
        result.human_review = "rejected"

    try:
        with tempfile.TemporaryDirectory(prefix=f"sdc-acceptance-{case.id}-") as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            prepare_repository(case, repo_path, allow_network=allow_network)

            try:
                mapping_payload, patch_payload = generate(repo_path, case.spec)
                result.mapping_target_count = int(mapping_payload.get("target_count", 0))
                result.mapping_confidence = float(mapping_payload.get("confidence", 0.0))
                result.mapping_ok = result.mapping_target_count > 0
                if not result.mapping_ok:
                    raise RuntimeError("Mapping produced no target locations.")
            except Exception as exc:
                result.failure_stage = "mapping_or_generation"
                result.error = str(exc)
                return result

            result.patch_generated = bool(
                patch_payload.get("new_files") or patch_payload.get("transformations")
            )
            if not result.patch_generated:
                result.failure_stage = "generation"
                result.error = "Patch payload is empty."
                return result

            try:
                result.changed_files = apply_patch_payload(repo_path, patch_payload)
                result.patch_applied = True
            except Exception as exc:
                result.failure_stage = "patch_apply"
                result.error = str(exc)
                return result

            try:
                compile_changed_python(repo_path, result.changed_files)
                result.compile_ok = True
            except Exception as exc:
                result.failure_stage = "compile"
                result.error = str(exc)
                return result

            if case.test_command:
                completed = _run_command(case.test_command, repo_path, case.timeout_seconds)
                result.tests_status = "passed" if completed.returncode == 0 else "failed"
                if completed.returncode != 0:
                    result.failure_stage = "tests"
                    result.error = (completed.stderr or completed.stdout)[-2000:]
                    return result

            if case.run_validation:
                result.validation_status = "passed" if validate(repo_path, patch_payload) else "failed"
                if result.validation_status == "failed":
                    result.failure_stage = "validation"
                    result.error = "Validation pipeline returned a failure."
    except Exception as exc:
        result.failure_stage = result.failure_stage or "repository_setup"
        result.error = str(exc)
    finally:
        result.duration_seconds = round(time.perf_counter() - started, 3)

    return result


def _rate(passed: int, attempted: int) -> float | None:
    if attempted == 0:
        return None
    return round(passed / attempted, 3)


def build_report(
    results: list[AcceptanceResult], thresholds: AcceptanceThresholds
) -> AcceptanceReport:
    total = len(results)
    patch_rate = _rate(sum(result.patch_applied for result in results), total) or 0.0

    tested = [result for result in results if result.tests_status != "not_configured"]
    test_rate = _rate(sum(result.tests_status == "passed" for result in tested), len(tested))
    test_coverage = _rate(len(tested), total) or 0.0

    reviewed = [result for result in results if result.human_review != "unreviewed"]
    human_rate = _rate(sum(result.human_review == "accepted" for result in reviewed), len(reviewed))
    review_coverage = _rate(len(reviewed), total) or 0.0

    failure_counts: dict[str, int] = {}
    for result in results:
        if result.failure_stage:
            failure_counts[result.failure_stage] = failure_counts.get(result.failure_stage, 0) + 1

    complete = test_coverage == 1.0 and review_coverage == 1.0
    rates_pass = (
        patch_rate >= thresholds.patch_apply_rate
        and test_rate is not None
        and test_rate >= thresholds.test_pass_rate
        and human_rate is not None
        and human_rate >= thresholds.human_acceptance_rate
    )
    gate_status = "passed" if complete and rates_pass else "incomplete" if not complete else "failed"

    return AcceptanceReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        thresholds=thresholds,
        total_cases=total,
        patch_apply_rate=patch_rate,
        test_pass_rate=test_rate,
        test_coverage_rate=test_coverage,
        human_acceptance_rate=human_rate,
        human_review_coverage_rate=review_coverage,
        gate_status=gate_status,
        failure_counts=failure_counts,
        results=results,
    )


def _serialize_report(report: AcceptanceReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["thresholds"] = asdict(report.thresholds)
    return payload


def build_markdown_summary(report: AcceptanceReport) -> str:
    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# ScholarDevClaw V1 Acceptance Summary",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Gate status: **{report.gate_status.upper()}**",
        f"- Patch apply rate: `{percentage(report.patch_apply_rate)}` "
        f"(target `{percentage(report.thresholds.patch_apply_rate)}`)",
        f"- Test pass rate: `{percentage(report.test_pass_rate)}` "
        f"(coverage `{percentage(report.test_coverage_rate)}`, "
        f"target `{percentage(report.thresholds.test_pass_rate)}`)",
        f"- Human acceptance rate: `{percentage(report.human_acceptance_rate)}` "
        f"(coverage `{percentage(report.human_review_coverage_rate)}`, "
        f"target `{percentage(report.thresholds.human_acceptance_rate)}`)",
        "",
        "| Case | Repository | Spec | Map | Apply | Compile | Tests | Validation | Review | Failure |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for result in report.results:
        lines.append(
            "| {id} | {repo} | {spec} | {mapping} | {apply} | {compile} | {tests} | "
            "{validation} | {review} | {failure} |".format(
                id=result.id,
                repo=result.repository,
                spec=result.spec,
                mapping="yes" if result.mapping_ok else "no",
                apply="yes" if result.patch_applied else "no",
                compile="yes" if result.compile_ok else "no",
                tests=result.tests_status,
                validation=result.validation_status,
                review=result.human_review,
                failure=result.failure_stage or "-",
            )
        )
    if report.failure_counts:
        lines.extend(["", "## Failure Categories", ""])
        for stage, count in sorted(report.failure_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{stage}`: {count}")
    return "\n".join(lines) + "\n"


def write_report(
    report: AcceptanceReport,
    output_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> None:
    output_path.write_text(json.dumps(_serialize_report(report), indent=2), encoding="utf-8")
    summary_path.write_text(build_markdown_summary(report), encoding="utf-8")


def load_human_reviews(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviews = payload.get("reviews", payload)
    if not isinstance(reviews, dict):
        raise ValueError("Human review file must contain an object or a 'reviews' object.")
    return {str(key): str(value) for key, value in reviews.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ScholarDevClaw V1 acceptance matrix.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--reviews", default=None, help="JSON file mapping case IDs to accept/reject.")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds, cases = load_manifest(Path(args.manifest).resolve())
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.id in selected]
    reviews = load_human_reviews(Path(args.reviews).resolve() if args.reviews else None)

    results = [
        run_case(case, allow_network=args.allow_network, human_reviews=reviews) for case in cases
    ]
    report = build_report(results, thresholds)
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    write_report(report, output_path, summary_path)

    print("ScholarDevClaw V1 acceptance matrix")
    print(f"  Cases              : {report.total_cases}")
    print(f"  Patch apply rate   : {report.patch_apply_rate:.1%}")
    print(
        "  Test pass rate     : "
        + ("n/a" if report.test_pass_rate is None else f"{report.test_pass_rate:.1%}")
    )
    print(
        "  Human acceptance  : "
        + (
            "n/a"
            if report.human_acceptance_rate is None
            else f"{report.human_acceptance_rate:.1%}"
        )
    )
    print(f"  Gate status        : {report.gate_status}")
    print(f"  JSON report        : {output_path}")
    print(f"  Markdown summary   : {summary_path}")

    if args.fail_on_gate and report.gate_status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
