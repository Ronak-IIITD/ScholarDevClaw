from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from .schema_contract import SCHEMA_VERSION, with_meta


@dataclass(slots=True)
class PipelineResult:
    ok: bool
    title: str
    payload: dict[str, Any]
    logs: list[str]
    error: str | None = None


LogCallback = Callable[[str], None]
PIPELINE_SCHEMA_VERSION = SCHEMA_VERSION


def _log(logs: list[str], message: str, log_callback: LogCallback | None = None) -> None:
    logs.append(message)
    if log_callback is not None:
        log_callback(message)


def _ensure_repo(repo_path: str) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Repository not found: {repo_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Repository is not a directory: {repo_path}")
    return path


def run_preflight(
    repo_path: str,
    *,
    require_clean: bool = False,
    log_callback: LogCallback | None = None,
) -> PipelineResult:
    logs: list[str] = []
    _log(logs, f"Running preflight checks for: {repo_path}", log_callback)

    try:
        path = _ensure_repo(repo_path)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Preflight",
            payload={"repo_path": repo_path},
            logs=logs,
            error=str(exc),
        )

    is_writable = os.access(path, os.W_OK)
    has_git_dir = (path / ".git").exists()
    python_file_count = len(list(path.rglob("*.py")))
    warnings: list[str] = []
    recommendations: list[str] = []

    git_available = False
    is_clean = True
    changed_files: list[str] = []
    git_error: str | None = None

    if has_git_dir:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            git_available = status.returncode == 0
            if git_available:
                lines = [line for line in status.stdout.splitlines() if line.strip()]
                changed_files = lines
                is_clean = len(lines) == 0
        except Exception as exc:
            git_error = str(exc)
            git_available = False

    if not is_writable:
        warnings.append("Repository directory is not writable")
        recommendations.append("Grant write permission for the repository before running integrate")

    if python_file_count == 0:
        warnings.append("No Python files detected")
        recommendations.append("Verify repository path and project language before integration")

    if has_git_dir and not git_available:
        warnings.append("Git repository detected but git status check failed")
        recommendations.append(
            "Ensure git is installed and repository permissions allow status checks"
        )

    if require_clean:
        if not has_git_dir:
            warnings.append("Clean-check requested but .git directory is missing")
            recommendations.append("Run integration inside a git clone or disable --require-clean")
        elif not git_available:
            warnings.append("Clean-check requested but git status is unavailable")
            recommendations.append("Fix git availability or disable --require-clean to continue")

    checks = {
        "repo_exists": True,
        "repo_is_writable": is_writable,
        "python_file_count": python_file_count,
        "has_git_dir": has_git_dir,
        "git_available": git_available,
        "is_clean": is_clean,
        "changed_file_entries": changed_files,
        "git_error": git_error,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    _log(logs, f"Preflight: python files detected = {python_file_count}", log_callback)
    if has_git_dir:
        if git_available:
            _log(logs, f"Preflight: git working tree clean = {is_clean}", log_callback)
        else:
            _log(logs, "Preflight: git check unavailable, continuing", log_callback)
    else:
        _log(logs, "Preflight: .git not found, continuing", log_callback)

    for warning in warnings:
        _log(logs, f"Preflight warning: {warning}", log_callback)

    if not is_writable:
        error = "Repository is not writable; integration cannot proceed"
        _log(logs, f"Failed: {error}", log_callback)
        return PipelineResult(
            ok=False,
            title="Preflight",
            payload=checks,
            logs=logs,
            error=error,
        )

    if require_clean and not has_git_dir:
        error = "Repository is not a git checkout; require_clean=True cannot be enforced"
        _log(logs, f"Failed: {error}", log_callback)
        return PipelineResult(
            ok=False,
            title="Preflight",
            payload=checks,
            logs=logs,
            error=error,
        )

    if require_clean and has_git_dir and not git_available:
        error = "Git status check failed; require_clean=True cannot be enforced"
        _log(logs, f"Failed: {error}", log_callback)
        return PipelineResult(
            ok=False,
            title="Preflight",
            payload=checks,
            logs=logs,
            error=error,
        )

    if require_clean and has_git_dir and git_available and not is_clean:
        error = "Repository has uncommitted changes; require_clean=True blocked execution"
        _log(logs, f"Failed: {error}", log_callback)
        return PipelineResult(
            ok=False,
            title="Preflight",
            payload=checks,
            logs=logs,
            error=error,
        )

    return PipelineResult(ok=True, title="Preflight", payload=checks, logs=logs)


def run_analyze(repo_path: str, *, log_callback: LogCallback | None = None) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    logs: list[str] = []
    _log(logs, f"Analyzing repository: {repo_path}", log_callback)
    try:
        path = _ensure_repo(repo_path)
        analyzer = TreeSitterAnalyzer(path)
        result = analyzer.analyze()

        payload = {
            "root_path": str(result.root_path),
            "languages": result.languages,
            "language_stats": [
                {
                    "language": s.language,
                    "file_count": s.file_count,
                    "line_count": s.line_count,
                }
                for s in result.language_stats
            ],
            "frameworks": result.frameworks,
            "entry_points": result.entry_points,
            "test_files": result.test_files,
            "patterns": result.patterns,
        }
        _log(
            logs,
            f"Detected languages: {', '.join(result.languages) if result.languages else 'None'}",
            log_callback,
        )
        _log(
            logs,
            f"Detected frameworks: {', '.join(result.frameworks) if result.frameworks else 'None'}",
            log_callback,
        )
        return PipelineResult(ok=True, title="Repository Analysis", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Repository Analysis",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_suggest(repo_path: str, *, log_callback: LogCallback | None = None) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    logs: list[str] = []
    _log(logs, f"Scanning for improvement opportunities: {repo_path}", log_callback)
    try:
        path = _ensure_repo(repo_path)
        analyzer = TreeSitterAnalyzer(path)
        suggestions = analyzer.suggest_research_papers()

        _log(logs, f"Suggestions found: {len(suggestions)}", log_callback)
        return PipelineResult(
            ok=True,
            title="Research Suggestions",
            payload={"repo_path": str(path), "suggestions": suggestions},
            logs=logs,
        )
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Research Suggestions",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_search(
    query: str,
    *,
    include_arxiv: bool = False,
    include_web: bool = False,
    language: str = "python",
    max_results: int = 10,
    log_callback: LogCallback | None = None,
) -> PipelineResult:
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    _log(logs, f"Searching for: {query}", log_callback)
    try:
        extractor = ResearchExtractor()
        local_results = extractor.search_by_keyword(query, max_results=max_results)

        payload: dict[str, Any] = {
            "query": query,
            "local": local_results,
            "arxiv": [],
            "web": {},
        }
        _log(logs, f"Local specs found: {len(local_results)}", log_callback)

        if include_arxiv:
            import asyncio

            from scholardevclaw.research_intelligence.extractor import ResearchQuery

            research_query = ResearchQuery(keywords=query.split(), max_results=max_results)
            arxiv_papers = asyncio.run(extractor.search_arxiv(research_query))
            payload["arxiv"] = [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "categories": p.categories,
                    "arxiv_id": p.arxiv_id,
                    "pdf_url": p.pdf_url,
                    "published": p.published.isoformat() if getattr(p, "published", None) else None,
                    "summary": p.summary,
                }
                for p in arxiv_papers
            ]
            _log(logs, f"arXiv papers found: {len(arxiv_papers)}", log_callback)

        if include_web:
            from scholardevclaw.research_intelligence.web_research import SyncWebResearchEngine

            engine = SyncWebResearchEngine()
            web_results = engine.search_all(query, language, max_results)
            payload["web"] = {
                "github_repos": [
                    {
                        "owner": r.owner,
                        "name": r.name,
                        "stars": r.stars,
                        "url": r.url,
                        "description": r.description,
                    }
                    for r in web_results.get("github_repos", [])
                ],
                "papers_with_code": [
                    {
                        "title": p.title,
                        "url": p.url,
                        "task": p.task,
                        "stars": p.stars,
                    }
                    for p in web_results.get("papers_with_code", [])
                ],
            }
            _log(
                logs,
                "Web sources found: "
                f"{len(payload['web'].get('github_repos', []))} repos, "
                f"{len(payload['web'].get('papers_with_code', []))} papers",
                log_callback,
            )

        return PipelineResult(ok=True, title="Research Search", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Research Search",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_specs(
    *, detailed: bool = False, by_category: bool = False, log_callback: LogCallback | None = None
) -> PipelineResult:
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    _log(logs, "Loading specification index", log_callback)
    try:
        extractor = ResearchExtractor()

        specs = extractor.list_available_specs()
        categories = extractor.get_categories()

        payload: dict[str, Any] = {
            "spec_names": specs,
            "categories": categories,
        }

        if detailed:
            detailed_specs = {}
            for spec_name in specs:
                spec = extractor.get_spec(spec_name)
                if spec:
                    detailed_specs[spec_name] = spec
            payload["details"] = detailed_specs

        payload["view"] = "categories" if by_category else "detailed" if detailed else "simple"
        _log(logs, f"Total specs: {len(specs)}", log_callback)
        _log(logs, f"Categories: {len(categories)}", log_callback)

        return PipelineResult(ok=True, title="Specifications", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Specifications",
            payload={},
            logs=logs,
            error=str(exc),
        )


def _build_mapping_result(
    repo_path: Path,
    spec_name: str,
    *,
    log_callback: LogCallback | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    analyzer = TreeSitterAnalyzer(repo_path)
    if log_callback is not None:
        log_callback("Analyzing repository structure for mapping")
    analysis = analyzer.analyze()

    extractor = ResearchExtractor()
    if log_callback is not None:
        log_callback(f"Resolving specification: {spec_name}")
    spec = extractor.get_spec(spec_name)
    if spec is None:
        raise ValueError(f"Unknown spec: {spec_name}")

    engine = MappingEngine(analysis.__dict__, spec)
    if log_callback is not None:
        log_callback("Running mapping engine")
    mapping = engine.map()

    mapping_result = {
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
        "research_spec": mapping.research_spec,
    }

    return mapping_result, spec


def run_map(
    repo_path: str, spec_name: str, *, log_callback: LogCallback | None = None
) -> PipelineResult:
    logs: list[str] = []
    _log(logs, f"Mapping spec '{spec_name}' to repository: {repo_path}", log_callback)
    try:
        path = _ensure_repo(repo_path)
        mapping_result, spec = _build_mapping_result(path, spec_name, log_callback=log_callback)
        targets = mapping_result.get("targets", [])

        payload = {
            "spec": spec_name,
            "algorithm": spec.get("algorithm", {}).get("name", "Unknown"),
            "strategy": mapping_result.get("strategy", "none"),
            "confidence": mapping_result.get("confidence", 0),
            "targets": targets,
            "target_count": len(targets),
            "mapping": mapping_result,
        }

        _log(logs, f"Algorithm: {payload['algorithm']}", log_callback)
        _log(logs, f"Targets found: {len(targets)}", log_callback)
        _log(
            logs,
            f"Strategy: {payload['strategy']}, confidence: {payload['confidence']}%",
            log_callback,
        )

        return PipelineResult(ok=True, title="Mapping", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(ok=False, title="Mapping", payload={}, logs=logs, error=str(exc))


def run_generate(
    repo_path: str,
    spec_name: str,
    *,
    output_dir: str | None = None,
    log_callback: LogCallback | None = None,
) -> PipelineResult:
    from scholardevclaw.patch_generation.generator import PatchGenerator

    logs: list[str] = []
    _log(
        logs,
        f"Generating patch for spec '{spec_name}' in repository: {repo_path}",
        log_callback,
    )
    try:
        path = _ensure_repo(repo_path)
        mapping_result, spec = _build_mapping_result(path, spec_name, log_callback=log_callback)

        generator = PatchGenerator(path)
        _log(logs, "Generating patch artifacts", log_callback)
        patch = generator.generate(mapping_result)

        written_files: list[str] = []
        output_dir_path: Path | None = None
        if output_dir:
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            for new_file in patch.new_files:
                destination = output_dir_path / new_file.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(new_file.content)
                written_files.append(str(destination))
                _log(logs, f"Wrote file: {destination}", log_callback)

        payload = {
            "spec": spec_name,
            "algorithm": spec.get("algorithm", {}).get("name", "Unknown"),
            "branch_name": patch.branch_name,
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
            "output_dir": str(output_dir_path) if output_dir_path else None,
            "written_files": written_files,
        }

        _log(logs, f"Branch: {patch.branch_name}", log_callback)
        _log(logs, f"New files: {len(patch.new_files)}", log_callback)
        _log(logs, f"Transformations: {len(patch.transformations)}", log_callback)
        if output_dir_path:
            _log(logs, f"Patch artifacts written to: {output_dir_path}", log_callback)

        return PipelineResult(ok=True, title="Patch Generation", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Patch Generation",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_validate(repo_path: str, *, log_callback: LogCallback | None = None) -> PipelineResult:
    from scholardevclaw.validation.runner import ValidationRunner

    def _metric_dict(metric: Any) -> dict[str, float] | None:
        if metric is None:
            return None
        return {
            "loss": float(getattr(metric, "loss", 0.0)),
            "perplexity": float(getattr(metric, "perplexity", 0.0)),
            "tokens_per_second": float(getattr(metric, "tokens_per_second", 0.0)),
            "memory_mb": float(getattr(metric, "memory_mb", 0.0)),
            "runtime_seconds": float(getattr(metric, "runtime_seconds", 0.0)),
        }

    def _build_scorecard(
        *,
        passed: bool,
        stage: str,
        comparison: dict[str, Any] | None,
        baseline: dict[str, float] | None,
        new: dict[str, float] | None,
    ) -> dict[str, Any]:
        speedup = None
        loss_change = None
        if comparison:
            speedup = comparison.get("speedup")
            loss_change = comparison.get("loss_change")

        checks = [
            {
                "name": "validation_passed",
                "status": "pass" if passed else "fail",
                "value": bool(passed),
            }
        ]
        if speedup is not None:
            checks.append(
                {
                    "name": "speedup",
                    "status": "pass" if float(speedup) >= 1.0 else "warn",
                    "value": float(speedup),
                }
            )
        if loss_change is not None:
            checks.append(
                {
                    "name": "loss_change_pct",
                    "status": "pass" if abs(float(loss_change)) <= 5.0 else "warn",
                    "value": float(loss_change),
                }
            )

        highlights: list[str] = []
        if passed:
            highlights.append(f"Validation passed at stage '{stage}'")
        else:
            highlights.append(f"Validation failed at stage '{stage}'")
        if speedup is not None:
            highlights.append(f"Speedup: {float(speedup):.3f}x")
        if loss_change is not None:
            highlights.append(f"Loss change: {float(loss_change):.3f}%")

        return {
            "version": "1.0",
            "summary": "pass" if passed else "fail",
            "stage": stage,
            "checks": checks,
            "deltas": {
                "speedup": float(speedup) if speedup is not None else None,
                "loss_change_pct": float(loss_change) if loss_change is not None else None,
            },
            "baseline_metrics": baseline,
            "new_metrics": new,
            "highlights": highlights,
        }

    logs: list[str] = []
    _log(logs, f"Running validation in repository: {repo_path}", log_callback)
    try:
        path = _ensure_repo(repo_path)
        runner = ValidationRunner(path)
        result = runner.run({}, str(path))
        baseline_metrics = _metric_dict(getattr(result, "baseline_metrics", None))
        new_metrics = _metric_dict(getattr(result, "new_metrics", None))
        scorecard = _build_scorecard(
            passed=bool(result.passed),
            stage=str(result.stage),
            comparison=result.comparison if isinstance(result.comparison, dict) else None,
            baseline=baseline_metrics,
            new=new_metrics,
        )

        payload = with_meta(
            {
                "passed": result.passed,
                "stage": result.stage,
                "comparison": result.comparison,
                "baseline_metrics": baseline_metrics,
                "new_metrics": new_metrics,
                "scorecard": scorecard,
                "logs": result.logs,
                "error": result.error,
            },
            "validation",
        )
        _log(logs, f"Stage: {result.stage}", log_callback)
        _log(logs, f"Passed: {result.passed}", log_callback)
        if scorecard["deltas"].get("speedup") is not None:
            _log(logs, f"Speedup: {scorecard['deltas']['speedup']:.3f}x", log_callback)
        if scorecard["deltas"].get("loss_change_pct") is not None:
            _log(logs, f"Loss change: {scorecard['deltas']['loss_change_pct']:.3f}%", log_callback)
        if result.error:
            _log(logs, f"Error: {result.error}", log_callback)

        return PipelineResult(ok=result.passed, title="Validation", payload=payload, logs=logs)
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Validation",
            payload=with_meta({}, "validation"),
            logs=logs,
            error=str(exc),
        )


def run_integrate(
    repo_path: str,
    spec_name: str | None = None,
    *,
    dry_run: bool = False,
    require_clean: bool = False,
    output_dir: str | None = None,
    create_rollback: bool = True,
    log_callback: LogCallback | None = None,
) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    rollback_snapshot_id: str | None = None
    _log(logs, f"Starting integration workflow for repository: {repo_path}", log_callback)
    try:
        path = _ensure_repo(repo_path)

        preflight = run_preflight(
            str(path),
            require_clean=require_clean,
            log_callback=log_callback,
        )
        logs.extend(preflight.logs)
        if not preflight.ok:
            return PipelineResult(
                ok=False,
                title="Integration",
                payload=with_meta(
                    {
                        "step": "preflight",
                        "preflight": preflight.payload,
                        "guidance": preflight.payload.get("recommendations", []),
                    },
                    "integration",
                ),
                logs=logs,
                error=preflight.error,
            )

        analyzer = TreeSitterAnalyzer(path)
        analysis = analyzer.analyze()
        _log(
            logs,
            f"Analyze: {len(analysis.languages)} languages, {len(analysis.elements)} elements",
            log_callback,
        )

        extractor = ResearchExtractor()
        selected_spec_name = spec_name
        selected_spec: dict[str, Any] | None = None

        if selected_spec_name:
            selected_spec = extractor.get_spec(selected_spec_name)
            if not selected_spec:
                raise ValueError(f"Unknown spec: {selected_spec_name}")
            _log(logs, f"Research: selected explicit spec '{selected_spec_name}'", log_callback)
        else:
            suggestions = analyzer.suggest_research_papers()
            if not suggestions:
                raise ValueError("No suitable improvements found for this repository")
            selected_spec_name = suggestions[0]["paper"]["name"]
            selected_spec = extractor.get_spec(selected_spec_name)
            if not selected_spec:
                raise ValueError(f"Suggested spec is unavailable: {selected_spec_name}")
            confidence = suggestions[0].get("confidence", 0)
            _log(
                logs,
                f"Research: auto-selected spec '{selected_spec_name}' ({confidence:.0f}% confidence)",
                log_callback,
            )

        if selected_spec_name is None:
            raise ValueError("Failed to resolve integration spec")

        mapping_result, _ = _build_mapping_result(
            path,
            selected_spec_name,
            log_callback=log_callback,
        )
        _log(logs, f"Mapping: {len(mapping_result.get('targets', []))} targets", log_callback)

        if create_rollback and not dry_run:
            from scholardevclaw.rollback import RollbackManager

            rollback_manager = RollbackManager()
            rollback_snapshot = rollback_manager.create_snapshot(
                str(path),
                selected_spec_name,
                description=f"Pre-integration snapshot for {selected_spec_name}",
                log_callback=log_callback,
            )
            rollback_snapshot_id = rollback_snapshot.id
            _log(logs, f"Created rollback snapshot: {rollback_snapshot_id}", log_callback)

        if dry_run:
            _log(logs, "Dry run enabled: skipping patch generation and validation", log_callback)
            payload = with_meta(
                {
                    "dry_run": True,
                    "spec": selected_spec_name,
                    "analysis": {
                        "languages": analysis.languages,
                        "frameworks": analysis.frameworks,
                        "entry_points": analysis.entry_points,
                        "patterns": analysis.patterns,
                    },
                    "preflight": preflight.payload,
                    "mapping": mapping_result,
                    "generation": None,
                    "validation": None,
                    "output_dir": output_dir,
                },
                "integration",
            )
            return PipelineResult(
                ok=True,
                title="Integration",
                payload=payload,
                logs=logs,
            )

        generate_result = run_generate(
            str(path),
            selected_spec_name,
            output_dir=output_dir,
            log_callback=log_callback,
        )
        logs.extend(generate_result.logs)
        if not generate_result.ok:
            return PipelineResult(
                ok=False,
                title="Integration",
                payload=with_meta({"step": "generate"}, "integration"),
                logs=logs,
                error=generate_result.error,
            )

        validate_result = run_validate(str(path), log_callback=log_callback)
        logs.extend(validate_result.logs)

        payload = with_meta(
            {
                "dry_run": False,
                "spec": selected_spec_name,
                "analysis": {
                    "languages": analysis.languages,
                    "frameworks": analysis.frameworks,
                    "entry_points": analysis.entry_points,
                    "patterns": analysis.patterns,
                },
                "preflight": preflight.payload,
                "mapping": mapping_result,
                "generation": generate_result.payload,
                "validation": validate_result.payload,
                "rollback_snapshot_id": rollback_snapshot_id,
            },
            "integration",
        )

        if rollback_snapshot_id and validate_result.ok:
            from scholardevclaw.rollback import RollbackManager

            rollback_manager = RollbackManager()
            rollback_manager.mark_applied(str(path), rollback_snapshot_id)
            _log(logs, f"Marked rollback snapshot as applied: {rollback_snapshot_id}", log_callback)

        return PipelineResult(
            ok=validate_result.ok,
            title="Integration",
            payload=payload,
            logs=logs,
            error=validate_result.error,
        )
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Integration",
            payload=with_meta({}, "integration"),
            logs=logs,
            error=str(exc),
        )


def run_planner(
    repo_path: str,
    *,
    max_specs: int = 5,
    target_categories: list[str] | None = None,
    log_callback: LogCallback | None = None,
):
    from scholardevclaw.planner import run_planner as _run_planner

    return _run_planner(
        repo_path,
        max_specs=max_specs,
        target_categories=target_categories,
        log_callback=log_callback,
    )


def run_multi_integrate(
    repo_path: str,
    spec_names: list[str],
    *,
    output_dir: str | None = None,
    require_clean: bool = False,
    log_callback: LogCallback | None = None,
) -> PipelineResult:
    from .schema_contract import with_meta

    logs: list[str] = []
    _log(
        logs,
        f"Starting multi-spec integration for: {repo_path} with {len(spec_names)} specs",
        log_callback,
    )

    try:
        path = _ensure_repo(repo_path)

        preflight = run_preflight(
            str(path),
            require_clean=require_clean,
            log_callback=log_callback,
        )
        logs.extend(preflight.logs)
        if not preflight.ok:
            return PipelineResult(
                ok=False,
                title="Multi-Integration",
                payload=with_meta(
                    {
                        "step": "preflight",
                        "preflight": preflight.payload,
                        "guidance": preflight.payload.get("recommendations", []),
                    },
                    "multi_integration",
                ),
                logs=logs,
                error=preflight.error,
            )

        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer(path)
        analysis = analyzer.analyze()
        _log(
            logs,
            f"Analyze: {len(analysis.languages)} languages, {len(analysis.elements)} elements",
            log_callback,
        )

        results: list[dict[str, Any]] = []

        for i, spec_name in enumerate(spec_names, 1):
            _log(logs, f"\n--- Spec {i}/{len(spec_names)}: {spec_name} ---", log_callback)

            mapping_result, spec = _build_mapping_result(path, spec_name, log_callback=log_callback)
            _log(
                logs,
                f"Mapping: {len(mapping_result.get('targets', []))} targets",
                log_callback,
            )

            generate_result = run_generate(
                str(path),
                spec_name,
                output_dir=output_dir,
                log_callback=log_callback,
            )
            logs.extend(generate_result.logs)

            if generate_result.ok:
                results.append(
                    {
                        "spec": spec_name,
                        "mapping": mapping_result,
                        "generation": generate_result.payload,
                    }
                )
            else:
                _log(
                    logs,
                    f"Generation failed for {spec_name}: {generate_result.error}",
                    log_callback,
                )

        validate_result = run_validate(str(path), log_callback=log_callback)
        logs.extend(validate_result.logs)

        payload = with_meta(
            {
                "specs": spec_names,
                "specs_applied": len(results),
                "analysis": {
                    "languages": analysis.languages,
                    "frameworks": analysis.frameworks,
                },
                "preflight": preflight.payload,
                "spec_results": results,
                "validation": validate_result.payload,
                "output_dir": output_dir,
            },
            "multi_integration",
        )

        return PipelineResult(
            ok=validate_result.ok,
            title="Multi-Integration",
            payload=payload,
            logs=logs,
            error=validate_result.error,
        )
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PipelineResult(
            ok=False,
            title="Multi-Integration",
            payload=with_meta({}, "multi_integration"),
            logs=logs,
            error=str(exc),
        )
