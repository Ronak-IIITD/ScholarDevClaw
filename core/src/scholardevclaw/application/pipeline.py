from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineResult:
    ok: bool
    title: str
    payload: dict[str, Any]
    logs: list[str]
    error: str | None = None


def _ensure_repo(repo_path: str) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Repository not found: {repo_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Repository is not a directory: {repo_path}")
    return path


def run_analyze(repo_path: str) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    logs = [f"Analyzing repository: {repo_path}"]
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
        logs.append(f"Detected languages: {', '.join(result.languages) if result.languages else 'None'}")
        logs.append(f"Detected frameworks: {', '.join(result.frameworks) if result.frameworks else 'None'}")
        return PipelineResult(ok=True, title="Repository Analysis", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(
            ok=False,
            title="Repository Analysis",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_suggest(repo_path: str) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    logs = [f"Scanning for improvement opportunities: {repo_path}"]
    try:
        path = _ensure_repo(repo_path)
        analyzer = TreeSitterAnalyzer(path)
        suggestions = analyzer.suggest_research_papers()

        logs.append(f"Suggestions found: {len(suggestions)}")
        return PipelineResult(
            ok=True,
            title="Research Suggestions",
            payload={"repo_path": str(path), "suggestions": suggestions},
            logs=logs,
        )
    except Exception as exc:
        logs.append(f"Failed: {exc}")
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
) -> PipelineResult:
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs = [f"Searching for: {query}"]
    try:
        extractor = ResearchExtractor()
        local_results = extractor.search_by_keyword(query, max_results=max_results)

        payload: dict[str, Any] = {
            "query": query,
            "local": local_results,
            "arxiv": [],
            "web": {},
        }
        logs.append(f"Local specs found: {len(local_results)}")

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
            logs.append(f"arXiv papers found: {len(arxiv_papers)}")

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
            logs.append(
                "Web sources found: "
                f"{len(payload['web'].get('github_repos', []))} repos, "
                f"{len(payload['web'].get('papers_with_code', []))} papers"
            )

        return PipelineResult(ok=True, title="Research Search", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(
            ok=False,
            title="Research Search",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_specs(*, detailed: bool = False, by_category: bool = False) -> PipelineResult:
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs = ["Loading specification index"]
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
        logs.append(f"Total specs: {len(specs)}")
        logs.append(f"Categories: {len(categories)}")

        return PipelineResult(ok=True, title="Specifications", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(
            ok=False,
            title="Specifications",
            payload={},
            logs=logs,
            error=str(exc),
        )


def _build_mapping_result(repo_path: Path, spec_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    analyzer = TreeSitterAnalyzer(repo_path)
    analysis = analyzer.analyze()

    extractor = ResearchExtractor()
    spec = extractor.get_spec(spec_name)
    if spec is None:
        raise ValueError(f"Unknown spec: {spec_name}")

    engine = MappingEngine(analysis.__dict__, spec)
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


def run_map(repo_path: str, spec_name: str) -> PipelineResult:
    logs = [f"Mapping spec '{spec_name}' to repository: {repo_path}"]
    try:
        path = _ensure_repo(repo_path)
        mapping_result, spec = _build_mapping_result(path, spec_name)
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

        logs.append(f"Algorithm: {payload['algorithm']}")
        logs.append(f"Targets found: {len(targets)}")
        logs.append(f"Strategy: {payload['strategy']}, confidence: {payload['confidence']}%")

        return PipelineResult(ok=True, title="Mapping", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(ok=False, title="Mapping", payload={}, logs=logs, error=str(exc))


def run_generate(repo_path: str, spec_name: str, *, output_dir: str | None = None) -> PipelineResult:
    from scholardevclaw.patch_generation.generator import PatchGenerator

    logs = [f"Generating patch for spec '{spec_name}' in repository: {repo_path}"]
    try:
        path = _ensure_repo(repo_path)
        mapping_result, spec = _build_mapping_result(path, spec_name)

        generator = PatchGenerator(path)
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

        logs.append(f"Branch: {patch.branch_name}")
        logs.append(f"New files: {len(patch.new_files)}")
        logs.append(f"Transformations: {len(patch.transformations)}")
        if output_dir_path:
            logs.append(f"Patch artifacts written to: {output_dir_path}")

        return PipelineResult(ok=True, title="Patch Generation", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(
            ok=False,
            title="Patch Generation",
            payload={},
            logs=logs,
            error=str(exc),
        )


def run_validate(repo_path: str) -> PipelineResult:
    from scholardevclaw.validation.runner import ValidationRunner

    logs = [f"Running validation in repository: {repo_path}"]
    try:
        path = _ensure_repo(repo_path)
        runner = ValidationRunner(path)
        result = runner.run({}, str(path))

        payload = {
            "passed": result.passed,
            "stage": result.stage,
            "comparison": result.comparison,
            "logs": result.logs,
            "error": result.error,
        }
        logs.append(f"Stage: {result.stage}")
        logs.append(f"Passed: {result.passed}")
        if result.error:
            logs.append(f"Error: {result.error}")

        return PipelineResult(ok=result.passed, title="Validation", payload=payload, logs=logs)
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(ok=False, title="Validation", payload={}, logs=logs, error=str(exc))


def run_integrate(repo_path: str, spec_name: str | None = None) -> PipelineResult:
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs = [f"Starting integration workflow for repository: {repo_path}"]
    try:
        path = _ensure_repo(repo_path)

        analyzer = TreeSitterAnalyzer(path)
        analysis = analyzer.analyze()
        logs.append(f"Analyze: {len(analysis.languages)} languages, {len(analysis.elements)} elements")

        extractor = ResearchExtractor()
        selected_spec_name = spec_name
        selected_spec: dict[str, Any] | None = None

        if selected_spec_name:
            selected_spec = extractor.get_spec(selected_spec_name)
            if not selected_spec:
                raise ValueError(f"Unknown spec: {selected_spec_name}")
            logs.append(f"Research: selected explicit spec '{selected_spec_name}'")
        else:
            suggestions = analyzer.suggest_research_papers()
            if not suggestions:
                raise ValueError("No suitable improvements found for this repository")
            selected_spec_name = suggestions[0]["paper"]["name"]
            selected_spec = extractor.get_spec(selected_spec_name)
            if not selected_spec:
                raise ValueError(f"Suggested spec is unavailable: {selected_spec_name}")
            confidence = suggestions[0].get("confidence", 0)
            logs.append(
                f"Research: auto-selected spec '{selected_spec_name}' ({confidence:.0f}% confidence)"
            )

        if selected_spec_name is None:
            raise ValueError("Failed to resolve integration spec")

        mapping_result, _ = _build_mapping_result(path, selected_spec_name)
        logs.append(f"Mapping: {len(mapping_result.get('targets', []))} targets")

        generate_result = run_generate(str(path), selected_spec_name)
        logs.extend(generate_result.logs)
        if not generate_result.ok:
            return PipelineResult(
                ok=False,
                title="Integration",
                payload={"step": "generate"},
                logs=logs,
                error=generate_result.error,
            )

        validate_result = run_validate(str(path))
        logs.extend(validate_result.logs)

        payload = {
            "spec": selected_spec_name,
            "analysis": {
                "languages": analysis.languages,
                "frameworks": analysis.frameworks,
                "entry_points": analysis.entry_points,
                "patterns": analysis.patterns,
            },
            "mapping": mapping_result,
            "generation": generate_result.payload,
            "validation": validate_result.payload,
        }

        return PipelineResult(
            ok=validate_result.ok,
            title="Integration",
            payload=payload,
            logs=logs,
            error=validate_result.error,
        )
    except Exception as exc:
        logs.append(f"Failed: {exc}")
        return PipelineResult(ok=False, title="Integration", payload={}, logs=logs, error=str(exc))
