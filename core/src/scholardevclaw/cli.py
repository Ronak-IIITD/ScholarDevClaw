#!/usr/bin/env python3
"""
ScholarDevClaw CLI - Autonomous Research-to-Code Agent

An AI-powered tool that analyzes your codebase, researches relevant papers,
and automatically implements improvements. Supports any programming language.

Commands:
  analyze     - Analyze repository structure (multi-language)
  search      - Search for research papers and implementations
  ingest      - Ingest a paper (PDF/DOI/arXiv/URL) into structured JSON
  understand  - Extract structured paper understanding via LLM
  plan        - Plan implementation from understanding JSON
  generate    - Generate implementation code from plan + understanding
  suggest     - Get AI-powered improvement suggestions
  integrate   - Full integration workflow
  tui         - Interactive terminal UI workflow
  specs       - List available paper specifications
  demo        - Run demo with nanoGPT

Examples:
  scholardevclaw analyze ./my-project
  scholardevclaw search "normalization" --arxiv --web
  scholardevclaw ingest arxiv:1706.03762 --output-dir ./artifacts
  scholardevclaw understand ./paper_document.json --output-dir ./artifacts
  scholardevclaw plan ./understanding.json --output-dir ./plan_output
  scholardevclaw generate ./implementation_plan.json ./understanding.json --output-dir ./generated
  scholardevclaw suggest ./my-project
  scholardevclaw integrate ./my-project rmsnorm

Version: 2.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

from scholardevclaw.application.schema_contract import evaluate_payload_compatibility
from scholardevclaw.auth.cli import cmd_auth


def _print_compatibility_report(payload: dict, expected_type: str, *, stderr: bool = False) -> None:
    report = evaluate_payload_compatibility(payload, expected_types={expected_type})
    stream = sys.stderr if stderr else sys.stdout
    if report.issues:
        print("Compatibility issues:", file=stream)
        for issue in report.issues:
            print(f"  - {issue}", file=stream)
    if report.warnings:
        print("Compatibility warnings:", file=stream)
        for warning in report.warnings:
            print(f"  - {warning}", file=stream)
    if report.notes:
        print("Compatibility notes:", file=stream)
        for note in report.notes:
            print(f"  - {note}", file=stream)


def _build_mapping_result(repo_path: Path, spec_name: str) -> tuple[dict, dict]:
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


def _resolve_confined_destination(base_dir: Path, relative_path: str) -> Path | None:
    destination = (base_dir / relative_path).resolve()
    try:
        destination.relative_to(base_dir)
    except ValueError:
        return None
    return destination


def cmd_analyze(args):
    """Analyze a repository (multi-language support)"""
    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing repository: {path}")
    print("-" * 50)

    # Use tree-sitter analyzer for multi-language support
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer(path)
    result = analyzer.analyze()

    print(f"\nLanguages detected: {', '.join(result.languages)}")
    print("\nLanguage statistics:")
    for stat in result.language_stats:
        print(f"  {stat.language}: {stat.file_count} files, {stat.line_count} lines")

    print(f"\nFrameworks detected: {', '.join(result.frameworks) if result.frameworks else 'None'}")

    print(f"\nEntry points: {len(result.entry_points)}")
    for ep in result.entry_points[:5]:
        print(f"  - {ep}")

    print(f"\nTest files: {len(result.test_files)}")

    # Find patterns for improvement
    if result.patterns:
        print("\nPatterns found for improvement:")
        for pattern_name, locations in result.patterns.items():
            print(f"  {pattern_name}: {len(locations)} locations")

    if args.output_json:
        output = {
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
        print(json.dumps(output, indent=2))


def cmd_search(args):
    """Search for research papers and implementations"""
    query = args.query

    print(f"Searching for: '{query}'")
    print("-" * 50)

    # Search local paper specs
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    extractor = ResearchExtractor()
    local_results = extractor.search_by_keyword(query, max_results=10)

    if local_results:
        print(f"\nLocal paper specs found: {len(local_results)}")
        for spec in local_results:
            print(f"\n  {spec['title']}")
            print(f"    Category: {spec['category']}")
            print(f"    Replaces: {spec['replaces']}")
            if spec.get("arxiv"):
                print(f"    arXiv: {spec['arxiv']}")

    # Search arXiv
    if args.arxiv:
        print("\nSearching arXiv...")
        try:
            from scholardevclaw.research_intelligence.extractor import ResearchQuery

            research_query = ResearchQuery(keywords=query.split(), max_results=args.max_results)

            import asyncio

            papers = asyncio.run(extractor.search_arxiv(research_query))

            if papers:
                print(f"\narXiv papers found: {len(papers)}")
                for paper in papers[:5]:
                    print(f"\n  {paper.title}")
                    print(f"    Authors: {', '.join(paper.authors[:3])}")
                    print(f"    Categories: {', '.join(paper.categories[:3])}")
                    print(f"    arXiv ID: {paper.arxiv_id}")
            else:
                print("\nNo papers found on arXiv.")
        except Exception as e:
            print(f"\nNote: arXiv search requires 'arxiv' package. Error: {e}")

    # Search web sources
    if args.web:
        print("\nSearching web sources...")
        try:
            from scholardevclaw.research_intelligence.web_research import SyncWebResearchEngine

            engine = SyncWebResearchEngine()
            web_results = engine.search_all(query, args.language, args.max_results)

            if web_results.get("github_repos"):
                print(f"\nGitHub repositories: {len(web_results['github_repos'])}")
                for repo in web_results["github_repos"][:3]:
                    print(f"  - {repo.owner}/{repo.name} ({repo.stars} stars)")

            if web_results.get("papers_with_code"):
                print(f"\nPapers with Code: {len(web_results['papers_with_code'])}")
                for paper in web_results["papers_with_code"][:3]:
                    print(f"  - {paper.title}")
        except Exception as e:
            print(f"Web search error: {e}")


def cmd_ingest(args):
    """Ingest a paper source into a structured PaperDocument JSON artifact."""
    from scholardevclaw.ingestion.paper_fetcher import (
        PaperFetchError,
        PaperIngester,
        PaperSourceResolutionError,
    )

    source = args.source.strip()
    if not source:
        print("Error: paper source must not be empty", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.cwd().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paper_document.json"

    print(f"Ingesting paper source: {source}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    ingester = PaperIngester()
    try:
        document = ingester.ingest(source, output_dir)
    except PaperSourceResolutionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except PaperFetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        print(f"Error: failed to ingest source '{source}': {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.write_text(json.dumps(document.to_dict(), indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"Title: {document.title or 'Unknown'}")
    print(f"Algorithms: {len(document.algorithms)}")
    print(f"Equations: {len(document.equations)}")
    print(f"Domain: {document.domain}")


def cmd_understand(args):
    """Run paper understanding agent over a paper_document JSON artifact."""
    input_path = Path(args.paper_document_json).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"Error: paper document JSON not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(payload, dict):
        print(
            f"Error: expected top-level JSON object in '{input_path}'",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    understanding_path = output_dir / "understanding.json"
    graph_path = output_dir / "concept_graph.json"

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY is required for understand command",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Understanding paper document: {input_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        from scholardevclaw.ingestion.models import PaperDocument
        from scholardevclaw.understanding.agent import UnderstandingAgent
        from scholardevclaw.understanding.graph import build_concept_graph, export_graph_json

        document = PaperDocument.from_dict(payload)
        agent = UnderstandingAgent(api_key=api_key, model=args.model)
        understanding = agent.understand(document)
        concept_graph = build_concept_graph(understanding)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: failed to understand paper: {exc}", file=sys.stderr)
        sys.exit(1)

    understanding_path.write_text(
        json.dumps(understanding.to_dict(), indent=2),
        encoding="utf-8",
    )
    graph_path.write_text(
        json.dumps(export_graph_json(concept_graph), indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {understanding_path}")
    print(f"Saved: {graph_path}")
    print(f"Complexity: {understanding.complexity}")
    print(f"Requirements: {len(understanding.requirements)}")
    print(f"Concept nodes: {len(understanding.concept_nodes)}")
    print(f"Concept edges: {len(understanding.concept_edges)}")


def cmd_plan(args):
    """Generate implementation plan from understanding.json artifact."""
    input_path = Path(args.understanding_json).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"Error: understanding JSON not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(payload, dict):
        print(
            f"Error: expected top-level JSON object in '{input_path}'",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "implementation_plan.json"

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is required for plan command", file=sys.stderr)
        sys.exit(1)

    print(f"Planning implementation from: {input_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        from scholardevclaw.ingestion.models import PaperDocument
        from scholardevclaw.planning import ImplementationPlanner
        from scholardevclaw.understanding.models import PaperUnderstanding

        understanding = PaperUnderstanding.from_dict(payload)
        year = payload.get("year")
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None

        domain = str(payload.get("domain", "")).strip().casefold()
        if domain not in {"cv", "nlp", "rl", "systems", "theory"}:
            domain_text = " ".join(
                [
                    understanding.paper_title,
                    understanding.problem_statement,
                    understanding.key_insight,
                    understanding.core_algorithm_description,
                    " ".join(req.name for req in understanding.requirements),
                ]
            ).casefold()
            if any(token in domain_text for token in ["transformer", "bert", "gpt", "attention"]):
                domain = "nlp"
            elif any(
                token in domain_text
                for token in ["convolution", "resnet", "yolo", "segmentation", "vision"]
            ):
                domain = "cv"
            elif any(
                token in domain_text for token in ["reward", "policy", "q-learning", "environment"]
            ):
                domain = "rl"
            elif any(token in domain_text for token in ["kernel", "mutex", "scheduler", "memory"]):
                domain = "systems"
            else:
                domain = "theory"

        doc = PaperDocument(
            title=understanding.paper_title or str(payload.get("paper_title", "Untitled Paper")),
            authors=[str(author) for author in payload.get("authors", [])],
            arxiv_id=str(payload["arxiv_id"]) if payload.get("arxiv_id") is not None else None,
            doi=str(payload["doi"]) if payload.get("doi") is not None else None,
            year=year,
            abstract=str(payload.get("abstract", ""))
            or understanding.one_line_summary
            or understanding.problem_statement,
            sections=[],
            equations=[],
            algorithms=[],
            figures=[],
            full_text=understanding.core_algorithm_description,
            pdf_path=None,
            references=[str(item) for item in payload.get("references", [])],
            keywords=[str(item) for item in payload.get("keywords", [])],
            domain=domain,
        )

        forced_stack = "numpy-only" if args.stack == "numpy" else args.stack
        planner = ImplementationPlanner(api_key=api_key, model="claude-opus-4-5")
        plan = planner.plan(understanding, doc, forced_stack=forced_stack)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: failed to plan implementation: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    stack_name = (
        plan.tech_stack or ("numpy-only" if args.stack == "numpy" else args.stack) or "unknown"
    )
    print(f"Saved: {output_path}")
    print(
        f"Project: {plan.project_name or 'unknown'} | "
        f"Stack: {stack_name} | Modules: {len(plan.modules)}"
    )


def cmd_suggest(args):
    """Get improvement suggestions based on code patterns"""
    if getattr(args, "use_specs", False):
        print("Using legacy specs-based workflow for suggest.")

    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing code patterns in: {path}")
    print("-" * 50)

    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer(path)
    suggestions = analyzer.suggest_research_papers()

    if not suggestions:
        print("\nNo improvement suggestions found.")
        print("Your code might already be using modern techniques, or")
        print("the patterns weren't recognized.")
        return

    print(f"\nFound {len(suggestions)} improvement opportunities:\n")

    for i, suggestion in enumerate(suggestions[:10], 1):
        print(f"{i}. {suggestion['paper']['title']}")
        print(f"   Pattern: {suggestion['pattern']}")
        print(f"   Confidence: {suggestion['confidence']:.0f}%")
        print(f"   Found in: {len(suggestion['locations'])} locations")
        print(f"   Category: {suggestion['paper']['category']}")
        print()


def cmd_map(args):
    """Map a research specification to repository locations"""
    if getattr(args, "use_specs", False):
        print("Using legacy specs-based workflow for map.")

    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Mapping spec '{args.spec}' to: {path}")
    print("-" * 50)

    try:
        mapping_result, spec = _build_mapping_result(path, args.spec)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    targets = mapping_result["targets"]
    print(f"\nAlgorithm: {spec['algorithm']['name']}")
    print(f"Strategy: {mapping_result['strategy']}")
    print(f"Confidence: {mapping_result['confidence']}%")
    print(f"Targets: {len(targets)}")

    for target in targets[:10]:
        print(f"  - {target['file']}:{target['line']} -> {target['current_code']}")

    if args.output_json:
        print(json.dumps(mapping_result, indent=2))


def _cmd_generate_legacy_specs(args) -> None:
    path = Path(args.arg1)
    spec_name = str(args.arg2)

    if not path.exists():
        print(f"Error: Repository not found: {args.arg1}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating patch for spec '{spec_name}' in: {path}")
    print("-" * 50)

    try:
        mapping_result, _ = _build_mapping_result(path, spec_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from scholardevclaw.patch_generation.generator import PatchGenerator

    generator = PatchGenerator(path)
    patch = generator.generate(mapping_result)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else path / "integration-patch"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for nf in patch.new_files:
        out_file = _resolve_confined_destination(output_dir, nf.path)
        if out_file is None:
            print(f"  ⚠ Skipped unsafe output path: {nf.path}")
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(nf.content)
        print(f"  ✓ Created: {out_file}")

    print(f"\nBranch: {patch.branch_name}")
    print(f"New files: {len(patch.new_files)}")
    print(f"Transformations: {len(patch.transformations)}")
    print(f"Patch output: {output_dir}")

    if args.output_json:
        output = {
            "branch_name": patch.branch_name,
            "algorithm_name": patch.algorithm_name,
            "paper_reference": patch.paper_reference,
            "new_files": [{"path": nf.path} for nf in patch.new_files],
            "transformations": [
                {
                    "file": t.file,
                    "changes": t.changes,
                }
                for t in patch.transformations
            ],
            "output_dir": str(output_dir),
        }
        print(json.dumps(output, indent=2))


def cmd_generate(args):
    """Generate implementation artifacts from plan+understanding, or legacy spec patches."""
    if getattr(args, "use_specs", False):
        print("Using legacy specs-based workflow for generate.")
        _cmd_generate_legacy_specs(args)
        return

    plan_path = Path(args.arg1).expanduser().resolve()
    understanding_path = Path(args.arg2).expanduser().resolve()

    if not plan_path.exists() or not plan_path.is_file():
        print(f"Error: implementation plan JSON not found: {plan_path}", file=sys.stderr)
        sys.exit(1)
    if not understanding_path.exists() or not understanding_path.is_file():
        print(f"Error: understanding JSON not found: {understanding_path}", file=sys.stderr)
        sys.exit(1)
    if args.max_parallel < 1:
        print("Error: --max-parallel must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in '{plan_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        understanding_payload = json.loads(understanding_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in '{understanding_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(plan_payload, dict):
        print(f"Error: expected top-level JSON object in '{plan_path}'", file=sys.stderr)
        sys.exit(1)
    if not isinstance(understanding_payload, dict):
        print(f"Error: expected top-level JSON object in '{understanding_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        from scholardevclaw.generation import CodeOrchestrator
        from scholardevclaw.planning.models import ImplementationPlan
        from scholardevclaw.understanding.models import PaperUnderstanding

        plan = ImplementationPlan.from_dict(plan_payload)
        understanding = PaperUnderstanding.from_dict(understanding_payload)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is required for generate command", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (plan_path.parent / (plan.project_name or "generated_project")).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generation_report.json"

    print(f"Generating implementation from: {plan_path}")
    print(f"Understanding source: {understanding_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        orchestrator = CodeOrchestrator(api_key=api_key, model=args.model)
        result = orchestrator.generate_sync(
            plan=plan,
            understanding=understanding,
            output_dir=output_dir,
            max_parallel=args.max_parallel,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: failed to generate implementation: {exc}", file=sys.stderr)
        sys.exit(1)

    report_payload = result.to_dict()
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Saved: {report_path}")
    print(
        "Success rate: "
        f"{result.success_rate:.0%} | "
        f"Modules: {len(result.module_results)} | "
        f"Duration: {result.duration_seconds:.2f}s"
    )

    if args.output_json:
        print(json.dumps(report_payload, indent=2))


def cmd_validate(args):
    """Run validation on a repository"""
    from scholardevclaw.application.pipeline import run_validate

    print(f"Validating repository: {args.repo_path}")
    print("-" * 50)

    result = run_validate(args.repo_path)
    if not result.ok and not result.payload:
        print(f"Error: {result.error or 'Validation failed'}", file=sys.stderr)
        sys.exit(1)

    payload = result.payload
    compat_report = evaluate_payload_compatibility(payload, expected_types={"validation"})
    print(f"Stage: {payload.get('stage')}")
    print(f"Passed: {'Yes' if payload.get('passed') else 'No'}")

    comparison = payload.get("comparison")
    if comparison:
        print("Comparison:")
        for key, value in comparison.items():
            print(f"  {key}: {value}")

    scorecard = payload.get("scorecard")
    if isinstance(scorecard, dict):
        print("Scorecard:")
        print(f"  Summary: {scorecard.get('summary')}")
        for highlight in scorecard.get("highlights", [])[:4]:
            print(f"  - {highlight}")

    if payload.get("error"):
        print(f"Error: {payload.get('error')}")

    if compat_report.issues or compat_report.warnings or compat_report.notes:
        _print_compatibility_report(payload, expected_type="validation")

    if args.output_json:
        print(json.dumps(payload, indent=2))


def cmd_integrate(args):
    """Full integration workflow"""
    if getattr(args, "use_specs", False):
        print("Using legacy specs-based workflow for integrate.")

    from scholardevclaw.application.pipeline import run_integrate

    def _print_log(line: str) -> None:
        print(f"  • {line}")

    print(f"Starting integration workflow for: {args.repo_path}")
    print(f"Target: {args.spec or 'auto-detect'}")
    print("=" * 60)

    result = run_integrate(
        args.repo_path,
        args.spec,
        dry_run=args.dry_run,
        require_clean=args.require_clean,
        output_dir=args.output_dir,
        log_callback=_print_log,
    )

    if not result.ok:
        print("\nIntegration failed.", file=sys.stderr)
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
        guidance = result.payload.get("guidance") if isinstance(result.payload, dict) else None
        if isinstance(guidance, list) and guidance:
            print("Guidance:", file=sys.stderr)
            for item in guidance:
                print(f"  - {item}", file=sys.stderr)
        _print_compatibility_report(result.payload, expected_type="integration", stderr=True)
        sys.exit(1)

    print("\n" + "=" * 60)
    compat_report = evaluate_payload_compatibility(result.payload, expected_types={"integration"})
    if result.payload.get("dry_run"):
        print("Dry-run complete (no patch generation/validation executed).")
    else:
        print("Integration complete!")

    print(f"Selected spec: {result.payload.get('spec')}")
    preflight = result.payload.get("preflight") or {}
    if preflight:
        print(f"Preflight clean state: {preflight.get('is_clean', 'unknown')}")

    validation = result.payload.get("validation")
    if validation:
        print(f"Validation stage: {validation.get('stage')}")
        print(f"Validation passed: {'Yes' if validation.get('passed') else 'No'}")
        scorecard = validation.get("scorecard")
        if isinstance(scorecard, dict):
            print(f"Validation summary: {scorecard.get('summary')}")
            for highlight in scorecard.get("highlights", [])[:3]:
                print(f"  - {highlight}")

    if compat_report.issues or compat_report.warnings or compat_report.notes:
        _print_compatibility_report(result.payload, expected_type="integration")

    if args.output_json:
        print(json.dumps(result.payload, indent=2))


def cmd_planner(args):
    """Plan multi-spec migration strategy"""
    from scholardevclaw.planner import run_planner

    print(f"Planning multi-spec migration for: {args.repo_path}")
    print("=" * 60)

    result = run_planner(
        args.repo_path,
        max_specs=args.max_specs,
        target_categories=args.categories.split(",") if args.categories else None,
    )

    if not result.ok:
        print(f"\nPlanning failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    payload = result.payload

    print(f"\nRepository: {payload.get('repo_path')}")
    print(f"Languages: {', '.join(payload.get('languages', []))}")
    print(f"Frameworks: {', '.join(payload.get('frameworks', []))}")
    print(f"\nOpportunities found: {payload.get('opportunities_found', 0)}")

    selected = payload.get("selected_specs", [])
    if selected:
        print(f"\nSelected specs ({len(selected)}):")
        for i, spec in enumerate(selected, 1):
            print(f"  {i}. {spec['name']} ({spec['category']})")
            print(f"     Confidence: {spec['confidence']:.0f}%")
            print(f"     Replaces: {spec['replaces']}")
            if spec.get("expected_benefits"):
                print(f"     Benefits: {', '.join(spec['expected_benefits'][:2])}")

        print(f"\nExecution order: {' -> '.join(payload.get('dependency_order', []))}")

        reasoning = payload.get("dependency_reasoning", [])
        if reasoning:
            print("\nDependency reasoning:")
            for r in reasoning:
                print(f"  • {r}")

        print(f"\n{payload.get('total_expected_improvement', 'N/A')}")

    available_cats = payload.get("available_categories", {})
    if available_cats and args.categories is None:
        print(f"\nAvailable categories: {', '.join(available_cats.keys())}")

    if args.output_json:
        print(json.dumps(payload, indent=2))


def cmd_critic(args):
    """Run critic to verify generated patches"""
    from scholardevclaw.critic import run_critic

    print(f"Running critic for: {args.repo_path}")
    print("=" * 60)

    patch_result = None
    if args.patch_json:
        import json as json_module

        try:
            patch_result = json_module.loads(args.patch_json)
        except json_module.JSONDecodeError as e:
            print(f"Error parsing patch JSON: {e}", file=sys.stderr)
            sys.exit(1)

    result = run_critic(
        args.repo_path,
        spec_name=args.spec,
        patch_result=patch_result,
    )

    if not result.ok:
        print("\nCritic found issues!", file=sys.stderr)
    else:
        print("\nCritic passed!")

    payload = result.payload

    print(f"\nSummary: {payload.get('summary', 'N/A').upper()}")

    severity = payload.get("severity_counts", {})
    if severity.get("error", 0) > 0:
        print(f"Errors: {severity.get('error', 0)}")
    if severity.get("warning", 0) > 0:
        print(f"Warnings: {severity.get('warning', 0)}")

    issues = payload.get("issues", [])
    if issues:
        print("\nIssues:")
        for issue in issues[:10]:
            severity_marker = "✗" if issue.get("severity") == "error" else "⚠"
            print(f"  {severity_marker} [{issue.get('type')}] {issue.get('message')}")
            if issue.get("file"):
                print(f"    File: {issue.get('file')}")

    warnings = payload.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for warning in warnings[:10]:
            print(f"  ⚠ [{warning.get('type')}] {warning.get('message')}")

    checks = payload.get("checks_passed", [])
    if checks:
        print("\nPassed checks:")
        for check in checks:
            print(f"  ✓ {check}")

    if args.output_json:
        print(json.dumps(payload, indent=2))


def cmd_specs(args):
    """List available paper specifications"""
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    extractor = ResearchExtractor()

    if args.list:
        specs = extractor.list_available_specs()
        categories = extractor.get_categories()

        print("Available paper specifications:\n")

        for category, spec_names in sorted(categories.items()):
            print(f"\n{category.upper()}")
            print("-" * 40)

            for spec_name in spec_names:
                spec = extractor.get_spec(spec_name)
                if spec:
                    print(f"\n  {spec_name}")
                    print(f"    Title: {spec['paper']['title']}")
                    print(f"    Replaces: {spec['algorithm']['replaces']}")
                    if spec["paper"].get("arxiv"):
                        print(f"    arXiv: {spec['paper']['arxiv']}")
                    print(f"    Benefits: {', '.join(spec['changes']['expected_benefits'][:2])}")
    elif args.categories:
        categories = extractor.get_categories()
        print("Categories:\n")
        for category, specs in sorted(categories.items()):
            print(f"  {category}: {len(specs)} specs")
    else:
        specs = extractor.list_available_specs()
        print(f"Available specs ({len(specs)} total):")
        print(", ".join(specs))
        print("\nUse --list for detailed info or --categories for category view")


def cmd_context(args):
    """Manage project context and memory"""
    from scholardevclaw.context_engine import ContextEngine
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    engine = ContextEngine()

    if args.context_action == "init":
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        print(f"Initializing context for: {args.repo_path}")

        analyzer = TreeSitterAnalyzer(Path(args.repo_path))
        analysis = analyzer.analyze()

        context_data = {
            "languages": analysis.languages,
            "frameworks": analysis.frameworks,
            "entry_points": analysis.entry_points,
            "patterns": analysis.patterns,
        }

        engine.initialize_project(args.repo_path, context_data)
        print("Context initialized!")
        print(f"  Languages: {', '.join(context_data['languages'])}")
        print(f"  Frameworks: {', '.join(context_data['frameworks'])}")

    elif args.context_action == "history":
        history = engine.get_integration_history(args.repo_path)

        if not history:
            print("No integration history found.")
            return

        print(f"Integration history for: {args.repo_path}")
        print("-" * 50)

        for record in history[-10:]:
            status_icon = (
                "✓"
                if record.get("validation_passed")
                else "✗"
                if record.get("status") == "failed"
                else "?"
            )
            print(f"  {status_icon} {record['spec']} - {record['status']}")
            print(f"    {record['timestamp'][:10]}")

    elif args.context_action == "summary":
        summary = engine.get_context_summary(args.repo_path)

        print(f"Context summary for: {summary['repo_path']}")
        print("=" * 50)

        print(f"\nLanguages: {', '.join(summary['languages']) or 'None'}")
        print(f"Frameworks: {', '.join(summary['frameworks']) or 'None'}")

        stats = summary.get("stats", {})
        print("\nStatistics:")
        print(f"  Total runs: {stats.get('total_runs', 0)}")
        print(f"  Successful: {stats.get('successful_runs', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.0%}")

        print(f"\nSuccessful specs: {', '.join(stats.get('successful_specs', [])) or 'None'}")
        print(f"Failed specs: {', '.join(stats.get('failed_specs', [])) or 'None'}")

        prefs = summary.get("preferences", {})
        print("\nPreferences:")
        print(f"  Preferred specs: {', '.join(prefs.get('preferred_specs', [])) or 'None'}")
        print(
            f"  Preferred categories: {', '.join(prefs.get('preferred_categories', [])) or 'None'}"
        )
        print(f"  Require validation: {prefs.get('require_validation', True)}")

    elif args.context_action == "recommend":
        extractor = ResearchExtractor()
        available_specs = extractor.list_available_specs()

        result = engine.get_recommendation(args.repo_path, available_specs)

        print(f"Agent Brain Recommendation for: {args.repo_path}")
        print("=" * 50)

        print(f"\n{result.recommendation}")
        print(f"Confidence: {result.confidence:.0%}")

        if result.reasoning:
            print("\nReasoning:")
            for reason in result.reasoning:
                print(f"  • {reason}")

        context_used = result.context_used
        print("\nContext used:")
        print(f"  Total runs: {context_used.get('total_runs', 0)}")
        print(f"  Success rate: {context_used.get('success_rate', 0):.0%}")

    elif args.context_action == "set":
        engine.set_preference(args.repo_path, args.pref_type, args.pref_value)
        print(f"Set preference: {args.pref_type} = {args.pref_value}")

    elif args.context_action == "clear":
        confirm = input(f"Clear all context for {args.repo_path}? (y/n): ")
        if confirm.lower() == "y":
            engine.clear_project_memory(args.repo_path)
            print("Context cleared!")
        else:
            print("Cancelled.")

    elif args.context_action == "list":
        projects = engine.list_tracked_projects()

        if not projects:
            print("No tracked projects.")
            return

        print("Tracked projects:")
        for project in projects:
            print(f"  - {project}")


def cmd_experiment(args):
    """Run experiment loop for hypothesis testing"""
    from scholardevclaw.experiment import run_experiment

    print(f"Running experiment for: {args.repo_path}")
    print(f"Spec: {args.spec}")
    print(f"Variants: {args.variants}")
    print("=" * 60)

    def _log(line):
        print(f"  • {line}")

    result = run_experiment(
        args.repo_path,
        args.spec,
        variant_count=args.variants,
        output_dir=args.output_dir,
        log_callback=_log,
    )

    if not result.get("ok"):
        print(f"\nExperiment failed: {result.get('error')}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    summary = result.get("summary", {})
    print(f"\nTotal variants: {summary.get('total_variants', 0)}")
    print(f"Completed: {summary.get('completed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")

    if summary.get("best_variant"):
        print(f"\n🏆 Best variant: {summary.get('best_variant')}")
        print(f"   Score: {summary.get('best_score', 0):.2f}")
        print(f"   {summary.get('recommendation', '')}")

    print("\nRanked Results:")
    print("-" * 50)
    for r in result.get("results", []):
        rank = r.get("rank", "?")
        name = r.get("variant_name", "unknown")
        score = r.get("score", 0)
        status = r.get("status", "unknown")

        if status == "completed":
            metrics = r.get("metrics", {})
            speedup = metrics.get("speedup", 0)
            print(f"  {rank}. {name} - score: {score:.2f}, speedup: {speedup:.2f}x")
        else:
            print(f"  {rank}. {name} - {status}")

    if args.output_json:
        print(json.dumps(result, indent=2))


def cmd_plugin(args):
    """Manage plugins"""
    from scholardevclaw.plugins import get_plugin_manager

    manager = get_plugin_manager()

    if args.plugin_action == "list":
        print("ScholarDevClaw Plugins")
        print("=" * 50)

        discovered = manager.discover_plugins()
        loaded = manager.list_plugins()

        print(f"\nDiscovered plugins ({len(discovered)}):")
        for p in discovered:
            enabled = manager.is_enabled(p.name) if hasattr(manager, "is_enabled") else True
            status = "enabled" if enabled else "disabled"
            print(f"  {'*' if enabled else '-'} {p.name} ({p.plugin_type}) - {p.description}")
            print(f"    Version: {p.version} | Author: {p.author} | Status: {status}")

        print(f"\nLoaded plugins ({len(loaded)}):")
        for p in loaded:
            print(f"  + {p.name} ({p.plugin_type})")

    elif args.plugin_action == "load":
        plugin = manager.load_plugin(args.plugin_name)
        if plugin:
            print(f"Loaded plugin: {plugin.metadata.name}")
            print(f"  Type: {plugin.metadata.plugin_type}")
            print(f"  Version: {plugin.metadata.version}")
        else:
            print(f"Failed to load plugin: {args.plugin_name}", file=sys.stderr)
            sys.exit(1)

    elif args.plugin_action == "unload":
        manager.unload_plugin(args.plugin_name)
        print(f"Unloaded plugin: {args.plugin_name}")

    elif args.plugin_action == "enable":
        if not args.plugin_name:
            print("Error: plugin name required for enable", file=sys.stderr)
            sys.exit(1)
        if hasattr(manager, "enable_plugin"):
            manager.enable_plugin(args.plugin_name)
            print(f"Enabled plugin: {args.plugin_name}")
        else:
            print("Enable/disable not supported by this plugin manager", file=sys.stderr)
            sys.exit(1)

    elif args.plugin_action == "disable":
        if not args.plugin_name:
            print("Error: plugin name required for disable", file=sys.stderr)
            sys.exit(1)
        if hasattr(manager, "disable_plugin"):
            manager.disable_plugin(args.plugin_name)
            print(f"Disabled plugin: {args.plugin_name}")
        else:
            print("Enable/disable not supported by this plugin manager", file=sys.stderr)
            sys.exit(1)

    elif args.plugin_action == "hooks":
        from scholardevclaw.plugins.hooks import HookPoint, get_hook_registry

        registry = get_hook_registry()

        # Load all plugins so their hooks are registered
        manager.load_all()

        all_hooks = registry.list_hooks()
        if not all_hooks:
            print("No hooks registered. Load plugins first with 'plugin load <name>'.")
            return

        print("Registered Plugin Hooks")
        print("=" * 60)
        print(f"Total callbacks: {registry.hook_count}")
        print()

        # Group by hook point
        by_point: dict[str, list[dict]] = {}
        for h in all_hooks:
            by_point.setdefault(h["hook"], []).append(h)

        for hp in HookPoint:
            entries = by_point.get(hp.value, [])
            if entries:
                print(f"  {hp.value} ({len(entries)} callbacks):")
                for e in entries:
                    print(f"    [{e['priority']:>3}] {e['plugin']}")
        print()

        # Show execution log if any
        log = registry.get_execution_log()
        if log:
            print(f"Recent execution log ({len(log)} entries):")
            for entry in log[-10:]:
                status = "OK" if entry.get("error") is None else "ERR"
                print(
                    f"  [{status}] {entry['hook']} <- {entry['plugin']} ({entry['elapsed_ms']}ms)"
                )

    elif args.plugin_action == "analyze":
        analyzer = manager.get_analyzer(args.plugin_name)
        if analyzer:
            result = analyzer.analyze(args.repo_path)
            print(f"Analysis results for: {args.repo_path}")
            print(f"  Languages: {', '.join(result.get('languages', []))}")
            print(f"  Frameworks: {', '.join(result.get('frameworks', []))}")
        else:
            print(f"Analyzer not found: {args.plugin_name}", file=sys.stderr)
            sys.exit(1)

    elif args.plugin_action == "validate":
        validator = manager.get_validator(args.plugin_name)
        if validator:
            patch_result = {"new_files": [], "transformations": []}
            result = validator.validate(args.repo_path, patch_result)
            print(f"Validation results ({validator.get_validation_type()}):")
            print(f"  Passed: {result.get('passed', False)}")
            issues = result.get("issues", [])
            if issues:
                print(f"  Issues: {len(issues)}")
                for issue in issues[:5]:
                    print(f"    - {issue.get('message')}")
        else:
            print(f"Validator not found: {args.plugin_name}", file=sys.stderr)
            sys.exit(1)

    elif args.plugin_action == "scaffold":
        scaffold_file = manager.create_plugin_scaffold(
            args.plugin_name,
            args.plugin_type or "custom",
        )
        print(f"Created plugin scaffold: {scaffold_file}")
        print("  Edit this file to implement your plugin")

    elif args.plugin_action == "info":
        plugin = manager.get_plugin(args.plugin_name)
        if plugin:
            meta = plugin.metadata
            print(f"Plugin: {meta.name}")
            print("=" * 50)
            print(f"  Version: {meta.version}")
            print(f"  Type: {meta.plugin_type}")
            print(f"  Description: {meta.description}")
            print(f"  Author: {meta.author}")
            print(f"  Entry point: {meta.entry_point}")
            if hasattr(meta, "hooks") and meta.hooks:
                print(f"  Hooks: {', '.join(meta.hooks)}")
            enabled = manager.is_enabled(meta.name) if hasattr(manager, "is_enabled") else True
            print(f"  Enabled: {enabled}")
        else:
            print(f"Plugin not found: {args.plugin_name}", file=sys.stderr)
            sys.exit(1)


def cmd_rollback(args):
    """Manage rollback snapshots and revert integrations"""
    from scholardevclaw.rollback import (
        RollbackManager,
        RollbackStatus,
        get_rollback_status,
        list_rollback_snapshots,
    )

    manager = RollbackManager()

    if args.rollback_action == "list":
        status_filter = None
        if args.status:
            status_filter = RollbackStatus(args.status)

        snapshots = list_rollback_snapshots(args.repo_path, status=status_filter)

        if not snapshots:
            print("No rollback snapshots found.")
            return

        print("Rollback Snapshots")
        print("=" * 70)
        print(f"{'ID':<30} {'Spec':<15} {'Status':<12} {'Changes':<8} {'Timestamp'}")
        print("-" * 70)

        for s in snapshots:
            print(
                f"{s['id']:<30} {s['spec']:<15} {s['status']:<12} {s['changes_count']:<8} {s['timestamp'][:19]}"
            )

        if args.output_json:
            print(json.dumps(snapshots, indent=2))

    elif args.rollback_action == "show":
        if not args.snapshot_id:
            print("Error: --snapshot-id is required for 'show' action", file=sys.stderr)
            sys.exit(1)

        snapshot = manager.get_snapshot(args.repo_path, args.snapshot_id)
        if not snapshot:
            print(f"Snapshot not found: {args.snapshot_id}", file=sys.stderr)
            sys.exit(1)

        print(f"Rollback Snapshot: {snapshot.id}")
        print("=" * 70)
        print(f"  Spec: {snapshot.spec}")
        print(f"  Status: {snapshot.status.value}")
        print(f"  Timestamp: {snapshot.timestamp}")
        print(f"  Description: {snapshot.description or 'N/A'}")

        if snapshot.git_snapshot:
            print("\nGit State:")
            print(f"  Branch: {snapshot.git_snapshot.branch or 'N/A'}")
            print(
                f"  Commit: {snapshot.git_snapshot.commit_sha[:8] if snapshot.git_snapshot.commit_sha else 'N/A'}"
            )
            print(f"  Was clean: {snapshot.git_snapshot.is_clean}")
            if snapshot.git_snapshot.created_branch:
                print(f"  Created branch: {snapshot.git_snapshot.created_branch}")

        if snapshot.changes:
            print(f"\nChanges ({len(snapshot.changes)}):")
            for change in snapshot.changes:
                print(f"  • {change.change_type.value}: {change.path}")

        if snapshot.rolled_back_at:
            print(f"\nRolled back at: {snapshot.rolled_back_at}")

        if args.output_json:
            print(json.dumps(snapshot.to_dict(), indent=2))

    elif args.rollback_action == "run":

        def _log(line):
            print(f"  • {line}")

        result = manager.rollback(
            args.repo_path,
            args.snapshot_id,
            force=args.force,
            log_callback=_log,
        )

        print("\nRollback Result")
        print("=" * 70)
        print(f"  Status: {'SUCCESS' if result.ok else 'FAILED'}")
        print(f"  Snapshot: {result.snapshot_id}")
        print(f"  Changes reverted: {result.changes_reverted}")
        print(f"  Files restored: {len(result.files_restored)}")

        if result.files_restored:
            print("\nRestored files:")
            for f in result.files_restored[:10]:
                print(f"  • {f}")

        if result.branches_deleted:
            print("\nDeleted branches:")
            for b in result.branches_deleted:
                print(f"  • {b}")

        if result.error:
            print(f"\nError: {result.error}")

        if args.output_json:
            print(
                json.dumps(
                    {
                        "ok": result.ok,
                        "snapshot_id": result.snapshot_id,
                        "status": result.status.value,
                        "changes_reverted": result.changes_reverted,
                        "files_restored": result.files_restored,
                        "branches_deleted": result.branches_deleted,
                        "error": result.error,
                    },
                    indent=2,
                )
            )

        if not result.ok:
            sys.exit(1)

    elif args.rollback_action == "status":
        status_info = get_rollback_status(args.repo_path, args.snapshot_id)

        if not status_info.get("found"):
            print("No active rollback snapshot found.")
            return

        print("Rollback Status")
        print("=" * 70)
        print(f"  Snapshot ID: {status_info['id']}")
        print(f"  Spec: {status_info['spec']}")
        print(f"  Status: {status_info['status']}")
        print(f"  Timestamp: {status_info['timestamp']}")
        print(f"  Description: {status_info.get('description', 'N/A')}")
        print(f"  Changes: {status_info['changes_count']}")

        if status_info.get("git_branch"):
            print(f"  Git branch: {status_info['git_branch']}")

        if status_info.get("rolled_back_at"):
            print(f"  Rolled back at: {status_info['rolled_back_at']}")

        if args.output_json:
            print(json.dumps(status_info, indent=2))

    elif args.rollback_action == "delete":
        if not args.snapshot_id:
            print("Error: --snapshot-id is required for 'delete' action", file=sys.stderr)
            sys.exit(1)

        snapshot = manager.get_snapshot(args.repo_path, args.snapshot_id)
        if not snapshot:
            print(f"Snapshot not found: {args.snapshot_id}", file=sys.stderr)
            sys.exit(1)

        if snapshot.status == RollbackStatus.APPLIED:
            print(f"Warning: Snapshot '{args.snapshot_id}' is in 'applied' state.")
            print("  Consider running 'rollback run' first to revert the changes.")
            if not args.force:
                print("  Use --force to delete anyway.")
                sys.exit(1)

        deleted = manager.delete_snapshot(args.repo_path, args.snapshot_id)
        if deleted:
            print(f"Deleted snapshot: {args.snapshot_id}")
        else:
            print(f"Failed to delete snapshot: {args.snapshot_id}", file=sys.stderr)
            sys.exit(1)


def cmd_demo(args):
    """Run end-to-end demo: clone nanoGPT (if needed), then analyze -> suggest -> map -> generate -> validate."""
    import subprocess as _sp
    import time as _time

    # ── Resolve demo repository ──────────────────────────────────────
    cli_dir = Path(__file__).parent
    project_root = cli_dir.parent.parent.parent

    repo_path: Path | None = None
    if getattr(args, "repo", None):
        repo_path = Path(args.repo).expanduser().resolve()
        if not repo_path.exists():
            print(f"Error: repository not found at {repo_path}", file=sys.stderr)
            sys.exit(1)
    else:
        repo_path = project_root / "test_repos" / "nanogpt"
        if not repo_path.exists():
            print("[setup] nanoGPT not found locally — cloning from GitHub...")
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _sp.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/karpathy/nanoGPT.git",
                        str(repo_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                print(f"[setup] Cloned to {repo_path}")
            except Exception as exc:
                print(f"Error: Failed to clone nanoGPT: {exc}", file=sys.stderr)
                print(
                    "Clone manually: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
                )
                sys.exit(1)

    # ── Resolve which specs to demo ──────────────────────────────────
    from scholardevclaw.research_intelligence.extractor import PAPER_SPECS

    demo_all = getattr(args, "all_specs", False)
    chosen_spec = getattr(args, "spec", None)
    output_dir = getattr(args, "output_dir", None)
    skip_validate = getattr(args, "skip_validate", False)
    output_json = getattr(args, "output_json", False)

    if demo_all:
        spec_names = sorted(PAPER_SPECS.keys())
    elif chosen_spec:
        if chosen_spec not in PAPER_SPECS:
            print(f"Error: Unknown spec '{chosen_spec}'", file=sys.stderr)
            print(f"Available: {', '.join(sorted(PAPER_SPECS.keys()))}")
            sys.exit(1)
        spec_names = [chosen_spec]
    else:
        # Default: demonstrate a curated set that covers all categories
        spec_names = ["rmsnorm", "swiglu", "flashattention", "rope", "cosine_warmup"]
        spec_names = [s for s in spec_names if s in PAPER_SPECS]

    # ── Imports ───────────────────────────────────────────────────────
    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.patch_generation.generator import PatchGenerator
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor
    from scholardevclaw.validation.runner import ValidationRunner

    total_steps = 3 + len(spec_names) * (2 if skip_validate else 3)
    step = 0
    t_start = _time.perf_counter()

    def _step(msg: str) -> None:
        nonlocal step
        step += 1
        print(f"\n[{step}/{total_steps}] {msg}")

    # ── Header ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ScholarDevClaw — End-to-End Demo")
    print("=" * 60)
    print(f"  Repository : {repo_path}")
    print(f"  Specs      : {', '.join(spec_names)}")
    if output_dir:
        print(f"  Output     : {output_dir}")
    print()

    # ── Step 1: Analyze ───────────────────────────────────────────────
    _step("Analyzing repository with tree-sitter AST extraction...")
    t0 = _time.perf_counter()
    analyzer = TreeSitterAnalyzer(repo_path)
    analysis = analyzer.analyze()
    dt = _time.perf_counter() - t0

    lang_list = ", ".join(analysis.languages) if analysis.languages else "none"
    print(f"  Languages     : {lang_list}")
    print(f"  Files scanned : {sum(s.file_count for s in analysis.language_stats)}")
    print(f"  Code elements : {len(analysis.elements)}")
    print(f"  Imports       : {len(analysis.imports)}")
    print(f"  Patterns      : {dict(analysis.patterns)}")
    print(f"  Frameworks    : {', '.join(analysis.frameworks) if analysis.frameworks else 'none'}")
    print(f"  Time          : {dt:.2f}s")

    # ── Step 2: Suggest ───────────────────────────────────────────────
    _step("Scanning for research improvement opportunities...")
    t0 = _time.perf_counter()
    suggestions = analyzer.suggest_research_papers()
    dt = _time.perf_counter() - t0

    print(f"  Found {len(suggestions)} opportunities ({dt:.2f}s)")
    for s in suggestions[:8]:
        paper = s.get("paper", {})
        title = paper.get("title", paper.get("name", "?"))
        conf = s.get("confidence", 0)
        print(f"    - {s['pattern']}: {title} ({conf:.0f}%)")

    # ── Step 3: Research specs ────────────────────────────────────────
    _step("Loading research specifications...")
    extractor = ResearchExtractor()
    specs_loaded = {}
    for name in spec_names:
        spec = extractor.get_spec(name)
        if spec:
            specs_loaded[name] = spec
            algo = spec.get("algorithm", {})
            print(
                f"  [{name}] {algo.get('name', '?')} — replaces {algo.get('replaces', '?')} ({algo.get('category', '?')})"
            )
        else:
            print(f"  [{name}] WARNING: spec not found, skipping")
    if not specs_loaded:
        print("Error: No specs could be loaded", file=sys.stderr)
        sys.exit(1)

    # ── Per-spec pipeline: Map -> Generate -> Validate ────────────────
    results = []
    output_path = Path(output_dir).expanduser().resolve() if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    for spec_name, spec in specs_loaded.items():
        spec_result: dict = {
            "spec": spec_name,
            "algorithm": spec.get("algorithm", {}).get("name", "?"),
        }

        # ── Map ───────────────────────────────────────────────────────
        _step(f"Mapping [{spec_name}] to repository code locations...")
        t0 = _time.perf_counter()
        engine = MappingEngine(analysis.__dict__, spec)
        mapping = engine.map()
        dt = _time.perf_counter() - t0

        targets = mapping.targets
        print(f"  Targets found : {len(targets)} ({dt:.2f}s)")
        print(f"  Strategy      : {mapping.strategy}")
        print(f"  Confidence    : {mapping.confidence}%")
        for t in targets[:5]:
            print(f"    - {t.file}:{t.line}")
        if len(targets) > 5:
            print(f"    ... and {len(targets) - 5} more")

        mapping_result = {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "current_code": t.current_code,
                    "replacement_required": t.replacement_required,
                    "context": t.context,
                }
                for t in targets
            ],
            "strategy": mapping.strategy,
            "confidence": mapping.confidence,
            "research_spec": spec,
        }
        spec_result["mapping"] = {
            "target_count": len(targets),
            "strategy": mapping.strategy,
            "confidence": mapping.confidence,
        }

        # ── Generate ──────────────────────────────────────────────────
        _step(f"Generating patch artifacts for [{spec_name}]...")
        t0 = _time.perf_counter()
        generator = PatchGenerator(repo_path)
        patch = generator.generate(mapping_result)
        dt = _time.perf_counter() - t0

        print(f"  Branch          : {patch.branch_name}")
        print(f"  New files       : {len(patch.new_files)}")
        print(f"  Transformations : {len(patch.transformations)}")
        print(f"  Time            : {dt:.2f}s")

        written_files: list[str] = []
        if output_path:
            spec_dir = output_path / spec_name
            spec_dir.mkdir(parents=True, exist_ok=True)
            for new_file in patch.new_files:
                dest = _resolve_confined_destination(spec_dir, new_file.path)
                if dest is None:
                    print(f"    ⚠ Skipped unsafe output path: {new_file.path}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_file.content)
                written_files.append(str(dest))
                print(f"    Wrote: {dest}")
            for xform in patch.transformations:
                xform_file_name = f"transform_{xform.file.replace('/', '_')}.diff"
                xform_file = _resolve_confined_destination(spec_dir, xform_file_name)
                if xform_file is None:
                    print(f"    ⚠ Skipped unsafe transformation output path: {xform_file_name}")
                    continue
                diff_content = f"--- a/{xform.file}\n+++ b/{xform.file}\n"
                if xform.changes:
                    for c in xform.changes:
                        diff_content += f"# {c.get('type', 'change')}: {c.get('from', '?')} -> {c.get('to', '?')}\n"
                diff_content += f"\n-{xform.original[:200]}\n+{xform.modified[:200]}\n"
                xform_file.write_text(diff_content)
                written_files.append(str(xform_file))

        spec_result["generation"] = {
            "branch": patch.branch_name,
            "new_files": len(patch.new_files),
            "transformations": len(patch.transformations),
            "written_files": written_files,
        }

        # ── Validate ──────────────────────────────────────────────────
        if not skip_validate:
            _step(f"Validating [{spec_name}] with real benchmarks...")
            t0 = _time.perf_counter()
            runner = ValidationRunner(repo_path)
            validation = runner.run(
                {
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
                },
                str(repo_path),
            )
            dt = _time.perf_counter() - t0

            print(f"  Stage  : {validation.stage}")
            print(f"  Passed : {'Yes' if validation.passed else 'No'}")
            if hasattr(validation, "comparison") and validation.comparison:
                comp = validation.comparison
                if isinstance(comp, dict):
                    speedup = comp.get("speedup", "N/A")
                    print(f"  Speedup: {speedup}")
            print(f"  Time   : {dt:.2f}s")

            spec_result["validation"] = {
                "passed": validation.passed,
                "stage": validation.stage,
            }

        results.append(spec_result)

    # ── Summary report ────────────────────────────────────────────────
    dt_total = _time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print("  Demo Results Summary")
    print("=" * 60)
    print(f"  Repository    : {repo_path}")
    print(f"  Total time    : {dt_total:.1f}s")
    print(f"  Specs demoed  : {len(results)}")
    print()

    # Per-spec summary table
    print(f"  {'Spec':<24} {'Targets':>8} {'Files':>6} {'Xforms':>7} {'Conf':>5}  {'Valid':>6}")
    print(f"  {'-' * 24} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 5}  {'-' * 6}")
    for r in results:
        m = r.get("mapping", {})
        g = r.get("generation", {})
        v = r.get("validation", {})
        valid_str = "Yes" if v.get("passed") else ("No" if v else "skip")
        print(
            f"  {r['spec']:<24} {m.get('target_count', 0):>8} "
            f"{g.get('new_files', 0):>6} {g.get('transformations', 0):>7} "
            f"{m.get('confidence', 0):>4}%  {valid_str:>6}"
        )

    if output_path:
        print(f"\n  Patch artifacts written to: {output_path}")

    print("\n  Your repository is ready for research-driven improvements.")
    print(f"  Next: scholardevclaw integrate {repo_path} <spec>\n")

    # ── JSON output ───────────────────────────────────────────────────
    if output_json:
        import json as _json

        report = {
            "repository": str(repo_path),
            "total_time_seconds": round(dt_total, 2),
            "analysis": {
                "languages": analysis.languages,
                "file_count": sum(s.file_count for s in analysis.language_stats),
                "elements": len(analysis.elements),
                "imports": len(analysis.imports),
                "frameworks": analysis.frameworks,
            },
            "suggestions": len(suggestions),
            "specs": results,
        }
        print(_json.dumps(report, indent=2))


def cmd_github_app(args):
    """Manage GitHub App and webhooks"""
    from scholardevclaw.github_app import GitHubAppClient, create_app
    from scholardevclaw.github_app.types import GitHubAppConfig

    if args.github_action == "setup":
        print("GitHub App Setup")
        print("=" * 50)

        config = GitHubAppConfig.from_env()

        if config.is_configured():
            print("GitHub App is already configured!")
            print(f"  App ID: {config.app_id}")
            if config.allowed_repositories:
                print(f"  Allowed repos: {', '.join(config.allowed_repositories)}")
            print("\nTo update configuration, set environment variables:")
            print("  GITHUB_APP_ID")
            print("  GITHUB_APP_PRIVATE_KEY")
            print("  GITHUB_APP_WEBHOOK_SECRET")
        else:
            print("GitHub App is not configured.")
            print("\nTo configure, set these environment variables:")
            print("  GITHUB_APP_ID=your_app_id")
            print("  GITHUB_APP_PRIVATE_KEY=/path/to/private-key.pem")
            print("  GITHUB_APP_WEBHOOK_SECRET=your_webhook_secret")
            print("\nThen create a GitHub App using the manifest:")
            print("  scholardevclaw github-app manifest")

    elif args.github_action == "manifest":
        from pathlib import Path

        manifest_path = Path(__file__).parent.parent / "github_app" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if args.server_url:
                manifest["hook_attributes"]["url"] = f"{args.server_url}/webhook"

            print("GitHub App Manifest")
            print("=" * 50)
            print(json.dumps(manifest, indent=2))
            print("\nTo create the GitHub App:")
            print("1. Go to https://github.com/settings/apps/new")
            print("2. Paste the manifest JSON above")
            print("3. Install the app on your repositories")
        else:
            print("Manifest file not found", file=sys.stderr)
            sys.exit(1)

    elif args.github_action == "server":
        config = GitHubAppConfig.from_env()

        if not config.is_configured():
            print("Error: GitHub App is not configured", file=sys.stderr)
            print("Run 'scholardevclaw github-app setup' for instructions")
            sys.exit(1)

        print(f"Starting GitHub App webhook server on port {args.port}...")
        print("Press Ctrl+C to stop")

        import uvicorn

        app = create_app(config)
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    elif args.github_action == "status":
        config = GitHubAppConfig.from_env()

        print("GitHub App Status")
        print("=" * 50)
        print(f"Configured: {'Yes' if config.is_configured() else 'No'}")

        if config.is_configured():
            print(f"App ID: {config.app_id}")
            print(f"Auto-apply safe patches: {config.auto_apply_safe_patches}")
            print(f"Require approval: {config.require_approval}")
            print(f"Notify on complete: {config.notify_on_complete}")

            if config.allowed_repositories:
                print(f"Allowed repositories: {', '.join(config.allowed_repositories)}")

            client = GitHubAppClient(config)
            print("\nWebhook endpoint: /webhook")
            print("Health check: /health")

        if args.output_json:
            print(
                json.dumps(
                    {
                        "configured": config.is_configured(),
                        "app_id": config.app_id,
                        "auto_apply": config.auto_apply_safe_patches,
                        "require_approval": config.require_approval,
                        "allowed_repos": config.allowed_repositories,
                    },
                    indent=2,
                )
            )

    elif args.github_action == "test-webhook":
        config = GitHubAppConfig.from_env()

        if not config.is_configured():
            print("Error: GitHub App is not configured", file=sys.stderr)
            sys.exit(1)

        print("Testing webhook signature verification...")

        import hashlib
        import hmac

        test_payload = b'{"action": "opened", "repository": {"name": "test"}}'
        signature = (
            "sha256="
            + hmac.new(config.webhook_secret.encode(), test_payload, hashlib.sha256).hexdigest()
        )

        client = GitHubAppClient(config)
        is_valid = client.verify_webhook_signature(test_payload, signature)

        if is_valid:
            print("✓ Webhook signature verification working!")
        else:
            print("✗ Webhook signature verification failed!")
            sys.exit(1)


def cmd_security(args):
    """Run security scans on repository"""
    from scholardevclaw.security.scanner import SecurityScanner

    print(f"Running security scan on: {args.repo_path}")
    print("=" * 50)

    scanner = SecurityScanner()

    availability = scanner.is_available()
    if args.security_action == "check":
        print("Security Scanner Availability")
        print("-" * 50)
        print(f"Bandit (Python):  {'✓ Available' if availability['bandit'] else '✗ Not installed'}")
        print(
            f"Semgrep (Multi):  {'✓ Available' if availability['semgrep'] else '✗ Not installed'}"
        )

        if args.output_json:
            print(json.dumps(availability, indent=2))
        return

    tools = None
    if args.tools:
        tools = args.tools.split(",")
        print(f"Running tools: {', '.join(tools)}")
    else:
        print("Running all available tools...")

    result = scanner.scan(args.repo_path, tools=tools)

    print("\nSecurity Scan Results")
    print("=" * 50)
    print(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")

    for scan in result.scans:
        print(f"\n{scan.tool.value.upper()}:")
        print(f"  Scan time: {scan.scan_time_seconds:.2f}s")
        print(f"  Findings: {len(scan.findings)}")
        print(f"    High:   {scan.high_severity_count}")
        print(f"    Medium: {scan.medium_severity_count}")
        print(f"    Low:    {scan.low_severity_count}")

        if scan.findings and args.verbose:
            print("\n  Top findings:")
            for f in scan.findings[:5]:
                print(
                    f"    [{f.severity.value.upper()}] {f.file_path}:{f.line_number or '?'} - {f.message[:60]}..."
                )

    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2))

    if not result.passed:
        sys.exit(1)


def cmd_workspace(args):
    """Manage CLI-first multi-repo workspace."""
    from scholardevclaw.multi_repo.manager import MultiRepoManager

    action = args.workspace_action
    repo_ref = args.repo_id_or_path
    manager = MultiRepoManager()

    if action == "add":
        if not repo_ref:
            print("Error: repository path is required for 'workspace add'", file=sys.stderr)
            sys.exit(1)

        repo_path = Path(repo_ref).expanduser().resolve()
        if not repo_path.exists():
            print(f"Error: repository not found: {repo_ref}", file=sys.stderr)
            sys.exit(1)
        if not repo_path.is_dir():
            print(f"Error: repository path must be a directory: {repo_ref}", file=sys.stderr)
            sys.exit(1)

        profile = manager.add_repo(str(repo_path), name=args.name)
        if args.output_json:
            print(
                json.dumps(
                    {
                        "action": "add",
                        "repo": {
                            "repo_id": profile.repo_id,
                            "name": profile.name,
                            "repo_path": profile.repo_path,
                        },
                    },
                    indent=2,
                )
            )
            return

        print(f"Added workspace repo: {profile.name} ({profile.repo_id})")
        return

    if action == "list":
        profiles = sorted(manager.list_profiles(), key=lambda p: (p.name, p.repo_id))
        if args.output_json:
            print(json.dumps([p.to_dict() for p in profiles], indent=2))
            return

        if not profiles:
            print("Workspace is empty.")
            return

        for profile in profiles:
            print(
                f"[{profile.status.value.upper()}] {profile.name} ({profile.repo_id}) {profile.repo_path}"
            )
        return

    if action == "remove":
        if not repo_ref:
            print("Error: repo id/name/path is required for 'workspace remove'", file=sys.stderr)
            sys.exit(1)

        removed = manager.remove_repo(repo_ref)
        if args.output_json:
            print(json.dumps({"action": "remove", "removed": removed, "repo": repo_ref}, indent=2))
            if not removed:
                sys.exit(1)
            return

        if not removed:
            print(f"Error: workspace repo not found: {repo_ref}", file=sys.stderr)
            sys.exit(1)

        print(f"Removed workspace repo: {repo_ref}")
        return

    if action == "analyze":
        if args.all and repo_ref:
            print("Error: cannot specify both --all and repo", file=sys.stderr)
            sys.exit(1)
        if not args.all and not repo_ref:
            print("Error: provide repo id/name/path or use --all", file=sys.stderr)
            sys.exit(1)

        if args.all:
            profiles = manager.list_profiles()
            if not profiles:
                if args.output_json:
                    print(
                        json.dumps(
                            {
                                "action": "analyze",
                                "mode": "all",
                                "total": 0,
                                "ready": 0,
                                "errors": 0,
                                "repos": [],
                            },
                            indent=2,
                        )
                    )
                else:
                    print("Workspace is empty.")
                return

            analyzed = manager.analyze_all()
            analyzed_sorted = sorted(analyzed, key=lambda p: (p.name, p.repo_id))
            ready_count = sum(1 for p in analyzed_sorted if p.status.value == "ready")
            error_count = sum(1 for p in analyzed_sorted if p.status.value == "error")

            if args.output_json:
                print(
                    json.dumps(
                        {
                            "action": "analyze",
                            "mode": "all",
                            "total": len(analyzed_sorted),
                            "ready": ready_count,
                            "errors": error_count,
                            "repos": [
                                {
                                    "repo_id": p.repo_id,
                                    "name": p.name,
                                    "status": p.status.value,
                                    "error": p.error,
                                }
                                for p in analyzed_sorted
                            ],
                        },
                        indent=2,
                    )
                )
                return

            print(
                f"Workspace analyze summary: total={len(analyzed_sorted)} ready={ready_count} errors={error_count}"
            )
            for p in analyzed_sorted:
                suffix = f" error={p.error}" if p.error else ""
                print(f"- {p.name} ({p.repo_id}) [{p.status.value.upper()}]{suffix}")
            return

        profile = manager.analyze_repo(repo_ref)
        if args.output_json:
            print(
                json.dumps(
                    {
                        "action": "analyze",
                        "mode": "single",
                        "repo": {
                            "repo_id": profile.repo_id,
                            "name": profile.name,
                            "status": profile.status.value,
                            "error": profile.error,
                        },
                    },
                    indent=2,
                )
            )
            return

        status = profile.status.value.upper()
        suffix = f" error={profile.error}" if profile.error else ""
        print(f"Analyzed workspace repo: {profile.name} ({profile.repo_id}) [{status}]{suffix}")


def cmd_multi_repo(args):
    """Multi-repo analysis, comparison, and knowledge transfer"""
    import json
    from pathlib import Path as _Path

    action = args.multi_repo_action
    repo_paths = args.repo_paths or []
    ws_path = _Path(args.workspace) if args.workspace else None

    if action == "add":
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        if not repo_paths:
            print("Error: provide at least one repo path to add", file=sys.stderr)
            sys.exit(1)
        mgr = MultiRepoManager(workspace_path=ws_path)
        for rp in repo_paths:
            profile = mgr.add_repo(rp)
            print(f"Added: {profile.name} ({profile.repo_id})")

    elif action == "remove":
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        if not repo_paths:
            print("Error: provide repo ID, path, or name to remove", file=sys.stderr)
            sys.exit(1)
        mgr = MultiRepoManager(workspace_path=ws_path)
        for rp in repo_paths:
            if mgr.remove_repo(rp):
                print(f"Removed: {rp}")
            else:
                print(f"Not found: {rp}")

    elif action == "list":
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        mgr = MultiRepoManager(workspace_path=ws_path)
        profiles = mgr.list_profiles()
        if not profiles:
            print("No repos in workspace")
            return
        if args.output_json:
            print(json.dumps([p.to_dict() for p in profiles], indent=2))
            return
        for p in profiles:
            status = p.status.value.upper()
            lang = ", ".join(p.languages[:3]) if p.languages else "not analyzed"
            print(f"  [{status:9s}] {p.name:20s} ({p.repo_id})  {lang}")

    elif action == "analyze":
        from scholardevclaw.application.pipeline import run_multi_repo_analyze

        ws_str = str(ws_path) if ws_path else None
        result = run_multi_repo_analyze(
            repo_paths,
            workspace_path=ws_str,
            log_callback=print,
        )
        if args.output_json:
            print(json.dumps(result.payload, indent=2))
        elif not result.ok:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)

    elif action == "compare":
        from scholardevclaw.application.pipeline import run_multi_repo_compare

        ws_str = str(ws_path) if ws_path else None
        result = run_multi_repo_compare(
            repo_paths or None,
            workspace_path=ws_str,
            log_callback=print,
        )
        if args.output_json:
            print(json.dumps(result.payload, indent=2))
        elif not result.ok:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)

    elif action == "transfer":
        from scholardevclaw.application.pipeline import run_multi_repo_transfer

        ws_str = str(ws_path) if ws_path else None
        result = run_multi_repo_transfer(
            repo_paths or None,
            source_id=args.source,
            target_id=args.target,
            workspace_path=ws_str,
            log_callback=print,
        )
        if args.output_json:
            print(json.dumps(result.payload, indent=2))
        elif not result.ok:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)

    elif action == "status":
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        mgr = MultiRepoManager(workspace_path=ws_path)
        profiles = mgr.list_profiles()
        ready = [p for p in profiles if p.status.value == "ready"]
        pending = [p for p in profiles if p.status.value == "pending"]
        errors = [p for p in profiles if p.status.value == "error"]

        if args.output_json:
            print(
                json.dumps(
                    {
                        "total": len(profiles),
                        "ready": len(ready),
                        "pending": len(pending),
                        "errors": len(errors),
                        "profiles": [p.to_dict() for p in profiles],
                    },
                    indent=2,
                )
            )
            return

        print(f"Multi-repo workspace: {len(profiles)} repo(s)")
        print(f"  Ready:   {len(ready)}")
        print(f"  Pending: {len(pending)}")
        print(f"  Errors:  {len(errors)}")
        if profiles:
            print()
            for p in profiles:
                status = p.status.value.upper()
                extra = ""
                if p.analyzed_at > 0:
                    import time as _time

                    elapsed = _time.time() - p.analyzed_at
                    if elapsed < 3600:
                        extra = f" ({elapsed / 60:.0f}m ago)"
                    else:
                        extra = f" ({elapsed / 3600:.1f}h ago)"
                print(f"  [{status:9s}] {p.name}{extra}")
                if p.error:
                    print(f"             Error: {p.error}")


def cmd_doctor(args):
    """Run self-diagnosis and health checks"""
    import json

    from scholardevclaw.utils.health import HealthChecker

    check = getattr(args, "check", "all")
    verbose = getattr(args, "verbose", False)
    checker = HealthChecker()

    if check == "quick":
        result = checker.run_quick_check()
        if result:
            print("Quick health check passed")
        else:
            print("Quick health check failed")
            sys.exit(1)
        return

    # Run specific check or all
    if check != "all" and check != "environment":
        result = checker.run_check(check)
        status = "OK" if result.healthy else "FAIL"
        print(f"[{status}] {result.name}: {result.message}")
        if verbose and result.details:
            print(f"  Details: {json.dumps(result.details)}")
        if not result.healthy:
            sys.exit(1)
    else:
        # Run all checks
        health = checker.run_all_checks()
        print("=" * 50)
        print("  ScholarDevClaw Health Check")
        print("=" * 50)
        print(f"  Version     : {health.version}")
        print(f"  Python      : {health.python_version}")
        print(f"  Platform    : {health.platform}")
        print(f"  Uptime      : {health.uptime_seconds:.1f}s")
        print()
        all_healthy = True
        for result in health.checks:
            status = "OK" if result.healthy else "FAIL"
            print(f"[{status}] {result.name}: {result.message}")
            if verbose and result.details:
                print(f"    Details: {json.dumps(result.details)}")
            if not result.healthy:
                all_healthy = False
        print()
        if all_healthy:
            print("All checks passed")
        else:
            print("Some checks failed")
            sys.exit(1)


def cmd_deploy_check(args):
    """Run deployment environment preflight checks."""
    from scholardevclaw.deploy.preflight import run_preflight

    try:
        result = run_preflight(args.env_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(
            json.dumps(
                {
                    "env_file": result.env_file,
                    "ok": result.ok,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "recommendations": result.recommendations,
                },
                indent=2,
            )
        )
    else:
        status = "passed" if result.ok else "failed"
        print(f"Deploy preflight {status}: {result.env_file}")
        if result.errors:
            print("Errors:")
            for issue in result.errors:
                print(f"  - {issue}")
        if result.warnings:
            print("Warnings:")
            for issue in result.warnings:
                print(f"  - {issue}")
        if result.recommendations:
            print("Recommendations:")
            for item in result.recommendations:
                print(f"  - {item}")

    if not result.ok:
        sys.exit(1)


def cmd_tui(args):
    """Launch interactive terminal UI (wizard mode)"""
    try:
        from scholardevclaw.tui import run_tui
    except ImportError as e:
        print("Error: TUI dependencies are not installed.", file=sys.stderr)
        print('Install with: pip install -e ".[tui]"', file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    run_tui()


def cmd_agent(args):
    """Start interactive AI agent"""
    from scholardevclaw.agent import run_agent_command
    from scholardevclaw.agent.repl import run_agent_repl

    repo_path = getattr(args, "repo", None)

    if args.query:
        result = run_agent_command(args.query, repo_path=repo_path)

        if args.output_json:
            print(
                json.dumps(
                    {
                        "ok": result.ok,
                        "message": result.message,
                        "output": result.output,
                        "error": result.error,
                        "suggestions": result.suggestions,
                        "next_steps": result.next_steps,
                    },
                    indent=2,
                )
            )
        else:
            print(result.message)
            if result.output and "languages" in result.output:
                print(f"\nLanguages: {', '.join(result.output['languages'])}")
                if result.output.get("frameworks"):
                    print(f"Frameworks: {', '.join(result.output['frameworks'])}")
            if result.suggestions:
                print("\nSuggestions:")
                for s in result.suggestions:
                    print(f"  - {s}")
            if result.error:
                print(f"\nError: {result.error}", file=sys.stderr)
                sys.exit(1)
    else:
        run_agent_repl(repo_path=repo_path)


def main():
    parser = argparse.ArgumentParser(
        description="ScholarDevClaw - Autonomous Research-to-Code Agent v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ScholarDevClaw analyzes your codebase, researches relevant papers,
and automatically implements improvements. Supports any programming language.

Examples:
  # Analyze any codebase (Python, JS/TS, Go, Rust, Java, etc.)
  scholardevclaw analyze ./my-project

  # Search for research papers and implementations
  scholardevclaw search "layer normalization" --arxiv --web

  # Ingest paper into structured document JSON
  scholardevclaw ingest arxiv:1706.03762 --output-dir ./artifacts

  # Understand ingested paper and build concept graph
  scholardevclaw understand ./paper_document.json --output-dir ./artifacts

  # Plan implementation from understanding artifact
  scholardevclaw plan ./understanding.json --stack pytorch --output-dir ./artifacts

  # Generate project code from implementation plan + understanding
  scholardevclaw generate ./implementation_plan.json ./understanding.json --output-dir ./generated

  # Get AI-powered improvement suggestions
  scholardevclaw suggest ./my-project

  # Full integration workflow
  scholardevclaw integrate ./my-project rmsnorm --output-dir ./patches

  # List available paper specifications
  scholardevclaw specs --list

  # Launch interactive terminal UI
    scholardevclaw tui

  # Run demo with nanoGPT
  scholardevclaw demo

For more information: https://github.com/Ronak-IIITD/ScholarDevClaw
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze repository structure")
    p_analyze.add_argument("repo_path", help="Path to repository")
    p_analyze.add_argument("--output-json", action="store_true", help="Output JSON")

    # search
    p_search = subparsers.add_parser("search", help="Search for papers and implementations")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--arxiv", action="store_true", help="Search arXiv")
    p_search.add_argument("--web", action="store_true", help="Search web sources (GitHub, etc.)")
    p_search.add_argument("--language", default="python", help="Programming language filter")
    p_search.add_argument("--max-results", type=int, default=10, help="Maximum results")

    # ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest paper source (PDF path, DOI, arXiv ID, or URL)",
    )
    p_ingest.add_argument("source", help="Paper source: local PDF path, DOI, arXiv ID, or URL")
    p_ingest.add_argument(
        "--output-dir",
        help="Directory to store ingestion artifacts (paper_document.json)",
    )

    # understand
    p_understand = subparsers.add_parser(
        "understand",
        help="Understand a paper_document.json and build concept graph",
    )
    p_understand.add_argument(
        "paper_document_json",
        help="Path to paper_document.json produced by ingest",
    )
    p_understand.add_argument(
        "--model",
        default="claude-opus-4-5",
        help="LLM model name for understanding extraction",
    )
    p_understand.add_argument(
        "--output-dir",
        help="Directory to store understanding.json and concept_graph.json",
    )

    # plan
    p_plan = subparsers.add_parser(
        "plan",
        help="Plan implementation from understanding.json",
    )
    p_plan.add_argument(
        "understanding_json",
        help="Path to understanding.json produced by understand",
    )
    p_plan.add_argument(
        "--stack",
        choices=["pytorch", "jax", "numpy", "numpy-only"],
        help="Force tech stack selection (pytorch|jax|numpy|numpy-only)",
    )
    p_plan.add_argument(
        "--output-dir",
        help="Directory to store implementation_plan.json",
    )

    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Get improvement suggestions")
    p_suggest.add_argument("repo_path", help="Path to repository")
    p_suggest.add_argument(
        "--use-specs",
        action="store_true",
        help="Use legacy specs-based workflow",
    )

    # integrate
    p_integrate = subparsers.add_parser("integrate", help="Full integration workflow")
    p_integrate.add_argument("repo_path", help="Path to repository")
    p_integrate.add_argument(
        "spec", nargs="?", help="Paper specification (auto-detect if not provided)"
    )
    p_integrate.add_argument("--output-dir", help="Output directory for generated files")
    p_integrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Run integration planning without generation and validation",
    )
    p_integrate.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail integration when git working tree has uncommitted changes",
    )
    p_integrate.add_argument("--output-json", action="store_true", help="Output JSON")
    p_integrate.add_argument(
        "--use-specs",
        action="store_true",
        help="Use legacy specs-based workflow",
    )

    # map
    p_map = subparsers.add_parser("map", help="Map a paper specification to repository locations")
    p_map.add_argument("repo_path", help="Path to repository")
    p_map.add_argument("spec", help="Paper specification name (e.g., rmsnorm)")
    p_map.add_argument("--output-json", action="store_true", help="Output JSON")
    p_map.add_argument(
        "--use-specs",
        action="store_true",
        help="Use legacy specs-based workflow",
    )

    # generate
    p_generate = subparsers.add_parser(
        "generate",
        help="Generate implementation code from implementation_plan.json and understanding.json",
    )
    p_generate.add_argument(
        "arg1",
        help=("Path to implementation_plan.json (or repo path when using --use-specs legacy mode)"),
    )
    p_generate.add_argument(
        "arg2",
        help=("Path to understanding.json (or spec name when using --use-specs legacy mode)"),
    )
    p_generate.add_argument("--output-dir", help="Output directory for generated project/report")
    p_generate.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum number of modules to generate in parallel",
    )
    p_generate.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="LLM model name for code generation",
    )
    p_generate.add_argument("--output-json", action="store_true", help="Output JSON")
    p_generate.add_argument(
        "--use-specs",
        action="store_true",
        help="Use legacy specs-based workflow (arg1=repo_path arg2=spec)",
    )

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate tests and benchmark")
    p_validate.add_argument("repo_path", help="Path to repository")
    p_validate.add_argument("--output-json", action="store_true", help="Output JSON")

    # specs
    p_specs = subparsers.add_parser("specs", help="List paper specifications")
    p_specs.add_argument("--list", action="store_true", help="Detailed list")
    p_specs.add_argument("--categories", action="store_true", help="Show categories")

    # planner
    p_planner = subparsers.add_parser("planner", help="Plan multi-spec migration strategy")
    p_planner.add_argument("repo_path", help="Path to repository")
    p_planner.add_argument("--max-specs", type=int, default=5, help="Maximum specs to recommend")
    p_planner.add_argument("--categories", help="Comma-separated categories to focus on")
    p_planner.add_argument("--output-json", action="store_true", help="Output JSON")

    # critic
    p_critic = subparsers.add_parser("critic", help="Verify generated patches for issues")
    p_critic.add_argument("repo_path", help="Path to repository")
    p_critic.add_argument("spec", nargs="?", help="Paper specification to verify")
    p_critic.add_argument("--patch-json", help="JSON string containing patch result to verify")
    p_critic.add_argument("--output-json", action="store_true", help="Output JSON")

    # context
    p_context = subparsers.add_parser("context", help="Manage project context and memory")
    p_context.add_argument("repo_path", nargs="?", help="Path to repository")
    p_context.add_argument(
        "context_action",
        choices=["init", "history", "summary", "recommend", "set", "clear", "list"],
        help="Action to perform",
    )
    p_context.add_argument("--pref-type", help="Preference type (for set action)")
    p_context.add_argument("--pref-value", help="Preference value (for set action)")

    # experiment
    p_experiment = subparsers.add_parser(
        "experiment", help="Run experiment loop for hypothesis testing"
    )
    p_experiment.add_argument("repo_path", help="Path to repository")
    p_experiment.add_argument("spec", help="Paper specification to experiment with")
    p_experiment.add_argument(
        "--variants", type=int, default=3, help="Number of variants to generate"
    )
    p_experiment.add_argument("--output-dir", help="Output directory for variants")
    p_experiment.add_argument("--output-json", action="store_true", help="Output JSON")

    # plugin
    p_plugin = subparsers.add_parser("plugin", help="Manage plugins")
    p_plugin.add_argument(
        "plugin_action",
        choices=[
            "list",
            "load",
            "unload",
            "enable",
            "disable",
            "hooks",
            "analyze",
            "validate",
            "scaffold",
            "info",
        ],
        help="Action to perform",
    )
    p_plugin.add_argument("plugin_name", nargs="?", help="Plugin name")
    p_plugin.add_argument("repo_path", nargs="?", help="Repository path (for analyze/validate)")
    p_plugin.add_argument(
        "--plugin-type", choices=["analyzer", "spec_provider", "validator", "hook", "custom"]
    )

    # rollback
    p_rollback = subparsers.add_parser(
        "rollback", help="Manage rollback snapshots and revert integrations"
    )
    p_rollback.add_argument(
        "rollback_action",
        choices=["list", "show", "run", "status", "delete"],
        help="Action to perform",
    )
    p_rollback.add_argument("repo_path", nargs="?", help="Path to repository")
    p_rollback.add_argument("--snapshot-id", help="Snapshot ID to show/run/delete")
    p_rollback.add_argument(
        "--status",
        choices=["pending", "applied", "rolled_back", "failed", "partial"],
        help="Filter by status (for list)",
    )
    p_rollback.add_argument("--force", action="store_true", help="Force rollback/delete")
    p_rollback.add_argument("--output-json", action="store_true", help="Output JSON")

    # github-app
    p_github = subparsers.add_parser("github-app", help="Manage GitHub App and webhooks")
    p_github.add_argument(
        "github_action",
        choices=["setup", "server", "status", "test-webhook", "manifest"],
        help="Action to perform",
    )
    p_github.add_argument("--server-url", help="Server URL for webhook (for setup)")
    p_github.add_argument("--port", type=int, default=8000, help="Server port (for server)")
    p_github.add_argument("--output-json", action="store_true", help="Output JSON")

    # security
    p_security = subparsers.add_parser("security", help="Run security scans")
    p_security.add_argument("repo_path", help="Path to repository")
    p_security.add_argument(
        "security_action",
        nargs="?",
        default="scan",
        choices=["scan", "check"],
        help="Action: scan or check tool availability",
    )
    p_security.add_argument(
        "--tools",
        help="Comma-separated tools to run (bandit,semgrep)",
    )
    p_security.add_argument("--verbose", "-v", action="store_true", help="Show detailed findings")
    p_security.add_argument("--output-json", action="store_true", help="Output JSON")

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Run self-diagnosis and health checks")
    p_doctor.add_argument(
        "check",
        nargs="?",
        default="all",
        choices=["all", "quick", "ollama", "openrouter", "auth_store", "environment", "production"],
        help="Specific check to run (default: all)",
    )
    p_doctor.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    # deploy-check
    p_deploy_check = subparsers.add_parser(
        "deploy-check", help="Validate deployment environment file"
    )
    p_deploy_check.add_argument("--env-file", default="docker/.env", help="Path to env file")
    p_deploy_check.add_argument("--output-json", action="store_true", help="Output JSON")

    # agent
    p_agent = subparsers.add_parser("agent", help="Start interactive AI agent")
    p_agent.add_argument(
        "query",
        nargs="?",
        help="Query to process (opens interactive mode if not provided)",
    )
    p_agent.add_argument("--repo", help="Set repository path")
    p_agent.add_argument("--output-json", action="store_true", help="Output JSON")

    # auth
    p_auth = subparsers.add_parser("auth", help="Manage authentication and API keys")
    p_auth.add_argument(
        "auth_action",
        nargs="?",
        default="status",
        choices=["setup", "login", "logout", "status", "list", "add", "remove", "default"],
        help="Action to perform",
    )
    p_auth.add_argument("--key", help="API key value")
    p_auth.add_argument("--name", help="Key name")
    p_auth.add_argument("--provider", help="Provider (anthropic, openai, github, custom)")
    p_auth.add_argument("--default", action="store_true", help="Set as default key")
    p_auth.add_argument("--force", action="store_true", help="Force action without confirmation")
    p_auth.add_argument("--key-id", help="Key ID for remove/default actions")
    p_auth.add_argument("--output-json", action="store_true", help="Output JSON")

    # multi-repo
    p_multi_repo = subparsers.add_parser(
        "multi-repo", help="Multi-repo analysis, comparison, and knowledge transfer"
    )
    p_multi_repo.add_argument(
        "multi_repo_action",
        choices=["add", "remove", "list", "analyze", "compare", "transfer", "status"],
        help="Action to perform",
    )
    p_multi_repo.add_argument(
        "repo_paths",
        nargs="*",
        help="Repository path(s) — used by add, remove, analyze, compare, transfer",
    )
    p_multi_repo.add_argument(
        "--workspace", help="Custom workspace JSON path (defaults to ~/.scholardevclaw/)"
    )
    p_multi_repo.add_argument(
        "--source", help="Source repo ID or path (for transfer with specific pair)"
    )
    p_multi_repo.add_argument(
        "--target", help="Target repo ID or path (for transfer with specific pair)"
    )
    p_multi_repo.add_argument("--output-json", action="store_true", help="Output JSON")

    # workspace
    p_workspace = subparsers.add_parser("workspace", help="Manage multi-repo workspace (CLI-first)")
    p_workspace.add_argument(
        "workspace_action",
        choices=["add", "remove", "list", "analyze"],
        help="Workspace action",
    )
    p_workspace.add_argument(
        "repo_id_or_path",
        nargs="?",
        help="Repo path (add) or repo id/name/path (remove/analyze)",
    )
    p_workspace.add_argument("--name", help="Optional display name for 'add'")
    p_workspace.add_argument(
        "--all",
        action="store_true",
        help="Analyze all workspace repos (for analyze action)",
    )
    p_workspace.add_argument("--output-json", action="store_true", help="Output JSON")

    # tui
    subparsers.add_parser("tui", help="Launch interactive terminal UI")

    # demo
    p_demo = subparsers.add_parser(
        "demo",
        help="Run end-to-end demo (clones nanoGPT if needed, runs full pipeline)",
    )
    p_demo.add_argument("--repo", help="Custom repository path (defaults to test_repos/nanogpt)")
    p_demo.add_argument("--spec", help="Run demo for a specific spec (e.g. rmsnorm, swiglu)")
    p_demo.add_argument("--all", dest="all_specs", action="store_true", help="Demo all 16 specs")
    p_demo.add_argument("--output-dir", help="Write patch artifacts to this directory")
    p_demo.add_argument(
        "--skip-validate", action="store_true", help="Skip validation benchmarks (faster)"
    )
    p_demo.add_argument(
        "--output-json", action="store_true", help="Output machine-readable JSON report"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "analyze": cmd_analyze,
        "search": cmd_search,
        "ingest": cmd_ingest,
        "understand": cmd_understand,
        "plan": cmd_plan,
        "suggest": cmd_suggest,
        "map": cmd_map,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "integrate": cmd_integrate,
        "tui": cmd_tui,
        "specs": cmd_specs,
        "planner": cmd_planner,
        "critic": cmd_critic,
        "context": cmd_context,
        "experiment": cmd_experiment,
        "plugin": cmd_plugin,
        "rollback": cmd_rollback,
        "github-app": cmd_github_app,
        "security": cmd_security,
        "agent": cmd_agent,
        "auth": cmd_auth,
        "demo": cmd_demo,
        "multi-repo": cmd_multi_repo,
        "workspace": cmd_workspace,
        "doctor": cmd_doctor,
        "deploy-check": cmd_deploy_check,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()


# Entry point for package
def cli_main():
    """Entry point for pip installation"""
    main()
