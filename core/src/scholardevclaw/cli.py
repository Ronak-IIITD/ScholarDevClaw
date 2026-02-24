#!/usr/bin/env python3
"""
ScholarDevClaw CLI - Autonomous Research-to-Code Agent

An AI-powered tool that analyzes your codebase, researches relevant papers,
and automatically implements improvements. Supports any programming language.

Commands:
  analyze     - Analyze repository structure (multi-language)
  search      - Search for research papers and implementations
  suggest     - Get AI-powered improvement suggestions
  integrate   - Full integration workflow
    tui         - Interactive terminal UI workflow
  specs       - List available paper specifications
  demo        - Run demo with nanoGPT

Examples:
  scholardevclaw analyze ./my-project
  scholardevclaw search "normalization" --arxiv --web
  scholardevclaw suggest ./my-project
  scholardevclaw integrate ./my-project rmsnorm

Version: 2.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor
    from scholardevclaw.mapping.engine import MappingEngine

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
    print(f"\nLanguage statistics:")
    for stat in result.language_stats:
        print(f"  {stat.language}: {stat.file_count} files, {stat.line_count} lines")

    print(f"\nFrameworks detected: {', '.join(result.frameworks) if result.frameworks else 'None'}")

    print(f"\nEntry points: {len(result.entry_points)}")
    for ep in result.entry_points[:5]:
        print(f"  - {ep}")

    print(f"\nTest files: {len(result.test_files)}")

    # Find patterns for improvement
    if result.patterns:
        print(f"\nPatterns found for improvement:")
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
        print(f"\nSearching arXiv...")
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
        print(f"\nSearching web sources...")
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


def cmd_suggest(args):
    """Get improvement suggestions based on code patterns"""
    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing code patterns in: {path}")
    print("-" * 50)

    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

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


def cmd_generate(args):
    """Generate patch artifacts for a research specification"""
    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating patch for spec '{args.spec}' in: {path}")
    print("-" * 50)

    try:
        mapping_result, _ = _build_mapping_result(path, args.spec)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from scholardevclaw.patch_generation.generator import PatchGenerator

    generator = PatchGenerator(path)
    patch = generator.generate(mapping_result)

    output_dir = Path(args.output_dir) if args.output_dir else path / "integration-patch"
    output_dir.mkdir(parents=True, exist_ok=True)

    for nf in patch.new_files:
        out_file = output_dir / nf.path
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
        print(f"\nCritic found issues!", file=sys.stderr)
    else:
        print(f"\nCritic passed!")

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
    from scholardevclaw.context_engine import ContextEngine, get_agent_brain
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
        print(f"\nStatistics:")
        print(f"  Total runs: {stats.get('total_runs', 0)}")
        print(f"  Successful: {stats.get('successful_runs', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.0%}")

        print(f"\nSuccessful specs: {', '.join(stats.get('successful_specs', [])) or 'None'}")
        print(f"Failed specs: {', '.join(stats.get('failed_specs', [])) or 'None'}")

        prefs = summary.get("preferences", {})
        print(f"\nPreferences:")
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
        print(f"\nContext used:")
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
            print(f"  • {p.name} ({p.plugin_type}) - {p.description}")
            print(f"    Version: {p.version} | Author: {p.author}")

        print(f"\nLoaded plugins ({len(loaded)}):")
        for p in loaded:
            print(f"  ✓ {p.name} ({p.plugin_type})")

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
        print(f"  Edit this file to implement your plugin")

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
            print(f"\nGit State:")
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
    """Run demo with nanoGPT"""
    # Find project root
    cli_dir = Path(__file__).parent
    project_root = cli_dir.parent.parent.parent
    demo_path = project_root / "test_repos" / "nanogpt"

    if not demo_path.exists():
        print(f"Error: nanoGPT not found at {demo_path}")
        print("Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt")
        sys.exit(1)

    print("ScholarDevClaw Demo")
    print("=" * 50)
    print(f"Repository: {demo_path}\n")

    # Step 1: Analyze
    print("[1/5] Analyzing repository...")
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer(demo_path)
    result = analyzer.analyze()

    print(f"  Found {len(result.languages)} languages: {', '.join(result.languages)}")
    print(f"  Total files: {sum(s.file_count for s in result.language_stats)}")

    # Step 2: Research
    print("\n[2/5] Extracting RMSNorm specification...")
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    extractor = ResearchExtractor()
    spec = extractor.get_spec("rmsnorm")

    if spec is None:
        print("Error: RMSNorm spec not found")
        sys.exit(1)

    print(f"  Algorithm: {spec['algorithm']['name']}")
    print(f"  Replaces: {spec['algorithm']['replaces']}")
    print(f"  Category: {spec['algorithm']['category']}")

    # Step 3: Suggest
    print("\n[3/5] Checking for improvement opportunities...")
    suggestions = analyzer.suggest_research_papers()

    if suggestions:
        print(f"  Found {len(suggestions)} opportunities")
        for s in suggestions[:3]:
            print(f"    - {s['pattern']}: {s['paper']['title']} ({s['confidence']:.0f}%)")

    # Step 4: Generate
    print("\n[4/5] Generating patch...")
    from scholardevclaw.patch_generation.generator import PatchGenerator

    mapping_result = {
        "targets": [],
        "research_spec": spec,
    }

    generator = PatchGenerator(demo_path)
    patch = generator.generate(mapping_result)

    print(f"  Branch: {patch.branch_name}")
    print(f"  New files: {len(patch.new_files)}")
    print(f"  Transformations: {len(patch.transformations)}")

    # Step 5: Validate
    print("\n[5/5] Validating...")
    from scholardevclaw.validation.runner import ValidationRunner

    runner = ValidationRunner(demo_path)
    validation = runner.run({}, str(demo_path))

    print(f"  Stage: {validation.stage}")
    print(f"  Passed: {'Yes' if validation.passed else 'No'}")

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nYour repository is ready for research-driven improvements.")


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
            print(f"\nWebhook endpoint: /webhook")
            print(f"Health check: /health")

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

        import hmac
        import hashlib

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
    from scholardevclaw.security import SecurityScanner, Severity

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

    print(f"\nSecurity Scan Results")
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
    from scholardevclaw.agent import AgentREPL, run_agent_command
    from scholardevclaw.agent.repl import run_agent_repl

    if args.query:
        result = run_agent_command(args.query)

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
                print("\n💡 Suggestions:")
                for s in result.suggestions:
                    print(f"  • {s}")
            if result.error:
                print(f"\n❌ Error: {result.error}", file=sys.stderr)
                sys.exit(1)
    else:
        run_agent_repl()


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

    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Get improvement suggestions")
    p_suggest.add_argument("repo_path", help="Path to repository")

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

    # map
    p_map = subparsers.add_parser("map", help="Map a paper specification to repository locations")
    p_map.add_argument("repo_path", help="Path to repository")
    p_map.add_argument("spec", help="Paper specification name (e.g., rmsnorm)")
    p_map.add_argument("--output-json", action="store_true", help="Output JSON")

    # generate
    p_generate = subparsers.add_parser("generate", help="Generate patch artifacts")
    p_generate.add_argument("repo_path", help="Path to repository")
    p_generate.add_argument("spec", help="Paper specification name (e.g., rmsnorm)")
    p_generate.add_argument("--output-dir", help="Output directory for generated files")
    p_generate.add_argument("--output-json", action="store_true", help="Output JSON")

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
        choices=["list", "load", "unload", "analyze", "validate", "scaffold", "info"],
        help="Action to perform",
    )
    p_plugin.add_argument("plugin_name", nargs="?", help="Plugin name")
    p_plugin.add_argument("repo_path", nargs="?", help="Repository path (for analyze/validate)")
    p_plugin.add_argument(
        "--plugin-type", choices=["analyzer", "spec_provider", "validator", "custom"]
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

    # tui
    subparsers.add_parser("tui", help="Launch interactive terminal UI")

    # demo
    p_demo = subparsers.add_parser("demo", help="Run demo with nanoGPT")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "analyze": cmd_analyze,
        "search": cmd_search,
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
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()


# Entry point for package
def cli_main():
    """Entry point for pip installation"""
    main()
