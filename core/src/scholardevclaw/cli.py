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
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()


# Entry point for package
def cli_main():
    """Entry point for pip installation"""
    main()
