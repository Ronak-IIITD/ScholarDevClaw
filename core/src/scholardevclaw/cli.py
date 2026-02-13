#!/usr/bin/env python3
"""ScholarDevClaw CLI - Command-line interface for the ML Research Integration Engine"""

import argparse
import json
import sys
from pathlib import Path

from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser
from scholardevclaw.research_intelligence.extractor import ResearchExtractor
from scholardevclaw.mapping.engine import MappingEngine
from scholardevclaw.patch_generation.generator import PatchGenerator
from scholardevclaw.validation.runner import ValidationRunner


def cmd_analyze(args):
    """Analyze a repository"""
    path = Path(args.repo_path)
    if not path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    parser = PyTorchRepoParser(path)
    result = parser.parse()

    output = {
        "repo_name": result.repo_name,
        "architecture": {
            "models": [
                {
                    "name": m.name,
                    "file": m.file,
                    "line": m.line,
                    "parent": m.parent,
                    "components": m.components,
                }
                for m in result.models
            ],
            "training_loop": {
                "file": result.training_loop.file,
                "line": result.training_loop.line,
                "optimizer": result.training_loop.optimizer,
                "loss_fn": result.training_loop.loss_fn,
            }
            if result.training_loop
            else None,
        },
        "test_files": result.test_files,
    }

    if args.output_json:
        print(json.dumps(output, indent=2))
    else:
        print(f"Repository: {result.repo_name}")
        print(f"Models found: {len(result.models)}")
        print(f"Modules: {len(result.modules)}")
        for model in result.models:
            print(f"  - {model.name} ({model.file})")


def cmd_specs(args):
    """List available paper specifications"""
    extractor = ResearchExtractor()
    specs = extractor.list_available_specs()

    if args.list:
        for spec_name in specs:
            spec = extractor.get_spec(spec_name)
            if spec:
                print(f"\n{spec_name.upper()}")
                print(f"  Paper: {spec['paper']['title']}")
                print(f"  Authors: {', '.join(spec['paper']['authors'])}")
                if spec["paper"].get("arxiv"):
                    print(f"  arXiv: {spec['paper']['arxiv']}")
                print(f"  Replaces: {spec['algorithm']['replaces']}")
    else:
        print("Available specs:", ", ".join(specs))


def cmd_extract(args):
    """Extract research specification"""
    extractor = ResearchExtractor()

    if args.spec:
        spec = extractor.get_spec(args.spec)
        if spec:
            if args.output_json:
                print(json.dumps(spec, indent=2))
            else:
                print(f"Algorithm: {spec['algorithm']['name']}")
                print(f"Replaces: {spec['algorithm']['replaces']}")
                print(f"Description: {spec['algorithm']['description']}")
        else:
            print(f"Error: Unknown spec '{args.spec}'", file=sys.stderr)
            sys.exit(1)
    else:
        print("Available specs:", ", ".join(extractor.list_available_specs()))


def cmd_map(args):
    """Map research to repository"""
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    # Analyze repo
    parser = PyTorchRepoParser(repo_path)
    repo_analysis = parser.parse()

    repo_data = {
        "repo_name": repo_analysis.repo_name,
        "architecture": {
            "models": [
                {
                    "name": m.name,
                    "file": m.file,
                    "line": m.line,
                    "parent": m.parent,
                    "components": m.components,
                }
                for m in repo_analysis.models
            ],
        },
    }

    # Get spec
    extractor = ResearchExtractor()
    spec = extractor.get_spec(args.spec)

    if not spec:
        print(f"Error: Unknown spec '{args.spec}'", file=sys.stderr)
        sys.exit(1)

    # Map
    engine = MappingEngine(repo_data, spec)
    result = engine.map()

    if args.output_json:
        print(
            json.dumps(
                {
                    "targets": [
                        {
                            "file": t.file,
                            "line": t.line,
                            "current_code": t.current_code,
                            "replacement_required": t.replacement_required,
                        }
                        for t in result.targets
                    ],
                    "strategy": result.strategy,
                    "confidence": result.confidence,
                },
                indent=2,
            )
        )
    else:
        print(f"Targets: {len(result.targets)}")
        print(f"Strategy: {result.strategy}")
        print(f"Confidence: {result.confidence}%")


def cmd_generate(args):
    """Generate patch"""
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    # Analyze repo
    parser = PyTorchRepoParser(repo_path)
    repo_analysis = parser.parse()

    repo_data = {
        "repo_name": repo_analysis.repo_name,
        "architecture": {
            "models": [
                {
                    "name": m.name,
                    "file": m.file,
                    "line": m.line,
                    "parent": m.parent,
                    "components": m.components,
                }
                for m in repo_analysis.models
            ],
        },
    }

    # Get spec
    extractor = ResearchExtractor()
    spec = extractor.get_spec(args.spec)

    if not spec:
        print(f"Error: Unknown spec '{args.spec}'", file=sys.stderr)
        sys.exit(1)

    # Map
    engine = MappingEngine(repo_data, spec)
    mapping = engine.map()

    # Generate
    generator = PatchGenerator(repo_path)
    patch = generator.generate(
        {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "current_code": t.current_code,
                    "context": t.context,
                }
                for t in mapping.targets
            ],
            "research_spec": spec,
        }
    )

    # Write new files
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for nf in patch.new_files:
            out_file = out_dir / nf.path
            out_file.write_text(nf.content)
            print(f"Created: {out_file}")

        print(f"\nPatch written to: {out_dir}")
    else:
        if args.output_json:
            print(
                json.dumps(
                    {
                        "branch_name": patch.branch_name,
                        "new_files": [f.path for f in patch.new_files],
                        "transformations": [t.file for t in patch.transformations],
                    },
                    indent=2,
                )
            )
        else:
            print(f"Branch: {patch.branch_name}")
            print(f"New files: {len(patch.new_files)}")
            print(f"Transformations: {len(patch.transformations)}")


def cmd_validate(args):
    """Run validation"""
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    runner = ValidationRunner(repo_path)
    result = runner.run({}, str(repo_path))

    if args.output_json:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "stage": result.stage,
                    "comparison": result.comparison,
                    "logs": result.logs[:500],
                },
                indent=2,
            )
        )
    else:
        print(f"Stage: {result.stage}")
        print(f"Passed: {'Yes' if result.passed else 'No'}")
        if result.comparison:
            print(f"Speedup: {result.comparison.get('speedup', 'N/A'):.2f}x")
            print(f"Loss change: {result.comparison.get('loss_change', 'N/A'):.2f}%")


def cmd_demo(args):
    """Run demo with nanoGPT"""
    # Find project root (two levels up from core/src/scholardevclaw/cli.py)
    cli_dir = Path(__file__).parent
    project_root = cli_dir.parent.parent.parent
    demo_path = project_root / "test_repos" / "nanogpt"

    if not demo_path.exists():
        print(f"Error: nanoGPT not found at {demo_path}", file=sys.stderr)
        print("Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt")
        sys.exit(1)

    print("ScholarDevClaw Demo")
    print("=" * 40)
    print(f"Repository: {demo_path}")
    print()

    # Step 1: Analyze
    print("Step 1: Analyzing repository...")
    parser = PyTorchRepoParser(demo_path)
    result = parser.parse()
    print(f"  Found {len(result.models)} models, {len(result.modules)} modules")

    # Step 2: Extract
    print("\nStep 2: Extracting RMSNorm specification...")
    extractor = ResearchExtractor()
    spec = extractor.get_spec("rmsnorm")
    print(f"  Algorithm: {spec['algorithm']['name']}")
    print(f"  Replaces: {spec['algorithm']['replaces']}")

    # Step 3: Map
    print("\nStep 3: Mapping to repository...")
    repo_data = {
        "repo_name": result.repo_name,
        "architecture": {
            "models": [
                {
                    "name": m.name,
                    "file": m.file,
                    "line": m.line,
                    "parent": m.parent,
                    "components": m.components,
                }
                for m in result.models
            ],
        },
    }
    engine = MappingEngine(repo_data, spec)
    mapping = engine.map()
    print(f"  Targets: {len(mapping.targets)}")
    print(f"  Strategy: {mapping.strategy}")
    print(f"  Confidence: {mapping.confidence}%")

    # Step 4: Generate
    print("\nStep 4: Generating patch...")
    generator = PatchGenerator(demo_path)
    patch = generator.generate(
        {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "current_code": t.current_code,
                    "context": t.context,
                }
                for t in mapping.targets
            ],
            "research_spec": spec,
        }
    )
    print(f"  Branch: {patch.branch_name}")
    print(f"  New files: {len(patch.new_files)}")
    print(f"  Transformations: {len(patch.transformations)}")

    # Step 5: Validate
    print("\nStep 5: Running validation...")
    runner = ValidationRunner(demo_path)
    validation = runner.run({}, str(demo_path))
    print(f"  Stage: {validation.stage}")
    print(f"  Passed: {'Yes' if validation.passed else 'No'}")

    print("\n" + "=" * 40)
    print("Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="ScholarDevClaw - Autonomous ML Research Integration Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a repository")
    p_analyze.add_argument("repo_path", help="Path to repository")
    p_analyze.add_argument("--output-json", action="store_true", help="Output JSON")

    # specs
    p_specs = subparsers.add_parser("specs", help="List paper specifications")
    p_specs.add_argument("--list", action="store_true", help="List all specs")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract research specification")
    p_extract.add_argument("--spec", help="Specification name")
    p_extract.add_argument("--output-json", action="store_true", help="Output JSON")

    # map
    p_map = subparsers.add_parser("map", help="Map research to repository")
    p_map.add_argument("repo_path", help="Path to repository")
    p_map.add_argument("spec", help="Research specification name")
    p_map.add_argument("--output-json", action="store_true", help="Output JSON")

    # generate
    p_generate = subparsers.add_parser("generate", help="Generate patch")
    p_generate.add_argument("repo_path", help="Path to repository")
    p_generate.add_argument("spec", help="Research specification name")
    p_generate.add_argument("--output-dir", help="Output directory for patch files")
    p_generate.add_argument("--output-json", action="store_true", help="Output JSON")

    # validate
    p_validate = subparsers.add_parser("validate", help="Run validation")
    p_validate.add_argument("repo_path", help="Path to repository")
    p_validate.add_argument("--output-json", action="store_true", help="Output JSON")

    # demo
    p_demo = subparsers.add_parser("demo", help="Run demo with nanoGPT")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "specs":
        cmd_specs(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "map":
        cmd_map(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
