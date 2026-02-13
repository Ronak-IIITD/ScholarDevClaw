import typer
from pathlib import Path
import json
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser
from scholardevclaw.research_intelligence.extractor import ResearchExtractor
from scholardevclaw.mapping.engine import MappingEngine
from scholardevclaw.patch_generation.generator import PatchGenerator
from scholardevclaw.validation.runner import ValidationRunner

app = typer.Typer(
    name="scholardevclaw",
    help="ScholarDevClaw - Autonomous ML Research Integration Engine",
    add_completion=False,
)

console = Console()


@app.command()
def analyze(
    repo_path: str = typer.Argument(..., help="Path to repository"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
) -> None:
    """Analyze a PyTorch repository structure."""
    console.print(f"[bold blue]Analyzing repository:[/bold blue] {repo_path}")

    path = Path(repo_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Repository path does not exist: {repo_path}")
        raise typer.Exit(1)

    parser = PyTorchRepoParser(path)
    result = parser.parse()

    console.print(
        f"[bold green]Found:[/bold green] {len(result.models)} models, {len(result.modules)} modules"
    )

    if result.models:
        console.print("\n[bold]Models:[/bold]")
        for model in result.models:
            console.print(f"  - {model.name} ({model.file}:{model.line})")
            if model.components:
                for key, value in model.components.items():
                    console.print(f"    - {key}: {value}")

    if output:
        output_data = {
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

        Path(output).write_text(json.dumps(output_data, indent=2))
        console.print(f"[bold green]Output saved to:[/bold green] {output}")


@app.command()
def specs(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all available paper specs"),
) -> None:
    """List available research paper specifications."""
    extractor = ResearchExtractor()

    if list_all:
        available = extractor.list_available_specs()
        console.print("[bold]Available paper specifications:[/bold]")
        for spec in available:
            info = extractor.get_spec(spec)
            if info:
                console.print(f"\n[bold cyan]{spec.upper()}[/bold cyan]")
                console.print(f"  Paper: {info['paper']['title']}")
                console.print(f"  Authors: {', '.join(info['paper']['authors'])}")
                if info["paper"].get("arxiv"):
                    console.print(f"  arXiv: {info['paper']['arxiv']}")
                console.print(f"  Replaces: {info['algorithm']['replaces']}")
    else:
        console.print("[bold]Usage:[/bold] scholardevclaw specs --list")


@app.command()
def extract(
    source: str = typer.Argument(..., help="Paper source (name, arxiv ID, or pdf path)"),
    source_type: str = typer.Option("name", "--type", "-t", help="Source type: name, arxiv, pdf"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
) -> None:
    """Extract research specification from paper."""
    console.print(f"[bold blue]Extracting research:[/bold blue] {source}")

    extractor = ResearchExtractor()
    result = extractor.extract(source, source_type)

    console.print(f"[bold green]Algorithm:[/bold green] {result['algorithm']['name']}")
    console.print(f"[bold green]Replaces:[/bold green] {result['algorithm']['replaces']}")
    console.print(f"[bold green]Description:[/bold green] {result['algorithm']['description']}")

    if output:
        Path(output).write_text(json.dumps(result, indent=2))
        console.print(f"[bold green]Output saved to:[/bold green] {output}")


@app.command()
def map(
    repo_path: str = typer.Argument(..., help="Path to repository"),
    spec_name: str = typer.Argument(..., help="Research specification name (e.g., rmsnorm)"),
) -> None:
    """Map research changes to repository."""
    console.print(f"[bold blue]Mapping:[/bold blue] {spec_name} -> {repo_path}")

    repo = Path(repo_path)
    if not repo.exists():
        console.print(f"[bold red]Error:[/bold red] Repository not found")
        raise typer.Exit(1)

    parser = PyTorchRepoParser(repo)
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

    extractor = ResearchExtractor()
    spec = extractor.get_spec(spec_name)

    if not spec:
        console.print(f"[bold red]Error:[/bold red] Unknown spec: {spec_name}")
        available = extractor.list_available_specs()
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    engine = MappingEngine(repo_data, spec)
    result = engine.map()

    console.print(f"\n[bold]Targets found:[/bold] {len(result.targets)}")
    console.print(f"[bold]Strategy:[/bold] {result.strategy}")
    console.print(f"[bold]Confidence:[/bold] {result.confidence}%")

    for target in result.targets:
        console.print(f"\n  File: {target.file}:{target.line}")
        console.print(f"    Current: {target.current_code}")
        console.print(f"    Replacement: {target.context.get('replacement', 'N/A')}")


@app.command()
def generate(
    repo_path: str = typer.Argument(..., help="Path to repository"),
    spec_name: str = typer.Argument(..., help="Research specification name"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-d", help="Output directory for patch"
    ),
) -> None:
    """Generate patch for research integration."""
    console.print(f"[bold blue]Generating patch:[/bold blue] {spec_name} -> {repo_path}")

    repo = Path(repo_path)
    parser = PyTorchRepoParser(repo)
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

    extractor = ResearchExtractor()
    spec = extractor.get_spec(spec_name)

    if not spec:
        console.print(f"[bold red]Error:[/bold red] Unknown spec: {spec_name}")
        raise typer.Exit(1)

    engine = MappingEngine(repo_data, spec)
    mapping = engine.map()

    generator = PatchGenerator(repo)
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

    console.print(f"\n[bold]Branch:[/bold] {patch.branch_name}")
    console.print(f"[bold]New files:[/bold] {len(patch.new_files)}")
    console.print(f"[bold]Transformations:[/bold] {len(patch.transformations)}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for nf in patch.new_files:
            (out_path / nf.path).write_text(nf.content)
            console.print(f"  Created: {out_path / nf.path}")

        console.print(f"[bold green]Patch files written to:[/bold green] {output_dir}")


@app.command()
def validate(
    repo_path: str = typer.Argument(..., help="Path to repository"),
) -> None:
    """Run validation tests and benchmarks."""
    console.print(f"[bold blue]Validating:[/bold blue] {repo_path}")

    repo = Path(repo_path)
    runner = ValidationRunner(repo)

    result = runner.run({}, str(repo))

    console.print(f"\n[bold]Stage:[/bold] {result.stage}")
    console.print(
        f"[bold]Passed:[/bold] {'[green]Yes[/green]' if result.passed else '[red]No[/red]'}"
    )

    if result.comparison:
        console.print(f"[bold]Speedup:[/bold] {result.comparison.get('speedup', 'N/A'):.2f}x")
        console.print(
            f"[bold]Loss change:[/bold] {result.comparison.get('loss_change', 'N/A'):.2f}%"
        )

    if result.logs:
        console.print(f"\n[bold]Logs:[/bold]")
        console.print(result.logs[:500])


@app.command()
def demo() -> None:
    """Run a demo with nanoGPT and RMSNorm."""
    console.print("[bold cyan]ScholarDevClaw Demo[/bold cyan]")
    console.print("Testing RMSNorm integration on nanoGPT...\n")

    demo_path = Path(__file__).parent.parent.parent.parent / "test_repos" / "nanogpt"

    if not demo_path.exists():
        console.print(f"[yellow]nanoGPT not found at {demo_path}[/yellow]")
        console.print("Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt")
        raise typer.Exit(1)

    console.print(f"[bold]Repository:[/bold] {demo_path}")

    console.print("\n[bold cyan]Step 1: Analyzing repository...[/bold cyan]")
    parser = PyTorchRepoParser(demo_path)
    result = parser.parse()
    console.print(f"  Found {len(result.models)} models, {len(result.modules)} modules")

    console.print("\n[bold cyan]Step 2: Extracting RMSNorm specification...[/bold cyan]")
    extractor = ResearchExtractor()
    spec = extractor.get_spec("rmsnorm")
    if spec is None:
        console.print("[bold red]Error:[/bold red] RMSNorm spec not found")
        raise typer.Exit(1)
    console.print(f"  Algorithm: {spec['algorithm']['name']}")
    console.print(f"  Replaces: {spec['algorithm']['replaces']}")

    console.print("\n[bold cyan]Step 3: Mapping to repository...[/bold cyan]")
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
    console.print(f"  Targets: {len(mapping.targets)}")
    console.print(f"  Strategy: {mapping.strategy}")
    console.print(f"  Confidence: {mapping.confidence}%")

    console.print("\n[bold cyan]Step 4: Generating patch...[/bold cyan]")
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
    console.print(f"  Branch: {patch.branch_name}")
    console.print(f"  New files: {len(patch.new_files)}")
    console.print(f"  Transformations: {len(patch.transformations)}")

    console.print("\n[bold cyan]Step 5: Validation...[/bold cyan]")
    runner = ValidationRunner(demo_path)
    validation = runner.run({}, str(demo_path))
    console.print(f"  Stage: {validation.stage}")
    console.print(f"  Passed: {'[green]Yes[/green]' if validation.passed else '[red]No[/red]'}")

    console.print("\n[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    app()


def main():
    app()
