from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding


class ProductScaffolder:
    def __init__(self, templates_dir: Path | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))

    def scaffold(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        reproducibility_report: ReproducibilityReport,
    ) -> None:
        project_dir.mkdir(parents=True, exist_ok=True)
        self._generate_api(project_dir, plan, understanding)
        self._generate_gradio_demo(project_dir, plan, understanding)
        self._generate_pyproject(project_dir, plan, understanding)
        self._generate_dockerfile(project_dir, plan)
        self._generate_readme(project_dir, plan, understanding, reproducibility_report)
        self._generate_github_actions(project_dir)
        # Quick wins for researchers
        self._generate_bibtex(project_dir, understanding)
        self._generate_gitignore(project_dir)
        self._generate_seed_utils(project_dir)

    def _generate_api(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
    ) -> Path:
        template = self.env.get_template("api_main.py.j2")
        out = template.render(
            project_name=plan.project_name,
            summary=understanding.one_line_summary,
            io_spec=understanding.input_output_spec,
            entry_module=self._entry_module(plan),
        )
        api_dir = project_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)
        output_path = api_dir / "main.py"
        output_path.write_text(out, encoding="utf-8")
        return output_path

    def _generate_gradio_demo(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
    ) -> Path:
        template = self.env.get_template("gradio_demo.py.j2")
        out = template.render(
            project_name=plan.project_name,
            summary=understanding.one_line_summary,
            io_spec=understanding.input_output_spec,
            entry_module=self._entry_module(plan),
        )
        output_path = project_dir / "demo.py"
        output_path.write_text(out, encoding="utf-8")
        return output_path

    def _generate_pyproject(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
    ) -> Path:
        deps = [f"{name}{version}" for name, version in sorted(plan.environment.items())]

        tomlkit = None
        try:
            import tomlkit as _tomlkit  # type: ignore[import-not-found]

            tomlkit = _tomlkit
        except ImportError:
            tomlkit = None

        if tomlkit is not None:
            doc = tomlkit.document()
            project = tomlkit.table()
            project.add("name", plan.project_name)
            project.add("version", "0.1.0")
            project.add("description", understanding.one_line_summary)
            project.add("requires-python", ">=3.11")
            project.add("dependencies", deps)
            doc.add("project", project)
            content = tomlkit.dumps(doc)
        else:
            dep_lines = "\n".join(f'    "{dep}",' for dep in deps)
            content = (
                "[project]\n"
                f'name = "{self._toml_escape(plan.project_name)}"\n'
                'version = "0.1.0"\n'
                f'description = "{self._toml_escape(understanding.one_line_summary)}"\n'
                'requires-python = ">=3.11"\n'
                "dependencies = [\n"
                f"{dep_lines}\n"
                "]\n"
            )

        output_path = project_dir / "pyproject.toml"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def _generate_dockerfile(self, project_dir: Path, plan: ImplementationPlan) -> Path:
        template = self.env.get_template("Dockerfile.j2")
        out = template.render(
            project_name=plan.project_name,
            stack=plan.tech_stack,
            deps=sorted(plan.environment.keys()),
        )
        output_path = project_dir / "Dockerfile"
        output_path.write_text(out, encoding="utf-8")
        return output_path

    def _generate_readme(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        reproducibility_report: ReproducibilityReport,
    ) -> Path:
        template = self.env.get_template("README.md.j2")
        out = template.render(
            project_name=plan.project_name,
            paper_title=understanding.paper_title,
            summary=understanding.one_line_summary,
            key_insight=understanding.key_insight,
            problem=understanding.problem_statement,
            io_spec=understanding.input_output_spec,
            evaluation=understanding.evaluation_protocol,
            repro_score=reproducibility_report.score,
            repro_verdict=reproducibility_report.verdict,
            claimed_metrics=reproducibility_report.claimed_metrics,
            achieved_metrics=reproducibility_report.achieved_metrics,
            requirements=[req.name for req in understanding.requirements],
            stack=plan.tech_stack,
        )
        output_path = project_dir / "README.md"
        output_path.write_text(out, encoding="utf-8")
        return output_path

    def _generate_github_actions(self, project_dir: Path) -> Path:
        ci = (
            "name: CI\n"
            "on: [push, pull_request]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            '        with: {python-version: "3.11"}\n'
            '      - run: pip install -e ".[dev]"\n'
            "      - run: pytest tests/ -v\n"
            "      - run: docker build -t app .\n"
        )
        actions_dir = project_dir / ".github" / "workflows"
        actions_dir.mkdir(parents=True, exist_ok=True)
        output_path = actions_dir / "ci.yml"
        output_path.write_text(ci, encoding="utf-8")
        return output_path

    def _entry_module(self, plan: ImplementationPlan) -> str:
        if not plan.entry_points:
            return "main"

        candidate = str(plan.entry_points[0]).strip()
        if not candidate:
            return "main"

        candidate = candidate.replace("\\", "/")
        if candidate.startswith("python -m "):
            candidate = candidate[len("python -m ") :]

        if candidate.endswith(".py"):
            candidate = candidate[:-3]
        if candidate.startswith("src/"):
            candidate = candidate[len("src/") :]
        if candidate.startswith("src."):
            candidate = candidate[len("src.") :]

        normalized = candidate.replace("/", ".").strip(".")
        if not normalized:
            return "main"

        parts = [part for part in normalized.split(".") if part]
        safe_parts = [self._sanitize_identifier(part) for part in parts]
        safe_parts = [part for part in safe_parts if part]
        return ".".join(safe_parts) if safe_parts else "main"

    def _sanitize_identifier(self, text: str) -> str:
        chars = [ch if (ch.isalnum() or ch == "_") else "_" for ch in text.strip()]
        result = "".join(chars)
        if not result:
            return ""
        if result[0].isdigit():
            result = f"_{result}"
        return result

    def _toml_escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _generate_bibtex(
        self,
        project_dir: Path,
        understanding: PaperUnderstanding,
    ) -> Path:
        """Generate CITATION.bib with the source paper's BibTeX entry."""
        # Build a reasonable BibTeX key
        title_words = understanding.paper_title.split()
        first_word = title_words[0].lower() if title_words else "paper"
        year = ""
        # Try to extract year from any requirements or known fields
        for req in understanding.requirements:
            if req.name and req.name.isdigit() and len(req.name) == 4:
                year = req.name
                break

        bib_key = f"{first_word}{year or '2024'}"

        authors = ""
        # Construct author list from contributions if available
        if understanding.contributions:
            authors = "ScholarDevClaw Auto-Generated"

        bibtex = (
            f"@article{{{bib_key},\n"
            f"  title = {{{understanding.paper_title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year or '2024'}}},\n"
            f"  note = {{Implementation generated by ScholarDevClaw}},\n"
            f"  abstract = {{{understanding.one_line_summary}}},\n"
            f"}}\n"
        )

        output_path = project_dir / "CITATION.bib"
        output_path.write_text(bibtex, encoding="utf-8")
        return output_path

    def _generate_gitignore(self, project_dir: Path) -> Path:
        """Generate a research-specific .gitignore."""
        gitignore = (
            "# Python\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            "*$py.class\n"
            "*.so\n"
            "*.egg-info/\n"
            "dist/\n"
            "build/\n"
            ".eggs/\n"
            "\n"
            "# Virtual environments\n"
            ".venv/\n"
            "venv/\n"
            "env/\n"
            "\n"
            "# Research artifacts (large files)\n"
            "checkpoints/\n"
            "*.ckpt\n"
            "*.pt\n"
            "*.pth\n"
            "*.safetensors\n"
            "*.h5\n"
            "*.pkl\n"
            "*.pickle\n"
            "\n"
            "# Data\n"
            "data/\n"
            "datasets/\n"
            "*.csv\n"
            "*.tsv\n"
            "*.parquet\n"
            "\n"
            "# Experiment tracking\n"
            "wandb/\n"
            "mlruns/\n"
            "outputs/\n"
            "logs/\n"
            "*.log\n"
            "\n"
            "# Jupyter\n"
            ".ipynb_checkpoints/\n"
            "\n"
            "# IDE\n"
            ".vscode/\n"
            ".idea/\n"
            "*.swp\n"
            "*.swo\n"
            "\n"
            "# OS\n"
            ".DS_Store\n"
            "Thumbs.db\n"
            "\n"
            "# Docker\n"
            ".docker/\n"
        )

        output_path = project_dir / ".gitignore"
        output_path.write_text(gitignore, encoding="utf-8")
        return output_path

    def _generate_seed_utils(self, project_dir: Path) -> Path:
        """Generate a seed-locking utility for reproducible experiments."""
        seed_code = (
            '"""Reproducibility utilities — seed locking for all RNG sources."""\n'
            "\n"
            "import os\n"
            "import random\n"
            "\n"
            "import numpy as np\n"
            "\n"
            "\n"
            "def set_seed(seed: int = 42) -> None:\n"
            '    """Lock all random number generators for reproducibility.\n'
            "\n"
            "    Call this at the start of training and evaluation.\n"
            '    """\n'
            "    random.seed(seed)\n"
            "    np.random.seed(seed)\n"
            '    os.environ["PYTHONHASHSEED"] = str(seed)\n'
            "\n"
            "    try:\n"
            "        import torch\n"
            "\n"
            "        torch.manual_seed(seed)\n"
            "        torch.cuda.manual_seed_all(seed)\n"
            "        torch.backends.cudnn.deterministic = True\n"
            "        torch.backends.cudnn.benchmark = False\n"
            "    except ImportError:\n"
            "        pass\n"
            "\n"
            "    try:\n"
            "        import tensorflow as tf\n"
            "\n"
            "        tf.random.set_seed(seed)\n"
            "    except ImportError:\n"
            "        pass\n"
            "\n"
            "    try:\n"
            "        import jax\n"
            "\n"
            "        # JAX uses explicit PRNG keys, but we can set the global seed\n"
            "        jax.config.update('jax_enable_custom_prng', True)\n"
            "    except ImportError:\n"
            "        pass\n"
        )

        utils_dir = project_dir / "src" / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)
        (utils_dir / "__init__.py").write_text("", encoding="utf-8")

        output_path = utils_dir / "reproducibility.py"
        output_path.write_text(seed_code, encoding="utf-8")
        return output_path
