from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.product import ProductScaffolder
from scholardevclaw.understanding.models import PaperUnderstanding, Requirement


def _make_plan(entry_module: str = "main") -> ImplementationPlan:
    return ImplementationPlan.from_dict(
        {
            "project_name": "demo-product",
            "target_language": "python",
            "tech_stack": "pytorch",
            "modules": [],
            "environment": {
                "fastapi": ">=0.111.0",
                "uvicorn": ">=0.30.0",
                "pytest": ">=8.0.0",
            },
            "entry_points": [f"src/{entry_module}.py"],
        }
    )


def _make_understanding() -> PaperUnderstanding:
    return PaperUnderstanding(
        paper_title="Demo Paper",
        one_line_summary="A compact reproducible demo.",
        problem_statement="Demonstrate paper-driven scaffolding.",
        key_insight="Generate shipping artifacts deterministically.",
        input_output_spec="Input dictionary to output dictionary.",
        evaluation_protocol="Accuracy: 0.95",
        requirements=[
            Requirement(
                name="Python 3.11", requirement_type="runtime", is_optional=False, notes=""
            ),
        ],
    )


def _make_reproducibility_report() -> ReproducibilityReport:
    return ReproducibilityReport(
        paper_title="Demo Paper",
        claimed_metrics={"accuracy": 0.95},
        achieved_metrics={"accuracy": 0.94},
        delta={"accuracy": -0.01},
        score=0.99,
        verdict="reproduced",
    )


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_src_module_cache() -> None:
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]


def _stub_src_module(project_dir: Path, entry_module: str) -> None:
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / f"{entry_module}.py").write_text(
        'def run(payload: dict) -> dict:\n    return {"echo": payload}\n',
        encoding="utf-8",
    )


def test_scaffolder_creates_required_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "generated_project"
    scaffolder = ProductScaffolder()

    scaffolder.scaffold(
        project_dir,
        _make_plan(entry_module="main"),
        _make_understanding(),
        _make_reproducibility_report(),
    )

    expected = {
        project_dir / "api" / "main.py",
        project_dir / "demo.py",
        project_dir / "pyproject.toml",
        project_dir / "Dockerfile",
        project_dir / "README.md",
        project_dir / ".github" / "workflows" / "ci.yml",
    }
    for path in expected:
        assert path.exists()


def test_generated_fastapi_app_imports_and_health_endpoint_callable(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "api_project"
    entry_module = "entrypoint"
    _stub_src_module(project_dir, entry_module)

    scaffolder = ProductScaffolder()
    scaffolder.scaffold(
        project_dir,
        _make_plan(entry_module=entry_module),
        _make_understanding(),
        _make_reproducibility_report(),
    )

    _clear_src_module_cache()
    monkeypatch.syspath_prepend(str(project_dir))
    api_module = _load_module("generated_api_main", project_dir / "api" / "main.py")
    client = TestClient(api_module.app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "demo-product"}

    predict = client.post("/predict", json={"data": {"value": 1}})
    assert predict.status_code == 200
    assert predict.json() == {"result": {"echo": {"value": 1}}}


def test_generated_readme_has_non_empty_reproducibility_table(tmp_path: Path) -> None:
    project_dir = tmp_path / "readme_project"
    scaffolder = ProductScaffolder()
    scaffolder.scaffold(
        project_dir,
        _make_plan(entry_module="main"),
        _make_understanding(),
        _make_reproducibility_report(),
    )

    readme = (project_dir / "README.md").read_text(encoding="utf-8")
    assert "| Metric | Claimed | Achieved |" in readme
    assert "| accuracy | 0.95 | 0.94 |" in readme


def test_generated_demo_module_imports_without_errors(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "demo_project"
    entry_module = "serve"
    _stub_src_module(project_dir, entry_module)

    class _FakeTextbox:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class _FakeInterface:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def launch(self) -> None:
            return None

    fake_gradio = SimpleNamespace(Interface=_FakeInterface, Textbox=_FakeTextbox)
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    scaffolder = ProductScaffolder()
    scaffolder.scaffold(
        project_dir,
        _make_plan(entry_module=entry_module),
        _make_understanding(),
        _make_reproducibility_report(),
    )

    _clear_src_module_cache()
    monkeypatch.syspath_prepend(str(project_dir))
    demo_module = _load_module("generated_demo", project_dir / "demo.py")

    assert demo_module.run_inference("hello") == "{'echo': {'text': 'hello'}}"
