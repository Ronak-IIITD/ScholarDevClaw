from __future__ import annotations

import pytest

from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.planning.planner import ImplementationPlanner
from scholardevclaw.understanding.models import PaperUnderstanding, Requirement


def _mock_doc(domain: str = "unknown") -> PaperDocument:
    return PaperDocument(
        title="Paper",
        authors=[],
        arxiv_id=None,
        doi=None,
        year=None,
        abstract="",
        sections=[],
        equations=[],
        algorithms=[],
        figures=[],
        full_text="",
        pdf_path=None,
        references=[],
        keywords=[],
        domain=domain,
    )


def _mock_understanding(requirements: list[Requirement] | None = None) -> PaperUnderstanding:
    return PaperUnderstanding(
        paper_title="Test Paper",
        one_line_summary="summary",
        core_algorithm_description="algorithm",
        requirements=requirements or [],
    )


def is_topologically_ordered(plan: ImplementationPlan) -> bool:
    priorities = {module.id: module.priority for module in plan.modules}
    for module in plan.modules:
        for dep in module.depends_on:
            if dep not in priorities:
                return False
            if priorities[dep] >= module.priority:
                return False
    return True


def test_implementation_plan_roundtrip_with_multiple_modules() -> None:
    raw = {
        "project_name": "paper_impl",
        "target_language": "python",
        "tech_stack": "pytorch",
        "modules": [
            {
                "id": "data_loader",
                "name": "Data Loader",
                "description": "Load data",
                "file_path": "src/data/loader.py",
                "depends_on": [],
                "priority": 1,
                "estimated_lines": 80,
                "test_file_path": "tests/data/test_loader.py",
                "tech_stack": "pytorch",
            },
            {
                "id": "model_core",
                "name": "Model Core",
                "description": "Model implementation",
                "file_path": "src/model/core.py",
                "depends_on": ["data_loader"],
                "priority": 2,
                "estimated_lines": 200,
                "test_file_path": "tests/model/test_core.py",
                "tech_stack": "pytorch",
            },
        ],
        "directory_structure": {"src/": {"data/": {}, "model/": {}}, "tests/": {}},
        "environment": {"torch": ">=2.0.0", "numpy": ">=1.26.0"},
        "entry_points": ["python -m src.train"],
        "estimated_total_lines": 280,
    }

    plan = ImplementationPlan.from_dict(raw)
    roundtrip = ImplementationPlan.from_dict(plan.to_dict())

    assert len(roundtrip.modules) == 2
    assert roundtrip.to_dict() == plan.to_dict()


def test_select_tech_stack_prefers_jax_when_required() -> None:
    planner = object.__new__(ImplementationPlanner)
    understanding = _mock_understanding(
        requirements=[Requirement(name="JAX", type="library", is_optional=False, notes="")]
    )

    stack = planner._select_tech_stack(understanding, _mock_doc(domain="ml"))

    assert stack == "jax"


def test_select_tech_stack_prefers_numpy_for_systems_without_pytorch() -> None:
    planner = object.__new__(ImplementationPlanner)
    understanding = _mock_understanding(
        requirements=[
            Requirement(name="SIMD kernels", type="hardware", is_optional=False, notes="")
        ]
    )

    stack = planner._select_tech_stack(understanding, _mock_doc(domain="systems"))

    assert stack == "numpy-only"


def test_select_tech_stack_defaults_to_pytorch() -> None:
    planner = object.__new__(ImplementationPlanner)
    understanding = _mock_understanding()

    stack = planner._select_tech_stack(understanding, _mock_doc(domain="vision"))

    assert stack == "pytorch"


def test_planner_parse_json_response_handles_fenced_json() -> None:
    planner = object.__new__(ImplementationPlanner)
    raw = '```json\n{"project_name": "demo", "modules": []}\n```'

    parsed = planner._parse_json_response(raw)

    assert parsed["project_name"] == "demo"


def test_planner_parse_json_response_extracts_embedded_json_block() -> None:
    planner = object.__new__(ImplementationPlanner)
    raw = 'Result:\n{"project_name": "embedded", "modules": []}\nDone.'

    parsed = planner._parse_json_response(raw)

    assert parsed["project_name"] == "embedded"


def test_is_topologically_ordered_valid_and_invalid_cases() -> None:
    valid = ImplementationPlan.from_dict(
        {
            "project_name": "valid",
            "target_language": "python",
            "tech_stack": "pytorch",
            "modules": [
                {
                    "id": "data",
                    "name": "data",
                    "description": "",
                    "file_path": "src/data.py",
                    "depends_on": [],
                    "priority": 1,
                    "estimated_lines": 50,
                    "test_file_path": "tests/test_data.py",
                    "tech_stack": "pytorch",
                },
                {
                    "id": "model",
                    "name": "model",
                    "description": "",
                    "file_path": "src/model.py",
                    "depends_on": ["data"],
                    "priority": 2,
                    "estimated_lines": 120,
                    "test_file_path": "tests/test_model.py",
                    "tech_stack": "pytorch",
                },
            ],
        }
    )
    invalid = ImplementationPlan.from_dict(
        {
            "project_name": "invalid",
            "target_language": "python",
            "tech_stack": "pytorch",
            "modules": [
                {
                    "id": "data",
                    "name": "data",
                    "description": "",
                    "file_path": "src/data.py",
                    "depends_on": [],
                    "priority": 2,
                    "estimated_lines": 50,
                    "test_file_path": "tests/test_data.py",
                    "tech_stack": "pytorch",
                },
                {
                    "id": "model",
                    "name": "model",
                    "description": "",
                    "file_path": "src/model.py",
                    "depends_on": ["data"],
                    "priority": 1,
                    "estimated_lines": 120,
                    "test_file_path": "tests/test_model.py",
                    "tech_stack": "pytorch",
                },
            ],
        }
    )

    assert is_topologically_ordered(valid) is True
    assert is_topologically_ordered(invalid) is False


def test_planner_plan_raises_for_invalid_topological_order() -> None:
    planner = object.__new__(ImplementationPlanner)

    def _fake_llm_plan(_understanding: PaperUnderstanding, _stack: str):
        return {
            "project_name": "invalid",
            "target_language": "python",
            "tech_stack": "pytorch",
            "modules": [
                {
                    "id": "data",
                    "name": "data",
                    "description": "",
                    "file_path": "src/data.py",
                    "depends_on": [],
                    "priority": 2,
                    "estimated_lines": 40,
                    "test_file_path": "tests/test_data.py",
                    "tech_stack": "pytorch",
                },
                {
                    "id": "model",
                    "name": "model",
                    "description": "",
                    "file_path": "src/model.py",
                    "depends_on": ["data"],
                    "priority": 1,
                    "estimated_lines": 100,
                    "test_file_path": "tests/test_model.py",
                    "tech_stack": "pytorch",
                },
            ],
        }

    planner._llm_plan = _fake_llm_plan
    planner._select_tech_stack = lambda _u, _d: "pytorch"

    with pytest.raises(ValueError, match="topologically ordered"):
        planner.plan(_mock_understanding(), _mock_doc())


def test_planner_plan_uses_forced_stack_when_provided() -> None:
    planner = object.__new__(ImplementationPlanner)

    observed: dict[str, str] = {}

    def _fake_llm_plan(_understanding: PaperUnderstanding, stack: str):
        observed["stack"] = stack
        return {
            "project_name": "forced",
            "target_language": "python",
            "tech_stack": stack,
            "modules": [
                {
                    "id": "data",
                    "name": "data",
                    "description": "",
                    "file_path": "src/data.py",
                    "depends_on": [],
                    "priority": 1,
                    "estimated_lines": 10,
                    "test_file_path": "tests/test_data.py",
                    "tech_stack": stack,
                }
            ],
        }

    planner._llm_plan = _fake_llm_plan

    plan = planner.plan(
        _mock_understanding(),
        _mock_doc(domain="systems"),
        forced_stack="jax",
    )

    assert observed["stack"] == "jax"
    assert plan.tech_stack == "jax"
