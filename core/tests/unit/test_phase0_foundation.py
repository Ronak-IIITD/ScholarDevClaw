from __future__ import annotations

import ast
import json
from pathlib import Path

from scholardevclaw.exceptions import (
    ExecutionError,
    GenerationError,
    IngestionError,
    KnowledgeBaseError,
    PaperFetchError,
    PaperNotAccessibleError,
    PaperSourceResolutionError,
    PlanningError,
    SandboxError,
    ScholarDevClawError,
    UnderstandingError,
)
from scholardevclaw.execution.sandbox import DEFAULT_SANDBOX_IMAGE
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.ingestion.paper_fetcher import (
    PaperFetchError as FetcherPaperFetchError,
)
from scholardevclaw.ingestion.paper_fetcher import (
    PaperNotAccessibleError as FetcherPaperNotAccessibleError,
)
from scholardevclaw.ingestion.paper_fetcher import (
    PaperSourceResolutionError as FetcherPaperSourceResolutionError,
)
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_shared_exception_hierarchy_is_available() -> None:
    assert issubclass(IngestionError, ScholarDevClawError)
    assert issubclass(PaperFetchError, IngestionError)
    assert issubclass(PaperNotAccessibleError, PaperFetchError)
    assert issubclass(PaperSourceResolutionError, PaperFetchError)
    assert issubclass(UnderstandingError, ScholarDevClawError)
    assert issubclass(PlanningError, ScholarDevClawError)
    assert issubclass(GenerationError, ScholarDevClawError)
    assert issubclass(ExecutionError, ScholarDevClawError)
    assert issubclass(SandboxError, ExecutionError)
    assert issubclass(KnowledgeBaseError, ScholarDevClawError)


def test_ingestion_module_reuses_shared_fetch_exceptions() -> None:
    assert FetcherPaperFetchError is PaperFetchError
    assert FetcherPaperNotAccessibleError is PaperNotAccessibleError
    assert FetcherPaperSourceResolutionError is PaperSourceResolutionError


def test_phase0_seed_fixtures_roundtrip() -> None:
    paper_payload = json.loads((FIXTURES_DIR / "attention_paper_document.json").read_text())
    understanding_payload = json.loads((FIXTURES_DIR / "attention_understanding.json").read_text())
    plan_payload = json.loads((FIXTURES_DIR / "simple_plan.json").read_text())

    paper = PaperDocument.from_dict(paper_payload)
    understanding = PaperUnderstanding.from_dict(understanding_payload)
    plan = ImplementationPlan.from_dict(plan_payload)

    assert paper.domain == "nlp"
    assert len(paper.algorithms) >= 2
    assert len(paper.equations) >= 10
    assert understanding.complexity == "medium"
    assert len(understanding.concept_nodes) >= 6
    assert len(plan.modules) == 3
    assert plan.entry_points == ["train_loop"]


def test_phase0_broken_module_fixture_has_syntax_error() -> None:
    broken_source = (FIXTURES_DIR / "broken_module.py").read_text()

    try:
        ast.parse(broken_source)
    except SyntaxError:
        return

    raise AssertionError("broken_module.py fixture must contain a syntax error")


def test_phase0_passing_project_fixture_contains_runnable_layout() -> None:
    project_dir = FIXTURES_DIR / "passing_project"

    assert (project_dir / "src" / "__init__.py").exists()
    assert (project_dir / "src" / "demo.py").exists()
    assert (project_dir / "tests" / "test_demo.py").exists()
    assert DEFAULT_SANDBOX_IMAGE == "sdc-sandbox:latest"
