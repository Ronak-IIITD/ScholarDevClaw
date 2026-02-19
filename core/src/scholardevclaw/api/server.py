from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..repo_intelligence.parser import PyTorchRepoParser
from ..research_intelligence.extractor import ResearchExtractor
from ..mapping.engine import MappingEngine
from ..patch_generation.generator import PatchGenerator
from ..validation.runner import ValidationRunner
from ..application.schema_contract import SCHEMA_VERSION
from .docs import setup_openapi, setup_docs_routes, setup_exception_handlers
from .metrics_middleware import setup_metrics


app = FastAPI(
    title="ScholarDevClaw API",
    description="Autonomous ML Research Integration Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

setup_openapi(app)
setup_docs_routes(app)
setup_exception_handlers(app)
setup_metrics(app)


class RepoAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repoPath: str

    @field_validator("repoPath")
    @classmethod
    def validate_repo_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("repoPath cannot be empty")
        return value


class ResearchExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    sourceType: Literal["pdf", "arxiv"] = "pdf"

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source cannot be empty")
        return value


class MappingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repoAnalysis: dict[str, Any]
    researchSpec: dict[str, Any]


class PatchGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mapping: dict[str, Any]
    repoPath: str | None = None


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: dict[str, Any]
    repoPath: str


class ModelEntry(BaseModel):
    name: str
    file: str
    line: int
    parent: str
    components: dict[str, Any] = Field(default_factory=dict)


class TrainingLoopEntry(BaseModel):
    file: str
    line: int
    optimizer: str
    lossFn: str


class ArchitectureEntry(BaseModel):
    models: list[ModelEntry]
    trainingLoop: TrainingLoopEntry | None = None


class TestSuiteEntry(BaseModel):
    runner: str
    testFiles: list[str]


class RepoAnalyzeResponse(BaseModel):
    repoName: str
    architecture: ArchitectureEntry
    dependencies: dict[str, Any]
    testSuite: TestSuiteEntry


class ResearchPaper(BaseModel):
    title: str
    authors: list[str]
    arxiv: str | None = None
    year: int


class ResearchAlgorithm(BaseModel):
    name: str
    replaces: str | None = None
    description: str
    formula: str | None = None


class ResearchImplementation(BaseModel):
    moduleName: str
    parentClass: str
    parameters: list[str]
    codeTemplate: str | None = None


class ResearchChanges(BaseModel):
    type: str
    targetPattern: str
    insertionPoints: list[str]
    replacement: str | None = None
    expectedBenefits: list[str] = Field(default_factory=list)


class ResearchExtractResponse(BaseModel):
    paper: ResearchPaper
    algorithm: ResearchAlgorithm
    implementation: ResearchImplementation
    changes: ResearchChanges
    validation: dict[str, Any] | None = None


class MappingTargetResponse(BaseModel):
    file: str
    line: int
    currentCode: str
    replacementRequired: bool


class MappingResponse(BaseModel):
    targets: list[MappingTargetResponse]
    strategy: str
    confidence: int


class PatchFileResponse(BaseModel):
    path: str
    content: str


class TransformationResponse(BaseModel):
    file: str
    original: str
    modified: str
    changes: list[dict[str, Any]] = Field(default_factory=list)


class PatchGenerateResponse(BaseModel):
    newFiles: list[PatchFileResponse]
    transformations: list[TransformationResponse]
    branchName: str


class MetricsResponse(BaseModel):
    loss: float
    perplexity: float
    tokensPerSecond: float
    memoryMb: float


class ValidationResponse(BaseModel):
    passed: bool
    stage: str
    baselineMetrics: MetricsResponse | None = None
    newMetrics: MetricsResponse | None = None
    comparison: dict[str, Any] | None = None
    logs: str
    error: str | None = None
    schemaVersion: str | None = None
    payloadType: str | None = None


def _resolve_existing_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Repository path not found")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Repository path must be a directory")
    return path


def _normalize_research_spec(raw: dict[str, Any]) -> ResearchExtractResponse:
    paper = raw.get("paper", {})
    algorithm = raw.get("algorithm", {})
    implementation = raw.get("implementation", {})
    changes = raw.get("changes", {})

    target_patterns = changes.get("target_patterns") or []
    insertion_points = changes.get("insertion_points") or []
    expected_benefits = changes.get("expected_benefits") or []

    normalized = {
        "paper": {
            "title": paper.get("title", "Unknown"),
            "authors": paper.get("authors") or [],
            "arxiv": paper.get("arxiv"),
            "year": int(paper.get("year", 2024)),
        },
        "algorithm": {
            "name": algorithm.get("name", "Unknown"),
            "replaces": algorithm.get("replaces"),
            "description": algorithm.get("description", ""),
            "formula": algorithm.get("formula"),
        },
        "implementation": {
            "moduleName": implementation.get("module_name", implementation.get("moduleName", "")),
            "parentClass": implementation.get(
                "parent_class", implementation.get("parentClass", "")
            ),
            "parameters": implementation.get("parameters") or [],
            "codeTemplate": implementation.get("code_template", implementation.get("codeTemplate")),
        },
        "changes": {
            "type": changes.get("type", "replace"),
            "targetPattern": target_patterns[0] if target_patterns else "",
            "insertionPoints": insertion_points,
            "replacement": changes.get("replacement"),
            "expectedBenefits": expected_benefits,
        },
        "validation": raw.get("validation"),
    }

    return ResearchExtractResponse.model_validate(normalized)


def _metrics_to_response(metrics: Any) -> MetricsResponse | None:
    if metrics is None:
        return None
    return MetricsResponse(
        loss=metrics.loss,
        perplexity=metrics.perplexity,
        tokensPerSecond=metrics.tokens_per_second,
        memoryMb=metrics.memory_mb,
    )


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API is healthy and responsive",
)
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/repo/analyze",
    response_model=RepoAnalyzeResponse,
    tags=["repo"],
    summary="Analyze repository",
    description="Analyze repository structure to identify models, training loops, and patterns",
)
async def analyze_repo(request: RepoAnalyzeRequest):
    try:
        repo_path = _resolve_existing_repo_path(request.repoPath)

        parser = PyTorchRepoParser(repo_path)
        repo_map = parser.parse()

        result = {
            "repoName": repo_map.repo_name,
            "architecture": {
                "models": [
                    {
                        "name": m.name,
                        "file": m.file,
                        "line": m.line,
                        "parent": m.parent,
                        "components": m.components,
                    }
                    for m in repo_map.models
                ],
                "trainingLoop": {
                    "file": repo_map.training_loop.file,
                    "line": repo_map.training_loop.line,
                    "optimizer": repo_map.training_loop.optimizer,
                    "lossFn": repo_map.training_loop.loss_fn,
                }
                if repo_map.training_loop
                else None,
            },
            "dependencies": {},
            "testSuite": {
                "runner": "pytest",
                "testFiles": repo_map.test_files,
            },
        }

        return RepoAnalyzeResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/research/extract",
    response_model=ResearchExtractResponse,
    tags=["research"],
    summary="Extract research spec",
    description="Extract implementation specification from a research paper",
)
async def extract_research(request: ResearchExtractRequest):
    try:
        extractor = ResearchExtractor()
        raw_result = extractor.extract(request.source, request.sourceType)

        if not isinstance(raw_result, dict):
            raise HTTPException(status_code=500, detail="Invalid research extraction result")

        return _normalize_research_spec(raw_result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/mapping/map",
    response_model=MappingResponse,
    tags=["mapping"],
    summary="Map spec to code",
    description="Map a research specification to specific code locations in the repository",
)
async def map_architecture(request: MappingRequest):
    try:
        engine = MappingEngine(request.repoAnalysis, request.researchSpec)
        result = engine.map()

        response = {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "currentCode": t.current_code,
                    "replacementRequired": t.replacement_required,
                }
                for t in result.targets
            ],
            "strategy": result.strategy,
            "confidence": result.confidence,
        }
        return MappingResponse.model_validate(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/patch/generate",
    response_model=PatchGenerateResponse,
    tags=["patch"],
    summary="Generate patch",
    description="Generate code patches implementing the research specification",
)
async def generate_patch(request: PatchGenerateRequest):
    try:
        generator_repo_path = (
            _resolve_existing_repo_path(request.repoPath)
            if request.repoPath
            else Path(".").resolve()
        )
        generator = PatchGenerator(generator_repo_path)
        patch = generator.generate(request.mapping)

        response = {
            "newFiles": [{"path": f.path, "content": f.content} for f in patch.new_files],
            "transformations": [
                {
                    "file": t.file,
                    "original": t.original[:500],
                    "modified": t.modified[:500],
                    "changes": t.changes,
                }
                for t in patch.transformations
            ],
            "branchName": patch.branch_name,
        }
        return PatchGenerateResponse.model_validate(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/validation/run",
    response_model=ValidationResponse,
    tags=["validation"],
    summary="Run validation",
    description="Run tests and benchmarks to validate the generated patches",
)
async def run_validation(request: ValidationRequest):
    try:
        repo_path = _resolve_existing_repo_path(request.repoPath)

        runner = ValidationRunner(repo_path)
        result = runner.run(request.patch, request.repoPath)

        response = {
            "passed": result.passed,
            "stage": result.stage,
            "baselineMetrics": _metrics_to_response(result.baseline_metrics),
            "newMetrics": _metrics_to_response(result.new_metrics),
            "comparison": result.comparison,
            "logs": result.logs,
            "error": result.error,
            "schemaVersion": SCHEMA_VERSION,
            "payloadType": "validation",
        }
        return ValidationResponse.model_validate(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
