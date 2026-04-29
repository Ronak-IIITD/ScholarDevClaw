import hmac
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..application.schema_contract import SCHEMA_VERSION
from ..mapping.engine import MappingEngine
from ..patch_generation.generator import PatchGenerator
from ..repo_intelligence.tree_sitter_analyzer import RepoAnalysis, TreeSitterAnalyzer
from ..research_intelligence.extractor import ResearchExtractionError, ResearchExtractor
from ..utils.health import health_checker, liveness_probe, readiness_probe
from ..utils.shutdown import shutdown_manager
from ..validation.runner import ValidationRunner
from .docs import setup_docs_routes, setup_exception_handlers, setup_openapi
from .metrics_middleware import setup_metrics
from .rate_limit_middleware import setup_rate_limiting

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security: Allowed base directories for repo path confinement
# ---------------------------------------------------------------------------
_ALLOWED_BASE_DIRS: list[Path] = []
# Per-user allowed directories: X-User-ID -> list of allowed Path
_USER_ALLOWED_DIRS: dict[str, list[Path]] = {}

_env_allowed = os.environ.get("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", "")
if _env_allowed:
    for d in _env_allowed.split(":"):
        d = d.strip()
        if d:
            _ALLOWED_BASE_DIRS.append(Path(d).resolve())
elif os.environ.get("SCHOLARDEVCLAW_DEV_MODE", "").lower() != "true":
    raise RuntimeError(
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS is not set and SCHOLARDEVCLAW_DEV_MODE is not 'true'. "
        "Set SCHOLARDEVCLAW_ALLOWED_REPO_DIRS to a colon-separated list of allowed repo roots "
        "(e.g. /repos:/workspace) or set SCHOLARDEVCLAW_DEV_MODE=true for local development."
    )

# Load per-user allowed directories from environment
# Format: SCHOLARDEVCLAW_USER_ALLOWED_DIRS_user123=/home/user123/repos:/home/user123/work
for key, val in os.environ.items():
    if key.startswith("SCHOLARDEVCLAW_USER_ALLOWED_DIRS_"):
        user_id = key.removeprefix("SCHOLARDEVCLAW_USER_ALLOWED_DIRS_")
        user_dirs = []
        for d in val.split(":"):
            d = d.strip()
            if d:
                user_dirs.append(Path(d).resolve())
        if user_dirs:
            _USER_ALLOWED_DIRS[user_id] = user_dirs


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
setup_rate_limiting(app)

# Dashboard routes (specs, pipeline run, WebSocket)
from .routes.dashboard import router as dashboard_router  # noqa: E402

app.include_router(dashboard_router)

# CORS: restrict origins in production via env var; allow Vite dev server in development
_allowed_origins = os.environ.get("SCHOLARDEVCLAW_CORS_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if not _allowed_origins:
    # Development defaults: allow local Vite dev server
    _allowed_origins = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if os.environ.get("SCHOLARDEVCLAW_ENABLE_HSTS", "").lower() == "true":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# API key authentication middleware
# ---------------------------------------------------------------------------
_API_AUTH_KEY = os.environ.get("SCHOLARDEVCLAW_API_AUTH_KEY", "")
_DEV_MODE = os.environ.get("SCHOLARDEVCLAW_DEV_MODE", "").lower() == "true"

# SECURITY: Warn if dev mode is active but not in development environment
if _DEV_MODE:
    import warnings
    env = os.environ.get("ENV", os.environ.get("NODE_ENV", "")).lower()
    if env not in ("development", "dev", "local"):
        warnings.warn(
            "SCHOLARDEVCLAW_DEV_MODE is enabled but ENV is not 'development'. "
            "Authentication is DISABLED - do NOT use this in production!",
            UserWarning,
        )
        logger.warning("SECURITY WARNING: Dev mode active in non-dev environment %s", env)

if not _API_AUTH_KEY and not _DEV_MODE:
    raise RuntimeError(
        "SCHOLARDEVCLAW_API_AUTH_KEY is not set and SCHOLARDEVCLAW_DEV_MODE is not 'true'. "
        "Set SCHOLARDEVCLAW_API_AUTH_KEY to a strong secret or set SCHOLARDEVCLAW_DEV_MODE=true "
        "for local development."
    )
_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/", "/metrics"}


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Require API key for all non-exempt endpoints when configured.

    SECURITY: Extracts X-User-ID for per-user repo isolation. If no X-User-ID
    is provided, uses global _ALLOWED_BASE_DIRS (legacy single-tenant behavior).
    """
    if _API_AUTH_KEY:
        # Use prefix matching for exempt paths (e.g. /docs/json matches /docs)
        is_exempt = any(
            request.url.path == ep or request.url.path.startswith(ep + "/")
            for ep in _AUTH_EXEMPT_PATHS
        )
        if not is_exempt:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not provided or not hmac.compare_digest(provided, _API_AUTH_KEY):
                import json as _json

                return Response(
                    content=_json.dumps({"detail": "Unauthorized"}),
                    status_code=401,
                    media_type="application/json",
                    headers={"X-Request-ID": getattr(request.state, "request_id", "")},
                )

    # SECURITY: Extract user identity for per-user repo isolation
    user_id = request.headers.get("X-User-ID", "").strip() or None
    request.state.user_id = user_id

    return await call_next(request)


class RepoAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repoPath: str  # noqa: N815

    @field_validator("repoPath")
    @classmethod
    def validate_repo_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "What failed: request validation for 'repoPath'. "
                "Why: repoPath is empty. "
                "Fix: provide a non-empty repository path string."
            )
        return value


class ResearchExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    sourceType: Literal["pdf", "arxiv"] = "pdf"  # noqa: N815

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "What failed: request validation for 'source'. "
                "Why: source is empty. "
                "Fix: provide a non-empty paper source (arXiv ID, DOI, URL, or PDF path)."
            )
        return value


class MappingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    repo_analysis: dict[str, Any] = Field(alias="repoAnalysis")
    research_spec: dict[str, Any] = Field(alias="researchSpec")


class PatchGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mapping: dict[str, Any]
    repo_path: str = Field(alias="repoPath")


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    patch: dict[str, Any]
    repo_path: str = Field(alias="repoPath")


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
    lossFn: str  # noqa: N815


class ArchitectureEntry(BaseModel):
    models: list[ModelEntry]
    trainingLoop: TrainingLoopEntry | None = None  # noqa: N815


class TestSuiteEntry(BaseModel):
    runner: str
    testFiles: list[str]  # noqa: N815


class RepoAnalyzeResponse(BaseModel):
    repoName: str  # noqa: N815
    architecture: ArchitectureEntry
    dependencies: dict[str, Any]
    testSuite: TestSuiteEntry  # noqa: N815


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
    moduleName: str  # noqa: N815
    parentClass: str  # noqa: N815
    parameters: list[str]
    codeTemplate: str | None = None  # noqa: N815


class ResearchChanges(BaseModel):
    type: str
    targetPattern: str = ""  # noqa: N815
    targetPatterns: list[str] = Field(default_factory=list)  # noqa: N815
    insertionPoints: list[str]  # noqa: N815
    replacement: str | None = None
    expectedBenefits: list[str] = Field(default_factory=list)  # noqa: N815


class ResearchExtractResponse(BaseModel):
    paper: ResearchPaper
    algorithm: ResearchAlgorithm
    implementation: ResearchImplementation
    changes: ResearchChanges
    validation: dict[str, Any] | None = None


class MappingTargetResponse(BaseModel):
    file: str
    line: int
    currentCode: str  # noqa: N815
    replacementRequired: bool  # noqa: N815
    context: dict[str, Any] = Field(default_factory=dict)
    original: str | None = None
    replacement: str | None = None


class MappingResponse(BaseModel):
    targets: list[MappingTargetResponse]
    strategy: str
    confidence: int
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    research_spec: dict[str, Any] = Field(default_factory=dict)
    researchSpec: dict[str, Any] = Field(default_factory=dict)  # noqa: N815


class PatchFileResponse(BaseModel):
    path: str
    content: str


class TransformationResponse(BaseModel):
    file: str
    original: str
    modified: str
    changes: list[dict[str, Any]] = Field(default_factory=list)


class PatchGenerateResponse(BaseModel):
    newFiles: list[PatchFileResponse]  # noqa: N815
    transformations: list[TransformationResponse]
    branchName: str  # noqa: N815


class MetricsResponse(BaseModel):
    loss: float
    perplexity: float
    tokensPerSecond: float  # noqa: N815
    memoryMb: float  # noqa: N815


class ValidationResponse(BaseModel):
    passed: bool
    stage: str
    baselineMetrics: MetricsResponse | None = None  # noqa: N815
    newMetrics: MetricsResponse | None = None  # noqa: N815
    comparison: dict[str, Any] | None = None
    logs: str
    error: str | None = None
    schemaVersion: str | None = None  # noqa: N815
    payloadType: str | None = None  # noqa: N815


def _get_allowed_dirs_for_user(user_id: str | None) -> list[Path]:
    """Get allowed directories for a user. Falls back to global dirs if no user or no per-user config."""
    if user_id and user_id in _USER_ALLOWED_DIRS:
        return _USER_ALLOWED_DIRS[user_id]
    return _ALLOWED_BASE_DIRS


def _resolve_existing_repo_path(repo_path: str, user_id: str | None = None) -> Path:
    """Resolve and validate a repo path, enforcing confinement.

    SECURITY: Uses per-user allowed directories when X-User-ID is provided and
    per-user configs exist. Falls back to global _ALLOWED_BASE_DIRS for backward
    compatibility (single-tenant deployments).
    """
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "What failed: repository path resolution. "
                f"Why: repository path '{repo_path}' does not exist. "
                "Fix: provide an existing repository directory."
            ),
        )
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "What failed: repository path resolution. "
                f"Why: path '{repo_path}' is not a directory. "
                "Fix: provide a directory path."
            ),
        )

    # SECURITY: Path confinement with per-user isolation
    allowed_dirs = _get_allowed_dirs_for_user(user_id)
    if allowed_dirs:
        if not any(path == base or path.is_relative_to(base) for base in allowed_dirs):
            logger.warning(
                "Path confinement violation for user=%s: %s (allowed=%s)",
                user_id,
                path,
                allowed_dirs,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    "What failed: repository path confinement check. "
                    "Why: resolved path is outside user's allowed directories. "
                    "Fix: use a repository within your allowed directories or contact admin."
                ),
            )

    return path


def _normalize_research_spec(raw: dict[str, Any]) -> ResearchExtractResponse:
    paper = raw.get("paper", {})
    algorithm = raw.get("algorithm", {})
    implementation = raw.get("implementation", {})
    changes = raw.get("changes", {})

    target_patterns = _as_str_list(changes.get("target_patterns"))
    if not target_patterns:
        target_patterns = _as_str_list(changes.get("targetPatterns"))
    if not target_patterns and isinstance(changes.get("targetPattern"), str):
        single_target = str(changes.get("targetPattern") or "").strip()
        if single_target:
            target_patterns = [single_target]

    insertion_points = _as_str_list(changes.get("insertion_points"))
    if not insertion_points:
        insertion_points = _as_str_list(changes.get("insertionPoints"))

    expected_benefits = _as_str_list(changes.get("expected_benefits"))
    if not expected_benefits:
        expected_benefits = _as_str_list(changes.get("expectedBenefits"))

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
            "targetPatterns": target_patterns,
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


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        candidate_values = [value]
    elif isinstance(value, (list, tuple, set)):
        candidate_values = list(value)
    else:
        return []

    normalized: list[str] = []
    for item in candidate_values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


def _coerce_int(value: Any, *, field_name: str, default: int | None = None) -> int:
    if value in (None, ""):
        if default is not None:
            return default
        raise HTTPException(
            status_code=422,
            detail=(
                "What failed: integer field validation. "
                f"Why: required field '{field_name}' is missing. "
                "Fix: provide this field with an integer value."
            ),
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "What failed: integer field validation. "
                f"Why: field '{field_name}' is not a valid integer. "
                "Fix: provide a numeric integer value."
            ),
        ) from exc


def _resolve_llm_selection() -> tuple[str | None, str | None]:
    provider = os.environ.get("SCHOLARDEVCLAW_API_PROVIDER", "").strip().lower()
    model = os.environ.get("SCHOLARDEVCLAW_API_MODEL", "").strip()
    if not provider or provider == "auto":
        return None, None
    return provider, model or None


def _create_llm_assistant() -> Any | None:
    provider, model = _resolve_llm_selection()
    if not provider:
        return None
    try:
        from scholardevclaw.llm.research_assistant import LLMResearchAssistant

        assistant = LLMResearchAssistant.create(provider=provider, model=model)
        return assistant if assistant.is_available else None
    except Exception as exc:
        logger.debug("LLM assistant unavailable for provider=%s: %s", provider, exc)
        return None


def _normalize_research_spec_for_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    paper = raw.get("paper", {}) if isinstance(raw.get("paper"), dict) else {}
    algorithm = raw.get("algorithm", {}) if isinstance(raw.get("algorithm"), dict) else {}
    implementation = (
        raw.get("implementation", {}) if isinstance(raw.get("implementation"), dict) else {}
    )
    changes = raw.get("changes", {}) if isinstance(raw.get("changes"), dict) else {}

    target_patterns = _as_str_list(changes.get("target_patterns"))
    if not target_patterns:
        target_patterns = _as_str_list(changes.get("targetPatterns"))
    if not target_patterns and isinstance(changes.get("targetPattern"), str):
        single_target = str(changes.get("targetPattern") or "").strip()
        if single_target:
            target_patterns = [single_target]

    insertion_points = _as_str_list(changes.get("insertion_points"))
    if not insertion_points:
        insertion_points = _as_str_list(changes.get("insertionPoints"))

    expected_benefits = _as_str_list(changes.get("expected_benefits"))
    if not expected_benefits:
        expected_benefits = _as_str_list(changes.get("expectedBenefits"))

    return {
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
            "category": algorithm.get("category"),
        },
        "implementation": {
            "module_name": implementation.get("module_name", implementation.get("moduleName", "")),
            "parent_class": implementation.get(
                "parent_class", implementation.get("parentClass", "")
            ),
            "parameters": implementation.get("parameters") or [],
            "code_template": implementation.get(
                "code_template", implementation.get("codeTemplate", "")
            ),
        },
        "changes": {
            "type": changes.get("type", "replace"),
            "target_patterns": target_patterns,
            "replacement": changes.get("replacement", ""),
            "insertion_points": insertion_points,
            "expected_benefits": expected_benefits,
        },
        "validation": raw.get("validation") if isinstance(raw.get("validation"), dict) else {},
    }


def _normalize_mapping_for_patch(mapping: dict[str, Any]) -> dict[str, Any]:
    raw_spec = mapping.get("research_spec") or mapping.get("researchSpec") or {}
    research_spec = (
        _normalize_research_spec_for_mapping(raw_spec) if isinstance(raw_spec, dict) else {}
    )
    default_replacement = research_spec.get("changes", {}).get("replacement", "")

    targets: list[dict[str, Any]] = []
    for raw_target in mapping.get("targets", []):
        if not isinstance(raw_target, dict):
            continue
        raw_context = raw_target.get("context")
        context = dict(raw_context) if isinstance(raw_context, dict) else {}

        current_code = raw_target.get("current_code", raw_target.get("currentCode", ""))
        if not current_code and isinstance(context.get("original"), str):
            current_code = context.get("original", "")

        original = raw_target.get("original") or context.get("original") or current_code
        replacement = (
            raw_target.get("replacement") or context.get("replacement") or default_replacement
        )

        context.setdefault("original", original)
        if replacement:
            context.setdefault("replacement", replacement)

        targets.append(
            {
                "file": raw_target.get("file", ""),
                "line": _coerce_int(
                    raw_target.get("line", 0),
                    field_name="mapping.targets[].line",
                    default=0,
                ),
                "current_code": current_code,
                "replacement_required": bool(
                    raw_target.get(
                        "replacement_required", raw_target.get("replacementRequired", True)
                    )
                ),
                "context": context,
            }
        )

    return {
        "targets": targets,
        "strategy": mapping.get("strategy", "none"),
        "confidence": _coerce_int(
            mapping.get("confidence", 0),
            field_name="mapping.confidence",
            default=0,
        ),
        "research_spec": research_spec,
    }


def _select_model_elements(analysis: RepoAnalysis) -> list[dict[str, Any]]:
    class_elements = [element for element in analysis.elements if element.type == "class"]
    preferred = [
        element
        for element in class_elements
        if any(
            token in element.name.lower()
            for token in ("model", "transformer", "encoder", "decoder", "module", "net", "block")
        )
    ]
    selected = preferred or class_elements

    return [
        {
            "name": element.name,
            "file": element.file,
            "line": element.line,
            "parent": element.dependencies[0] if element.dependencies else "",
            "components": {
                "language": element.language,
                "decorators": element.decorators,
                "parameters": element.parameters,
            },
        }
        for element in selected[:25]
    ]


def _select_training_loop(analysis: RepoAnalysis) -> dict[str, Any] | None:
    optimizer_names = {"adam", "adamw", "sgd", "rmsprop", "adagrad", "lion"}
    loss_names = {"cross_entropy", "mse", "nllloss", "bceloss", "l1loss"}

    optimizer = "unknown"
    loss_fn = "unknown"

    for statement in analysis.imports:
        module_name = statement.module.lower()
        imported_names = [name.lower() for name in statement.names]
        if "optim" in module_name:
            optimizer = next(
                (name for name in statement.names if name.lower() in optimizer_names),
                statement.names[0] if statement.names else "optimizer",
            )
        if any(name in imported_names for name in loss_names):
            loss_fn = next(
                (name for name in statement.names if name.lower() in loss_names),
                statement.names[0] if statement.names else "loss",
            )

    for element in analysis.elements:
        if element.type not in {"function", "async_function", "method", "async_method"}:
            continue
        if any(token in element.name.lower() for token in ("train", "fit", "training_step")):
            return {
                "file": element.file,
                "line": element.line,
                "optimizer": optimizer,
                "lossFn": loss_fn,
            }

    return None


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API is healthy and responsive",
)
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get(
    "/health/live",
    tags=["health"],
    summary="Liveness probe",
    description="Kubernetes/container liveness probe",
)
async def health_live():
    probe_status = liveness_probe.check()
    if not probe_status["alive"]:
        return JSONResponse(status_code=503, content=probe_status)
    return probe_status


@app.get(
    "/health/ready",
    tags=["health"],
    summary="Readiness probe",
    description="Kubernetes/container readiness probe",
)
async def health_ready():
    probe = readiness_probe.check()
    ready = bool(probe.get("ready", False))
    reasons = list(probe.get("reasons", []))

    if shutdown_manager.is_shutting_down():
        ready = False
        reasons.append("System is shutting down")

    if not health_checker.run_quick_check():
        ready = False
        reasons.append("Health checks failing")

    payload = {"ready": ready, "reasons": reasons}
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post(
    "/repo/analyze",
    response_model=RepoAnalyzeResponse,
    tags=["repo"],
    summary="Analyze repository",
    description="Analyze repository structure to identify models, training loops, and patterns",
)
async def analyze_repo(request: RepoAnalyzeRequest, http_request: Request):
    try:
        user_id = getattr(http_request.state, "user_id", None)
        repo_path = _resolve_existing_repo_path(request.repoPath, user_id=user_id)

        analyzer = TreeSitterAnalyzer(repo_path)
        analysis = analyzer.analyze()

        result = {
            "repoName": repo_path.name,
            "architecture": {
                "models": _select_model_elements(analysis),
                "trainingLoop": _select_training_loop(analysis),
            },
            "dependencies": analysis.dependencies,
            "testSuite": {
                "runner": "pytest" if analysis.test_files else "unknown",
                "testFiles": analysis.test_files,
            },
        }

        return RepoAnalyzeResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_repo failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "What failed: repository analysis request. "
                "Why: an unexpected internal error occurred while analyzing the repository. "
                "Fix: check server logs using X-Request-ID and retry."
            ),
        )


@app.post(
    "/research/extract",
    response_model=ResearchExtractResponse,
    tags=["research"],
    summary="Extract research spec",
    description="Extract implementation specification from a research paper",
)
async def extract_research(request: ResearchExtractRequest):
    try:
        llm_assistant = _create_llm_assistant()
        extractor = ResearchExtractor(llm_assistant=llm_assistant)
        raw_result = extractor.extract(request.source, request.sourceType)

        if not isinstance(raw_result, dict):
            raise HTTPException(
                status_code=500,
                detail=(
                    "What failed: research extraction response validation. "
                    "Why: internal extractor returned a non-object payload. "
                    "Fix: check extraction logs and retry with a valid source."
                ),
            )

        return _normalize_research_spec(raw_result)
    except ResearchExtractionError as e:
        raise HTTPException(status_code=422, detail=e.to_error_metadata())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("extract_research failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "What failed: research extraction request. "
                "Why: an unexpected internal error occurred during extraction. "
                "Fix: check server logs using X-Request-ID and retry."
            ),
        )


@app.post(
    "/mapping/map",
    response_model=MappingResponse,
    tags=["mapping"],
    summary="Map spec to code",
    description="Map a research specification to specific code locations in the repository",
)
async def map_architecture(request: MappingRequest, http_request: Request):
    try:
        # user_id extracted for consistency but this endpoint doesn't access filesystem
        _ = getattr(http_request.state, "user_id", None)
        llm_assistant = _create_llm_assistant()
        normalized_spec = _normalize_research_spec_for_mapping(request.research_spec)
        engine = MappingEngine(
            request.repo_analysis,
            normalized_spec,
            llm_assistant=llm_assistant,
        )
        result = engine.map()

        response = {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "currentCode": t.current_code,
                    "replacementRequired": t.replacement_required,
                    "context": {
                        **(t.context if isinstance(t.context, dict) else {}),
                        "original": (
                            t.context.get("original", t.current_code)
                            if isinstance(t.context, dict)
                            else t.current_code
                        ),
                        "replacement": (
                            t.context.get(
                                "replacement", normalized_spec.get("changes", {}).get("replacement")
                            )
                            if isinstance(t.context, dict)
                            else normalized_spec.get("changes", {}).get("replacement")
                        ),
                    },
                    "original": (
                        t.context.get("original", t.current_code)
                        if isinstance(t.context, dict)
                        else t.current_code
                    ),
                    "replacement": (
                        t.context.get(
                            "replacement", normalized_spec.get("changes", {}).get("replacement")
                        )
                        if isinstance(t.context, dict)
                        else normalized_spec.get("changes", {}).get("replacement")
                    ),
                }
                for t in result.targets
            ],
            "strategy": result.strategy,
            "confidence": result.confidence,
            "confidence_breakdown": getattr(result, "confidence_breakdown", {}) or {},
            "research_spec": result.research_spec,
            "researchSpec": request.research_spec,
        }
        return MappingResponse.model_validate(response)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("map_architecture failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "What failed: mapping request. "
                "Why: an unexpected internal error occurred while mapping spec to code. "
                "Fix: check server logs using X-Request-ID and retry."
            ),
        )


@app.post(
    "/patch/generate",
    response_model=PatchGenerateResponse,
    tags=["patch"],
    summary="Generate patch",
    description="Generate code patches implementing the research specification",
)
async def generate_patch(request: PatchGenerateRequest, http_request: Request):
    try:
        user_id = getattr(http_request.state, "user_id", None)
        generator_repo_path = _resolve_existing_repo_path(request.repo_path, user_id=user_id)
        llm_assistant = _create_llm_assistant()
        generator = PatchGenerator(generator_repo_path, llm_assistant=llm_assistant)
        normalized_mapping = _normalize_mapping_for_patch(request.mapping)
        patch = generator.generate(normalized_mapping)

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
        logger.exception("generate_patch failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "What failed: patch generation request. "
                "Why: an unexpected internal error occurred while generating patch artifacts. "
                "Fix: check server logs using X-Request-ID and retry."
            ),
        )


@app.post(
    "/validation/run",
    response_model=ValidationResponse,
    tags=["validation"],
    summary="Run validation",
    description="Run tests and benchmarks to validate the generated patches",
)
async def run_validation(request: ValidationRequest, http_request: Request):
    try:
        user_id = getattr(http_request.state, "user_id", None)
        repo_path = _resolve_existing_repo_path(request.repo_path, user_id=user_id)

        runner = ValidationRunner(repo_path)
        result = runner.run(request.patch, str(repo_path))

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
        logger.exception("run_validation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "What failed: validation request. "
                "Why: an unexpected internal error occurred while running validation. "
                "Fix: check server logs using X-Request-ID and retry."
            ),
        )
