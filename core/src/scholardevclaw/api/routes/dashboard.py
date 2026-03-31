"""
Dashboard API routes for the ScholarDevClaw web dashboard.

Provides:
- GET  /api/specs          - list all available paper specs
- GET  /api/specs/{name}   - get a single spec
- POST /api/pipeline/run   - run the full pipeline (non-blocking)
- GET  /api/pipeline/status - get current pipeline run status
- WS   /api/ws/pipeline    - real-time pipeline progress via WebSocket
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_API_AUTH_KEY = os.environ.get("SCHOLARDEVCLAW_API_AUTH_KEY", "")


def _validate_repo_path(repo_path: str) -> Path:
    """Validate and confine repo_path to allowed directories."""
    p = Path(repo_path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {repo_path}")

    allowed = os.environ.get("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", "")
    if allowed:
        allowed_dirs = [Path(d.strip()).resolve() for d in allowed.split(":") if d.strip()]
        if not any(_is_subpath(p, ad) for ad in allowed_dirs):
            raise HTTPException(
                status_code=403,
                detail=f"Repository path is outside the allowed directories: {repo_path}",
            )
    return p


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_dir(output_dir: str | None, repo_path: Path) -> Path | None:
    """Validate output_dir is within or adjacent to the repo path."""
    if output_dir is None:
        return None
    p = Path(output_dir).expanduser().resolve()
    # Allow output dirs that are subpaths of repo or siblings
    if not _is_subpath(p, repo_path.parent) and not _is_subpath(p, repo_path):
        raise HTTPException(
            status_code=403,
            detail=f"Output directory must be within or adjacent to the repository: {output_dir}",
        )
    return p


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SpecSummary(BaseModel):
    name: str
    title: str
    algorithm: str
    category: str
    replaces: str
    arxiv_id: str = ""
    description: str = ""


class PipelineRunRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the repository to analyze")
    spec_names: list[str] = Field(
        default_factory=list,
        description="Specs to run (empty = auto-suggest)",
    )
    skip_validate: bool = Field(default=False, description="Skip validation benchmarks")
    output_dir: str | None = Field(default=None, description="Write patch artifacts here")


class PipelineStepResult(BaseModel):
    step: str
    status: str  # "running", "completed", "failed"
    duration_seconds: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineRunStatus(BaseModel):
    run_id: str
    status: str  # "idle", "running", "completed", "failed"
    repo_path: str = ""
    spec_names: list[str] = Field(default_factory=list)
    steps: list[PipelineStepResult] = Field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    total_seconds: float = 0.0


# ---------------------------------------------------------------------------
# In-memory state (single-server; for multi-server use Redis/Convex)
# ---------------------------------------------------------------------------

_current_run: PipelineRunStatus | None = None
_ws_clients: list[WebSocket] = []


async def _broadcast(msg: dict[str, Any]) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    payload = json.dumps(msg)
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Spec endpoints
# ---------------------------------------------------------------------------


@router.get("/specs", response_model=list[SpecSummary])
async def list_specs():
    """List all available paper specifications."""
    from scholardevclaw.research_intelligence.extractor import PAPER_SPECS

    result = []
    for name, spec in sorted(PAPER_SPECS.items()):
        algo = spec.get("algorithm", {})
        paper = spec.get("paper", {})
        result.append(
            SpecSummary(
                name=name,
                title=paper.get("title", ""),
                algorithm=algo.get("name", ""),
                category=algo.get("category", ""),
                replaces=algo.get("replaces", ""),
                arxiv_id=paper.get("arxiv", ""),
                description=algo.get("description", ""),
            )
        )
    return result


@router.get("/specs/{name}", response_model=SpecSummary)
async def get_spec(name: str):
    """Get a single spec by name."""
    from scholardevclaw.research_intelligence.extractor import PAPER_SPECS

    spec = PAPER_SPECS.get(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Spec '{name}' not found")
    algo = spec.get("algorithm", {})
    paper = spec.get("paper", {})
    return SpecSummary(
        name=name,
        title=paper.get("title", ""),
        algorithm=algo.get("name", ""),
        category=algo.get("category", ""),
        replaces=algo.get("replaces", ""),
        arxiv_id=paper.get("arxiv", ""),
        description=algo.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Pipeline run endpoints
# ---------------------------------------------------------------------------


@router.get("/pipeline/status", response_model=PipelineRunStatus)
async def pipeline_status():
    """Get current pipeline run status."""
    if _current_run is None:
        return PipelineRunStatus(run_id="", status="idle")
    return _current_run


@router.post("/pipeline/run", response_model=PipelineRunStatus)
async def pipeline_run(request: PipelineRunRequest):
    """Start a full pipeline run (non-blocking; progress via WebSocket)."""
    global _current_run

    if _current_run and _current_run.status == "running":
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress")

    # Validate and confine repo_path
    repo_path = _validate_repo_path(request.repo_path)
    output_dir = _validate_output_dir(request.output_dir, repo_path)

    run_id = str(uuid.uuid4())[:8]
    _current_run = PipelineRunStatus(
        run_id=run_id,
        status="running",
        repo_path=str(repo_path),
        spec_names=request.spec_names,
        started_at=time.time(),
    )

    asyncio.create_task(
        _run_pipeline_async(
            run_id=run_id,
            repo_path=str(repo_path),
            spec_names=request.spec_names,
            skip_validate=request.skip_validate,
            output_dir=str(output_dir) if output_dir else None,
        )
    )

    return _current_run


async def _run_pipeline_async(
    run_id: str,
    repo_path: str,
    spec_names: list[str],
    skip_validate: bool,
    output_dir: str | None,
) -> None:
    """Execute pipeline steps in a background task, broadcasting progress."""
    global _current_run
    assert _current_run is not None

    loop = asyncio.get_event_loop()

    def _add_step(
        step: str,
        status: str,
        duration: float = 0.0,
        data: dict | None = None,
        error: str | None = None,
    ) -> PipelineStepResult:
        result = PipelineStepResult(
            step=step,
            status=status,
            duration_seconds=round(duration, 3),
            data=data or {},
            error=error,
        )
        _current_run.steps.append(result)  # type: ignore[union-attr]
        return result

    async def _step_broadcast(step: str, status: str, **kwargs: Any) -> None:
        msg = {"type": "pipeline_step", "run_id": run_id, "step": step, "status": status, **kwargs}
        await _broadcast(msg)

    try:
        # ---- 1. Analyze -----------------------------------------------
        await _step_broadcast("analyze", "running")
        t0 = time.time()

        def _do_analyze():
            from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

            analyzer = TreeSitterAnalyzer(Path(repo_path))
            return analyzer.analyze()

        analysis = await loop.run_in_executor(None, _do_analyze)
        dt = time.time() - t0

        analysis_data = {
            "languages": analysis.languages,
            "file_count": sum(s.file_count for s in analysis.language_stats),
            "elements": len(analysis.elements),
            "imports": len(analysis.imports),
            "frameworks": analysis.frameworks,
            "patterns": dict(analysis.patterns) if hasattr(analysis.patterns, "items") else {},
        }
        _add_step("analyze", "completed", dt, analysis_data)
        await _step_broadcast("analyze", "completed", data=analysis_data, duration=round(dt, 3))

        # ---- 2. Suggest -----------------------------------------------
        await _step_broadcast("suggest", "running")
        t0 = time.time()

        def _do_suggest():
            from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

            analyzer = TreeSitterAnalyzer(Path(repo_path))
            return analyzer.suggest_research_papers()

        suggestions = await loop.run_in_executor(None, _do_suggest)
        dt = time.time() - t0

        suggest_data = {
            "count": len(suggestions),
            "top": [
                {
                    "pattern": s.get("pattern", ""),
                    "title": s.get("paper", {}).get("title", ""),
                    "confidence": s.get("confidence", 0),
                }
                for s in suggestions[:10]
            ],
        }
        _add_step("suggest", "completed", dt, suggest_data)
        await _step_broadcast("suggest", "completed", data=suggest_data, duration=round(dt, 3))

        # ---- Resolve specs --------------------------------------------
        if not spec_names:
            from scholardevclaw.research_intelligence.extractor import PAPER_SPECS

            # Auto-select from suggestions
            seen = set()
            for s in suggestions:
                paper = s.get("paper", {})
                name = paper.get("name", "")
                if name and name in PAPER_SPECS and name not in seen:
                    spec_names.append(name)
                    seen.add(name)
            if not spec_names:
                spec_names = ["rmsnorm"]
            _current_run.spec_names = spec_names  # type: ignore[union-attr]
            await _step_broadcast("specs_resolved", "completed", data={"specs": spec_names})

        # ---- 3. Load specs --------------------------------------------
        from scholardevclaw.research_intelligence.extractor import ResearchExtractor

        extractor = ResearchExtractor()
        specs_loaded: dict[str, dict] = {}
        for name in spec_names:
            spec = extractor.get_spec(name)
            if spec:
                specs_loaded[name] = spec

        if not specs_loaded:
            raise RuntimeError(f"No specs could be loaded: {spec_names}")

        # ---- Per-spec: Map -> Generate -> Validate --------------------
        from scholardevclaw.mapping.engine import MappingEngine
        from scholardevclaw.patch_generation.generator import PatchGenerator
        from scholardevclaw.validation.runner import ValidationRunner

        for spec_name, spec in specs_loaded.items():
            # Map
            await _step_broadcast(f"map:{spec_name}", "running")
            t0 = time.time()

            def _do_map(s=spec):
                engine = MappingEngine(analysis.__dict__, s)
                return engine.map()

            mapping = await loop.run_in_executor(None, _do_map)
            dt = time.time() - t0

            map_data = {
                "spec": spec_name,
                "targets": len(mapping.targets),
                "strategy": mapping.strategy,
                "confidence": mapping.confidence,
                "target_files": [{"file": t.file, "line": t.line} for t in mapping.targets[:10]],
            }
            _add_step(f"map:{spec_name}", "completed", dt, map_data)
            await _step_broadcast(
                f"map:{spec_name}", "completed", data=map_data, duration=round(dt, 3)
            )

            # Generate
            await _step_broadcast(f"generate:{spec_name}", "running")
            t0 = time.time()

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
                "research_spec": spec,
            }

            def _do_generate(mr=mapping_result):
                gen = PatchGenerator(Path(repo_path))
                return gen.generate(mr)

            patch = await loop.run_in_executor(None, _do_generate)
            dt = time.time() - t0

            gen_data = {
                "spec": spec_name,
                "branch": patch.branch_name,
                "new_files": len(patch.new_files),
                "transformations": len(patch.transformations),
                "file_names": [f.path for f in patch.new_files],
            }

            # Write artifacts if output_dir
            if output_dir:
                out_path = Path(output_dir).expanduser().resolve() / spec_name
                out_path.mkdir(parents=True, exist_ok=True)
                for nf in patch.new_files:
                    dest = out_path / nf.path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(nf.content)
                gen_data["output_dir"] = str(out_path)

            _add_step(f"generate:{spec_name}", "completed", dt, gen_data)
            await _step_broadcast(
                f"generate:{spec_name}", "completed", data=gen_data, duration=round(dt, 3)
            )

            # Validate
            if not skip_validate:
                await _step_broadcast(f"validate:{spec_name}", "running")
                t0 = time.time()

                def _do_validate():
                    runner = ValidationRunner(Path(repo_path))
                    return runner.run({}, repo_path)

                validation = await loop.run_in_executor(None, _do_validate)
                dt = time.time() - t0

                val_data = {
                    "spec": spec_name,
                    "passed": validation.passed,
                    "stage": validation.stage,
                }
                _add_step(f"validate:{spec_name}", "completed", dt, val_data)
                await _step_broadcast(
                    f"validate:{spec_name}", "completed", data=val_data, duration=round(dt, 3)
                )

        # ---- Done -----------------------------------------------------
        _current_run.status = "completed"  # type: ignore[union-attr]
        _current_run.finished_at = time.time()  # type: ignore[union-attr]
        _current_run.total_seconds = round(  # type: ignore[union-attr]
            _current_run.finished_at - _current_run.started_at,
            2,  # type: ignore[union-attr]
        )
        await _broadcast(
            {
                "type": "pipeline_complete",
                "run_id": run_id,
                "status": "completed",
                "total_seconds": _current_run.total_seconds,  # type: ignore[union-attr]
            }
        )

    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        _add_step("error", "failed", error=str(exc))
        if _current_run:
            _current_run.status = "failed"
            _current_run.finished_at = time.time()
            _current_run.total_seconds = round(
                _current_run.finished_at - _current_run.started_at, 2
            )
        await _broadcast({"type": "pipeline_error", "run_id": run_id, "error": str(exc)})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline progress.

    Requires Bearer token via query parameter `token` when auth is configured.
    Limits connections to 20 concurrent clients per server.
    """
    # Auth check when API key is configured
    if _API_AUTH_KEY:
        token = websocket.query_params.get("token", "")
        if not token or not hmac.compare_digest(token, _API_AUTH_KEY):
            await websocket.close(code=4001, reason="Unauthorized")
            return

    # Connection cap
    if len(_ws_clients) >= 20:
        await websocket.close(code=4002, reason="Too many connections")
        return

    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))

    # Send current state on connect
    if _current_run:
        await websocket.send_text(_current_run.model_dump_json())

    try:
        while True:
            # Keep connection alive; handle client messages (ping/pong, etc.)
            data = await websocket.receive_text()
            # Clients can send {"type": "ping"} to keep alive
            if data:
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))
