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
from typing import Any, cast

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from scholardevclaw.application.pipeline import (
    run_analyze,
    run_generate,
    run_map,
    run_suggest,
    run_validate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_API_AUTH_KEY = os.environ.get("SCHOLARDEVCLAW_API_AUTH_KEY", "")
_WS_QUERY_TOKEN_COMPAT = (
    os.environ.get("SCHOLARDEVCLAW_WS_QUERY_TOKEN_COMPAT", "").lower() == "true"
)


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
    """Validate output_dir is strictly within the repo path."""
    if output_dir is None:
        return None
    p = Path(output_dir).expanduser().resolve()
    if not _is_subpath(p, repo_path):
        raise HTTPException(
            status_code=403,
            detail=f"Output directory must be inside the repository: {output_dir}",
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
    sequence: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
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
    current_run = cast(PipelineRunStatus, _current_run)

    loop = asyncio.get_event_loop()
    step_sequence = 0

    def _require_ok(step: str, result: Any) -> dict[str, Any]:
        if result.ok:
            return result.payload
        raise RuntimeError(f"{step} failed: {result.error or 'Unknown error'}")

    def _add_step(
        step: str,
        status: str,
        duration: float = 0.0,
        data: dict | None = None,
        error: str | None = None,
        started_at: float | None = None,
        finished_at: float | None = None,
    ) -> PipelineStepResult:
        nonlocal step_sequence
        step_sequence += 1
        started = started_at if started_at is not None else time.time()
        finished = finished_at if finished_at is not None else started
        measured_duration = duration if duration > 0 else max(0.0, finished - started)
        result = PipelineStepResult(
            step=step,
            status=status,
            sequence=step_sequence,
            started_at=started,
            finished_at=finished,
            duration_seconds=round(measured_duration, 3),
            data=data or {},
            error=error,
        )
        current_run.steps.append(result)
        return result

    async def _step_broadcast(step: str, status: str, **kwargs: Any) -> None:
        msg = {"type": "pipeline_step", "run_id": run_id, "step": step, "status": status, **kwargs}
        await _broadcast(msg)

    try:
        # ---- 1. Analyze -----------------------------------------------
        await _step_broadcast("analyze", "running")
        t0 = time.time()

        def _do_analyze():
            return run_analyze(repo_path)

        analyze_result = await loop.run_in_executor(None, _do_analyze)
        finished_at = time.time()
        dt = finished_at - t0
        analyze_payload = _require_ok("analyze", analyze_result)
        language_stats = analyze_payload.get("language_stats", [])

        analysis_data = {
            "languages": len(analyze_payload.get("languages", [])),
            "file_count": sum(
                int(stat.get("file_count", 0)) for stat in language_stats if isinstance(stat, dict)
            ),
            "frameworks": len(analyze_payload.get("frameworks", [])),
            "entry_points": len(analyze_payload.get("entry_points", [])),
            "test_files": len(analyze_payload.get("test_files", [])),
            "patterns": len(analyze_payload.get("patterns", {})),
        }
        analyze_step = _add_step(
            "analyze",
            "completed",
            dt,
            analysis_data,
            started_at=t0,
            finished_at=finished_at,
        )
        await _step_broadcast(
            "analyze",
            "completed",
            data=analysis_data,
            duration=round(dt, 3),
            sequence=analyze_step.sequence,
            started_at=analyze_step.started_at,
            finished_at=analyze_step.finished_at,
        )

        # ---- 2. Suggest -----------------------------------------------
        await _step_broadcast("suggest", "running")
        t0 = time.time()

        def _do_suggest():
            return run_suggest(repo_path)

        suggest_result = await loop.run_in_executor(None, _do_suggest)
        finished_at = time.time()
        dt = finished_at - t0
        suggest_payload = _require_ok("suggest", suggest_result)
        suggestions = suggest_payload.get("suggestions", [])

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
        suggest_step = _add_step(
            "suggest",
            "completed",
            dt,
            suggest_data,
            started_at=t0,
            finished_at=finished_at,
        )
        await _step_broadcast(
            "suggest",
            "completed",
            data=suggest_data,
            duration=round(dt, 3),
            sequence=suggest_step.sequence,
            started_at=suggest_step.started_at,
            finished_at=suggest_step.finished_at,
        )

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
            current_run.spec_names = spec_names
            await _step_broadcast("specs_resolved", "completed", data={"specs": spec_names})

        # ---- Per-spec: Map -> Generate -> Validate --------------------
        for spec_name in spec_names:
            # Map
            await _step_broadcast(f"map:{spec_name}", "running")
            t0 = time.time()

            def _do_map(name=spec_name):
                return run_map(repo_path, name)

            map_result = await loop.run_in_executor(None, _do_map)
            finished_at = time.time()
            dt = finished_at - t0
            map_payload = _require_ok(f"map:{spec_name}", map_result)
            targets = map_payload.get("targets", [])

            map_data = {
                "spec": spec_name,
                "targets": map_payload.get("target_count", len(targets)),
                "strategy": map_payload.get("strategy", "none"),
                "confidence": map_payload.get("confidence", 0),
                "target_files": [
                    {"file": t.get("file", ""), "line": t.get("line", 0)}
                    for t in targets[:10]
                    if isinstance(t, dict)
                ],
            }
            map_step = _add_step(
                f"map:{spec_name}",
                "completed",
                dt,
                map_data,
                started_at=t0,
                finished_at=finished_at,
            )
            await _step_broadcast(
                f"map:{spec_name}",
                "completed",
                data=map_data,
                duration=round(dt, 3),
                sequence=map_step.sequence,
                started_at=map_step.started_at,
                finished_at=map_step.finished_at,
            )

            # Generate
            await _step_broadcast(f"generate:{spec_name}", "running")
            t0 = time.time()

            def _do_generate(name=spec_name):
                return run_generate(repo_path, name, output_dir=output_dir)

            generate_result = await loop.run_in_executor(None, _do_generate)
            finished_at = time.time()
            dt = finished_at - t0
            generate_payload = _require_ok(f"generate:{spec_name}", generate_result)
            new_files = generate_payload.get("new_files", [])
            transformations = generate_payload.get("transformations", [])
            preview_files = [
                f.get("path", "") for f in new_files[:10] if isinstance(f, dict) and f.get("path")
            ]
            if not preview_files:
                preview_files = [
                    t.get("file", "")
                    for t in transformations[:10]
                    if isinstance(t, dict) and t.get("file")
                ]

            gen_data = {
                "spec": spec_name,
                "branch": generate_payload.get("branch_name", ""),
                "new_files": len(new_files),
                "transformations": len(transformations),
                "file_names": [f.get("path", "") for f in new_files[:10] if isinstance(f, dict)],
            }
            if preview_files:
                gen_data["preview_files"] = preview_files[:6]
            if generate_payload.get("output_dir"):
                gen_data["output_dir"] = generate_payload["output_dir"]

            generate_step = _add_step(
                f"generate:{spec_name}",
                "completed",
                dt,
                gen_data,
                started_at=t0,
                finished_at=finished_at,
            )
            await _step_broadcast(
                f"generate:{spec_name}",
                "completed",
                data=gen_data,
                duration=round(dt, 3),
                sequence=generate_step.sequence,
                started_at=generate_step.started_at,
                finished_at=generate_step.finished_at,
            )

            # Validate
            if not skip_validate:
                await _step_broadcast(f"validate:{spec_name}", "running")
                t0 = time.time()

                def _do_validate(payload=generate_payload):
                    return run_validate(repo_path, payload)

                validate_result = await loop.run_in_executor(None, _do_validate)
                finished_at = time.time()
                dt = finished_at - t0
                validate_payload = _require_ok(f"validate:{spec_name}", validate_result)
                scorecard = validate_payload.get("scorecard", {})
                scorecard_data: dict[str, Any] = {}
                if isinstance(scorecard, dict):
                    summary = scorecard.get("summary", "")
                    if summary:
                        scorecard_data["summary"] = summary
                    checks = scorecard.get("checks", [])
                    if isinstance(checks, list):
                        scorecard_data["checks"] = checks[:8]
                    highlights = scorecard.get("highlights", [])
                    if isinstance(highlights, list):
                        scorecard_data["highlights"] = highlights[:6]
                    deltas = scorecard.get("deltas")
                    if isinstance(deltas, dict):
                        scorecard_data["deltas"] = deltas

                baseline_metrics = validate_payload.get("baseline_metrics")
                if baseline_metrics is None and isinstance(scorecard, dict):
                    baseline_metrics = scorecard.get("baseline_metrics")
                new_metrics = validate_payload.get("new_metrics")
                if new_metrics is None and isinstance(scorecard, dict):
                    new_metrics = scorecard.get("new_metrics")

                val_data = {
                    "spec": spec_name,
                    "passed": validate_payload.get("passed", False),
                    "stage": validate_payload.get("stage", "unknown"),
                    "summary": scorecard_data.get("summary", ""),
                }
                if scorecard_data:
                    val_data["scorecard"] = scorecard_data
                if "checks" in scorecard_data:
                    val_data["checks"] = scorecard_data["checks"]
                if "highlights" in scorecard_data:
                    val_data["highlights"] = scorecard_data["highlights"]
                if "deltas" in scorecard_data:
                    val_data["deltas"] = scorecard_data["deltas"]
                if isinstance(baseline_metrics, dict):
                    val_data["baseline_metrics"] = baseline_metrics
                if isinstance(new_metrics, dict):
                    val_data["new_metrics"] = new_metrics

                validate_step = _add_step(
                    f"validate:{spec_name}",
                    "completed",
                    dt,
                    val_data,
                    started_at=t0,
                    finished_at=finished_at,
                )
                await _step_broadcast(
                    f"validate:{spec_name}",
                    "completed",
                    data=val_data,
                    duration=round(dt, 3),
                    sequence=validate_step.sequence,
                    started_at=validate_step.started_at,
                    finished_at=validate_step.finished_at,
                )

        # ---- Done -----------------------------------------------------
        current_run.status = "completed"
        current_run.finished_at = time.time()
        current_run.total_seconds = round(
            current_run.finished_at - current_run.started_at,
            2,
        )
        await _broadcast(
            {
                "type": "pipeline_complete",
                "run_id": run_id,
                "status": "completed",
                "total_seconds": current_run.total_seconds,
            }
        )

    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        errored_at = time.time()
        _add_step("error", "failed", error=str(exc), started_at=errored_at, finished_at=errored_at)
        current_run.status = "failed"
        current_run.finished_at = time.time()
        current_run.total_seconds = round(current_run.finished_at - current_run.started_at, 2)
        await _broadcast({"type": "pipeline_error", "run_id": run_id, "error": str(exc)})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline progress.

    When auth is configured, clients must authenticate with
    a first message: {"type":"auth","token":"<api-key>"}.
    Query-string token auth is optional compatibility mode via
    SCHOLARDEVCLAW_WS_QUERY_TOKEN_COMPAT=true.
    Limits connections to 20 concurrent clients per server.
    """
    # Connection cap
    if len(_ws_clients) >= 20:
        await websocket.close(code=4002, reason="Too many connections")
        return

    await websocket.accept()

    authenticated = not bool(_API_AUTH_KEY)
    if _API_AUTH_KEY and _WS_QUERY_TOKEN_COMPAT:
        token = websocket.query_params.get("token", "")
        if token and hmac.compare_digest(token, _API_AUTH_KEY):
            authenticated = True

    if _API_AUTH_KEY and not authenticated:
        try:
            auth_payload = await websocket.receive_text()
            auth_msg = json.loads(auth_payload)
            token = str(auth_msg.get("token", ""))
            if (
                auth_msg.get("type") != "auth"
                or not token
                or not hmac.compare_digest(token, _API_AUTH_KEY)
            ):
                await websocket.close(code=4001, reason="Unauthorized")
                return
            await websocket.send_text(json.dumps({"type": "auth_ok"}))
            authenticated = True
        except Exception:
            await websocket.close(code=4001, reason="Unauthorized")
            return

    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))

    # Send current state on connect
    if _current_run:
        snapshot = _current_run.model_dump()
        await websocket.send_text(json.dumps({"type": "pipeline_snapshot", "run": snapshot}))
        # Compatibility for older clients that expect raw run payload
        await websocket.send_text(json.dumps(snapshot))

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
