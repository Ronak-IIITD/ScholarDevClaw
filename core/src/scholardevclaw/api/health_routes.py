from __future__ import annotations

from fastapi import APIRouter, Response
from pydantic import BaseModel

from scholardevclaw.utils.health import health_checker, liveness_probe, readiness_probe
from scholardevclaw.utils.circuit_breaker import circuit_registry
from scholardevclaw.utils.shutdown import shutdown_manager


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: dict[str, dict]


class LivenessResponse(BaseModel):
    alive: bool
    last_heartbeat: str
    seconds_since_heartbeat: float


class ReadinessResponse(BaseModel):
    ready: bool
    reasons: list[str]


class CircuitHealthResponse(BaseModel):
    healthy: bool
    total_circuits: int
    open_circuits: list[str]


@router.get("/health", response_model=HealthResponse)
async def health():
    """Full health check."""
    system_health = health_checker.run_all_checks()

    return HealthResponse(
        status="healthy" if system_health.overall_healthy else "unhealthy",
        version=system_health.version,
        uptime_seconds=system_health.uptime_seconds,
        checks={
            check.name: {
                "healthy": check.healthy,
                "message": check.message,
            }
            for check in system_health.checks
        },
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Liveness probe."""
    probe_status = liveness_probe.check()

    if not probe_status["alive"]:
        return Response(
            content=LivenessResponse(**probe_status).model_dump_json(),
            status_code=503,
            media_type="application/json",
        )

    return LivenessResponse(**probe_status)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness():
    """Readiness probe."""
    ready = readiness_probe.is_ready()
    reasons = []

    if not ready:
        reasons.extend(readiness_probe._reasons)

    if shutdown_manager.is_shutting_down():
        ready = False
        reasons.append("System is shutting down")

    if not health_checker.run_quick_check():
        ready = False
        reasons.append("Health checks failing")

    response = ReadinessResponse(ready=ready, reasons=reasons)

    if not ready:
        return Response(
            content=response.model_dump_json(),
            status_code=503,
            media_type="application/json",
        )

    return response


@router.get("/health/circuits", response_model=CircuitHealthResponse)
async def circuit_health():
    """Circuit breaker health."""
    health = circuit_registry.get_health()
    return CircuitHealthResponse(**health)


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health with all metrics."""
    system_health = health_checker.run_all_checks()
    circuit_health = circuit_registry.get_health()
    liveness_status = liveness_probe.check()
    readiness_status = readiness_probe.check()

    return {
        "system": {
            "overall_healthy": system_health.overall_healthy,
            "uptime_seconds": system_health.uptime_seconds,
            "version": system_health.version,
            "python_version": system_health.python_version,
            "platform": system_health.platform,
        },
        "checks": [
            {
                "name": check.name,
                "healthy": check.healthy,
                "message": check.message,
                "details": check.details,
            }
            for check in system_health.checks
        ],
        "circuits": circuit_health,
        "liveness": liveness_status,
        "readiness": readiness_status,
        "shutdown": {
            "shutting_down": shutdown_manager.is_shutting_down(),
            "reason": shutdown_manager.state.reason
            if shutdown_manager.is_shutting_down()
            else None,
        },
    }
