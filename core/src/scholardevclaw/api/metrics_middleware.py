import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..utils.metrics import (
    registry,
    ACTIVE_REQUESTS,
    track_request,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        ACTIVE_REQUESTS.inc()

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            track_request(
                method=request.method,
                path=self._normalize_path(request.url.path),
                status=response.status_code,
                duration=duration,
            )

            return response
        except Exception as e:
            duration = time.perf_counter() - start_time

            track_request(
                method=request.method,
                path=self._normalize_path(request.url.path),
                status=500,
                duration=duration,
            )
            raise
        finally:
            ACTIVE_REQUESTS.dec()

    def _normalize_path(self, path: str) -> str:
        parts = path.strip("/").split("/")
        normalized = []

        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            elif self._is_uuid(part):
                normalized.append("{uuid}")
            else:
                normalized.append(part)

        result = "/" + "/".join(normalized) if normalized else "/"
        return result

    def _is_uuid(self, s: str) -> bool:
        if len(s) == 36 and s.count("-") == 4:
            parts = s.split("-")
            return (
                len(parts[0]) == 8
                and len(parts[1]) == 4
                and len(parts[2]) == 4
                and len(parts[3]) == 4
                and len(parts[4]) == 12
            )
        return False


def setup_metrics(app) -> None:
    """Add metrics middleware and endpoint to FastAPI app."""
    app.add_middleware(MetricsMiddleware)

    @app.get(
        "/metrics",
        tags=["monitoring"],
        summary="Prometheus metrics",
        description="Export metrics in Prometheus text format",
        include_in_schema=False,
    )
    async def metrics():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(
            content=registry.export_prometheus(),
            media_type="text/plain; charset=utf-8",
        )
