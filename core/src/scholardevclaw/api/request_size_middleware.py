from __future__ import annotations

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""

    def __init__(
        self,
        app: ASGIApp,
        max_size: int = 10 * 1024 * 1024,  # 10 MB default
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.max_size = max_size
        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next):
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Check content-length header
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
                if length > self.max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body too large: {length} bytes > {self.max_size} bytes",
                    )
            except ValueError:
                # If header is not a valid integer, we'll let the server handle it
                pass

        # For chunked transfers or missing content-length, we cannot easily check size
        # without reading the body, which would interfere with the request.
        # We rely on the web server (e.g., uvicorn) to have its own limits.
        return await call_next(request)
