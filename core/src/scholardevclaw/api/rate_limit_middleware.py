from __future__ import annotations

import time
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from scholardevclaw.utils.rate_limit import RateLimiter, RateLimitConfig, RateLimitResult


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: RateLimiter | None = None,
        config: RateLimitConfig | None = None,
        key_prefix: str = "ratelimit",
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter(config)
        self.key_prefix = key_prefix
        self.exclude_paths = exclude_paths or ["/health", "/health/live", "/health/ready"]

    def _get_client_key(self, request: Request) -> str:
        client_ip = self._get_client_ip(request)
        return f"{self.key_prefix}:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _is_excluded(self, path: str) -> bool:
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if self._is_excluded(path):
            return await call_next(request)

        key = self._get_client_key(request)
        result = self.rate_limiter.check(key)

        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

        if not result.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": int(result.retry_after or 60),
                    "limit": result.limit,
                },
                headers={
                    "Retry-After": str(int(result.retry_after or 60)),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                },
            )

        return response


class EndpointRateLimiter:
    """Per-endpoint rate limiting configuration."""

    def __init__(self, default_config: RateLimitConfig | None = None):
        self.default_config = default_config or RateLimitConfig()
        self._endpoint_configs: dict[str, RateLimitConfig] = {}
        self._limiters: dict[str, RateLimiter] = {}

    def configure_endpoint(
        self,
        path_pattern: str,
        config: RateLimitConfig,
    ) -> None:
        self._endpoint_configs[path_pattern] = config
        self._limiters[path_pattern] = RateLimiter(config)

    def check(self, request: Request) -> RateLimitResult:
        path = request.url.path

        for pattern, limiter in self._limiters.items():
            if self._matches_pattern(path, pattern):
                client_ip = self._get_client_ip(request)
                key = f"{pattern}:{client_ip}"
                return limiter.check(key)

        default_limiter = RateLimiter(self.default_config)
        client_ip = self._get_client_ip(request)
        key = f"default:{client_ip}"
        return default_limiter.check(key)

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path == pattern

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        if request.client:
            return request.client.host

        return "unknown"


def setup_rate_limiting(
    app,
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_size: int = 10,
    exclude_paths: list[str] | None = None,
) -> RateLimitMiddleware:
    """Setup rate limiting middleware on a FastAPI app."""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        burst_size=burst_size,
    )

    middleware = RateLimitMiddleware(
        app=app,
        config=config,
        exclude_paths=exclude_paths,
    )

    app.add_middleware(RateLimitMiddleware, config=config, exclude_paths=exclude_paths)

    return middleware
