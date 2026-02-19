from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from typing import Any


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ScholarDevClaw API",
        version="2.0.0",
        description="""
# ScholarDevClaw - Autonomous Research-to-Code Agent API

ScholarDevClaw automatically integrates cutting-edge ML research into your codebase.

## Features

- **Repository Analysis**: Analyze code structure, patterns, and improvement opportunities
- **Research Extraction**: Extract implementation specs from papers (arXiv, PDF)
- **Mapping**: Map research specs to specific code locations
- **Patch Generation**: Generate code patches implementing research improvements
- **Validation**: Run tests and benchmarks to validate changes
- **Planner**: Plan multi-spec migration strategies
- **Critic**: Verify generated patches for safety
- **Context Engine**: Learn from past integrations
- **Experiment Loop**: Test hypotheses with variant comparison

## Authentication

Most endpoints don't require authentication. Rate limiting applies to all endpoints.

## Rate Limits

- 60 requests per minute
- 1000 requests per hour
- Burst: 10 requests

## Error Handling

All errors follow a standard format:
```json
{
  "error": "error_code",
  "message": "Human-readable message",
  "details": {}
}
```
        """,
        routes=app.routes,
        tags=[
            {
                "name": "health",
                "description": "Health check and monitoring endpoints",
            },
            {
                "name": "repo",
                "description": "Repository analysis operations",
            },
            {
                "name": "research",
                "description": "Research extraction and spec operations",
            },
            {
                "name": "mapping",
                "description": "Map specs to code locations",
            },
            {
                "name": "patch",
                "description": "Generate and manage patches",
            },
            {
                "name": "validation",
                "description": "Run validation and benchmarks",
            },
            {
                "name": "planner",
                "description": "Multi-spec migration planning",
            },
            {
                "name": "critic",
                "description": "Patch verification and safety checks",
            },
            {
                "name": "context",
                "description": "Project context and memory management",
            },
            {
                "name": "experiment",
                "description": "Hypothesis testing with variants",
            },
        ],
    )

    openapi_schema["info"]["x-logo"] = {"url": "https://scholardevclaw.dev/logo.png"}

    openapi_schema["info"]["contact"] = {
        "name": "ScholarDevClaw Support",
        "email": "support@scholardevclaw.dev",
        "url": "https://github.com/Ronak-IIITD/ScholarDevClaw",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_openapi(app: FastAPI) -> None:
    """Setup custom OpenAPI documentation."""
    app.openapi = lambda: custom_openapi(app)


def setup_docs_routes(app: FastAPI) -> None:
    """Setup additional documentation routes."""

    @app.get("/docs/json", tags=["docs"])
    async def get_openapi_json():
        """Get OpenAPI schema as JSON."""
        return app.openapi()

    @app.get("/docs/version", tags=["docs"])
    async def get_api_version():
        """Get API version information."""
        return {
            "version": "2.0.0",
            "schema_version": "1.0.0",
            "python_version": "3.10+",
        }


class APIError(BaseModel):
    """Standard API error response."""

    error: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: APIError
    request_id: str | None = None


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "details": {"type": type(exc).__name__},
            },
        )


from pydantic import BaseModel
