from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request, Response
from pydantic import BaseModel

from .client import GitHubAppClient
from .types import GitHubAppConfig
from .webhook import WebhookHandler, IntegrationHandler


class WebhookRequest(BaseModel):
    event: str
    payload: dict


class WebhookResponse(BaseModel):
    status: int
    message: str
    error: str | None = None


def create_github_app_app(
    config: GitHubAppConfig | None = None,
    integration_handler: IntegrationHandler | None = None,
) -> FastAPI:
    app = FastAPI(
        title="ScholarDevClaw GitHub App",
        description="GitHub App webhook server for ScholarDevClaw",
        version="1.0.0",
    )

    client = GitHubAppClient(config)
    handler = WebhookHandler(client, config, integration_handler)

    @app.get("/health")
    async def health():
        return {"status": "ok", "configured": client.is_configured}

    @app.post("/webhook")
    async def handle_webhook(request: Request) -> WebhookResponse:
        event_type = request.headers.get("X-GitHub-Event", "")
        signature = request.headers.get("X-Hub-Signature-256")

        body = await request.body()
        payload = None
        try:
            import json

            payload = json.loads(body)
        except Exception:
            return WebhookResponse(
                status=400,
                message="Invalid JSON",
                error="Could not parse webhook payload",
            )

        # SECURITY: Require signature header — reject if missing
        if not signature:
            return WebhookResponse(
                status=401,
                message="Missing signature",
                error="X-Hub-Signature-256 header is required",
            )

        if not client.verify_webhook_signature(body, signature):
            return WebhookResponse(
                status=401,
                message="Invalid signature",
                error="Webhook signature verification failed",
            )

        result = handler.handle_webhook(event_type, payload, signature, raw_body=body)

        return WebhookResponse(
            status=result.get("status", 200),
            message=result.get("message", ""),
            error=result.get("error"),
        )

    @app.get("/")
    async def root():
        return {
            "app": "ScholarDevClaw GitHub App",
            "version": "1.0.0",
            "configured": client.is_configured,
        }

    return app


def create_github_app_router(
    config: GitHubAppConfig | None = None,
    integration_handler: IntegrationHandler | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/github", tags=["github"])

    client = GitHubAppClient(config)
    handler = WebhookHandler(client, config, integration_handler)

    @router.get("/health")
    async def health():
        return {"status": "ok", "configured": client.is_configured}

    @router.post("/webhook")
    async def handle_webhook(
        request: Request,
    ) -> WebhookResponse:
        # SECURITY: Read event and signature from headers, not query params
        event = request.headers.get("X-GitHub-Event", "pull_request")
        signature = request.headers.get("X-Hub-Signature-256")

        body = await request.body()

        # SECURITY: Require signature header
        if not signature:
            return WebhookResponse(
                status=401,
                message="Missing signature",
                error="X-Hub-Signature-256 header is required",
            )

        if not client.verify_webhook_signature(body, signature):
            return WebhookResponse(
                status=401,
                message="Invalid signature",
                error="Webhook signature verification failed",
            )

        import json

        payload = json.loads(body)

        result = handler.handle_webhook(event, payload, signature, raw_body=body)

        return WebhookResponse(
            status=result.get("status", 200),
            message=result.get("message", ""),
            error=result.get("error"),
        )

    @router.get("/repos/{owner}/{repo}")
    async def get_repo(owner: str, repo: str):
        repo_obj = client.get_repository(owner, repo)
        if not repo_obj:
            return {"error": "Repository not found"}, 404
        return {
            "owner": repo_obj.owner,
            "name": repo_obj.name,
            "full_name": repo_obj.full_name,
            "html_url": repo_obj.html_url,
            "default_branch": repo_obj.default_branch,
            "private": repo_obj.private,
        }

    return router
