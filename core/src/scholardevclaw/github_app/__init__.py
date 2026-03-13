from __future__ import annotations

from .client import GitHubAppClient
from .server import create_github_app_app, create_github_app_router
from .types import (
    CheckConclusion,  # noqa: F401
    CheckRun,  # noqa: F401
    CheckStatus,  # noqa: F401
    GitHubAppConfig,
    IntegrationResult,  # noqa: F401
    PullRequest,  # noqa: F401
    Repository,  # noqa: F401
    ValidationResult,  # noqa: F401
    WebhookEventType,  # noqa: F401
    WebhookPayload,  # noqa: F401
)
from .webhook import WebhookHandler  # noqa: F401


def get_github_app_client(config: GitHubAppConfig | None = None) -> GitHubAppClient:
    return GitHubAppClient(config)


def create_app(
    config: GitHubAppConfig | None = None,
    integration_handler=None,
):
    return create_github_app_app(config, integration_handler)


def create_router(
    config: GitHubAppConfig | None = None,
    integration_handler=None,
):
    return create_github_app_router(config, integration_handler)
