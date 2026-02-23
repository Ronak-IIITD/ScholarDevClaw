from __future__ import annotations

from .client import GitHubAppClient
from .server import create_github_app_app, create_github_app_router
from .types import (
    CheckConclusion,
    CheckRun,
    CheckStatus,
    GitHubAppConfig,
    IntegrationResult,
    PullRequest,
    Repository,
    ValidationResult,
    WebhookEventType,
    WebhookPayload,
)
from .webhook import WebhookHandler


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
