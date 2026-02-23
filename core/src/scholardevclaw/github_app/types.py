from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WebhookEventType(str, Enum):
    PULL_REQUEST = "pull_request"
    PUSH = "push"
    CHECK_RUN = "check_run"
    CHECK_SUITE = "check_suite"
    ISSUE_COMMENT = "issue_comment"


class CheckStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckConclusion(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class WebhookPayload:
    event_type: str
    action: str | None = None
    repository: dict[str, Any] | None = None
    pull_request: dict[str, Any] | None = None
    sender: dict[str, Any] | None = None
    installation: dict[str, Any] | None = None
    organization: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Repository:
    owner: str
    name: str
    full_name: str
    html_url: str
    default_branch: str
    private: bool


@dataclass
class PullRequest:
    number: int
    title: str
    body: str | None
    state: str
    html_url: str
    head_branch: str
    head_sha: str
    base_branch: str
    user: dict[str, Any]


@dataclass
class CheckRun:
    id: int
    name: str
    status: CheckStatus
    conclusion: CheckConclusion | None = None
    output: dict[str, Any] = field(default_factory=dict)
    details_url: str | None = None


@dataclass
class ValidationResult:
    passed: bool
    stage: str
    scorecard: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IntegrationResult:
    ok: bool
    spec: str
    branch_name: str
    pr_url: str | None = None
    validation: ValidationResult | None = None
    changes_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class GitHubAppConfig:
    app_id: str = ""
    private_key: str = ""
    webhook_secret: str = ""
    installation_id: str = ""
    allowed_repositories: list[str] = field(default_factory=list)
    auto_apply_safe_patches: bool = False
    require_approval: bool = True
    notify_on_complete: bool = True

    @classmethod
    def from_env(cls) -> "GitHubAppConfig":
        import os

        return cls(
            app_id=os.environ.get("GITHUB_APP_ID", ""),
            private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY", ""),
            webhook_secret=os.environ.get("GITHUB_APP_WEBHOOK_SECRET", ""),
            installation_id=os.environ.get("GITHUB_APP_INSTALLATION_ID", ""),
            allowed_repositories=os.environ.get("GITHUB_APP_REPOS", "").split(",")
            if os.environ.get("GITHUB_APP_REPOS")
            else [],
            auto_apply_safe_patches=os.environ.get("GITHUB_APP_AUTO_APPLY", "false").lower()
            == "true",
            require_approval=os.environ.get("GITHUB_APP_REQUIRE_APPROVAL", "true").lower()
            == "true",
            notify_on_complete=os.environ.get("GITHUB_APP_NOTIFY", "true").lower() == "true",
        )

    def is_configured(self) -> bool:
        return bool(self.app_id and self.private_key and self.webhook_secret)
