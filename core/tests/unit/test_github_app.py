import json

import pytest

from scholardevclaw.github_app.types import (
    CheckConclusion,
    CheckStatus,
    GitHubAppConfig,
    IntegrationResult,
    PullRequest,
    Repository,
    ValidationResult,
    WebhookEventType,
    WebhookPayload,
)
from scholardevclaw.github_app.client import GitHubAppClient
from scholardevclaw.github_app.webhook import WebhookHandler


class TestGitHubAppConfig:
    def test_config_from_env_not_configured(self, monkeypatch):
        monkeypatch.delenv("GITHUB_APP_ID", raising=False)
        monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("GITHUB_APP_WEBHOOK_SECRET", raising=False)

        config = GitHubAppConfig.from_env()
        assert config.is_configured() is False

    def test_config_from_env_partial(self, monkeypatch):
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("GITHUB_APP_WEBHOOK_SECRET", raising=False)

        config = GitHubAppConfig.from_env()
        assert config.is_configured() is False

    def test_config_from_env_full(self, monkeypatch):
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "/path/to/key.pem")
        monkeypatch.setenv("GITHUB_APP_WEBHOOK_SECRET", "secret")

        config = GitHubAppConfig.from_env()
        assert config.is_configured() is True
        assert config.app_id == "12345"

    def test_config_auto_apply(self, monkeypatch):
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "/path/to/key.pem")
        monkeypatch.setenv("GITHUB_APP_WEBHOOK_SECRET", "secret")
        monkeypatch.setenv("GITHUB_APP_AUTO_APPLY", "true")

        config = GitHubAppConfig.from_env()
        assert config.auto_apply_safe_patches is True


class TestGitHubAppTypes:
    def test_webhook_payload_parsing(self):
        payload = {
            "action": "opened",
            "repository": {"name": "test-repo", "owner": {"login": "testuser"}},
            "pull_request": {"number": 1, "title": "Test PR"},
            "installation": {"id": "12345"},
        }

        result = WebhookPayload(
            event_type="pull_request",
            action="opened",
            repository=payload["repository"],
            pull_request=payload["pull_request"],
            installation=payload["installation"],
            raw=payload,
        )

        assert result.event_type == "pull_request"
        assert result.action == "opened"
        assert result.repository["name"] == "test-repo"

    def test_repository_dataclass(self):
        repo = Repository(
            owner="testuser",
            name="test-repo",
            full_name="testuser/test-repo",
            html_url="https://github.com/testuser/test-repo",
            default_branch="main",
            private=False,
        )

        assert repo.owner == "testuser"
        assert repo.full_name == "testuser/test-repo"

    def test_pull_request_dataclass(self):
        pr = PullRequest(
            number=1,
            title="Test PR",
            body="Test body",
            state="open",
            html_url="https://github.com/testuser/test-repo/pull/1",
            head_branch="feature",
            head_sha="abc123",
            base_branch="main",
            user={"login": "testuser"},
        )

        assert pr.number == 1
        assert pr.head_branch == "feature"

    def test_integration_result(self):
        result = IntegrationResult(
            ok=True,
            spec="rmsnorm",
            branch_name="integration/rmsnorm",
            pr_url="https://github.com/test/test/pull/1",
            validation=ValidationResult(
                passed=True,
                stage="completed",
                message="All tests passed",
            ),
            changes_summary={"files_changed": 5},
        )

        assert result.ok is True
        assert result.spec == "rmsnorm"
        assert result.validation.passed is True


class TestWebhookHandler:
    def test_handler_without_config(self):
        config = GitHubAppConfig()
        handler = WebhookHandler(config=config)
        assert handler.client.is_configured is False

    def test_webhook_event_types(self):
        assert WebhookEventType.PULL_REQUEST.value == "pull_request"
        assert WebhookEventType.PUSH.value == "push"
        assert WebhookEventType.CHECK_RUN.value == "check_run"

    def test_check_status_values(self):
        assert CheckStatus.QUEUED.value == "queued"
        assert CheckStatus.IN_PROGRESS.value == "in_progress"
        assert CheckStatus.COMPLETED.value == "completed"

    def test_check_conclusion_values(self):
        assert CheckConclusion.SUCCESS.value == "success"
        assert CheckConclusion.FAILURE.value == "failure"


class TestGitHubAppClient:
    def test_client_not_configured(self):
        client = GitHubAppClient(GitHubAppConfig())
        assert client.is_configured is False

    def test_verify_webhook_signature_empty_secret(self):
        client = GitHubAppClient(GitHubAppConfig())
        result = client.verify_webhook_signature(b"test", "sha256=abc")
        assert result is False


class TestWebhookHandlerIntegration:
    def test_handle_webhook_not_configured(self):
        config = GitHubAppConfig()
        handler = WebhookHandler(config=config)

        result = handler.handle_webhook(
            "pull_request",
            {"action": "opened", "repository": {}, "pull_request": {}},
        )

        assert "error" in result

    def test_parse_webhook(self):
        config = GitHubAppConfig()
        client = GitHubAppClient(config)

        payload = {
            "action": "opened",
            "repository": {
                "name": "test-repo",
                "owner": {"login": "testuser"},
            },
            "pull_request": {
                "number": 1,
                "title": "Test PR",
                "head": {"ref": "feature", "sha": "abc123"},
                "base": {"ref": "main"},
            },
            "installation": {"id": "12345"},
        }

        result = client.parse_webhook("pull_request", payload)
        assert result.event_type == "pull_request"
        assert result.action == "opened"
        assert result.repository["name"] == "test-repo"
