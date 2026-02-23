from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
import requests

from .types import (
    CheckConclusion,
    CheckRun,
    CheckStatus,
    GitHubAppConfig,
    PullRequest,
    Repository,
    WebhookEventType,
    WebhookPayload,
)


class GitHubAppClient:
    def __init__(self, config: GitHubAppConfig | None = None):
        self.config = config or GitHubAppConfig.from_env()
        self._jwt_token: str | None = None
        self._jwt_expires_at: float = 0
        self._installation_token: str = ""
        self._installation_expires_at: float = 0

    @property
    def is_configured(self) -> bool:
        return self.config.is_configured()

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        if not self.config.webhook_secret:
            return False

        expected_signature = (
            "sha256="
            + hmac.new(self.config.webhook_secret.encode(), payload, hashlib.sha256).hexdigest()
        )

        return hmac.compare_digest(expected_signature, signature)

    def parse_webhook(self, event_type: str, payload: dict[str, Any]) -> WebhookPayload:
        return WebhookPayload(
            event_type=event_type,
            action=payload.get("action"),
            repository=payload.get("repository"),
            pull_request=payload.get("pull_request"),
            sender=payload.get("sender"),
            installation=payload.get("installation"),
            organization=payload.get("organization"),
            raw=payload,
        )

    def _get_jwt_token(self) -> str:
        if self._jwt_token and time.time() < self._jwt_expires_at - 60:
            return self._jwt_token

        if not self.config.app_id or not self.config.private_key:
            raise ValueError("GitHub App not configured")

        import jwt

        now = int(time.time())
        payload = {
            "iat": now,
            "exp": now + 600,
            "iss": self.config.app_id,
        }

        with open(self.config.private_key) as f:
            private_key = f.read()

        encoded = jwt.encode(payload, private_key, algorithm="RS256")
        self._jwt_token = encoded if encoded else ""
        self._jwt_expires_at = now + 600
        return self._jwt_token

    def _get_installation_token(self, installation_id: str) -> str:
        if self._installation_token and time.time() < self._installation_expires_at - 60:
            return self._installation_token

        jwt_token = self._get_jwt_token()

        response = requests.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "permissions": {
                    "contents": "write",
                    "issues": "write",
                    "pull_requests": "write",
                    "statuses": "write",
                    "checks": "write",
                }
            },
        )

        if response.status_code != 201:
            raise RuntimeError(f"Failed to get installation token: {response.text}")

        data = response.json()
        self._installation_token = data["token"]
        self._installation_expires_at = time.time() + 3600
        return self._installation_token

    def _api_request(
        self,
        method: str,
        path: str,
        installation_id: str | None = None,
        **kwargs,
    ) -> requests.Response:
        headers = kwargs.pop("headers", {})
        headers["Accept"] = "application/vnd.github+json"

        if installation_id:
            token = self._get_installation_token(installation_id)
            headers["Authorization"] = f"Bearer {token}"
        elif self._installation_token:
            headers["Authorization"] = f"Bearer {self._installation_token}"

        response = requests.request(
            method,
            f"https://api.github.com{path}",
            headers=headers,
            **kwargs,
        )
        return response

    def get_repository(self, owner: str, repo: str) -> Repository | None:
        response = self._api_request("GET", f"/repos/{owner}/{repo}")
        if response.status_code != 200:
            return None

        data = response.json()
        return Repository(
            owner=data["owner"]["login"],
            name=data["name"],
            full_name=data["full_name"],
            html_url=data["html_url"],
            default_branch=data["default_branch"],
            private=data["private"],
        )

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequest | None:
        response = self._api_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        if response.status_code != 200:
            return None

        data = response.json()
        return PullRequest(
            number=data["number"],
            title=data["title"],
            body=data["body"],
            state=data["state"],
            html_url=data["html_url"],
            head_branch=data["head"]["ref"],
            head_sha=data["head"]["sha"],
            base_branch=data["base"]["ref"],
            user=data["user"],
        )

    def create_check_run(
        self,
        owner: str,
        repo: str,
        name: str,
        head_sha: str,
        status: CheckStatus,
        installation_id: str | None = None,
        conclusion: CheckConclusion | None = None,
        output: dict[str, Any] | None = None,
        details_url: str | None = None,
    ) -> CheckRun | None:
        payload: dict[str, Any] = {
            "name": name,
            "head_sha": head_sha,
            "status": status.value,
        }

        if conclusion:
            payload["conclusion"] = conclusion.value

        if output:
            payload["output"] = output

        if details_url:
            payload["details_url"] = details_url

        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/check-runs",
            installation_id=installation_id,
            json=payload,
        )

        if response.status_code != 201:
            return None

        data = response.json()
        return CheckRun(
            id=data["id"],
            name=data["name"],
            status=CheckStatus(data["status"]),
            conclusion=CheckConclusion(data["conclusion"]) if data.get("conclusion") else None,
            output=data.get("output", {}),
            details_url=data.get("details_url"),
        )

    def update_check_run(
        self,
        owner: str,
        repo: str,
        check_run_id: int,
        status: CheckStatus,
        conclusion: CheckConclusion | None = None,
        output: dict[str, Any] | None = None,
        installation_id: str | None = None,
    ) -> bool:
        payload: dict[str, Any] = {
            "status": status.value,
        }

        if conclusion:
            payload["conclusion"] = conclusion.value

        if output:
            payload["output"] = output

        response = self._api_request(
            "PATCH",
            f"/repos/{owner}/{repo}/check-runs/{check_run_id}",
            installation_id=installation_id,
            json=payload,
        )

        return response.status_code == 200

    def create_pull_request_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        installation_id: str | None = None,
    ) -> bool:
        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            installation_id=installation_id,
            json={"body": body},
        )
        return response.status_code == 201

    def create_pull_request_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        commit_id: str,
        path: str,
        line: int,
        installation_id: str | None = None,
    ) -> bool:
        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            installation_id=installation_id,
            json={
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
            },
        )
        return response.status_code == 201

    def create_or_update_commit_status(
        self,
        owner: str,
        repo: str,
        sha: str,
        state: str,
        description: str,
        target_url: str | None = None,
        installation_id: str | None = None,
    ) -> bool:
        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/statuses/{sha}",
            installation_id=installation_id,
            json={
                "state": state,
                "description": description,
                "target_url": target_url,
            },
        )
        return response.status_code == 201

    def get_pull_request_files(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        response = self._api_request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
            installation_id=installation_id,
        )

        if response.status_code != 200:
            return []

        return response.json()

    def get_branch(
        self,
        owner: str,
        repo: str,
        branch: str,
        installation_id: str | None = None,
    ) -> dict[str, Any] | None:
        response = self._api_request(
            "GET",
            f"/repos/{owner}/{repo}/branches/{branch}",
            installation_id=installation_id,
        )

        if response.status_code != 200:
            return None

        return response.json()

    def create_branch(
        self,
        owner: str,
        repo: str,
        branch: str,
        sha: str,
        installation_id: str | None = None,
    ) -> bool:
        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/git/refs",
            installation_id=installation_id,
            json={
                "ref": f"refs/heads/{branch}",
                "sha": sha,
            },
        )
        return response.status_code == 201

    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
        installation_id: str | None = None,
    ) -> str | None:
        url_path = f"/repos/{owner}/{repo}/contents/{path}"
        if ref:
            url_path += f"?ref={ref}"

        response = self._api_request(
            "GET",
            url_path,
            installation_id=installation_id,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if isinstance(data, dict) and data.get("content"):
            content = data["content"].replace("\n", "")
            return base64.b64decode(content).decode("utf-8")

        return None

    def update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        sha: str | None = None,
        branch: str | None = None,
        installation_id: str | None = None,
    ) -> bool:
        import base64

        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode("utf-8"),
        }

        if sha:
            payload["sha"] = sha

        if branch:
            payload["branch"] = branch

        response = self._api_request(
            "PUT",
            f"/repos/{owner}/{repo}/contents/{path}",
            installation_id=installation_id,
            json=payload,
        )

        return response.status_code in (200, 201)

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
        installation_id: str | None = None,
    ) -> dict[str, Any] | None:
        response = self._api_request(
            "POST",
            f"/repos/{owner}/{repo}/pulls",
            installation_id=installation_id,
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
            },
        )

        if response.status_code != 201:
            return None

        return response.json()
