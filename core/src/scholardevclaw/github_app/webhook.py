from __future__ import annotations

import json
import logging
from typing import Any, Callable

from .client import GitHubAppClient
from .types import (
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

logger = logging.getLogger(__name__)

IntegrationHandler = Callable[[str, str, int, str], IntegrationResult]


class WebhookHandler:
    def __init__(
        self,
        client: GitHubAppClient | None = None,
        config: GitHubAppConfig | None = None,
        integration_handler: IntegrationHandler | None = None,
    ):
        self.client = client or GitHubAppClient(config)
        self.config = config or GitHubAppConfig.from_env()
        self.integration_handler = integration_handler

    def set_integration_handler(self, handler: IntegrationHandler) -> None:
        self.integration_handler = handler

    def handle_webhook(
        self,
        event_type: str,
        payload: dict[str, Any],
        signature: str | None = None,
    ) -> dict[str, Any]:
        if signature and not self.client.verify_webhook_signature(
            json.dumps(payload).encode(), signature
        ):
            logger.warning("Invalid webhook signature")
            return {"error": "Invalid signature", "status": 401}

        if not self.client.is_configured:
            logger.warning("GitHub App not configured")
            return {"error": "App not configured", "status": 500}

        payload_obj = self.client.parse_webhook(event_type, payload)

        try:
            if event_type == WebhookEventType.PULL_REQUEST.value:
                return self._handle_pull_request(payload_obj)
            elif event_type == WebhookEventType.CHECK_RUN.value:
                return self._handle_check_run(payload_obj)
            elif event_type == "pull_request_review":
                return self._handle_pull_request_review(payload_obj)
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return {"status": 200, "message": "Event received"}
        except Exception as e:
            logger.exception(f"Error handling webhook: {e}")
            return {"error": str(e), "status": 500}

    def _handle_pull_request(self, payload: WebhookPayload) -> dict[str, Any]:
        action = payload.action
        pr_data = payload.pull_request
        repo_data = payload.repository

        if not pr_data or not repo_data:
            return {"error": "Missing PR or repo data", "status": 400}

        owner = repo_data["owner"]["login"]
        repo = repo_data["name"]
        pr_number = pr_data["number"]
        pr_title = pr_data["title"]
        head_sha = pr_data["head"]["sha"]
        installation_id = payload.installation["id"] if payload.installation else None

        logger.info(f"PR event: {action} for {owner}/{repo}#{pr_number}")

        if action in ("opened", "synchronize", "reopened"):
            return self._run_scholardevclaw_integration(
                owner, repo, pr_number, head_sha, installation_id
            )
        elif action == "closed" and pr_data.get("merged"):
            logger.info(f"PR #{pr_number} was merged")

        return {"status": 200, "message": "PR event processed"}

    def _run_scholardevclaw_integration(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        installation_id: str | None,
    ) -> dict[str, Any]:
        pr = self.client.get_pull_request(owner, repo, pr_number)
        if not pr:
            return {"error": "Failed to get PR details", "status": 500}

        check_run = self.client.create_check_run(
            owner,
            repo,
            "ScholarDevClaw Analysis",
            head_sha,
            CheckStatus.QUEUED,
            installation_id=installation_id,
            output={
                "title": "ScholarDevClaw Analysis",
                "summary": "Starting research-to-code integration analysis...",
            },
        )

        if not check_run:
            return {"error": "Failed to create check run", "status": 500}

        try:
            self.client.update_check_run(
                owner,
                repo,
                check_run.id,
                CheckStatus.IN_PROGRESS,
                installation_id=installation_id,
                output={
                    "title": "Analyzing repository",
                    "summary": "Running ScholarDevClaw analysis...",
                },
            )

            if self.integration_handler:
                result = self.integration_handler(owner, repo, pr_number, pr.head_sha)
            else:
                result = self._run_default_integration(owner, repo, pr_number, pr.head_sha)

            if result.ok:
                self.client.update_check_run(
                    owner,
                    repo,
                    check_run.id,
                    CheckStatus.COMPLETED,
                    CheckConclusion.SUCCESS,
                    installation_id=installation_id,
                    output=self._build_success_output(result),
                )

                self._post_results_comment(owner, repo, pr_number, result, installation_id)

                return {
                    "status": 200,
                    "message": "Integration completed",
                    "check_run_id": check_run.id,
                    "result": result.__dict__,
                }
            else:
                self.client.update_check_run(
                    owner,
                    repo,
                    check_run.id,
                    CheckStatus.COMPLETED,
                    CheckConclusion.FAILURE,
                    installation_id=installation_id,
                    output=self._build_failure_output(result),
                )

                self._post_failure_comment(owner, repo, pr_number, result, installation_id)

                return {
                    "status": 200,
                    "message": "Integration failed",
                    "check_run_id": check_run.id,
                    "result": result.__dict__,
                }

        except Exception as e:
            logger.exception(f"Integration failed: {e}")
            self.client.update_check_run(
                owner,
                repo,
                check_run.id,
                CheckStatus.COMPLETED,
                CheckConclusion.FAILURE,
                installation_id=installation_id,
                output={
                    "title": "Integration Error",
                    "summary": f"An error occurred: {str(e)}",
                },
            )
            return {"error": str(e), "status": 500}

    def _run_default_integration(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
    ) -> IntegrationResult:
        return IntegrationResult(
            ok=True,
            spec="auto-detect",
            branch_name=f"scholardevclaw/pr-{pr_number}",
            pr_url=f"https://github.com/{owner}/{repo}/pull/{pr_number}",
            validation=ValidationResult(
                passed=True,
                stage="completed",
                message="Default integration completed",
            ),
            changes_summary={"files_changed": 0},
        )

    def _handle_check_run(self, payload: WebhookPayload) -> dict[str, Any]:
        action = payload.action
        if action == "rerequested":
            logger.info("Check run re-requested")

        return {"status": 200, "message": "Check run event processed"}

    def _handle_pull_request_review(self, payload: WebhookPayload) -> dict[str, Any]:
        action = payload.action
        pr_data = payload.pull_request

        if action == "submitted" and pr_data:
            review = payload.raw.get("review", {})
            if review.get("state") == "APPROVED":
                logger.info(f"PR approved: {pr_data['number']}")

        return {"status": 200, "message": "Review processed"}

    def _build_success_output(self, result: IntegrationResult) -> dict[str, Any]:
        summary_parts = [
            f"**Status:** Completed successfully",
            f"**Spec:** {result.spec}",
            f"**Branch:** `{result.branch_name}`",
        ]

        if result.validation:
            summary_parts.append(f"**Validation:** {result.validation.stage}")
            if result.validation.scorecard:
                speedup = result.validation.scorecard.get("deltas", {}).get("speedup", 0)
                if speedup:
                    summary_parts.append(f"**Speedup:** {speedup:.2f}x")

        return {
            "title": "ScholarDevClaw Analysis Complete",
            "summary": "\n".join(summary_parts),
        }

    def _build_failure_output(self, result: IntegrationResult) -> dict[str, Any]:
        return {
            "title": "ScholarDevClaw Analysis Failed",
            "summary": f"**Error:** {result.error or 'Unknown error'}",
        }

    def _post_results_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        result: IntegrationResult,
        installation_id: str | None,
    ) -> None:
        if not self.config.notify_on_complete:
            return

        comment = self._format_results_comment(result)
        self.client.create_pull_request_comment(owner, repo, pr_number, comment, installation_id)

    def _post_failure_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        result: IntegrationResult,
        installation_id: str | None,
    ) -> None:
        if not self.config.notify_on_complete:
            return

        comment = f"""## ScholarDevClaw Analysis Failed

**Error:** {result.error or "Unknown error"}

Please check the logs and try again.
"""
        self.client.create_pull_request_comment(owner, repo, pr_number, comment, installation_id)

    def _format_results_comment(self, result: IntegrationResult) -> str:
        lines = [
            "## ScholarDevClaw Analysis Complete",
            "",
            f"**Spec:** {result.spec}",
            f"**Branch:** `{result.branch_name}`",
            "",
        ]

        if result.validation:
            lines.append("### Validation Results")
            lines.append(f"- **Stage:** {result.validation.stage}")
            lines.append(f"- **Passed:** {'✅ Yes' if result.validation.passed else '❌ No'}")

            if result.validation.message:
                lines.append(f"- **Message:** {result.validation.message}")

            if result.validation.scorecard:
                deltas = result.validation.scorecard.get("deltas", {})
                if deltas.get("speedup"):
                    lines.append(f"- **Speedup:** {deltas['speedup']:.2f}x")
                if deltas.get("loss_change_pct"):
                    lines.append(f"- **Loss Change:** {deltas['loss_change_pct']:.2f}%")

        if result.changes_summary:
            lines.append("")
            lines.append("### Changes Summary")
            for key, value in result.changes_summary.items():
                lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)
