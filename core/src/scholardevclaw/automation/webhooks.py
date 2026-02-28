"""
Webhook triggers for automated research-to-code on events.

Provides:
- Git push webhook handler
- PR creation/update webhook handler
- Custom webhook endpoints
- Event filtering and routing
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass
class WebhookTrigger:
    """A webhook trigger definition"""

    id: str
    name: str
    event_type: str  # "push", "pull_request", "custom"
    repo_pattern: str = ""  # glob pattern
    branch_pattern: str = ""  # glob pattern
    action_filter: str = ""  # for PR: "opened", "updated", "closed"
    enabled: bool = True
    secret: str = ""
    callback_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebhookEvent:
    """A received webhook event"""

    id: str
    trigger_id: str
    event_type: str
    payload: dict[str, Any]
    received_at: str
    processed: bool = False
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class WebhookRouter:
    """Route webhook events to triggers"""

    def __init__(self):
        self.triggers: dict[str, WebhookTrigger] = {}
        self.events: list[WebhookEvent] = []

    def add_trigger(
        self,
        name: str,
        event_type: str,
        repo_pattern: str = "",
        branch_pattern: str = "",
        action_filter: str = "",
        secret: str = "",
    ) -> WebhookTrigger:
        """Add a webhook trigger"""
        trigger = WebhookTrigger(
            id=str(uuid.uuid4()),
            name=name,
            event_type=event_type,
            repo_pattern=repo_pattern,
            branch_pattern=branch_pattern,
            action_filter=action_filter,
            secret=secret,
        )

        self.triggers[trigger.id] = trigger
        return trigger

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            return True
        return False

    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
            return True
        return False

    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
            return True
        return False

    def route_event(self, event_type: str, payload: dict[str, Any]) -> list[WebhookEvent]:
        """Route an event to matching triggers"""
        matched = []

        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue

            if trigger.event_type != event_type and trigger.event_type != "custom":
                continue

            if self._matches(trigger, payload):
                event = WebhookEvent(
                    id=str(uuid.uuid4()),
                    trigger_id=trigger.id,
                    event_type=event_type,
                    payload=payload,
                    received_at=datetime.now().isoformat(),
                )
                self.events.append(event)
                matched.append(event)

        return matched

    def _matches(self, trigger: WebhookTrigger, payload: dict[str, Any]) -> bool:
        """Check if payload matches trigger filters"""
        if trigger.repo_pattern:
            repo = payload.get("repository", {}).get("full_name", "")
            if not self._glob_match(repo, trigger.repo_pattern):
                return False

        if trigger.branch_pattern:
            ref = payload.get("ref", "")
            branch = ref.replace("refs/heads/", "")
            if not self._glob_match(branch, trigger.branch_pattern):
                return False

        if trigger.action_filter:
            action = payload.get("action", "")
            if action != trigger.action_filter:
                return False

        return True

    def _glob_match(self, text: str, pattern: str) -> bool:
        """Simple glob matching"""
        import fnmatch

        return fnmatch.fnmatch(text, pattern)

    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        if not secret or not signature:
            return False

        computed = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(f"sha256={computed}", signature)


class GitPushHandler:
    """Handle git push webhook events"""

    def __init__(self, router: WebhookRouter):
        self.router = router

    def handle(self, payload: dict[str, Any]) -> list[WebhookEvent]:
        """Process a push event"""
        return self.router.route_event("push", payload)

    def extract_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract useful info from push event"""
        repo = payload.get("repository", {})
        commits = payload.get("commits", [])
        ref = payload.get("ref", "")

        return {
            "repo": repo.get("full_name", ""),
            "repo_url": repo.get("html_url", ""),
            "branch": ref.replace("refs/heads/", ""),
            "pusher": payload.get("pusher", {}).get("name", ""),
            "commits_count": len(commits),
            "commit_messages": [c.get("message", "") for c in commits[:5]],
            "added_files": [f for c in commits for f in c.get("added", [])],
            "modified_files": [f for c in commits for f in c.get("modified", [])],
            "removed_files": [f for c in commits for f in c.get("removed", [])],
        }


class PullRequestHandler:
    """Handle pull request webhook events"""

    def __init__(self, router: WebhookRouter):
        self.router = router

    def handle(self, payload: dict[str, Any]) -> list[WebhookEvent]:
        """Process a PR event"""
        return self.router.route_event("pull_request", payload)

    def extract_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract useful info from PR event"""
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        return {
            "repo": repo.get("full_name", ""),
            "pr_number": pr.get("number", 0),
            "pr_title": pr.get("title", ""),
            "pr_body": pr.get("body", ""),
            "pr_url": pr.get("html_url", ""),
            "action": payload.get("action", ""),
            "branch": pr.get("head", {}).get("ref", ""),
            "base_branch": pr.get("base", {}).get("ref", ""),
            "author": pr.get("user", {}).get("login", ""),
            "is_draft": pr.get("draft", False),
            "changed_files": pr.get("changed_files", 0),
        }


class WebhookExecutor:
    """Execute actions based on webhook events"""

    def __init__(self, router: WebhookRouter):
        self.router = router
        self.handlers: dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for an event type"""
        self.handlers[event_type] = handler

    def execute(self, event: WebhookEvent, context: dict | None = None) -> dict[str, Any]:
        """Execute handler for an event"""
        try:
            if event.event_type in self.handlers:
                result = self.handlers[event.event_type](event.payload, context or {})
                event.result = result
                event.processed = True
                return result
            else:
                event.error = f"No handler for {event.event_type}"
                return {"error": event.error}

        except Exception as e:
            event.error = str(e)
            return {"error": event.error}


class WebhookServer:
    """Simple webhook server (FastAPI integration)"""

    @staticmethod
    def create_app(
        router: WebhookRouter,
        executor: WebhookExecutor | None = None,
        path: str = "/webhook",
    ):
        """Create FastAPI app for webhooks"""
        try:
            from fastapi import FastAPI, Request, HTTPException
            from fastapi.responses import JSONResponse
        except ImportError:
            return None

        app = FastAPI()

        @app.post(path)
        async def handle_webhook(request: Request):
            body = await request.body()
            payload = await request.json()

            signature = request.headers.get("X-Hub-Signature-256", "")
            event_type = request.headers.get("X-GitHub-Event", "custom")

            for trigger in router.triggers.values():
                if trigger.secret:
                    if not router.verify_signature(body, signature, trigger.secret):
                        raise HTTPException(status_code=401, detail="Invalid signature")

            events = router.route_event(event_type, payload)

            results = []
            for event in events:
                if executor:
                    result = executor.execute(event)
                    results.append(result)
                else:
                    results.append({"status": "queued"})

            return JSONResponse({"events": len(events), "results": results})

        return app


def create_push_trigger(router: WebhookRouter, name: str, branch: str = "*") -> WebhookTrigger:
    """Helper to create a push trigger"""
    return router.add_trigger(
        name=name,
        event_type="push",
        branch_pattern=branch,
    )


def create_pr_trigger(
    router: WebhookRouter,
    name: str,
    branch: str = "*",
    action: str = "",
) -> WebhookTrigger:
    """Helper to create a PR trigger"""
    return router.add_trigger(
        name=name,
        event_type="pull_request",
        branch_pattern=branch,
        action_filter=action,
    )
