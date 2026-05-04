from __future__ import annotations

import os
import logging
import httpx
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConvexIntegration:
    id: str
    status: str
    currentPhase: int
    repoUrl: str
    paperUrl: Optional[str] = None
    paperPdfPath: Optional[str] = None
    mode: str = "step_approval"

class ConvexClient:
    """
    Lightweight Python client for interacting with the ScholarDevClaw Convex backend.
    Used by the TUI to trigger agent runs and monitor progress.
    """
    def __init__(self):
        self.url = os.environ.get("CONVEX_URL")
        self.auth_key = os.environ.get("SCHOLARDEVCLAW_CONVEX_AUTH_KEY")

        if not self.url:
            logger.warning("CONVEX_URL not set; Convex integration will be unavailable")
        if not self.auth_key:
            logger.warning("SCHOLARDEVCLAW_CONVEX_AUTH_KEY not set; Convex mutations will fail")

    def _call_mutation(self, mutation_name: str, args: Dict[str, Any]) -> Any:
        """Helper to call Convex mutations via HTTP."""
        if not self.url or not self.auth_key:
            raise RuntimeError("Convex configuration missing (URL or Auth Key)")

        # Convex HTTP API endpoint for mutations
        endpoint = f"{self.url}/api/mutations/{mutation_name}"

        payload = {
            "authKey": self.auth_key,
            **args
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error("Convex mutation %s failed: %s", mutation_name, e)
            raise RuntimeError(f"Convex mutation {mutation_name} failed: {e}")

    def _call_query(self, query_name: str, args: Dict[str, Any] = None) -> Any:
        """Helper to call Convex queries via HTTP."""
        if not self.url:
            raise RuntimeError("Convex URL not set")

        endpoint = f"{self.url}/api/queries/{query_name}"
        params = args or {}

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(endpoint, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error("Convex query %s failed: %s", query_name, e)
            raise RuntimeError(f"Convex query {query_name} failed: {e}")

    def create_integration(self, repo_url: str, paper_url: Optional[str] = None, paper_pdf_path: Optional[str] = None, mode: str = "step_approval") -> str:
        """Creates a new integration record in Convex to trigger the agent."""
        args = {
            "repoUrl": repo_url,
            "paperUrl": paper_url,
            "paperPdfPath": paper_pdf_path,
            "mode": mode,
        }
        result = self._call_mutation("integrations:create", args)
        return result # Convex returns the ID of the created document

    def get_integration_status(self, integration_id: str) -> ConvexIntegration:
        """Polls the current status of an integration."""
        # We call the 'integrations:get' query
        data = self._call_query("integrations:get", {"id": integration_id})
        if not data:
            raise RuntimeError(f"Integration {integration_id} not found")

        return ConvexIntegration(
            id=data.get("_id"),
            status=data.get("status"),
            currentPhase=data.get("currentPhase", 0),
            repoUrl=data.get("repoUrl"),
            paperUrl=data.get("paperUrl"),
            paperPdfPath=data.get("paperPdfPath"),
            mode=data.get("mode", "step_approval")
        )

    def create_approval(self, integration_id: str, phase: int, action: str, notes: str = "") -> None:
        """Writes an approval/rejection to the Convex approvals table."""
        args = {
            "integrationId": integration_id,
            "phase": phase,
            "action": action, # 'approved' or 'rejected'
            "notes": notes
        }
        self._call_mutation("approvals:create", args)
