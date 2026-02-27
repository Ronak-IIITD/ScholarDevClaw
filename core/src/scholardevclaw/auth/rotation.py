"""Automated secret rotation for API keys.

Supports automatic rotation via provider APIs and scheduled rotation policies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


@dataclass
class RotationPolicy:
    """Policy for automatic key rotation."""

    id: str
    key_id: str
    provider: str
    rotation_days: int
    rotate_before_expiry_days: int = 7
    auto_rotate: bool = True
    notify_before_rotation: int = 3
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key_id": self.key_id,
            "provider": self.provider,
            "rotation_days": self.rotation_days,
            "rotate_before_expiry_days": self.rotate_before_expiry_days,
            "auto_rotate": self.auto_rotate,
            "notify_before_rotation": self.notify_before_rotation,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationPolicy:
        return cls(
            id=data["id"],
            key_id=data["key_id"],
            provider=data["provider"],
            rotation_days=data["rotation_days"],
            rotate_before_expiry_days=data.get("rotate_before_expiry_days", 7),
            auto_rotate=data.get("auto_rotate", True),
            notify_before_rotation=data.get("notify_before_rotation", 3),
            enabled=data.get("enabled", True),
        )


@dataclass
class RotationResult:
    """Result of a rotation operation."""

    success: bool
    key_id: str
    new_key_id: str | None = None
    old_key_id: str | None = None
    message: str | None = None
    error: str | None = None
    rotated_at: str | None = None


class RotationProvider:
    """Base class for provider-specific rotation."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key

    def __call__(self) -> tuple[str, str]:
        return self.get_new_key()

    def get_new_key(self) -> tuple[str, str]:
        """Get a new API key. Returns (key_value, key_id/name)."""
        raise NotImplementedError

    def revoke_old_key(self, key_identifier: str) -> bool:
        """Revoke the old key."""
        return True  # Default: don't revoke


class AnthropicRotationProvider(RotationProvider):
    """Anthropic API key rotation (via organization settings)."""

    def get_new_key(self) -> tuple[str, str]:
        """Create a new Anthropic API key."""
        import requests

        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

        response = requests.post(
            "https://api.anthropic.com/v1/organizations/current/api_keys",
            headers=headers,
            json={"name": f"scholardevclaw-{datetime.now().strftime('%Y%m%d-%H%M%S')}"},
            timeout=30,
        )

        if response.status_code == 201:
            data = response.json()
            return data["secret"], data["id"]
        else:
            raise RuntimeError(f"Failed to create Anthropic key: {response.text}")


class OpenAIRotationProvider(RotationProvider):
    """OpenAI API key rotation."""

    def get_new_key(self) -> tuple[str, str]:
        """Create a new OpenAI API key."""
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(
            "https://api.openai.com/v1/api_keys",
            headers=headers,
            json={"name": f"scholardevclaw-{datetime.now().strftime('%Y%m%d-%H%M%S')}"},
            timeout=30,
        )

        if response.status_code == 201:
            data = response.json()
            return data["secret_key"], data["id"]
        else:
            raise RuntimeError(f"Failed to create OpenAI key: {response.text}")


class GitHubRotationProvider(RotationProvider):
    """GitHub token rotation."""

    def get_new_key(self) -> tuple[str, str]:
        """Create a new GitHub PAT."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/vnd.github+json",
        }

        response = requests.post(
            "https://api.github.com/user/repos",
            headers=headers,
            json={
                "name": f"scholardevclaw-rotated-{datetime.now().strftime('%Y%m%d')}",
                "auto_init": False,
            },
            timeout=30,
        )

        # Note: GitHub doesn't have a direct API to create PATs
        # This would need organization admin or manual process
        raise NotImplementedError("GitHub PAT rotation requires manual process or org admin")


def get_rotation_provider(provider: str, api_key: str) -> RotationProvider:
    """Factory to get provider-specific rotation."""
    if provider == "anthropic":
        return AnthropicRotationProvider(api_key)
    elif provider == "openai":
        return OpenAIRotationProvider(api_key)
    elif provider == "github":
        return GitHubRotationProvider(api_key)
    else:
        raise ValueError(f"Rotation not supported for provider: {provider}")


class RotationScheduler:
    """Schedule and execute automatic key rotations."""

    POLICIES_FILE = "rotation_policies.json"
    ROTATION_LOG_FILE = "rotation_log.jsonl"

    def __init__(self, store_dir: str | Path, auth_store: Any = None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.store_dir, 0o700)
        self.policies_file = self.store_dir / self.POLICIES_FILE
        self.rotation_log_file = self.store_dir / self.ROTATION_LOG_FILE
        self.auth_store = auth_store

    def create_policy(
        self,
        key_id: str,
        provider: str,
        rotation_days: int = 90,
        auto_rotate: bool = True,
    ) -> RotationPolicy:
        """Create a rotation policy for a key."""
        import secrets

        policy = RotationPolicy(
            id=f"policy_{secrets.token_hex(8)}",
            key_id=key_id,
            provider=provider,
            rotation_days=rotation_days,
            auto_rotate=auto_rotate,
        )

        policies = self._load_policies()
        policies[policy.id] = policy.to_dict()
        self._save_policies(policies)

        return policy

    def update_policy(self, policy: RotationPolicy) -> None:
        """Update a rotation policy."""
        policies = self._load_policies()
        policies[policy.id] = policy.to_dict()
        self._save_policies(policies)

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a rotation policy."""
        policies = self._load_policies()
        if policy_id in policies:
            del policies[policy_id]
            self._save_policies(policies)
            return True
        return False

    def get_policy(self, policy_id: str) -> RotationPolicy | None:
        """Get a policy by ID."""
        policies = self._load_policies()
        data = policies.get(policy_id)
        return RotationPolicy.from_dict(data) if data else None

    def get_policy_for_key(self, key_id: str) -> RotationPolicy | None:
        """Get the rotation policy for a key."""
        policies = self._load_policies()
        for p in policies.values():
            if p["key_id"] == key_id:
                return RotationPolicy.from_dict(p)
        return None

    def list_policies(self) -> list[RotationPolicy]:
        """List all rotation policies."""
        policies = self._load_policies()
        return [RotationPolicy.from_dict(p) for p in policies.values()]

    def list_enabled_policies(self) -> list[RotationPolicy]:
        """List enabled rotation policies."""
        return [p for p in self.list_policies() if p.enabled]

    def get_keys_due_for_rotation(self) -> list[tuple[RotationPolicy, dict[str, Any]]]:
        """Get keys that are due for rotation."""
        due = []
        now = datetime.now()

        for policy in self.list_enabled_policies():
            if not self.auth_store:
                continue

            key = self.auth_store.get_api_key(policy.provider)
            if not key:
                continue

            # Check last rotation time
            rotation_history = self.auth_store.get_rotation_history(policy.key_id)
            if rotation_history:
                last_rotation = datetime.fromisoformat(rotation_history[-1].rotated_at)
                days_since = (now - last_rotation).days
            else:
                created = now - timedelta(days=365)  # Assume created a year ago
                days_since = (now - created).days

            if days_since >= policy.rotation_days:
                due.append((policy, {"days_since_rotation": days_since}))

            # Also check expiry if set
            if key.get("expires_at"):
                expiry = datetime.fromisoformat(key["expires_at"])
                days_until_expiry = (expiry - now).days
                if days_until_expiry <= policy.rotate_before_expiry_days:
                    due.append((policy, {"days_until_expiry": days_until_expiry}))

        return due

    def execute_rotation(self, policy: RotationPolicy, new_key_value: str) -> RotationResult:
        """Execute a key rotation."""
        if not self.auth_store:
            return RotationResult(
                success=False,
                key_id=policy.key_id,
                error="Auth store not configured",
            )

        try:
            old_key = self.auth_store.get_api_key(policy.provider)
            if not old_key:
                return RotationResult(
                    success=False,
                    key_id=policy.key_id,
                    error="Key not found",
                )

            rotated_key = self.auth_store.rotate_api_key(
                policy.key_id, new_key_value, reason="Automatic rotation"
            )

            if not rotated_key:
                return RotationResult(
                    success=False,
                    key_id=policy.key_id,
                    error="Rotation failed",
                )

            result = RotationResult(
                success=True,
                key_id=policy.key_id,
                new_key_id=rotated_key.id,
                old_key_id=policy.key_id,
                message=f"Successfully rotated key for {policy.provider}",
                rotated_at=datetime.now().isoformat(),
            )

            # Log rotation
            self._log_rotation(policy, result)

            return result

        except Exception as e:
            return RotationResult(
                success=False,
                key_id=policy.key_id,
                error=str(e),
            )

    def auto_rotate_due_keys(self) -> list[RotationResult]:
        """Automatically rotate all keys that are due."""
        results = []

        for policy, info in self.get_keys_due_for_rotation():
            if not policy.auto_rotate:
                continue

            try:
                provider = get_rotation_provider(policy.provider, "")
                new_key, _ = provider.get_new_key()
                result = self.execute_rotation(policy, new_key)
                results.append(result)
            except Exception as e:
                results.append(
                    RotationResult(
                        success=False,
                        key_id=policy.key_id,
                        error=f"Auto-rotation failed: {e}",
                    )
                )

        return results

    def get_rotation_history(self, key_id: str | None = None) -> list[dict[str, Any]]:
        """Get rotation history."""
        if not self.rotation_log_file.exists():
            return []

        history = []
        for line in self.rotation_log_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if key_id is None or data.get("key_id") == key_id:
                    history.append(data)
            except json.JSONDecodeError:
                continue

        return list(reversed(history))

    def _log_rotation(self, policy: RotationPolicy, result: RotationResult) -> None:
        """Log a rotation event."""
        entry = {
            "policy_id": policy.id,
            "key_id": result.key_id,
            "new_key_id": result.new_key_id,
            "old_key_id": result.old_key_id,
            "success": result.success,
            "message": result.message,
            "error": result.error,
            "rotated_at": result.rotated_at,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.rotation_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        os.chmod(self.rotation_log_file, 0o600)

    def _load_policies(self) -> dict[str, Any]:
        if not self.policies_file.exists():
            return {}
        try:
            return json.loads(self.policies_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_policies(self, policies: dict[str, Any]) -> None:
        self.policies_file.write_text(json.dumps(policies, indent=2))
        os.chmod(self.policies_file, 0o600)
