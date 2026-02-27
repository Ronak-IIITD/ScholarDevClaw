"""Import/export support for auth credentials.

Supports:
- Export to JSON, env file, dotenv format
- Import from JSON, env files, 1Password CSV

Security:
- Import operations enforce a maximum key count limit
- Keys are deduplicated by fingerprint during import
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import APIKey, AuthConfig, AuthProvider

# Maximum number of keys that can be imported in a single operation
MAX_IMPORT_KEYS = 100


def _key_fingerprint(key_value: str) -> str:
    """Compute a SHA256 fingerprint for deduplication."""
    return hashlib.sha256(key_value.encode()).hexdigest()


def _deduplicate_keys(keys: list[APIKey]) -> list[APIKey]:
    """Remove duplicate keys by fingerprint, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[APIKey] = []
    for k in keys:
        fp = _key_fingerprint(k.key)
        if fp not in seen:
            seen.add(fp)
            unique.append(k)
    return unique


@dataclass
class ImportResult:
    """Result of an import operation."""

    imported_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: list[str] | None = None
    imported_keys: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "errors": self.errors or [],
            "imported_keys": self.imported_keys or [],
        }


class AuthExporter:
    """Export auth credentials to various formats."""

    @staticmethod
    def to_json(config: AuthConfig, include_keys: bool = True, pretty: bool = True) -> str:
        """Export auth config to JSON string."""
        data = config.to_dict()
        if not include_keys:
            for key_data in data.get("api_keys", []):
                key_data["key"] = "***REDACTED***"
        indent = 2 if pretty else None
        return json.dumps(data, indent=indent)

    @staticmethod
    def to_env(config: AuthConfig, include_all: bool = False) -> str:
        """Export active key as .env format."""
        lines = [
            "# ScholarDevClaw Credentials Export",
            f"# Generated: {datetime.now().isoformat()}",
            "",
        ]

        if include_all:
            for i, key in enumerate(config.api_keys):
                if not key.is_valid():
                    continue
                prefix = f"SCHOLARDEVCLAW_API_KEY_{i}" if i > 0 else "SCHOLARDEVCLAW_API_KEY"
                lines.append(f"{prefix}={key.key}")
                lines.append(f"{prefix}_PROVIDER={key.provider.value}")
                lines.append(f"{prefix}_NAME={key.name}")
                lines.append("")
        else:
            active = config.get_active_key()
            if active:
                lines.append(f"SCHOLARDEVCLAW_API_KEY={active.key}")
                lines.append(f"SCHOLARDEVCLAW_API_PROVIDER={active.provider.value}")
                lines.append("")

        if config.profile:
            if config.profile.email:
                lines.append(f"SCHOLARDEVCLAW_USER_EMAIL={config.profile.email}")
            if config.profile.name:
                lines.append(f"SCHOLARDEVCLAW_USER_NAME={config.profile.name}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def to_dotenv_file(
        config: AuthConfig, output_path: str | Path, include_all: bool = False
    ) -> Path:
        """Write credentials to a .env file."""
        path = Path(output_path)
        content = AuthExporter.to_env(config, include_all)
        path.write_text(content)
        # Set restrictive permissions
        os.chmod(path, 0o600)
        return path

    @staticmethod
    def to_json_file(
        config: AuthConfig, output_path: str | Path, include_keys: bool = True
    ) -> Path:
        """Write config to JSON file."""
        path = Path(output_path)
        content = AuthExporter.to_json(config, include_keys)
        path.write_text(content)
        os.chmod(path, 0o600)
        return path


class AuthImporter:
    """Import auth credentials from various sources."""

    @staticmethod
    def _detect_provider(key: str) -> AuthProvider:
        """Auto-detect provider from key format."""
        if key.startswith("sk-ant"):
            return AuthProvider.ANTHROPIC
        elif key.startswith("sk-"):
            return AuthProvider.OPENAI
        elif key.startswith("ghp_") or key.startswith("github_pat_"):
            return AuthProvider.GITHUB
        elif key.startswith("ya29.") or key.startswith("1//"):
            return AuthProvider.GOOGLE
        return AuthProvider.CUSTOM

    @staticmethod
    def from_json(json_str: str) -> tuple[AuthConfig, ImportResult]:
        """Import from JSON string (ScholarDevClaw format)."""
        result = ImportResult(errors=[], imported_keys=[])
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            result.error_count = 1
            result.errors.append(f"Invalid JSON: {e}")
            return AuthConfig(), result

        try:
            config = AuthConfig.from_dict(data)
            # Deduplicate and enforce import limit
            config.api_keys = _deduplicate_keys(config.api_keys)
            if len(config.api_keys) > MAX_IMPORT_KEYS:
                result.errors.append(
                    f"Import truncated: {len(config.api_keys)} keys exceed limit of {MAX_IMPORT_KEYS}"
                )
                config.api_keys = config.api_keys[:MAX_IMPORT_KEYS]
            result.imported_count = len(config.api_keys)
            result.imported_keys = [k.name for k in config.api_keys]
        except (KeyError, ValueError) as e:
            result.error_count = 1
            result.errors.append(f"Invalid config format: {e}")
            return AuthConfig(), result

        return config, result

    @staticmethod
    def from_json_file(file_path: str | Path) -> tuple[AuthConfig, ImportResult]:
        """Import from JSON file."""
        path = Path(file_path)
        if not path.exists():
            result = ImportResult(errors=[f"File not found: {file_path}"])
            result.error_count = 1
            return AuthConfig(), result
        return AuthImporter.from_json(path.read_text())

    @staticmethod
    def from_env(env_content: str) -> tuple[list[APIKey], ImportResult]:
        """Import from .env file content.

        Looks for patterns like:
        - SCHOLARDEVCLAW_API_KEY=...
        - ANTHROPIC_API_KEY=...
        - OPENAI_API_KEY=...
        - GITHUB_TOKEN=...
        """
        result = ImportResult(errors=[], imported_keys=[])
        keys: list[APIKey] = []

        env_patterns = {
            r"SCHOLARDEVCLAW_API_KEY\s*=\s*(.+)": None,  # auto-detect
            r"ANTHROPIC_API_KEY\s*=\s*(.+)": AuthProvider.ANTHROPIC,
            r"OPENAI_API_KEY\s*=\s*(.+)": AuthProvider.OPENAI,
            r"GITHUB_TOKEN\s*=\s*(.+)": AuthProvider.GITHUB,
            r"GOOGLE_API_KEY\s*=\s*(.+)": AuthProvider.GOOGLE,
        }

        import secrets as _secrets

        for line in env_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            for pattern, provider in env_patterns.items():
                match = re.match(pattern, line)
                if match:
                    key_value = match.group(1).strip().strip("'\"")
                    if not key_value:
                        continue

                    if provider is None:
                        provider = AuthImporter._detect_provider(key_value)

                    api_key = APIKey(
                        id=f"key_{_secrets.token_hex(8)}",
                        name=f"imported-{provider.value}",
                        provider=provider,
                        key=key_value,
                    )
                    keys.append(api_key)
                    result.imported_count += 1
                    result.imported_keys.append(api_key.name)
                    break

        return keys, result

    @staticmethod
    def from_env_file(file_path: str | Path) -> tuple[list[APIKey], ImportResult]:
        """Import from .env file."""
        path = Path(file_path)
        if not path.exists():
            result = ImportResult(errors=[f"File not found: {file_path}"])
            result.error_count = 1
            return [], result
        return AuthImporter.from_env(path.read_text())

    @staticmethod
    def from_1password_csv(csv_content: str) -> tuple[list[APIKey], ImportResult]:
        """Import from 1Password CSV export.

        Expected columns: Title, Username, Password, URL, Notes
        We use Password as the API key and Title as the name.
        """
        result = ImportResult(errors=[], imported_keys=[])
        keys: list[APIKey] = []

        import secrets as _secrets

        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            try:
                title = row.get("Title", row.get("title", ""))
                password = row.get("Password", row.get("password", row.get("credential", "")))
                notes = row.get("Notes", row.get("notes", ""))
                url = row.get("URL", row.get("url", ""))

                if not password:
                    result.skipped_count += 1
                    continue

                # Try to detect provider from title, notes, or URL
                provider = AuthProvider.CUSTOM
                combined = f"{title} {notes} {url}".lower()

                if "anthropic" in combined or "claude" in combined:
                    provider = AuthProvider.ANTHROPIC
                elif "openai" in combined or "gpt" in combined:
                    provider = AuthProvider.OPENAI
                elif "github" in combined:
                    provider = AuthProvider.GITHUB
                elif "google" in combined:
                    provider = AuthProvider.GOOGLE

                # Also try auto-detect from key format
                if provider == AuthProvider.CUSTOM:
                    provider = AuthImporter._detect_provider(password)

                name = title or f"imported-{provider.value}"
                # Sanitize name
                name = re.sub(r"[^\w\s\-\.]", "", name)[:100] or f"imported-{provider.value}"

                api_key = APIKey(
                    id=f"key_{_secrets.token_hex(8)}",
                    name=name,
                    provider=provider,
                    key=password,
                )
                keys.append(api_key)
                result.imported_count += 1
                result.imported_keys.append(name)

            except Exception as e:
                result.error_count += 1
                result.errors.append(f"Row error: {e}")

        return keys, result

    @staticmethod
    def from_1password_csv_file(file_path: str | Path) -> tuple[list[APIKey], ImportResult]:
        """Import from 1Password CSV file."""
        path = Path(file_path)
        if not path.exists():
            result = ImportResult(errors=[f"File not found: {file_path}"])
            result.error_count = 1
            return [], result
        return AuthImporter.from_1password_csv(path.read_text())
