"""Tests for import/export (import_export.py)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from scholardevclaw.auth.import_export import AuthExporter, AuthImporter, ImportResult
from scholardevclaw.auth.types import APIKey, AuthConfig, AuthProvider, UserProfile


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_config(
    keys: list[tuple[str, str, AuthProvider]] | None = None,
    email: str | None = None,
    name: str | None = None,
) -> AuthConfig:
    """Helper to build an AuthConfig with keys and optional profile."""
    config = AuthConfig()
    if keys:
        for i, (key_val, key_name, provider) in enumerate(keys):
            api_key = APIKey(
                id=f"key_{i}",
                name=key_name,
                provider=provider,
                key=key_val,
            )
            config.api_keys.append(api_key)
        config.default_key_id = config.api_keys[0].id
        config.default_provider = config.api_keys[0].provider
    if email or name:
        config.profile = UserProfile(id="user_1", email=email, name=name)
    return config


class TestImportResult:
    def test_defaults(self):
        r = ImportResult()
        assert r.imported_count == 0
        assert r.skipped_count == 0
        assert r.error_count == 0

    def test_to_dict(self):
        r = ImportResult(
            imported_count=3,
            skipped_count=1,
            error_count=0,
            errors=[],
            imported_keys=["k1", "k2", "k3"],
        )
        d = r.to_dict()
        assert d["imported_count"] == 3
        assert d["skipped_count"] == 1
        assert len(d["imported_keys"]) == 3

    def test_to_dict_none_defaults(self):
        r = ImportResult()
        d = r.to_dict()
        assert d["errors"] == []
        assert d["imported_keys"] == []


# -------------------------------------------------------------------
# Exporter tests
# -------------------------------------------------------------------


class TestAuthExporter:
    def test_to_json_basic(self):
        config = _make_config(keys=[("sk-ant-test123456", "my-key", AuthProvider.ANTHROPIC)])
        result = AuthExporter.to_json(config)
        data = json.loads(result)
        assert len(data["api_keys"]) == 1
        assert data["api_keys"][0]["key"] == "sk-ant-test123456"

    def test_to_json_redacted(self):
        config = _make_config(keys=[("sk-ant-test123456", "my-key", AuthProvider.ANTHROPIC)])
        result = AuthExporter.to_json(config, include_keys=False)
        data = json.loads(result)
        assert data["api_keys"][0]["key"] == "***REDACTED***"

    def test_to_json_pretty(self):
        config = _make_config(keys=[("sk-test", "k", AuthProvider.CUSTOM)])
        pretty = AuthExporter.to_json(config, pretty=True)
        compact = AuthExporter.to_json(config, pretty=False)
        assert len(pretty) > len(compact)
        assert "\n" in pretty

    def test_to_json_multiple_keys(self):
        config = _make_config(
            keys=[
                ("sk-ant-key1", "key1", AuthProvider.ANTHROPIC),
                ("sk-openai-key2", "key2", AuthProvider.OPENAI),
            ]
        )
        result = AuthExporter.to_json(config)
        data = json.loads(result)
        assert len(data["api_keys"]) == 2

    def test_to_json_with_profile(self):
        config = _make_config(
            keys=[("sk-test", "k", AuthProvider.CUSTOM)],
            email="user@example.com",
            name="Test User",
        )
        result = AuthExporter.to_json(config)
        data = json.loads(result)
        assert data["profile"]["email"] == "user@example.com"

    def test_to_env_active_key(self):
        config = _make_config(keys=[("sk-ant-testkey123", "my-key", AuthProvider.ANTHROPIC)])
        env = AuthExporter.to_env(config)
        assert "SCHOLARDEVCLAW_API_KEY=sk-ant-testkey123" in env
        assert "SCHOLARDEVCLAW_API_PROVIDER=anthropic" in env

    def test_to_env_all_keys(self):
        config = _make_config(
            keys=[
                ("sk-ant-key1", "key1", AuthProvider.ANTHROPIC),
                ("sk-key2", "key2", AuthProvider.OPENAI),
            ]
        )
        env = AuthExporter.to_env(config, include_all=True)
        assert "SCHOLARDEVCLAW_API_KEY=" in env
        assert "SCHOLARDEVCLAW_API_KEY_1=" in env

    def test_to_env_with_profile(self):
        config = _make_config(
            keys=[("sk-test", "k", AuthProvider.CUSTOM)],
            email="user@example.com",
            name="Test User",
        )
        env = AuthExporter.to_env(config)
        assert "SCHOLARDEVCLAW_USER_EMAIL=user@example.com" in env
        assert "SCHOLARDEVCLAW_USER_NAME=Test User" in env

    def test_to_env_empty_config(self):
        config = AuthConfig()
        env = AuthExporter.to_env(config)
        assert "ScholarDevClaw Credentials Export" in env

    def test_to_dotenv_file(self, temp_dir):
        config = _make_config(keys=[("sk-test-key", "k", AuthProvider.CUSTOM)])
        path = AuthExporter.to_dotenv_file(config, temp_dir / "output.env")
        assert path.exists()
        content = path.read_text()
        assert "SCHOLARDEVCLAW_API_KEY=sk-test-key" in content
        # Check file permissions (should be 0o600)
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_to_json_file(self, temp_dir):
        config = _make_config(keys=[("sk-test-key", "k", AuthProvider.CUSTOM)])
        path = AuthExporter.to_json_file(config, temp_dir / "output.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["api_keys"]) == 1
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_to_json_file_redacted(self, temp_dir):
        config = _make_config(keys=[("sk-secret", "k", AuthProvider.CUSTOM)])
        path = AuthExporter.to_json_file(config, temp_dir / "out.json", include_keys=False)
        data = json.loads(path.read_text())
        assert data["api_keys"][0]["key"] == "***REDACTED***"


# -------------------------------------------------------------------
# Importer tests
# -------------------------------------------------------------------


class TestAuthImporterJSON:
    def test_from_json_valid(self):
        config = _make_config(keys=[("sk-test", "mykey", AuthProvider.CUSTOM)])
        json_str = AuthExporter.to_json(config)
        imported_config, result = AuthImporter.from_json(json_str)
        assert result.imported_count == 1
        assert result.error_count == 0
        assert imported_config.api_keys[0].key == "sk-test"

    def test_from_json_invalid_json(self):
        _, result = AuthImporter.from_json("{{not json")
        assert result.error_count == 1
        assert any("Invalid JSON" in e for e in result.errors)

    def test_from_json_invalid_schema(self):
        _, result = AuthImporter.from_json('{"api_keys": [{"bad": "data"}]}')
        assert result.error_count == 1

    def test_from_json_multiple_keys(self):
        config = _make_config(
            keys=[
                ("sk-key1", "k1", AuthProvider.ANTHROPIC),
                ("sk-key2", "k2", AuthProvider.OPENAI),
                ("sk-key3", "k3", AuthProvider.CUSTOM),
            ]
        )
        json_str = AuthExporter.to_json(config)
        imported, result = AuthImporter.from_json(json_str)
        assert result.imported_count == 3

    def test_from_json_file(self, temp_dir):
        config = _make_config(keys=[("sk-test", "k", AuthProvider.CUSTOM)])
        file_path = temp_dir / "import.json"
        file_path.write_text(AuthExporter.to_json(config))
        imported, result = AuthImporter.from_json_file(str(file_path))
        assert result.imported_count == 1

    def test_from_json_file_not_found(self):
        _, result = AuthImporter.from_json_file("/nonexistent/file.json")
        assert result.error_count == 1
        assert any("File not found" in e for e in result.errors)


class TestAuthImporterEnv:
    def test_from_env_scholardevclaw_key(self):
        content = "SCHOLARDEVCLAW_API_KEY=sk-ant-my-test-key-123\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].key == "sk-ant-my-test-key-123"
        assert keys[0].provider == AuthProvider.ANTHROPIC

    def test_from_env_anthropic_key(self):
        content = "ANTHROPIC_API_KEY=sk-ant-abcdef123456\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].provider == AuthProvider.ANTHROPIC

    def test_from_env_openai_key(self):
        content = "OPENAI_API_KEY=sk-openai-test-key-123\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].provider == AuthProvider.OPENAI

    def test_from_env_github_token(self):
        content = "GITHUB_TOKEN=ghp_abcdefghijklmnop123456\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].provider == AuthProvider.GITHUB

    def test_from_env_google_key(self):
        content = "GOOGLE_API_KEY=ya29.abcdefghij123456\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].provider == AuthProvider.GOOGLE

    def test_from_env_multiple_keys(self):
        content = """
ANTHROPIC_API_KEY=sk-ant-key1
OPENAI_API_KEY=sk-key2
GITHUB_TOKEN=ghp_key3
"""
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 3

    def test_from_env_comments_and_blanks_ignored(self):
        content = """
# This is a comment
ANTHROPIC_API_KEY=sk-ant-key1

# Another comment

"""
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1

    def test_from_env_quoted_values(self):
        content = 'SCHOLARDEVCLAW_API_KEY="sk-ant-quoted-key"\n'
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].key == "sk-ant-quoted-key"

    def test_from_env_single_quoted_values(self):
        content = "SCHOLARDEVCLAW_API_KEY='sk-ant-quoted-key'\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1
        assert keys[0].key == "sk-ant-quoted-key"

    def test_from_env_empty_value_skipped(self):
        content = "SCHOLARDEVCLAW_API_KEY=\n"
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 0

    def test_from_env_auto_detect_provider(self):
        content = "SCHOLARDEVCLAW_API_KEY=sk-ant-key123\n"
        keys, result = AuthImporter.from_env(content)
        assert keys[0].provider == AuthProvider.ANTHROPIC

    def test_from_env_custom_provider_fallback(self):
        content = "SCHOLARDEVCLAW_API_KEY=custom-key-no-prefix\n"
        keys, result = AuthImporter.from_env(content)
        assert keys[0].provider == AuthProvider.CUSTOM

    def test_from_env_file(self, temp_dir):
        env_file = temp_dir / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-file-key\n")
        keys, result = AuthImporter.from_env_file(str(env_file))
        assert result.imported_count == 1

    def test_from_env_file_not_found(self):
        keys, result = AuthImporter.from_env_file("/nonexistent/.env")
        assert result.error_count == 1
        assert keys == []

    def test_from_env_unrecognized_lines_ignored(self):
        content = """
SOME_OTHER_VAR=foobar
ANTHROPIC_API_KEY=sk-ant-key1
DATABASE_URL=postgres://localhost/db
"""
        keys, result = AuthImporter.from_env(content)
        assert result.imported_count == 1


class TestAuthImporter1Password:
    def test_from_1password_csv_basic(self):
        csv_content = """Title,Username,Password,URL,Notes
Anthropic API Key,user@example.com,sk-ant-1password-key,https://console.anthropic.com,Claude key
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 1
        assert keys[0].key == "sk-ant-1password-key"
        assert keys[0].provider == AuthProvider.ANTHROPIC

    def test_from_1password_csv_provider_detection_from_title(self):
        csv_content = """Title,Username,Password,URL,Notes
OpenAI GPT Key,user,custom-key-format,https://platform.openai.com,GPT
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 1
        assert keys[0].provider == AuthProvider.OPENAI

    def test_from_1password_csv_provider_detection_from_url(self):
        csv_content = """Title,Username,Password,URL,Notes
My Key,user,custom-key,https://github.com/settings/tokens,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert keys[0].provider == AuthProvider.GITHUB

    def test_from_1password_csv_skip_empty_password(self):
        csv_content = """Title,Username,Password,URL,Notes
No Password,user,,https://example.com,
Has Password,user,sk-test-key,https://example.com,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 1
        assert result.skipped_count == 1

    def test_from_1password_csv_multiple_rows(self):
        csv_content = """Title,Username,Password,URL,Notes
Key 1,u,sk-ant-key1,https://anthropic.com,
Key 2,u,sk-key2,https://openai.com,
Key 3,u,ghp_key3,https://github.com,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 3

    def test_from_1password_csv_name_sanitization(self):
        csv_content = """Title,Username,Password,URL,Notes
Key <script>alert(1)</script>,u,sk-test-key,https://example.com,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 1
        # Name should have special chars removed
        assert "<script>" not in keys[0].name

    def test_from_1password_csv_auto_detect_from_key_format(self):
        csv_content = """Title,Username,Password,URL,Notes
My API Key,u,sk-ant-auto-detect-key,https://example.com,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert keys[0].provider == AuthProvider.ANTHROPIC

    def test_from_1password_csv_file(self, temp_dir):
        csv_file = temp_dir / "1password.csv"
        csv_file.write_text("""Title,Username,Password,URL,Notes
Test Key,u,sk-test-key,https://example.com,
""")
        keys, result = AuthImporter.from_1password_csv_file(str(csv_file))
        assert result.imported_count == 1

    def test_from_1password_csv_file_not_found(self):
        keys, result = AuthImporter.from_1password_csv_file("/nonexistent/file.csv")
        assert result.error_count == 1
        assert keys == []

    def test_from_1password_csv_empty(self):
        csv_content = """Title,Username,Password,URL,Notes
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 0

    def test_from_1password_csv_no_title_fallback(self):
        csv_content = """Title,Username,Password,URL,Notes
,u,sk-test-key,https://example.com,
"""
        keys, result = AuthImporter.from_1password_csv(csv_content)
        assert result.imported_count == 1
        # Should use fallback name
        assert "imported-" in keys[0].name


class TestAuthImporterDetectProvider:
    def test_detect_anthropic(self):
        assert AuthImporter._detect_provider("sk-ant-abc") == AuthProvider.ANTHROPIC

    def test_detect_openai(self):
        assert AuthImporter._detect_provider("sk-abc123") == AuthProvider.OPENAI

    def test_detect_github_ghp(self):
        assert AuthImporter._detect_provider("ghp_abc123") == AuthProvider.GITHUB

    def test_detect_github_pat(self):
        assert AuthImporter._detect_provider("github_pat_abc123") == AuthProvider.GITHUB

    def test_detect_google_ya29(self):
        assert AuthImporter._detect_provider("ya29.abc123") == AuthProvider.GOOGLE

    def test_detect_google_oauth(self):
        assert AuthImporter._detect_provider("1//abc123") == AuthProvider.GOOGLE

    def test_detect_custom_fallback(self):
        assert AuthImporter._detect_provider("unknown-format") == AuthProvider.CUSTOM

    def test_detect_empty_string(self):
        assert AuthImporter._detect_provider("") == AuthProvider.CUSTOM


class TestImportExportWithStore:
    """Integration tests: import/export via AuthStore."""

    def test_export_json_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-123", "my-key", AuthProvider.CUSTOM)
        result = store.export_json()
        data = json.loads(result)
        assert len(data["api_keys"]) == 1

    def test_export_json_redacted_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-123", "my-key", AuthProvider.CUSTOM)
        result = store.export_json(include_keys=False)
        data = json.loads(result)
        assert data["api_keys"][0]["key"] == "***REDACTED***"

    def test_export_env_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-123", "my-key", AuthProvider.CUSTOM)
        env = store.export_env()
        assert "SCHOLARDEVCLAW_API_KEY=sk-test-123" in env

    def test_import_env_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        count, errors = store.import_keys_from_env("ANTHROPIC_API_KEY=sk-ant-imported\n")
        assert count == 1
        assert len(errors) == 0
        keys = store.list_api_keys()
        assert len(keys) == 1

    def test_import_json_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        config = _make_config(keys=[("sk-test", "k", AuthProvider.CUSTOM)])
        json_str = AuthExporter.to_json(config)

        count, errors = store.import_keys_from_json(json_str)
        assert count == 1
        assert len(errors) == 0

    def test_import_1password_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        csv = """Title,Username,Password,URL,Notes
My Key,u,sk-ant-key123,https://anthropic.com,
"""
        count, errors = store.import_keys_from_1password(csv)
        assert count == 1
        assert len(errors) == 0

    def test_import_json_invalid_via_store(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        count, errors = store.import_keys_from_json("{{invalid")
        assert count == 0
        assert len(errors) > 0

    def test_roundtrip_export_import(self, temp_dir):
        """Export then import should preserve keys."""
        from scholardevclaw.auth.store import AuthStore

        store1 = AuthStore(str(temp_dir / "store1"))
        store1.add_api_key("sk-ant-roundtrip-key", "my-key", AuthProvider.ANTHROPIC)
        json_str = store1.export_json()

        store2 = AuthStore(str(temp_dir / "store2"))
        count, errors = store2.import_keys_from_json(json_str)
        assert count == 1
        keys = store2.list_api_keys()
        assert keys[0].key == "sk-ant-roundtrip-key"
