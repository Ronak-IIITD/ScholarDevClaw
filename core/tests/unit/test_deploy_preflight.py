from __future__ import annotations

from scholardevclaw.deploy.preflight import is_placeholder_value, parse_env_file, run_preflight


def test_parse_env_file_ignores_comments_and_blank_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# comment

SCHOLARDEVCLAW_API_AUTH_KEY=abc123
export CORE_BRIDGE_MODE=http
GRAFANA_ADMIN_USER="admin"
""",
        encoding="utf-8",
    )

    parsed = parse_env_file(env_file)

    assert parsed["SCHOLARDEVCLAW_API_AUTH_KEY"] == "abc123"
    assert parsed["CORE_BRIDGE_MODE"] == "http"
    assert parsed["GRAFANA_ADMIN_USER"] == "admin"


def test_placeholder_detection_includes_double_underscore_markers():
    assert is_placeholder_value("__OPENCLAW_TOKEN__") is True
    assert is_placeholder_value("replace_with_real_value") is True
    assert is_placeholder_value("real-secret-value") is False


def test_run_preflight_passes_with_required_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
SCHOLARDEVCLAW_API_AUTH_KEY=real-api-key
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_CORS_ORIGINS=https://scholardevclaw.ai
OPENCLAW_TOKEN=real-openclaw-token
OPENCLAW_API_URL=https://api.openclaw.example
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=super-secure-password
CORE_BRIDGE_MODE=http
CORE_API_URL=http://core-api:8000
SCHOLARDEVCLAW_API_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-good
""",
        encoding="utf-8",
    )

    result = run_preflight(env_file)

    assert result.ok is True
    assert result.errors == []


def test_run_preflight_enforces_provider_key_and_bridge_mode(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
SCHOLARDEVCLAW_API_AUTH_KEY=real-api-key
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_CORS_ORIGINS=https://scholardevclaw.ai
OPENCLAW_TOKEN=real-openclaw-token
OPENCLAW_API_URL=https://api.openclaw.example
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=super-secure-password
CORE_BRIDGE_MODE=invalid
SCHOLARDEVCLAW_API_PROVIDER=openai
""",
        encoding="utf-8",
    )

    result = run_preflight(env_file)

    assert result.ok is False
    assert any("CORE_BRIDGE_MODE must be one of" in e for e in result.errors)
    assert any("OPENAI_API_KEY is required" in e for e in result.errors)


def test_run_preflight_warns_on_localhost_and_blocks_default_grafana_password(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
SCHOLARDEVCLAW_API_AUTH_KEY=real-api-key
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_CORS_ORIGINS=https://app.example,http://localhost:3000
OPENCLAW_TOKEN=real-openclaw-token
OPENCLAW_API_URL=https://api.openclaw.example
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_me_in_production
CORE_BRIDGE_MODE=subprocess
""",
        encoding="utf-8",
    )

    result = run_preflight(env_file)

    assert result.ok is False
    assert any(
        "GRAFANA_ADMIN_PASSWORD must not be change_me_in_production" in e for e in result.errors
    )
    assert any("localhost/127.0.0.1" in w for w in result.warnings)
