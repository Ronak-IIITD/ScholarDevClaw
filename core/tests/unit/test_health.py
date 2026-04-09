from __future__ import annotations

import os

from scholardevclaw.utils.health import HealthChecker


def test_environment_check_requires_core_vars_in_prod(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", raising=False)

    checker = HealthChecker()
    result = checker.run_check("environment")

    assert result.healthy is False
    assert "SCHOLARDEVCLAW_API_AUTH_KEY" in result.details["missing_required"]
    assert "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS" in result.details["missing_required"]


def test_environment_check_relaxes_required_vars_in_dev_mode(monkeypatch):
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")
    monkeypatch.delenv("SCHOLARDEVCLAW_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", raising=False)

    checker = HealthChecker()
    result = checker.run_check("environment")

    assert result.healthy is True
    assert result.details["dev_mode"] is True
    assert result.details["missing_required"] == []

    # Avoid cross-test contamination for environments that reuse process state.
    os.environ.pop("SCHOLARDEVCLAW_DEV_MODE", None)
