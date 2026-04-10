from __future__ import annotations

from scholardevclaw.research_intelligence.web_research import (
    _build_raw_github_url,
    _is_allowed_fixed_source_url,
)


def test_build_raw_github_url_rejects_spoofed_host():
    assert _build_raw_github_url("https://github.com.evil.com/a/b/blob/main/x.py") is None


def test_build_raw_github_url_rejects_http_scheme():
    assert _build_raw_github_url("http://github.com/a/b/blob/main/x.py") is None


def test_build_raw_github_url_accepts_valid_blob_url(monkeypatch):
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._validate_public_host",
        lambda _host: True,
    )

    raw = _build_raw_github_url("https://github.com/owner/repo/blob/main/src/file.py")

    assert raw == "https://raw.githubusercontent.com/owner/repo/main/src/file.py"


def test_build_raw_github_url_accepts_raw_host(monkeypatch):
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._validate_public_host",
        lambda _host: True,
    )

    raw = _build_raw_github_url("https://raw.githubusercontent.com/owner/repo/main/src/file.py")

    assert raw == "https://raw.githubusercontent.com/owner/repo/main/src/file.py"


def test_allowed_fixed_source_url_rejects_non_allowlisted_host():
    assert _is_allowed_fixed_source_url("https://github.com/some/repo") is False


def test_allowed_fixed_source_url_accepts_known_hosts(monkeypatch):
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._validate_public_host",
        lambda _host: True,
    )

    assert _is_allowed_fixed_source_url("https://api.github.com/repos/o/r") is True
    assert _is_allowed_fixed_source_url("https://paperswithcode.com/api/v0/search") is True
