from __future__ import annotations

from scholardevclaw.research_intelligence.web_research import _build_raw_github_url


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
