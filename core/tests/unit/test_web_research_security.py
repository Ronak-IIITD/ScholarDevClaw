from __future__ import annotations

import asyncio

from scholardevclaw.research_intelligence.web_research import (
    GitHubRepo,
    WebResearchEngine,
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


def test_lookup_paper_with_code_by_arxiv_falls_back_to_github_when_endpoint_is_html(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}

        def json(self):
            raise ValueError("not json")

    async def fake_safe_get(*_args, **_kwargs):
        return FakeResponse()

    async def fake_search_github(self, query: str, language: str = "python", max_results: int = 10):
        return [
            GitHubRepo(
                name="flash-attn",
                owner="dao-ai",
                url="https://github.com/dao-ai/flash-attn",
                description="FlashAttention reference implementation",
                stars=123,
                language=language,
                topics=["attention"],
                relevance_score=95.0,
            )
        ]

    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._safe_get",
        fake_safe_get,
    )
    monkeypatch.setattr(WebResearchEngine, "search_github", fake_search_github)

    engine = WebResearchEngine()
    engine.client = object()
    result = asyncio.run(
        engine.lookup_paper_with_code_by_arxiv(
            "2205.14135",
            paper_title="FlashAttention",
        )
    )

    assert result is not None
    assert result["source"] == "github_search_fallback"
    assert result["repositories"][0]["url"] == "https://github.com/dao-ai/flash-attn"
