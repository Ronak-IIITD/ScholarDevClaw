from __future__ import annotations

import asyncio

from scholardevclaw.research_intelligence.web_research import (
    GitHubRepo,
    WebResearchEngine,
    _build_raw_github_url,
    _compute_backoff,
    _is_allowed_fixed_source_url,
    _safe_get,
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


def test_compute_backoff_increases_exponentially():
    """Backoff should grow: base * 2^(attempt-1) with jitter in [0.5, 1.5]."""
    base = 1.0
    # Check that minimum bounds hold (jitter can be 0.5 minimum)
    assert _compute_backoff(1, base) >= base * 0.5
    assert _compute_backoff(2, base) >= base * 2 * 0.5
    assert _compute_backoff(3, base) >= base * 4 * 0.5
    # Check maximum bounds (jitter can be 1.5 maximum)
    assert _compute_backoff(1, base) <= base * 1.5
    assert _compute_backoff(2, base) <= base * 2 * 1.5
    assert _compute_backoff(3, base) <= base * 4 * 1.5


def test_safe_get_returns_immediately_on_success(monkeypatch):
    """Should return on first successful response without retrying."""
    call_count = 0

    class FakeClient:
        async def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse(200)

    class FakeResponse:
        status_code: int
        headers: dict

        def __init__(self, code: int):
            self.status_code = code
            self.headers = {}

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._is_allowed_fixed_source_url",
        lambda _url: True,
    )

    result = asyncio.run(
        _safe_get(FakeClient(), "https://api.github.com/repos/test", max_retries=3)
    )
    assert result.status_code == 200
    assert call_count == 1  # Only one call, no retries


def test_safe_get_retries_on_server_error(monkeypatch):
    """Should retry on 500 errors with backoff."""
    call_count = 0

    class FakeClient:
        async def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeResponse(500)
            return FakeResponse(200)

    class FakeResponse:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._is_allowed_fixed_source_url",
        lambda _url: True,
    )
    # Speed up tests by using tiny delays
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._compute_backoff",
        lambda _attempt, _base: 0.01,
    )

    result = asyncio.run(
        _safe_get(
            FakeClient(), "https://api.github.com/repos/test", max_retries=3, base_delay=0.001
        )
    )
    assert result.status_code == 200
    assert call_count == 3  # Two 500s + one success


def test_safe_get_gives_up_after_max_retries(monkeypatch):
    """Should exhaust retries and return the last 500 response."""
    call_count = 0

    class FakeClient:
        async def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse(502)

    class FakeResponse:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}

    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._is_allowed_fixed_source_url",
        lambda _url: True,
    )
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.web_research._compute_backoff",
        lambda _attempt, _base: 0.01,
    )

    result = asyncio.run(
        _safe_get(
            FakeClient(), "https://api.github.com/repos/test", max_retries=3, base_delay=0.001
        )
    )
    assert result.status_code == 502
    assert call_count == 3  # All retries exhausted


def test_safe_get_rejects_blocked_url():
    """Should raise ValueError for non-allowed URLs."""
    import pytest

    async def _run():
        await _safe_get(None, "https://evil.com/api/data")

    with pytest.raises(ValueError, match="Blocked outbound URL"):
        asyncio.run(_run())
