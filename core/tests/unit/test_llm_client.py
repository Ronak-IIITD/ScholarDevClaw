from __future__ import annotations

import json

import httpx
import pytest

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.llm.client import LLMAPIError, LLMClient
from scholardevclaw.utils.retry import RetryPolicy


def test_from_provider_ollama_uses_ollama_host_and_empty_api_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.internal:2244/")

    client = LLMClient.from_provider("ollama")
    try:
        assert client.provider == AuthProvider.OLLAMA
        assert client.api_key == ""
        assert client.base_url == "http://ollama.internal:2244"
    finally:
        client.close()


def test_from_provider_non_ollama_uses_provider_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123456")

    client = LLMClient.from_provider("openrouter")
    try:
        assert client.provider == AuthProvider.OPENROUTER
        assert client.api_key == "sk-or-test-123456"
    finally:
        client.close()


def test_from_provider_new_provider_uses_provider_env_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-123456")

    client = LLMClient.from_provider("gemini")
    try:
        assert client.provider == AuthProvider.GEMINI
        assert client.api_key == "gemini-test-123456"
    finally:
        client.close()


@pytest.mark.parametrize(
    ("provider_name", "env_var", "api_key", "default_model", "base_url"),
    [
        (
            "gemini",
            "GEMINI_API_KEY",
            "gemini-test-key-123456",
            "gemini-2.0-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        (
            "grok",
            "XAI_API_KEY",
            "xai-test-key-123456",
            "grok-4-0709",
            "https://api.x.ai/v1",
        ),
        (
            "moonshot",
            "MOONSHOT_API_KEY",
            "moonshot-test-key-123456",
            "kimi-k2.6",
            "https://api.moonshot.cn/v1",
        ),
        (
            "glm",
            "GLM_API_KEY",
            "glm-test-key-123456",
            "glm-5.1",
            "https://open.bigmodel.cn/api/paas/v4",
        ),
        (
            "minimax",
            "MINIMAX_API_KEY",
            "minimax-test-key-123456",
            "MiniMax-M2.7",
            "https://api.minimax.io/v1",
        ),
    ],
)
def test_from_provider_new_openai_compatible_providers_defaults(
    monkeypatch,
    provider_name,
    env_var,
    api_key,
    default_model,
    base_url,
):
    monkeypatch.setenv(env_var, api_key)

    client = LLMClient.from_provider(provider_name)
    try:
        assert client.api_key == api_key
        assert client.model == default_model
        assert client.base_url == base_url
        assert client._chat_path == "/chat/completions"
    finally:
        client.close()


def test_ollama_chat_url_joining_handles_trailing_slash_base_url():
    client = LLMClient.from_provider("ollama", base_url="http://127.0.0.1:11434/")
    calls: dict[str, str] = {}

    class _FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class _FakeHTTP:
        @staticmethod
        def post(url: str, **_: object) -> _FakeResponse:
            calls["url"] = url
            return _FakeResponse()

    client._http = _FakeHTTP()  # type: ignore[assignment]

    response = client.chat("hello")

    assert response.content == "ok"
    assert calls["url"] == "http://127.0.0.1:11434/v1/chat/completions"


def test_chat_stream_retries_transient_startup_429_and_then_succeeds():
    client = LLMClient.from_provider("ollama", base_url="http://127.0.0.1:11434/")
    client._retry_policy = RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0, jitter=False)

    class _FakeHTTP:
        def __init__(self) -> None:
            self.calls = 0

        @staticmethod
        def build_request(method: str, url: str, **_: object) -> httpx.Request:
            return httpx.Request(method, url)

        def send(self, request: httpx.Request, *, stream: bool = False) -> httpx.Response:
            assert stream is True
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(429, request=request, text="rate limited")
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "ok"},
                            "finish_reason": None,
                        }
                    ],
                    "model": "llama3.1",
                }
            )
            content = f"data: {payload}\n\ndata: [DONE]\n\n".encode()
            return httpx.Response(200, request=request, content=content)

    fake_http = _FakeHTTP()
    client._http = fake_http  # type: ignore[assignment]

    chunks = list(client.chat_stream("hello"))

    assert "".join(chunk.delta for chunk in chunks) == "ok"
    assert fake_http.calls == 2


def test_chat_stream_raises_when_no_parseable_chunks():
    client = LLMClient.from_provider("ollama", base_url="http://127.0.0.1:11434/")

    class _FakeHTTP:
        @staticmethod
        def build_request(method: str, url: str, **_: object) -> httpx.Request:
            return httpx.Request(method, url)

        @staticmethod
        def send(request: httpx.Request, *, stream: bool = False) -> httpx.Response:
            assert stream is True
            content = b"data: not-json\n\ndata: [DONE]\n\n"
            return httpx.Response(200, request=request, content=content)

    client._http = _FakeHTTP()  # type: ignore[assignment]

    try:
        list(client.chat_stream("hello"))
        raise AssertionError("Expected LLMAPIError for empty stream")
    except LLMAPIError as exc:
        assert exc.status_code == 0
        assert "no parseable chunks" in exc.detail.lower()
