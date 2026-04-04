from __future__ import annotations

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.llm.client import LLMClient


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


def test_ollama_chat_url_joining_handles_trailing_slash_base_url():
    client = LLMClient.from_provider("ollama", base_url="http://127.0.0.1:11434/")
    calls: dict[str, str] = {}

    class _FakeResponse:
        status_code = 200
        text = ""

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
