from __future__ import annotations

from scholardevclaw.llm import research_assistant as module


def test_auto_detect_ollama_uses_ollama_host_for_probe_and_client(monkeypatch):
    for env_var, _provider in module._AUTO_DETECT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.internal:2244/")
    calls: dict[str, object] = {}

    class _Response:
        status_code = 200

    def _fake_get(url: str, timeout: float):
        calls["probe_url"] = url
        calls["probe_timeout"] = timeout
        return _Response()

    class _FakeLLMClient:
        @staticmethod
        def from_provider(provider: str, **kwargs: object):
            calls["provider"] = provider
            calls["kwargs"] = kwargs
            return "fake-client"

    monkeypatch.setattr("httpx.get", _fake_get)
    monkeypatch.setattr(module, "LLMClient", _FakeLLMClient)

    client = module._auto_detect_client(model="llama3.2")

    assert client == "fake-client"
    assert calls["probe_url"] == "http://ollama.internal:2244/api/tags"
    assert calls["provider"] == "ollama"
    assert calls["kwargs"] == {
        "api_key": "",
        "base_url": "http://ollama.internal:2244/",
        "model": "llama3.2",
    }
