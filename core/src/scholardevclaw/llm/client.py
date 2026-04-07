"""
Unified LLM client supporting multiple providers.

Sends real HTTP requests to LLM APIs via httpx.  Each provider has its own
request/response format; the client normalises everything into a common
``LLMResponse`` dataclass.

Supported providers:
  - Anthropic (Messages API)
  - OpenAI / Azure OpenAI / DeepSeek / Groq / Together / Fireworks / OpenRouter
    (all use OpenAI-compatible chat completions)
  - Ollama (local, OpenAI-compatible endpoint)
  - Mistral (OpenAI-compatible with minor differences)
  - Cohere (Chat API v2)
  - GitHub Copilot (OpenAI-compatible with extra auth header)

Usage::

    from scholardevclaw.llm.client import LLMClient

    client = LLMClient.from_provider("anthropic", api_key="sk-ant-...")
    resp = client.chat("Explain RMSNorm in one paragraph.")
    print(resp.content)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.utils.retry import RetryPolicy, _extract_retry_after

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMStreamChunk:
    """A single chunk from a streaming response."""

    delta: str
    model: str = ""
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Provider-specific request builders
# ---------------------------------------------------------------------------


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _anthropic_body(
    messages: list[dict[str, str]],
    model: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    return body


def _openai_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _openai_body(
    messages: list[dict[str, str]],
    model: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    body: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return body


def _cohere_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _cohere_body(
    messages: list[dict[str, str]],
    model: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    system: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Cohere v2 Chat API body."""
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    return {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def _copilot_headers(api_key: str) -> dict[str, str]:
    """GitHub Copilot uses OpenAI-compatible format with a different auth header."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Editor-Version": "scholardevclaw/1.0",
    }


def _azure_headers(api_key: str) -> dict[str, str]:
    return {
        "api-key": api_key,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def _parse_anthropic(data: dict[str, Any]) -> tuple[str, int, int, str | None]:
    """Return (content, input_tokens, output_tokens, finish_reason)."""
    content_blocks = data.get("content", [])
    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
    content = "".join(text_parts)
    usage = data.get("usage", {})
    return (
        content,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        data.get("stop_reason"),
    )


def _parse_openai(data: dict[str, Any]) -> tuple[str, int, int, str | None]:
    """Parse OpenAI-compatible response (works for OpenAI, Groq, Together, etc.)."""
    choices = data.get("choices", [])
    content = ""
    finish_reason = None
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason")
    usage = data.get("usage", {})
    return (
        content,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        finish_reason,
    )


def _parse_cohere(data: dict[str, Any]) -> tuple[str, int, int, str | None]:
    """Parse Cohere v2 Chat response."""
    # v2 format: {"message": {"content": [{"text": "..."}]}, ...}
    message = data.get("message", {})
    content_blocks = message.get("content", [])
    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
    content = "".join(text_parts) if text_parts else data.get("text", "")
    usage = data.get("usage", data.get("meta", {}).get("tokens", {}))
    return (
        content,
        usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        usage.get("output_tokens", usage.get("completion_tokens", 0)),
        data.get("finish_reason"),
    )


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Maps AuthProvider -> (base_url, chat_path, header_builder, body_builder, response_parser)
_PROVIDER_CONFIG: dict[
    AuthProvider,
    tuple[
        str,  # default base_url
        str,  # chat endpoint path
        Any,  # header builder
        Any,  # body builder
        Any,  # response parser
    ],
] = {
    AuthProvider.ANTHROPIC: (
        "https://api.anthropic.com",
        "/v1/messages",
        _anthropic_headers,
        _anthropic_body,
        _parse_anthropic,
    ),
    AuthProvider.OPENAI: (
        "https://api.openai.com/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.OLLAMA: (
        "http://localhost:11434",
        "/v1/chat/completions",
        lambda _key: {"Content-Type": "application/json"},
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.GROQ: (
        "https://api.groq.com/openai/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.MISTRAL: (
        "https://api.mistral.ai/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.DEEPSEEK: (
        "https://api.deepseek.com",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.COHERE: (
        "https://api.cohere.com/v2",
        "/chat",
        _cohere_headers,
        _cohere_body,
        _parse_cohere,
    ),
    AuthProvider.OPENROUTER: (
        "https://openrouter.ai/api/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.TOGETHER: (
        "https://api.together.xyz/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.FIREWORKS: (
        "https://api.fireworks.ai/inference/v1",
        "/chat/completions",
        _openai_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.GITHUB_COPILOT: (
        "https://api.githubcopilot.com",
        "/chat/completions",
        _copilot_headers,
        _openai_body,
        _parse_openai,
    ),
    AuthProvider.AZURE_OPENAI: (
        "",  # user must supply base_url
        "/chat/completions",  # path includes deployment; caller overrides
        _azure_headers,
        _openai_body,
        _parse_openai,
    ),
}

# Default models per provider (used when caller doesn't specify)
DEFAULT_MODELS: dict[AuthProvider, str] = {
    AuthProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    AuthProvider.OPENAI: "gpt-4o",
    AuthProvider.OLLAMA: "llama3.1",
    AuthProvider.GROQ: "llama-3.1-70b-versatile",
    AuthProvider.MISTRAL: "mistral-large-latest",
    AuthProvider.DEEPSEEK: "deepseek-chat",
    AuthProvider.COHERE: "command-r-plus",
    AuthProvider.OPENROUTER: "openai/gpt-4.1-mini",
    AuthProvider.TOGETHER: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    AuthProvider.FIREWORKS: "accounts/fireworks/models/llama-v3p1-70b-instruct",
    AuthProvider.GITHUB_COPILOT: "gpt-4o",
    AuthProvider.AZURE_OPENAI: "gpt-4o",
}


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified client for all supported LLM providers.

    Parameters
    ----------
    provider : AuthProvider
        The LLM provider to use.
    api_key : str
        API key (empty string acceptable for Ollama).
    base_url : str | None
        Override the default base URL for the provider.
    model : str | None
        Override the default model for the provider.
    timeout : float
        HTTP request timeout in seconds (default 120).
    """

    def __init__(
        self,
        provider: AuthProvider,
        api_key: str = "",
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if provider not in _PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider.value}. "
                f"Supported: {', '.join(p.value for p in _PROVIDER_CONFIG)}"
            )

        self.provider = provider
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS.get(provider, "")
        self.timeout = timeout

        cfg = _PROVIDER_CONFIG[provider]
        self.base_url = (base_url or cfg[0]).rstrip("/")
        self._chat_path = cfg[1]
        self._build_headers = cfg[2]
        self._build_body = cfg[3]
        self._parse_response = cfg[4]

        self._http = httpx.Client(timeout=httpx.Timeout(timeout))

        # Retry policy for transient failures (429, 5xx, network errors)
        self._retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            retry_after=_extract_retry_after,
            on_retry=lambda exc, attempt, delay: logger.warning(
                "[%s] Retry attempt %d after %.1fs: %s",
                self.provider.value,
                attempt,
                delay,
                str(exc)[:100],
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat request and return a normalised ``LLMResponse``.

        You can pass either a simple ``prompt`` (converted to a single user
        message) or a full ``messages`` list.
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        used_model = model or self.model
        headers = self._build_headers(self.api_key)
        body = self._build_body(
            messages,
            used_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )

        url = self.base_url + self._chat_path
        # Azure OpenAI uses a different URL pattern with api-version
        if self.provider == AuthProvider.AZURE_OPENAI:
            url = f"{self.base_url}/openai/deployments/{used_model}/chat/completions?api-version=2024-02-01"

        t0 = time.monotonic()

        def _make_request() -> httpx.Response:
            response = self._http.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response

        try:
            resp = self._retry_policy.execute(_make_request)
        except Exception as exc:
            status_code = 0
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
            raise LLMAPIError(
                provider=self.provider.value,
                status_code=status_code,
                detail=str(exc)[:500],
            ) from exc

        latency = (time.monotonic() - t0) * 1000

        data = resp.json()
        content, in_tok, out_tok, finish = self._parse_response(data)

        return LLMResponse(
            content=content,
            model=used_model,
            provider=self.provider.value,
            input_tokens=in_tok,
            output_tokens=out_tok,
            finish_reason=finish,
            latency_ms=latency,
            raw=data,
        )

    def chat_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream a chat response, yielding ``LLMStreamChunk`` objects.

        Currently supported for Anthropic and OpenAI-compatible providers.
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        used_model = model or self.model
        headers = self._build_headers(self.api_key)
        body = self._build_body(
            messages,
            used_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )
        body["stream"] = True

        url = self.base_url + self._chat_path
        if self.provider == AuthProvider.AZURE_OPENAI:
            url = f"{self.base_url}/openai/deployments/{used_model}/chat/completions?api-version=2024-02-01"

        def _start_stream() -> httpx.Response:
            request = self._http.build_request("POST", url, headers=headers, json=body)
            response = self._http.send(request, stream=True)
            try:
                response.raise_for_status()
            except Exception:
                response.read()
                response.close()
                raise
            return response

        try:
            resp = self._retry_policy.execute(_start_stream)
        except Exception as exc:
            status_code = 0
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
            raise LLMAPIError(
                provider=self.provider.value,
                status_code=status_code,
                detail=str(exc)[:500],
            ) from exc

        try:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk
        finally:
            resp.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        """Ensure HTTP client is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_provider(
        cls,
        provider: str | AuthProvider,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create a client from a provider name or enum.

        If ``api_key`` is not provided, the client tries to read it from the
        provider's standard environment variable (e.g. ``ANTHROPIC_API_KEY``).
        """
        if isinstance(provider, str):
            provider = AuthProvider(provider)

        if provider == AuthProvider.OLLAMA:
            if api_key is None:
                api_key = ""
            if base_url is None:
                base_url = os.environ.get("OLLAMA_HOST", "").strip() or provider.default_base_url
        elif api_key is None:
            api_key = os.environ.get(provider.env_var_name, "")

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    @classmethod
    def from_auth_store(
        cls,
        provider: AuthProvider | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create a client using a key from the local AuthStore.

        Falls back to the default active key if no provider is specified.
        """
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore()
        config = store.get_config()
        key_obj = config.get_active_key(provider)
        if key_obj is None:
            raise LLMConfigError(
                "No API key found in auth store. Run 'scholardevclaw auth setup' to add one."
            )

        return cls(
            provider=key_obj.provider,
            api_key=key_obj.key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_stream_event(self, event: dict[str, Any]) -> LLMStreamChunk | None:
        """Parse a single SSE event into a stream chunk."""
        if self.provider == AuthProvider.ANTHROPIC:
            etype = event.get("type", "")
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                return LLMStreamChunk(
                    delta=delta.get("text", ""),
                    model=self.model,
                )
            if etype == "message_stop":
                return LLMStreamChunk(delta="", model=self.model, finish_reason="end_turn")
            return None

        # OpenAI-compatible SSE
        choices = event.get("choices", [])
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta", {})
        text = delta.get("content", "") or ""
        return LLMStreamChunk(
            delta=text,
            model=event.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMAPIError(Exception):
    """Raised when an LLM API returns a non-200 status."""

    def __init__(self, provider: str, status_code: int, detail: str = "") -> None:
        self.provider = provider
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{provider}] HTTP {status_code}: {detail}")


class LLMConfigError(Exception):
    """Raised for configuration problems (missing key, bad provider, etc.)."""
