"""
Multi-model support for flexible LLM selection.

Provides:
- Model registry and configuration
- Provider abstraction (Anthropic, OpenAI, Google, etc.)
- Automatic failover
- Cost optimization
- Request routing
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ModelProvider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Model capabilities"""

    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    FUNCTION = "function"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"


@dataclass
class ModelConfig:
    """Configuration for a model"""

    id: str
    name: str
    provider: ModelProvider
    model_id: str  # provider-specific model ID
    max_tokens: int = 100000
    context_window: int = 200000
    capabilities: list[ModelCapability] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_estimate_ms: int = 1000
    reliability: float = 1.0  # 0-1
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from a model"""

    content: str
    model_id: str
    provider: ModelProvider
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    raw_response: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry of available models"""

    DEFAULT_MODELS: list[ModelConfig] = [
        ModelConfig(
            id="claude-opus",
            name="Claude 4 Opus",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-opus-4-20250514",
            max_tokens=320000,
            context_window=200000,
            capabilities=[
                ModelCapability.TEXT,
                ModelCapability.CODE,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
                ModelCapability.LONG_CONTEXT,
            ],
            cost_per_1k_input=15.0,
            cost_per_1k_output=75.0,
            latency_estimate_ms=2000,
            reliability=0.95,
        ),
        ModelConfig(
            id="claude-sonnet",
            name="Claude 4 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            max_tokens=320000,
            context_window=200000,
            capabilities=[
                ModelCapability.TEXT,
                ModelCapability.CODE,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
            ],
            cost_per_1k_input=3.0,
            cost_per_1k_output=15.0,
            latency_estimate_ms=1500,
            reliability=0.98,
        ),
        ModelConfig(
            id="gpt-4o",
            name="GPT-4o",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o-2024-05-13",
            max_tokens=128000,
            context_window=128000,
            capabilities=[
                ModelCapability.TEXT,
                ModelCapability.CODE,
                ModelCapability.VISION,
                ModelCapability.FUNCTION,
                ModelCapability.STREAMING,
            ],
            cost_per_1k_input=5.0,
            cost_per_1k_output=15.0,
            latency_estimate_ms=1500,
            reliability=0.97,
        ),
        ModelConfig(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4-turbo-2024-04-09",
            max_tokens=128000,
            context_window=128000,
            capabilities=[
                ModelCapability.TEXT,
                ModelCapability.CODE,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
            ],
            cost_per_1k_input=10.0,
            cost_per_1k_output=30.0,
            latency_estimate_ms=1800,
            reliability=0.95,
        ),
        ModelConfig(
            id="gemini-pro",
            name="Gemini 1.5 Pro",
            provider=ModelProvider.GOOGLE,
            model_id="gemini-1.5-pro",
            max_tokens=128000,
            context_window=2000000,
            capabilities=[
                ModelCapability.TEXT,
                ModelCapability.CODE,
                ModelCapability.VISION,
                ModelCapability.LONG_CONTEXT,
            ],
            cost_per_1k_input=1.25,
            cost_per_1k_output=5.0,
            latency_estimate_ms=2000,
            reliability=0.93,
        ),
    ]

    def __init__(self):
        self.models: dict[str, ModelConfig] = {}
        for model in self.DEFAULT_MODELS:
            self.models[model.id] = model

    def register(self, config: ModelConfig):
        """Register a new model"""
        self.models[config.id] = config

    def get(self, model_id: str) -> ModelConfig | None:
        """Get a model by ID"""
        return self.models.get(model_id)

    def list_models(
        self,
        provider: ModelProvider | None = None,
        capability: ModelCapability | None = None,
        enabled_only: bool = True,
    ) -> list[ModelConfig]:
        """List models with filters"""
        models = list(self.models.values())

        if enabled_only:
            models = [m for m in models if m.enabled]

        if provider:
            models = [m for m in models if m.provider == provider]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        return models

    def find_cheapest(self, capability: ModelCapability | None = None) -> ModelConfig | None:
        """Find cheapest model"""
        models = self.list_models(capability=capability)
        if not models:
            return None
        return min(models, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

    def find_fastest(self, capability: ModelCapability | None = None) -> ModelConfig | None:
        """Find fastest model"""
        models = self.list_models(capability=capability)
        if not models:
            return None
        return min(models, key=lambda m: m.latency_estimate_ms)

    def find_most_reliable(self, capability: ModelCapability | None = None) -> ModelConfig | None:
        """Find most reliable model"""
        models = self.list_models(capability=capability)
        if not models:
            return None
        return max(models, key=lambda m: m.reliability)


class ModelRouter:
    """Route requests to appropriate models"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.fallback_chain: list[str] = []

    def set_fallback_chain(self, model_ids: list[str]):
        """Set fallback chain"""
        self.fallback_chain = model_ids

    def select_model(
        self,
        requirements: dict[str, Any] | None = None,
    ) -> ModelConfig | None:
        """Select best model based on requirements"""
        requirements = requirements or {}

        capability = requirements.get("capability")
        prefer_cheap = requirements.get("prefer_cheap", False)
        prefer_fast = requirements.get("prefer_fast", False)
        prefer_reliable = requirements.get("prefer_reliable", False)
        min_reliability = requirements.get("min_reliability", 0.0)

        models = self.registry.list_models(capability=capability)
        models = [m for m in models if m.reliability >= min_reliability]

        if not models:
            return None

        if prefer_cheap:
            return min(models, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
        elif prefer_fast:
            return min(models, key=lambda m: m.latency_estimate_ms)
        elif prefer_reliable:
            return max(models, key=lambda m: m.reliability)
        else:
            return models[0]

    def execute_with_fallback(
        self,
        prompt: str,
        executor: Callable[[ModelConfig], ModelResponse],
        requirements: dict[str, Any] | None = None,
    ) -> ModelResponse | None:
        """Execute with automatic fallback"""
        model = self.select_model(requirements)
        if not model:
            return None

        try:
            return executor(model)
        except Exception as e:
            if self.fallback_chain:
                for fallback_id in self.fallback_chain:
                    fallback = self.registry.get(fallback_id)
                    if fallback:
                        try:
                            return executor(fallback)
                        except Exception:
                            continue
            raise e


class ModelPool:
    """Pool of models for load balancing"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.usage_counts: dict[str, int] = {}

    def get_next_model(self, model_ids: list[str]) -> ModelConfig | None:
        """Get next model using round-robin"""
        available = [m for m in model_ids if self.registry.get(m)]

        if not available:
            return None

        for m_id in available:
            if m_id not in self.usage_counts:
                self.usage_counts[m_id] = 0

        selected = min(available, key=lambda m: self.usage_counts.get(m, 0))
        self.usage_counts[selected] += 1

        return self.registry.get(selected)

    def reset_counts(self):
        """Reset usage counts"""
        self.usage_counts = {}


def create_router() -> ModelRouter:
    """Create a model router with default models"""
    registry = ModelRegistry()
    router = ModelRouter(registry)
    router.set_fallback_chain(["claude-sonnet", "gpt-4o", "gemini-pro"])
    return router


def estimate_cost(
    model: ModelConfig,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a request"""
    input_cost = (input_tokens / 1000) * model.cost_per_1k_input
    output_cost = (output_tokens / 1000) * model.cost_per_1k_output
    return input_cost + output_cost
