# LLM module
from .client import (
    LLMClient,
    LLMResponse,
    LLMStreamChunk,
    LLMAPIError,
    LLMConfigError,
    DEFAULT_MODELS as LLM_DEFAULT_MODELS,
)
from .multi_model import (
    ModelProvider,
    ModelCapability,
    ModelConfig,
    ModelResponse,
    ModelRegistry,
    ModelRouter,
    ModelPool,
    create_router,
    estimate_cost,
)
from .rag_context import (
    DocumentChunk,
    RetrievedChunk,
    TextChunker,
    CodeAwareChunker,
    SimpleEmbedder,
    VectorStore,
    RAGContextBuilder,
    create_rag_context,
)
from .confidence import (
    ConfidenceLevel,
    Prediction,
    CalibrationMetrics,
    ConfidenceCalibrator,
    AdaptiveConfidence,
    UncertaintyEstimator,
    quick_confidence,
)
from .research_assistant import (
    LLMResearchAssistant,
    ExtractedSpec,
    CodeAnalysis,
    ImplementationPlan,
)

__all__ = [
    # Client
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMAPIError",
    "LLMConfigError",
    "LLM_DEFAULT_MODELS",
    # Multi-model
    "ModelProvider",
    "ModelCapability",
    "ModelConfig",
    "ModelResponse",
    "ModelRegistry",
    "ModelRouter",
    "ModelPool",
    "create_router",
    "estimate_cost",
    # RAG
    "DocumentChunk",
    "RetrievedChunk",
    "TextChunker",
    "CodeAwareChunker",
    "SimpleEmbedder",
    "VectorStore",
    "RAGContextBuilder",
    "create_rag_context",
    # Confidence
    "ConfidenceLevel",
    "Prediction",
    "CalibrationMetrics",
    "ConfidenceCalibrator",
    "AdaptiveConfidence",
    "UncertaintyEstimator",
    "quick_confidence",
    # Research Assistant
    "LLMResearchAssistant",
    "ExtractedSpec",
    "CodeAnalysis",
    "ImplementationPlan",
]
