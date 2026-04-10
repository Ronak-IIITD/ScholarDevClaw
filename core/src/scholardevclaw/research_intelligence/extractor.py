"""
Research paper extraction and specification lookup.

This module provides the :class:`ResearchExtractor` which:

- Maintains a dynamic registry of paper specs (seeded from ``PAPER_SPECS``).
- Searches the local registry by keyword, code pattern, or algorithm name.
- Queries arXiv for papers matching a :class:`ResearchQuery`.
- Extracts implementation specs from PDFs / arXiv IDs using an optional
  :class:`~scholardevclaw.llm.research_assistant.LLMResearchAssistant`.
- Dynamically discovers and registers new specs via arXiv + LLM when
  local results are insufficient.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from scholardevclaw.utils.retry import retry

if TYPE_CHECKING:
    from scholardevclaw.llm.research_assistant import LLMResearchAssistant

logger = logging.getLogger(__name__)


def _is_allowed_arxiv_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() == "export.arxiv.org"


# ---------------------------------------------------------------------------
# Spec registry — seeds the dynamic registry at construction time
# ---------------------------------------------------------------------------

PAPER_SPECS: dict[str, dict] = {
    # --- Normalization ---
    "rmsnorm": {
        "paper": {
            "title": "Root Mean Square Layer Normalization",
            "authors": ["Biao Zhang", "Rico Sennrich"],
            "arxiv": "1910.07467",
            "year": 2019,
        },
        "algorithm": {
            "name": "RMSNorm",
            "replaces": "LayerNorm",
            "description": "Simplified layer normalization without mean-centering",
            "formula": "x / sqrt(mean(x^2) + eps) * gamma",
            "complexity": "O(n)",
            "category": "normalization",
        },
        "implementation": {
            "module_name": "RMSNorm",
            "parent_class": "nn.Module",
            "parameters": ["ndim", "eps"],
            "forward_signature": "(x: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["LayerNorm", "nn.LayerNorm"],
            "replacement": "RMSNorm",
            "insertion_points": ["Block class", "GPT class"],
            "expected_benefits": [
                "5-10% training speedup",
                "Simplified computation",
                "Better gradient flow",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity", "tokens_per_second", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    "preln_transformer": {
        "paper": {
            "title": "On Layer Normalization in the Transformer Architecture",
            "authors": ["Ruibin Xiong", "Yunchang Yang", "et al."],
            "arxiv": "2002.04745",
            "year": 2020,
        },
        "algorithm": {
            "name": "Pre-LN Transformer",
            "replaces": "Post-LN Transformer",
            "description": "Moves layer normalization before the attention and FFN sub-layers for more stable training",
            "category": "normalization",
        },
        "implementation": {
            "module_name": "PreLNBlock",
            "parent_class": "nn.Module",
            "parameters": ["config"],
        },
        "changes": {
            "type": "reorder",
            "target_patterns": ["LayerNorm", "self.ln_1", "self.ln_2"],
            "replacement": "Pre-LN ordering",
            "insertion_points": ["Block class"],
            "expected_benefits": [
                "More stable training without learning rate warmup",
                "Enables deeper models",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    "qknorm": {
        "paper": {
            "title": "Scaling Vision Transformers to 22 Billion Parameters",
            "authors": ["Mostafa Dehghani", "Josip Djolonga", "et al."],
            "arxiv": "2302.05442",
            "year": 2023,
        },
        "algorithm": {
            "name": "QK-Norm",
            "replaces": "Standard attention (no Q/K normalization)",
            "description": "Applies layer normalization to query and key projections before dot product, stabilising large-scale training",
            "category": "normalization",
        },
        "implementation": {
            "module_name": "QKNormAttention",
            "parent_class": "nn.Module",
            "parameters": ["config"],
        },
        "changes": {
            "type": "augment",
            "target_patterns": ["CausalSelfAttention", "self.c_attn", "self.q_proj", "self.k_proj"],
            "replacement": "QK-Norm attention",
            "insertion_points": ["CausalSelfAttention class", "Attention class"],
            "expected_benefits": [
                "Training stability at scale",
                "Prevents attention logit growth",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    # --- Activation / FFN ---
    "swiglu": {
        "paper": {
            "title": "GLU Variants Improve Transformer",
            "authors": ["Noam Shazeer"],
            "arxiv": "2002.05202",
            "year": 2020,
        },
        "algorithm": {
            "name": "SwiGLU",
            "replaces": "MLP (GELU)",
            "description": "Swish-Gated Linear Unit — combines Swish activation with gated linear units for better FFN quality",
            "formula": "Swish(xW) * (xV)",
            "category": "activation",
        },
        "implementation": {
            "module_name": "SwiGLU",
            "parent_class": "nn.Module",
            "parameters": ["config"],
            "forward_signature": "(x: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class MLP", "self.gelu = nn.GELU()", "nn.GELU", "GELU"],
            "replacement": "class SwiGLU",
            "insertion_points": ["MLP class"],
            "expected_benefits": ["Improved FFN quality", "Better gradient flow"],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    "geglu": {
        "paper": {
            "title": "GLU Variants Improve Transformer",
            "authors": ["Noam Shazeer"],
            "arxiv": "2002.05202",
            "year": 2020,
        },
        "algorithm": {
            "name": "GEGLU",
            "replaces": "MLP (GELU)",
            "description": "GELU-Gated Linear Unit — gated variant of GELU activation",
            "formula": "GELU(xW) * (xV)",
            "category": "activation",
        },
        "implementation": {
            "module_name": "GEGLU",
            "parent_class": "nn.Module",
            "parameters": ["config"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class MLP", "nn.GELU", "GELU"],
            "replacement": "GEGLU",
            "insertion_points": ["MLP class"],
            "expected_benefits": ["Improved FFN quality"],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    # --- Attention ---
    "flashattention": {
        "paper": {
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "authors": [
                "Tri Dao",
                "Daniel Y. Fu",
                "Stefano Ermon",
                "Christopher Ré",
                "Zachary C. Brown",
            ],
            "arxiv": "2205.14135",
            "year": 2022,
        },
        "algorithm": {
            "name": "FlashAttention",
            "replaces": "CausalSelfAttention (slow)",
            "description": "IO-aware exact attention algorithm that reduces memory complexity from O(N^2) to O(N)",
            "formula": "Softmax(QK^T / sqrt(d))V with tiling and recomputation",
            "category": "attention",
        },
        "implementation": {
            "module_name": "FlashAttention",
            "parent_class": "nn.Module",
            "parameters": ["config"],
            "forward_signature": "(q, k, v: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class CausalSelfAttention", "self.flash"],
            "replacement": "FlashAttention",
            "insertion_points": ["CausalSelfAttention class"],
            "expected_benefits": ["2-4x speedup", "Reduced memory usage", "IO-awareness"],
        },
        "validation": {
            "test_type": "benchmark",
            "metrics": ["tokens_per_second", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    "flashattention2": {
        "paper": {
            "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
            "authors": ["Tri Dao"],
            "arxiv": "2307.08691",
            "year": 2023,
        },
        "algorithm": {
            "name": "FlashAttention-2",
            "replaces": "FlashAttention / standard attention",
            "description": "Improved FlashAttention with better work partitioning, reduced non-matmul FLOPs, and support for head dimensions up to 256",
            "category": "attention",
        },
        "implementation": {
            "module_name": "FlashAttention2",
            "parent_class": "nn.Module",
            "parameters": ["config"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["CausalSelfAttention", "flash_attn", "attention"],
            "replacement": "FlashAttention2",
            "insertion_points": ["CausalSelfAttention class", "Attention class"],
            "expected_benefits": ["1.5-2x over FlashAttention-1", "Better GPU utilization"],
        },
        "validation": {
            "test_type": "benchmark",
            "metrics": ["tokens_per_second", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    "grouped_query_attention": {
        "paper": {
            "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
            "authors": ["Joshua Ainslie", "James Lee-Thorp", "et al."],
            "arxiv": "2305.13245",
            "year": 2023,
        },
        "algorithm": {
            "name": "Grouped-Query Attention",
            "replaces": "Multi-Head Attention / Multi-Query Attention",
            "description": "Interpolates between multi-head and multi-query attention, sharing KV heads across groups of query heads to reduce KV-cache size",
            "category": "attention",
        },
        "implementation": {
            "module_name": "GroupedQueryAttention",
            "parent_class": "nn.Module",
            "parameters": ["config", "n_kv_heads"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": [
                "CausalSelfAttention",
                "MultiHeadAttention",
                "self.n_head",
                "self.c_attn",
            ],
            "replacement": "GroupedQueryAttention",
            "insertion_points": ["CausalSelfAttention class", "Attention class"],
            "expected_benefits": [
                "Reduced KV-cache memory",
                "Faster inference with minimal quality loss",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    # --- Positional encoding ---
    "rope": {
        "paper": {
            "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
            "authors": [
                "Jianlin Su",
                "Yu Lu",
                "Shengfeng Pan",
                "Ahmed Murtadha",
                "Bohan Zhou",
                "Yunfeng Liu",
            ],
            "arxiv": "2104.09864",
            "year": 2021,
        },
        "algorithm": {
            "name": "RoPE",
            "replaces": "Positional Encoding",
            "description": "Rotary Position Embedding — encodes position information through rotation matrices",
            "formula": "RoPE(x_m, x_n) = f(x_m, m) * f(x_n, n)^conj where f(x, k) = x * e^(ikθ)",
            "category": "position_encoding",
        },
        "implementation": {
            "module_name": "RoPE",
            "parent_class": "nn.Module",
            "parameters": ["dim", "max_seq_len"],
            "forward_signature": "(x: Tensor, positions: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["self.wpe", "nn.Embedding"],
            "replacement": "RoPE",
            "insertion_points": ["GPT class __init__", "GPT class forward"],
            "expected_benefits": [
                "Better length extrapolation",
                "Improved long-range dependencies",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    "alibi": {
        "paper": {
            "title": "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization",
            "authors": ["Ofir Press", "Noah A. Smith", "Mike Lewis"],
            "arxiv": "2108.12409",
            "year": 2021,
        },
        "algorithm": {
            "name": "ALiBi",
            "replaces": "Positional Encoding / learned embeddings",
            "description": "Replaces positional embeddings with linear biases added to attention scores, enabling length generalization",
            "category": "position_encoding",
        },
        "implementation": {
            "module_name": "ALiBi",
            "parent_class": "nn.Module",
            "parameters": ["n_heads"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["self.wpe", "nn.Embedding", "positional"],
            "replacement": "ALiBi",
            "insertion_points": ["GPT class", "Attention class"],
            "expected_benefits": [
                "Length generalization without fine-tuning",
                "No learned position parameters",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    # --- Optimiser ---
    "weight_decay_fused": {
        "paper": {
            "title": "Decoupled Weight Decay Regularization",
            "authors": ["Ilya Loshchilov", "Frank Hutter"],
            "arxiv": "1711.05101",
            "year": 2019,
        },
        "algorithm": {
            "name": "AdamW",
            "replaces": "Adam with L2 regularization",
            "description": "Decoupled weight decay that regularises weights directly instead of adding L2 to loss",
            "category": "optimizer",
        },
        "implementation": {
            "module_name": "AdamW",
            "parent_class": "torch.optim.Optimizer",
            "parameters": ["lr", "betas", "weight_decay"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["Adam(", "torch.optim.Adam", "optim.Adam"],
            "replacement": "AdamW",
            "insertion_points": ["configure_optimizers"],
            "expected_benefits": [
                "Better generalisation",
                "Correct weight decay behavior",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    "lion": {
        "paper": {
            "title": "Symbolic Discovery of Optimization Algorithms",
            "authors": ["Xiangning Chen", "Chen Liang", "Da Huang", "et al."],
            "arxiv": "2302.06675",
            "year": 2023,
        },
        "algorithm": {
            "name": "Lion",
            "replaces": "AdamW",
            "description": "EvoLved Sign Momentum optimiser — uses sign of momentum for updates, requiring less memory than Adam",
            "category": "optimizer",
        },
        "implementation": {
            "module_name": "Lion",
            "parent_class": "torch.optim.Optimizer",
            "parameters": ["lr", "betas", "weight_decay"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["AdamW", "torch.optim.AdamW", "optim.AdamW", "Adam"],
            "replacement": "Lion",
            "insertion_points": ["configure_optimizers", "training setup"],
            "expected_benefits": [
                "Lower memory usage (no second moment)",
                "Competitive or better quality",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    # --- Architecture / Mixture of Experts ---
    "mistral": {
        "paper": {
            "title": "Mixtral of Experts",
            "authors": ["Albert Q. Jiang", "Alexandre Sablayrolles", "et al."],
            "arxiv": "2401.04088",
            "year": 2024,
        },
        "algorithm": {
            "name": "Mixture of Experts",
            "replaces": "Dense MLP / feedforward",
            "description": "Sparse mixture-of-experts layer that routes tokens to a subset of expert FFN blocks",
            "category": "architecture",
        },
        "implementation": {
            "module_name": "MoELayer",
            "parent_class": "nn.Module",
            "parameters": ["config", "n_experts", "top_k"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class MLP", "feedforward", "self.mlp"],
            "replacement": "MoELayer",
            "insertion_points": ["MLP class", "Block class"],
            "expected_benefits": [
                "Higher capacity at constant compute",
                "Better scaling",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity", "tokens_per_second"],
            "max_benchmark_time": 600,
        },
    },
    # --- Learning rate scheduling ---
    "cosine_warmup": {
        "paper": {
            "title": "SGDR: Stochastic Gradient Descent with Warm Restarts",
            "authors": ["Ilya Loshchilov", "Frank Hutter"],
            "arxiv": "1608.03983",
            "year": 2017,
        },
        "algorithm": {
            "name": "Cosine Annealing with Warmup",
            "replaces": "Step LR / constant LR",
            "description": "Cosine learning rate schedule with optional linear warmup and warm restarts",
            "category": "scheduler",
        },
        "implementation": {
            "module_name": "CosineAnnealingWarmup",
            "parent_class": "torch.optim.lr_scheduler._LRScheduler",
            "parameters": ["optimizer", "warmup_steps", "max_steps"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["learning_rate", "lr_schedule", "get_lr", "StepLR", "MultiStepLR"],
            "replacement": "CosineAnnealingWarmup",
            "insertion_points": ["training loop", "configure_optimizers"],
            "expected_benefits": [
                "Smoother convergence",
                "Better final performance",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss"],
            "max_benchmark_time": 300,
        },
    },
    # --- Regularisation ---
    "dropout_variants": {
        "paper": {
            "title": "DropBlock: A regularization method for convolutional networks",
            "authors": ["Golnaz Ghiasi", "Tsung-Yi Lin", "Quoc V. Le"],
            "arxiv": "1810.12890",
            "year": 2018,
        },
        "algorithm": {
            "name": "DropBlock / DropPath",
            "replaces": "Standard Dropout",
            "description": "Structured dropout that drops contiguous regions or entire residual paths",
            "category": "regularization",
        },
        "implementation": {
            "module_name": "DropPath",
            "parent_class": "nn.Module",
            "parameters": ["drop_prob"],
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["nn.Dropout", "Dropout", "self.drop"],
            "replacement": "DropPath",
            "insertion_points": ["Block class", "MLP class"],
            "expected_benefits": [
                "Better regularization for transformers",
                "Enables deeper models",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """Represents a research paper"""

    id: str
    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str | None = None
    pdf_url: str | None = None
    published: str | None = None
    categories: list[str] = field(default_factory=list)
    year: int = 2024


@dataclass
class ResearchQuery:
    """Search query for research papers"""

    keywords: list[str]
    domain: str = "cs.AI"
    max_results: int = 10
    year_from: int | None = None


@dataclass
class ImplementationSpec:
    """Specification for implementing a paper"""

    paper: Paper
    target_language: str
    target_patterns: list[str]
    code_template: str
    expected_benefits: list[str]
    risks: list[str] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# PDF text extraction helper
# ---------------------------------------------------------------------------


def _read_pdf_text(pdf_path: str, max_pages: int = 20) -> str | None:
    """Best-effort plain-text extraction from a PDF file.

    Tries ``PyPDF2`` (commonly available) then ``pdfminer.six``.
    Returns ``None`` when no reader is installed or the file is unreadable.
    """
    path = Path(pdf_path)
    if not path.exists() or not path.suffix.lower() == ".pdf":
        return None

    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages]
        text_parts = [page.extract_text() or "" for page in pages]
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text as pm_extract  # type: ignore[import-untyped]

        text = pm_extract(str(path), maxpages=max_pages)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return None


def _fetch_arxiv_abstract(arxiv_id: str) -> str | None:
    """Fetch the abstract for an arXiv paper via the Atom feed API.

    This is a lightweight synchronous call — no heavy ``arxiv`` library needed.
    """
    clean_id = arxiv_id.strip().split("/")[-1]  # handle full URLs too
    try:
        import httpx

        url = f"https://export.arxiv.org/api/query?id_list={clean_id}"
        if not _is_allowed_arxiv_url(url):
            return None

        @retry(max_attempts=2, base_delay=1.0, max_delay=10.0)
        def _fetch_abstract() -> httpx.Response:
            return httpx.get(url, timeout=5.0, follow_redirects=False)

        resp = _fetch_abstract()
        if resp.status_code != 200:
            return None

        # Quick XML extraction (no lxml dependency)
        match = re.search(r"<summary[^>]*>(.*?)</summary>", resp.text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up whitespace artefacts
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract
    except Exception as exc:
        logger.debug("arXiv abstract fetch failed for %s: %s", arxiv_id, exc)

    return None


def _fetch_arxiv_papers(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search arXiv via the Atom API and return basic paper metadata.

    This is a lightweight synchronous helper that does NOT require the
    ``arxiv`` Python package — it uses httpx + regex XML extraction.
    Returns a list of dicts with ``title``, ``abstract``, ``arxiv_id``,
    ``authors``, ``year``, and ``published``.
    """
    try:
        import httpx
    except ImportError:
        return []

    clean_query = re.sub(r"[^\w\s]", " ", query).strip()
    if not clean_query:
        return []

    params = {
        "search_query": f"all:{clean_query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        import httpx

        @retry(max_attempts=2, base_delay=1.0, max_delay=10.0)
        def _search_arxiv() -> httpx.Response:
            return httpx.get(
                "https://export.arxiv.org/api/query",
                params=params,
                timeout=5.0,
                follow_redirects=False,
            )

        resp = _search_arxiv()
        if resp.status_code != 200:
            return []
    except Exception as exc:
        logger.debug("arXiv search failed for '%s': %s", query, exc)
        return []

    xml = resp.text
    entries: list[dict[str, Any]] = []

    for entry_match in re.finditer(r"<entry>(.*?)</entry>", xml, re.DOTALL):
        entry_xml = entry_match.group(1)

        title_m = re.search(r"<title[^>]*>(.*?)</title>", entry_xml, re.DOTALL)
        summary_m = re.search(r"<summary[^>]*>(.*?)</summary>", entry_xml, re.DOTALL)
        id_m = re.search(r"<id>(.*?)</id>", entry_xml)
        published_m = re.search(r"<published>(.*?)</published>", entry_xml)
        authors = re.findall(r"<name>(.*?)</name>", entry_xml)

        if not title_m:
            continue

        title = re.sub(r"\s+", " ", title_m.group(1)).strip()
        abstract = re.sub(r"\s+", " ", summary_m.group(1)).strip() if summary_m else ""
        entry_id = id_m.group(1).strip() if id_m else ""
        published = published_m.group(1).strip() if published_m else ""

        # Extract arXiv ID from the entry URL
        arxiv_id = ""
        aid_m = re.search(r"abs/(\d+\.\d+)", entry_id)
        if aid_m:
            arxiv_id = aid_m.group(1)

        year = 2024
        if published:
            yr_m = re.match(r"(\d{4})", published)
            if yr_m:
                year = int(yr_m.group(1))

        entries.append(
            {
                "title": title,
                "abstract": abstract,
                "arxiv_id": arxiv_id,
                "authors": authors,
                "year": year,
                "published": published,
            }
        )

    return entries


# ---------------------------------------------------------------------------
# ResearchExtractor
# ---------------------------------------------------------------------------


class ResearchExtractor:
    """Handles research paper extraction, search, and dynamic spec discovery.

    Parameters
    ----------
    llm_assistant : LLMResearchAssistant | None
        Optional LLM assistant for AI-powered spec extraction.
        When provided, extraction and search methods will attempt
        LLM-based analysis before falling back to hardcoded data.
    """

    def __init__(
        self,
        llm_assistant: LLMResearchAssistant | None = None,
    ) -> None:
        # Deep copy the seed data so each instance has its own mutable registry
        self.specs: dict[str, dict] = {k: dict(v) for k, v in PAPER_SPECS.items()}
        self._arxiv_client = None
        self._llm = llm_assistant

    def _get_arxiv_client(self):
        """Lazy load arxiv client"""
        if self._arxiv_client is None:
            try:
                import arxiv

                self._arxiv_client = arxiv.Client()
            except ImportError:
                return None
        return self._arxiv_client

    # ------------------------------------------------------------------
    # arXiv search (async, via the ``arxiv`` package when available)
    # ------------------------------------------------------------------

    async def search_arxiv(self, query: ResearchQuery) -> list[Paper]:
        """Search arXiv for papers matching the query"""
        client = self._get_arxiv_client()

        if client is None:
            # Fall back to the lightweight HTTP helper
            return self._search_arxiv_http(query)

        try:
            import arxiv  # noqa: F811 — lazy import matching _get_arxiv_client

            search_query = " AND ".join(query.keywords)
            search = arxiv.Search(
                query=search_query,
                max_results=query.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in client.results(search):
                papers.append(
                    Paper(
                        id=result.entry_id,
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        abstract=result.summary,
                        arxiv_id=result.get_short_id(),
                        pdf_url=result.pdf_url,
                        published=str(result.published) if result.published else None,
                        categories=result.categories,
                        year=result.published.year if result.published else 2024,
                    )
                )

            return papers
        except Exception as e:
            logger.warning("arXiv search error: %s", e)
            return []

    def _search_arxiv_http(self, query: ResearchQuery) -> list[Paper]:
        """Fallback arXiv search using the lightweight HTTP helper."""
        raw = _fetch_arxiv_papers(
            " ".join(query.keywords),
            max_results=query.max_results,
        )
        papers: list[Paper] = []
        for entry in raw:
            papers.append(
                Paper(
                    id=entry.get("arxiv_id", ""),
                    title=entry.get("title", ""),
                    authors=entry.get("authors", []),
                    abstract=entry.get("abstract", ""),
                    arxiv_id=entry.get("arxiv_id"),
                    published=entry.get("published"),
                    year=entry.get("year", 2024),
                )
            )
        return papers

    # ------------------------------------------------------------------
    # Keyword search (local registry)
    # ------------------------------------------------------------------

    def search_by_keyword(
        self,
        keyword: str,
        max_results: int = 10,
        *,
        include_arxiv: bool = False,
    ) -> list[dict]:
        """Search papers by keyword in the local registry.

        When *include_arxiv* is True **and** local results are fewer than
        *max_results*, the lightweight arXiv HTTP API is used to discover
        additional papers (which are also auto-registered as specs when an
        LLM is available).
        """
        results = self._search_local(keyword, max_results)

        if include_arxiv and len(results) < max_results:
            remaining = max_results - len(results)
            arxiv_papers = _fetch_arxiv_papers(keyword, max_results=remaining)
            seen_titles = {r.get("title", "").lower() for r in results}

            for entry in arxiv_papers:
                if entry["title"].lower() in seen_titles:
                    continue

                # Try to auto-register via LLM
                spec = self._try_register_arxiv_paper(entry)
                if spec:
                    results.append(
                        {
                            "name": _spec_key(spec),
                            "title": spec["paper"].get("title", entry["title"]),
                            "authors": spec["paper"].get("authors", entry.get("authors", [])),
                            "arxiv": entry.get("arxiv_id", ""),
                            "year": entry.get("year", 2024),
                            "category": spec.get("algorithm", {}).get("category", "unknown"),
                            "replaces": spec.get("algorithm", {}).get("replaces", ""),
                            "description": spec.get("algorithm", {}).get("description", ""),
                            "source": "arxiv_discovery",
                        }
                    )
                else:
                    # Even without LLM, return the arXiv hit as a result
                    results.append(
                        {
                            "name": re.sub(r"\W+", "_", entry["title"][:40]).strip("_").lower(),
                            "title": entry["title"],
                            "authors": entry.get("authors", []),
                            "arxiv": entry.get("arxiv_id", ""),
                            "year": entry.get("year", 2024),
                            "category": "unknown",
                            "replaces": "",
                            "description": entry.get("abstract", "")[:200],
                            "source": "arxiv_search",
                        }
                    )

                if len(results) >= max_results:
                    break

        return results[:max_results]

    def _search_local(self, query: str, max_results: int = 10) -> list[dict]:
        """Search across all fields in the local spec registry."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results: list[dict] = []

        for name, spec in self.specs.items():
            paper = spec.get("paper", {})
            algorithm = spec.get("algorithm", {})
            changes = spec.get("changes", {})

            # Collect searchable text
            searchable = " ".join(
                [
                    name,
                    paper.get("title", ""),
                    algorithm.get("name", ""),
                    algorithm.get("category", ""),
                    algorithm.get("replaces", ""),
                    algorithm.get("description", ""),
                    " ".join(changes.get("target_patterns", [])),
                    changes.get("replacement", ""),
                ]
            ).lower()

            # Check for any word match or substring match
            if query_lower in searchable or any(w in searchable for w in query_words):
                results.append(
                    {
                        "name": name,
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "arxiv": paper.get("arxiv", ""),
                        "year": paper.get("year", 2024),
                        "category": algorithm.get("category", ""),
                        "replaces": algorithm.get("replaces", ""),
                        "description": algorithm.get("description", ""),
                    }
                )

                if len(results) >= max_results:
                    break

        return results

    # ------------------------------------------------------------------
    # Code pattern → paper matching
    # ------------------------------------------------------------------

    def find_papers_for_code_pattern(
        self, code_pattern: str, language: str = "python"
    ) -> list[dict]:
        """Find papers relevant to a code pattern.

        Uses a broad keyword → spec mapping, then falls back to LLM-based
        code analysis when the local lookup returns nothing.
        """
        pattern_lower = code_pattern.lower()

        # Broad keyword → spec mapping (expanded from 4 → 16 specs)
        mapping: dict[str, list[str]] = {
            # Normalization
            "norm": ["rmsnorm", "preln_transformer", "qknorm"],
            "normalization": ["rmsnorm", "preln_transformer", "qknorm"],
            "layer": ["rmsnorm", "preln_transformer"],
            "layernorm": ["rmsnorm", "preln_transformer", "qknorm"],
            "batchnorm": ["rmsnorm"],
            "rms": ["rmsnorm"],
            # Attention
            "attention": [
                "flashattention",
                "flashattention2",
                "grouped_query_attention",
                "rope",
                "qknorm",
            ],
            "self_attention": ["flashattention", "flashattention2", "grouped_query_attention"],
            "causalselfattention": ["flashattention", "flashattention2", "grouped_query_attention"],
            "multihead": ["grouped_query_attention"],
            "flash": ["flashattention", "flashattention2"],
            "kv_cache": ["grouped_query_attention"],
            # Activation / FFN
            "mlp": ["swiglu", "geglu", "mistral"],
            "feedforward": ["swiglu", "geglu", "mistral"],
            "gelu": ["swiglu", "geglu"],
            "relu": ["swiglu", "geglu"],
            "activation": ["swiglu", "geglu"],
            "ffn": ["swiglu", "geglu", "mistral"],
            # Positional encoding
            "position": ["rope", "alibi"],
            "positional": ["rope", "alibi"],
            "embedding": ["rope", "alibi"],
            "wpe": ["rope", "alibi"],
            "sinusoidal": ["rope", "alibi"],
            # Optimizer
            "optimizer": ["weight_decay_fused", "lion"],
            "adam": ["weight_decay_fused", "lion"],
            "adamw": ["lion"],
            "sgd": ["weight_decay_fused", "cosine_warmup"],
            "weight_decay": ["weight_decay_fused"],
            "learning_rate": ["cosine_warmup"],
            "lr_schedule": ["cosine_warmup"],
            "scheduler": ["cosine_warmup"],
            # Architecture
            "expert": ["mistral"],
            "moe": ["mistral"],
            "mixture": ["mistral"],
            "router": ["mistral"],
            # Regularization
            "dropout": ["dropout_variants"],
            "droppath": ["dropout_variants"],
            "dropblock": ["dropout_variants"],
            "regulariz": ["dropout_variants"],
            # Transformer block
            "transformer": ["preln_transformer", "flashattention", "swiglu", "rope"],
            "block": ["preln_transformer", "dropout_variants"],
            "residual": ["preln_transformer", "dropout_variants"],
        }

        results: list[dict] = []
        seen_names: set = set()

        for key, spec_names in mapping.items():
            if key in pattern_lower:
                for spec_name in spec_names:
                    if spec_name in self.specs and spec_name not in seen_names:
                        seen_names.add(spec_name)
                        spec = self.specs[spec_name]
                        results.append(
                            {
                                "name": spec_name,
                                "title": spec["paper"]["title"],
                                "match_reason": f"Matches '{key}' pattern",
                                "category": spec["algorithm"]["category"],
                            }
                        )

        # LLM enhancement: if local lookup returned nothing, ask the LLM
        if not results and self._llm is not None and self._llm.is_available:
            try:
                analysis = self._llm.analyse_code(
                    code_pattern,
                    language=language,
                    focus="ML research improvements",
                )
                if analysis is not None:
                    for opp in analysis.improvement_opportunities:
                        results.append(
                            {
                                "name": opp.get("suggested_improvement", "unknown"),
                                "title": opp.get("paper_reference", ""),
                                "match_reason": opp.get("description", "LLM suggestion"),
                                "category": "llm_suggested",
                            }
                        )
            except Exception as exc:
                logger.debug("LLM code pattern analysis failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Dynamic spec discovery
    # ------------------------------------------------------------------

    def discover_specs_for_repo(
        self,
        patterns: dict[str, list[str]],
        frameworks: list[str],
        *,
        max_arxiv_queries: int = 3,
    ) -> list[dict]:
        """Dynamically discover relevant specs for a repository.

        Parameters
        ----------
        patterns : dict
            Pattern name → list of file locations (from tree-sitter analysis).
        frameworks : list
            Detected frameworks (e.g. ``["torch", "transformers"]``).
        max_arxiv_queries : int
            Maximum number of arXiv queries to make for discovery.

        Returns
        -------
        list[dict]
            Newly discovered and registered specs.
        """
        new_specs: list[dict] = []
        queries_made = 0

        for pattern_name in patterns:
            # Skip patterns that already have good local coverage
            existing = self.find_papers_for_code_pattern(pattern_name)
            if len(existing) >= 2:
                continue

            # Build a targeted arXiv query
            arxiv_query = self._build_discovery_query(pattern_name, frameworks)
            if not arxiv_query or queries_made >= max_arxiv_queries:
                continue

            arxiv_papers = _fetch_arxiv_papers(arxiv_query, max_results=3)
            queries_made += 1

            for entry in arxiv_papers:
                spec = self._try_register_arxiv_paper(entry)
                if spec:
                    new_specs.append(spec)

        return new_specs

    def _build_discovery_query(
        self,
        pattern_name: str,
        frameworks: list[str],
    ) -> str:
        """Build a targeted arXiv query from a code pattern and frameworks."""
        pattern_queries: dict[str, str] = {
            "normalization": "layer normalization transformer",
            "attention": "efficient attention transformer",
            "activation": "activation function neural network",
            "position_encoding": "positional encoding transformer",
            "optimizer": "optimizer deep learning training",
            "dropout": "regularization deep learning",
            "convolution": "efficient convolution neural network",
            "pooling": "pooling attention transformer",
            "loss_function": "loss function training",
            "data_augmentation": "data augmentation training",
        }

        base_query = pattern_queries.get(pattern_name, "")
        if not base_query:
            # Try partial matches
            for key, q in pattern_queries.items():
                if key in pattern_name or pattern_name in key:
                    base_query = q
                    break

        if not base_query:
            base_query = f"{pattern_name} deep learning"

        # Append framework context for better results
        if "torch" in frameworks or "pytorch" in frameworks:
            base_query += " pytorch"

        return base_query

    def _try_register_arxiv_paper(self, entry: dict[str, Any]) -> dict | None:
        """Attempt to register an arXiv paper as a spec using the LLM.

        Returns the registered spec dict, or ``None`` if no LLM is
        available or extraction fails.
        """
        if self._llm is None or not self._llm.is_available:
            return None

        abstract = entry.get("abstract", "")
        title = entry.get("title", "")
        if not abstract:
            return None

        try:
            extracted = self._llm.extract_paper_spec(
                abstract,
                paper_title=title,
            )
            if extracted is None:
                return None

            spec = _extracted_spec_to_dict(extracted)

            # Ensure paper metadata is populated
            paper_section = spec.setdefault("paper", {})
            paper_section.setdefault("title", title)
            paper_section.setdefault("authors", entry.get("authors", []))
            paper_section.setdefault("arxiv", entry.get("arxiv_id", ""))
            paper_section.setdefault("year", entry.get("year", 2024))

            key = _spec_key(spec)
            if key and key not in self.specs:
                self.specs[key] = spec
                logger.info("Auto-registered arXiv spec '%s' (%s)", key, title)
                return spec

        except Exception as exc:
            logger.debug("Failed to register arXiv paper '%s': %s", title, exc)

        return None

    # ------------------------------------------------------------------
    # Extraction entry point
    # ------------------------------------------------------------------

    def extract(self, source: str, source_type: str = "pdf") -> dict:
        """Extract research specification (backward compatible)"""
        if source_type == "pdf":
            return self._extract_from_pdf(source)
        elif source_type == "arxiv":
            return self._extract_from_arxiv(source)
        else:
            return self._extract_from_known_paper(source)

    # ------------------------------------------------------------------
    # PDF extraction — LLM-powered with hardcoded fallback
    # ------------------------------------------------------------------

    def _extract_from_pdf(self, pdf_path: str) -> dict:
        """Extract an implementation spec from a PDF file.

        Strategy:
        1. Try to read the PDF text.
        2. If text is available and an LLM is configured, use the LLM to
           extract a structured spec.
        3. Fall back to the hardcoded RMSNorm spec (legacy behaviour).
        """
        # Step 1: attempt PDF text extraction
        paper_text = _read_pdf_text(pdf_path)

        # Step 2: LLM-powered extraction
        if paper_text and self._llm is not None and self._llm.is_available:
            try:
                extracted = self._llm.extract_paper_spec(
                    paper_text,
                    paper_title=Path(pdf_path).stem.replace("_", " ").replace("-", " "),
                )
                if extracted is not None:
                    spec = _extracted_spec_to_dict(extracted)
                    # Register the new spec so subsequent lookups find it
                    key = _spec_key(spec)
                    if key:
                        self.specs[key] = spec
                        logger.info("LLM-extracted spec registered as '%s'", key)
                    return spec
            except Exception as exc:
                logger.warning("LLM PDF extraction failed, using fallback: %s", exc)

        # Step 3: hardcoded fallback
        return {
            "paper": {
                "title": "Root Mean Square Layer Normalization",
                "authors": ["Biao Zhang", "Rico Sennrich"],
                "year": 2019,
            },
            "algorithm": {
                "name": "RMSNorm",
                "replaces": "LayerNorm",
                "description": "Simplified layer normalization without mean-centering",
                "formula": "x / sqrt(mean(x^2) + eps) * gamma",
            },
            "implementation": {
                "module_name": "RMSNorm",
                "parent_class": "nn.Module",
                "parameters": ["ndim", "eps"],
                "code_template": self._get_rmsnorm_template(),
            },
            "changes": {
                "type": "replace",
                "target_patterns": ["LayerNorm", "nn.LayerNorm"],
                "replacement": "RMSNorm",
                "insertion_points": ["Block", "GPT"],
                "expected_benefits": ["5-10% training speedup", "Simplified computation"],
            },
            "validation": {
                "test_type": "training_comparison",
                "metrics": ["loss", "perplexity", "tokens_per_second"],
                "max_benchmark_time": 300,
            },
        }

    # ------------------------------------------------------------------
    # arXiv extraction — enhanced with abstract fetch + LLM
    # ------------------------------------------------------------------

    def _extract_from_arxiv(self, arxiv_id: str) -> dict:
        """Extract spec from an arXiv identifier.

        Strategy:
        1. Check local specs for a matching arXiv ID.
        2. Fetch the abstract from arXiv and use the LLM.
        3. Fall back to ``_extract_from_pdf`` (hardcoded default).
        """
        arxiv_id_clean = arxiv_id.strip().lower()

        # Step 1: check local registry
        for key, spec in self.specs.items():
            if key in arxiv_id_clean or spec["paper"].get("arxiv", "") in arxiv_id_clean:
                return spec

        # Step 2: fetch abstract and try LLM extraction
        if self._llm is not None and self._llm.is_available:
            abstract = _fetch_arxiv_abstract(arxiv_id)
            if abstract:
                try:
                    extracted = self._llm.extract_paper_spec(
                        abstract,
                        paper_title=f"arXiv:{arxiv_id}",
                    )
                    if extracted is not None:
                        spec = _extracted_spec_to_dict(extracted)
                        key = _spec_key(spec)
                        if key:
                            self.specs[key] = spec
                            logger.info(
                                "LLM-extracted arXiv spec registered as '%s'",
                                key,
                            )
                        return spec
                except Exception as exc:
                    logger.warning("LLM arXiv extraction failed for %s: %s", arxiv_id, exc)

        return self._extract_from_pdf(arxiv_id)

    def _extract_from_known_paper(self, paper_name: str) -> dict:
        paper_lower = paper_name.lower().replace("-", "").replace("_", "").replace(" ", "")

        for key, spec in self.specs.items():
            if (
                key in paper_lower
                or spec["algorithm"]["name"].lower().replace(" ", "") in paper_lower
            ):
                return spec

        return self._extract_from_pdf(paper_name)

    # ------------------------------------------------------------------
    # Templates and accessors (unchanged public interface)
    # ------------------------------------------------------------------

    def _get_rmsnorm_template(self) -> str:
        return """class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
"""

    def get_spec(self, name: str) -> dict | None:
        return self.specs.get(name.lower())

    def list_available_specs(self) -> list[str]:
        return list(self.specs.keys())

    def get_code_template(self, spec_name: str) -> str | None:
        spec = self.get_spec(spec_name)
        if not spec:
            return None

        name = spec["algorithm"]["name"].lower()

        templates = {
            "rmsnorm": self._get_rmsnorm_template(),
        }

        return templates.get(name)

    def get_categories(self) -> dict[str, list[str]]:
        """Get all available categories for filtering"""
        categories: dict[str, list[str]] = {}
        for name, spec in self.specs.items():
            cat = spec.get("algorithm", {}).get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories


# ---------------------------------------------------------------------------
# Helpers for converting LLM output → spec dict
# ---------------------------------------------------------------------------


def _extracted_spec_to_dict(extracted: Any) -> dict[str, Any]:
    """Convert an ``ExtractedSpec`` dataclass into the dict format
    that ``PAPER_SPECS`` uses."""
    return {
        "paper": extracted.paper if isinstance(extracted.paper, dict) else {},
        "algorithm": extracted.algorithm if isinstance(extracted.algorithm, dict) else {},
        "implementation": (
            extracted.implementation if isinstance(extracted.implementation, dict) else {}
        ),
        "changes": extracted.changes if isinstance(extracted.changes, dict) else {},
        "validation": extracted.validation if isinstance(extracted.validation, dict) else {},
    }


def _spec_key(spec: dict[str, Any]) -> str:
    """Derive a lowercase registry key from a spec dict."""
    alg_name = spec.get("algorithm", {}).get("name", "")
    if alg_name:
        return alg_name.lower().replace(" ", "_").replace("-", "_")
    title = spec.get("paper", {}).get("title", "")
    if title:
        # Use first meaningful word
        words = [w.lower() for w in title.split() if len(w) > 3]
        return "_".join(words[:2]) if words else ""
    return ""
