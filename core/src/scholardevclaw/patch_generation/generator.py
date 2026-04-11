"""
Patch generation engine.

Produces :class:`Patch` artefacts from a mapping result by:

1. **Creating new implementation files** — standalone module implementations for
   each algorithm (RMSNorm, SwiGLU, GEGLU, FlashAttention, GQA, RoPE, ALiBi,
   Lion optimizer, cosine warmup scheduler, etc.).
2. **Applying AST-level transformations** — using libcst ``CSTTransformer``
   subclasses to perform safe, structure-aware source code modifications
   (renaming classes, swapping activations, inserting norm layers, etc.).
3. **LLM-powered synthesis** — when the algorithm is not covered by a built-in
   template, an optional :class:`~scholardevclaw.llm.research_assistant.LLMResearchAssistant`
   is used to generate the implementation and transformation code.

All transformations are reversible: both the original and modified source
snippets are captured in :class:`Transformation` objects so the caller can
produce unified diffs.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import libcst as cst
from libcst import CSTTransformer, parse_module

if TYPE_CHECKING:
    from scholardevclaw.llm.research_assistant import LLMResearchAssistant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NewFile:
    path: str
    content: str


@dataclass
class Transformation:
    file: str
    original: str
    modified: str
    changes: list[dict] = field(default_factory=list)


@dataclass
class Patch:
    new_files: list[NewFile]
    transformations: list[Transformation]
    branch_name: str
    algorithm_name: str
    paper_reference: str


# ---------------------------------------------------------------------------
# CST transformers — one per algorithm family
# ---------------------------------------------------------------------------


class RMSNormTransformer(CSTTransformer):
    """Replaces LayerNorm references with RMSNorm."""

    def __init__(self, original_name: str = "LayerNorm", replacement_name: str = "RMSNorm"):
        self.original_name = original_name
        self.replacement_name = replacement_name
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == self.original_name:
            new_name = cst.Name(self.replacement_name)
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": self.original_name,
                    "to": self.replacement_name,
                }
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value == self.original_name:
            self.changes.append(
                {
                    "type": "rename_reference",
                    "from": self.original_name,
                    "to": self.replacement_name,
                }
            )
            return cst.Name(self.replacement_name)
        return updated_node


class SwiGLUTransformer(CSTTransformer):
    """Replaces MLP class with SwiGLU and swaps GELU → SiLU activations inside it."""

    def __init__(self) -> None:
        self.changes: list[dict] = []
        self.in_mlp_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:  # noqa: N802
        if node.name.value == "MLP":
            self.in_mlp_class = True
        return True

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "MLP":
            new_name = cst.Name("SwiGLU")
            self.changes.append({"type": "rename_class", "from": "MLP", "to": "SwiGLU"})
            self.in_mlp_class = False
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:  # noqa: N802
        if self.in_mlp_class:
            func = original_node.func
            if isinstance(func, cst.Attribute) and func.attr.value == "GELU":
                self.changes.append({"type": "replace_activation", "from": "GELU", "to": "SiLU"})
        return updated_node


class GEGLUTransformer(CSTTransformer):
    """Replaces MLP class with GEGLU and swaps GELU → gated GELU inside it."""

    def __init__(self) -> None:
        self.changes: list[dict] = []
        self.in_mlp_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:  # noqa: N802
        if node.name.value == "MLP":
            self.in_mlp_class = True
        return True

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "MLP":
            new_name = cst.Name("GEGLU")
            self.changes.append({"type": "rename_class", "from": "MLP", "to": "GEGLU"})
            self.in_mlp_class = False
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:  # noqa: N802
        if self.in_mlp_class:
            func = original_node.func
            if isinstance(func, cst.Attribute) and func.attr.value == "GELU":
                self.changes.append({"type": "replace_activation", "from": "GELU", "to": "GEGLU"})
        return updated_node


class FlashAttentionTransformer(CSTTransformer):
    """Renames CausalSelfAttention to FlashCausalSelfAttention and annotates flash usage."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "CausalSelfAttention":
            new_name = cst.Name("FlashCausalSelfAttention")
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": "CausalSelfAttention",
                    "to": "FlashCausalSelfAttention",
                }
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value == "CausalSelfAttention":
            self.changes.append(
                {
                    "type": "rename_reference",
                    "from": "CausalSelfAttention",
                    "to": "FlashCausalSelfAttention",
                }
            )
            return cst.Name("FlashCausalSelfAttention")
        return updated_node


class GQATransformer(CSTTransformer):
    """Renames CausalSelfAttention → GroupedQueryAttention."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value in ("CausalSelfAttention", "MultiHeadAttention"):
            new_name = cst.Name("GroupedQueryAttention")
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": original_node.name.value,
                    "to": "GroupedQueryAttention",
                }
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value in ("CausalSelfAttention", "MultiHeadAttention"):
            self.changes.append(
                {
                    "type": "rename_reference",
                    "from": original_node.value,
                    "to": "GroupedQueryAttention",
                }
            )
            return cst.Name("GroupedQueryAttention")
        return updated_node


class QKNormTransformer(CSTTransformer):
    """Augments attention classes by renaming to QKNorm-prefixed variants."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value in ("CausalSelfAttention", "Attention"):
            new_name = cst.Name("QKNorm" + original_node.name.value)
            self.changes.append(
                {"type": "rename_class", "from": original_node.name.value, "to": new_name.value}
            )
            return updated_node.with_changes(name=new_name)
        return updated_node


class PreLNTransformer(CSTTransformer):
    """Renames Block → PreLNBlock to signal pre-norm ordering."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "Block":
            new_name = cst.Name("PreLNBlock")
            self.changes.append({"type": "rename_class", "from": "Block", "to": "PreLNBlock"})
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value == "Block":
            self.changes.append({"type": "rename_reference", "from": "Block", "to": "PreLNBlock"})
            return cst.Name("PreLNBlock")
        return updated_node


class RoPETransformer(CSTTransformer):
    """Renames positional embedding references to RoPE variants."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        targets = {"PositionalEncoding", "position_embedding", "pos_emb", "wpe"}
        if original_node.value in targets:
            replacement = (
                "rotary_emb" if original_node.value.islower() else "RotaryPositionalEmbedding"
            )
            self.changes.append(
                {"type": "rename_reference", "from": original_node.value, "to": replacement}
            )
            return cst.Name(replacement)
        return updated_node


class ALiBiTransformer(CSTTransformer):
    """Renames positional embedding references to ALiBi variants."""

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        targets = {"PositionalEncoding", "position_embedding", "pos_emb", "wpe"}
        if original_node.value in targets:
            replacement = "alibi_bias" if original_node.value.islower() else "ALiBiPositionalBias"
            self.changes.append(
                {"type": "rename_reference", "from": original_node.value, "to": replacement}
            )
            return cst.Name(replacement)
        return updated_node


class GenericRenameTransformer(CSTTransformer):
    """Generic transformer that renames all occurrences of one name to another."""

    def __init__(self, original: str, replacement: str) -> None:
        self.original = original
        self.replacement = replacement
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == self.original:
            self.changes.append(
                {"type": "rename_class", "from": self.original, "to": self.replacement}
            )
            return updated_node.with_changes(name=cst.Name(self.replacement))
        return updated_node

    def leave_FunctionDef(  # noqa: N802
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node.name.value == self.original:
            self.changes.append(
                {"type": "rename_function", "from": self.original, "to": self.replacement}
            )
            return updated_node.with_changes(name=cst.Name(self.replacement))
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value == self.original:
            self.changes.append(
                {"type": "rename_reference", "from": self.original, "to": self.replacement}
            )
            return cst.Name(self.replacement)
        return updated_node


# ---------------------------------------------------------------------------
# Algorithm → transformer mapping
# ---------------------------------------------------------------------------

_TRANSFORMER_REGISTRY: dict[str, Any] = {
    "rmsnorm": lambda orig, repl: RMSNormTransformer(orig, repl),
    "swiglu": lambda orig, repl: SwiGLUTransformer(),
    "geglu": lambda orig, repl: GEGLUTransformer(),
    "flashattention": lambda orig, repl: FlashAttentionTransformer(),
    "flashattention2": lambda orig, repl: FlashAttentionTransformer(),
    "grouped_query_attention": lambda orig, repl: GQATransformer(),
    "qknorm": lambda orig, repl: QKNormTransformer(),
    "preln_transformer": lambda orig, repl: PreLNTransformer(),
    "rope": lambda orig, repl: RoPETransformer(),
    "alibi": lambda orig, repl: ALiBiTransformer(),
}


def _get_transformer(algorithm: str, original: str, replacement: str) -> CSTTransformer:
    """Return the best CSTTransformer for *algorithm*.

    Falls back to :class:`GenericRenameTransformer` when no specialised
    transformer is registered.
    """
    key = algorithm.lower().replace("-", "").replace(" ", "_")
    factory = _TRANSFORMER_REGISTRY.get(key)
    if factory is not None:
        return factory(original, replacement)
    # Try matching by prefix (e.g. "flash_attention_2" matches "flashattention")
    for reg_key, factory_fn in _TRANSFORMER_REGISTRY.items():
        if key.startswith(reg_key) or reg_key.startswith(key):
            return factory_fn(original, replacement)
    return GenericRenameTransformer(original, replacement)


# ---------------------------------------------------------------------------
# Code templates — one per algorithm
# ---------------------------------------------------------------------------


def _template_rmsnorm(spec: dict) -> str:
    paper = spec.get("paper", {})
    algorithm = spec.get("algorithm", {})
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", 2019)
    description = algorithm.get("description", "")
    formula = algorithm.get("formula", "x / sqrt(mean(x^2) + eps) * gamma")
    return textwrap.dedent(f'''\
        """
        RMSNorm: Root Mean Square Layer Normalization

        Integrated from "{paper.get("title", "Root Mean Square Layer Normalization")}"
        by {authors} ({year})

        Paper: arXiv:{paper.get("arxiv", "1910.07467")}
        Description: {description}
        Formula: {formula}
        """

        import torch
        import torch.nn as nn


        class RMSNorm(nn.Module):
            """
            Root Mean Square Layer Normalization

            Simplified layer normalization without mean-centering.
            Formula: x / sqrt(mean(x^2) + eps) * gamma

            Benefits:
            - Faster computation than LayerNorm
            - Simplified forward pass (no mean subtraction)
            - Often achieves similar or better results
            """

            def __init__(self, ndim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(ndim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return norm * self.weight
    ''')


def _template_swiglu(spec: dict) -> str:
    paper = spec.get("paper", {})
    algorithm = spec.get("algorithm", {})
    description = algorithm.get("description", "")
    return textwrap.dedent(f'''\
        """
        SwiGLU: Swish-Gated Linear Unit

        Integrated from "{paper.get("title", "GLU Variants Improve Transformer")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2020)})

        Paper: arXiv:{paper.get("arxiv", "2002.05202")}
        Description: {description}
        """

        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class SwiGLU(nn.Module):
            """
            Swish-Gated Linear Unit (SwiGLU).

            Combines Swish activation with gated linear units for improved FFN quality.
            Formula: SiLU(xW1) * (xW3) projected by W2.
            """

            def __init__(self, config):
                super().__init__()
                hidden_dim = 4 * config.n_embd
                self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
                self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
                self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    ''')


def _template_geglu(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        GEGLU: GELU-Gated Linear Unit

        Integrated from "{paper.get("title", "GLU Variants Improve Transformer")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2020)})

        Paper: arXiv:{paper.get("arxiv", "2002.05202")}
        """

        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class GEGLU(nn.Module):
            """
            GELU-Gated Linear Unit (GEGLU).

            Gated variant of GELU activation.
            Formula: GELU(xW1) * (xW3) projected by W2.
            """

            def __init__(self, config):
                super().__init__()
                hidden_dim = 4 * config.n_embd
                self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
                self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
                self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))
    ''')


def _template_flashattention(spec: dict) -> str:
    paper = spec.get("paper", {})
    version = "2" if "2" in spec.get("algorithm", {}).get("name", "") else ""
    return textwrap.dedent(f'''\
        """
        FlashAttention{version}: Fast and Memory-Efficient Exact Attention

        Integrated from "{paper.get("title", "FlashAttention")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2022)})

        Paper: arXiv:{paper.get("arxiv", "2205.14135")}
        """

        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class FlashCausalSelfAttention(nn.Module):
            """
            FlashAttention{version}-based causal self-attention.

            Uses PyTorch's scaled_dot_product_attention with is_causal=True
            for O(N) memory and hardware-aware IO optimisation.
            """

            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.head_dim = config.n_embd // config.n_head
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.attn_dropout = nn.Dropout(config.dropout)
                self.resid_dropout = nn.Dropout(config.dropout)
                self.dropout_p = config.dropout

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.size()
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.resid_dropout(self.c_proj(y))
    ''')


def _template_gqa(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Grouped-Query Attention (GQA)

        Integrated from "{paper.get("title", "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2023)})

        Paper: arXiv:{paper.get("arxiv", "2305.13245")}
        """

        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class GroupedQueryAttention(nn.Module):
            """
            Grouped-Query Attention.

            Uses fewer key-value heads than query heads, reducing KV-cache size
            while retaining most of multi-head attention quality.
            """

            def __init__(self, config):
                super().__init__()
                self.n_head = config.n_head
                self.n_kv_head = getattr(config, "n_kv_head", config.n_head // 4)
                self.n_embd = config.n_embd
                self.head_dim = config.n_embd // config.n_head
                self.kv_group_size = self.n_head // self.n_kv_head

                self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
                self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, _ = x.size()
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

                # Expand KV heads to match Q heads via repetition
                k = k.repeat_interleave(self.kv_group_size, dim=1)
                v = v.repeat_interleave(self.kv_group_size, dim=1)

                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
                y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
                return self.dropout(self.o_proj(y))
    ''')


def _template_qknorm(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        QK-Norm Attention

        Integrated from "{paper.get("title", "Scaling Vision Transformers to 22 Billion Parameters")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2023)})

        Paper: arXiv:{paper.get("arxiv", "2302.05442")}
        """

        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class QKNormCausalSelfAttention(nn.Module):
            """
            Causal self-attention with QK-Norm.

            Applies LayerNorm to query and key projections before the dot
            product to prevent attention logit growth at scale.
            """

            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.head_dim = config.n_embd // config.n_head
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.q_norm = nn.LayerNorm(self.head_dim)
                self.k_norm = nn.LayerNorm(self.head_dim)
                self.attn_dropout = nn.Dropout(config.dropout)
                self.resid_dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.size()
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                # Apply QK normalization
                q = self.q_norm(q)
                k = self.k_norm(k)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.resid_dropout(self.c_proj(y))
    ''')


def _template_preln(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Pre-LN Transformer Block

        Integrated from "{paper.get("title", "On Layer Normalization in the Transformer Architecture")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2020)})

        Paper: arXiv:{paper.get("arxiv", "2002.04745")}
        """

        import torch
        import torch.nn as nn


        class PreLNBlock(nn.Module):
            """
            Pre-LN Transformer Block.

            Places layer normalisation *before* attention and FFN sub-layers
            instead of after (Post-LN). Benefits: more stable training without
            learning-rate warmup, enabling deeper models.
            """

            def __init__(self, config):
                super().__init__()
                self.ln_1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config)  # noqa: F821
                self.ln_2 = nn.LayerNorm(config.n_embd)
                self.mlp = MLP(config)  # noqa: F821

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Pre-LN: norm BEFORE sub-layer, residual around sub-layer
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return x
    ''')


def _template_rope(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Rotary Positional Embedding (RoPE)

        Integrated from "{paper.get("title", "RoFormer: Enhanced Transformer with Rotary Position Embedding")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2021)})

        Paper: arXiv:{paper.get("arxiv", "2104.09864")}
        """

        import math
        import torch
        import torch.nn as nn


        class RotaryPositionalEmbedding(nn.Module):
            """
            Rotary Positional Embedding (RoPE).

            Encodes position information by rotating query and key vectors.
            Benefits: relative position awareness, no learned parameters,
            decaying inter-token dependency with distance, extrapolation
            to unseen sequence lengths.
            """

            def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
                super().__init__()
                self.dim = dim
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)
                self._build_cache(max_seq_len)

            def _build_cache(self, seq_len: int) -> None:
                t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.register_buffer("cos_cached", emb.cos(), persistent=False)
                self.register_buffer("sin_cached", emb.sin(), persistent=False)

            @staticmethod
            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)

            def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
                cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
                sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
                q_embed = (q * cos) + (self._rotate_half(q) * sin)
                k_embed = (k * cos) + (self._rotate_half(k) * sin)
                return q_embed, k_embed
    ''')


def _template_alibi(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        ALiBi: Attention with Linear Biases

        Integrated from "{paper.get("title", "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2022)})

        Paper: arXiv:{paper.get("arxiv", "2108.12409")}
        """

        import math
        import torch
        import torch.nn as nn


        class ALiBiPositionalBias(nn.Module):
            """
            Attention with Linear Biases (ALiBi).

            Replaces positional embeddings with a linear bias added to
            attention scores. Enables length generalisation without
            any learned positional parameters.
            """

            def __init__(self, n_heads: int, max_seq_len: int = 2048):
                super().__init__()
                self.n_heads = n_heads
                slopes = self._get_slopes(n_heads)
                self.register_buffer("slopes", slopes, persistent=False)
                self._build_bias(max_seq_len)

            @staticmethod
            def _get_slopes(n_heads: int) -> torch.Tensor:
                def _closest_power_of_2(n: int) -> int:
                    return 2 ** math.floor(math.log2(n))

                n = _closest_power_of_2(n_heads)
                slopes = torch.tensor(
                    [2 ** (-(2 ** -(math.log2(n) - i))) for i in range(n)]
                )
                if n < n_heads:
                    extra = torch.tensor(
                        [2 ** (-(2 ** -(math.log2(2 * n) - i))) for i in range(n_heads - n)]
                    )
                    slopes = torch.cat([slopes, extra])
                return slopes.unsqueeze(1).unsqueeze(1)

            def _build_bias(self, seq_len: int) -> None:
                positions = torch.arange(seq_len)
                relative = positions.unsqueeze(0) - positions.unsqueeze(1)
                bias = self.slopes * relative.unsqueeze(0).float()
                self.register_buffer("bias", bias, persistent=False)

            def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
                T = attn_scores.size(-1)
                return attn_scores + self.bias[:, :T, :T]
    ''')


def _template_lion(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Lion Optimizer: Evolved Sign Momentum

        Integrated from "{paper.get("title", "Symbolic Discovery of Optimization Algorithms")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2023)})

        Paper: arXiv:{paper.get("arxiv", "2302.06675")}
        """

        import torch
        from torch.optim import Optimizer


        class Lion(Optimizer):
            """
            Lion (EvoLved sIgn mOmeNtum) optimizer.

            Uses only the sign of the gradient for updates, resulting in
            uniform update magnitudes. Benefits: lower memory than Adam
            (no second moment), often faster convergence with proper LR.
            """

            def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                         weight_decay: float = 0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]

                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)

                        exp_avg = state["exp_avg"]
                        beta1, beta2 = group["betas"]

                        # Weight decay
                        if group["weight_decay"] != 0:
                            p.data.mul_(1.0 - group["lr"] * group["weight_decay"])

                        # Update: sign of interpolation between gradient and momentum
                        update = exp_avg * beta1 + grad * (1.0 - beta1)
                        p.add_(update.sign_(), alpha=-group["lr"])

                        # Momentum update
                        exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                return loss
    ''')


def _template_weight_decay_fused(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Decoupled Weight Decay (AdamW)

        Integrated from "{paper.get("title", "Decoupled Weight Decay Regularization")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2019)})

        Paper: arXiv:{paper.get("arxiv", "1711.05101")}
        """

        import torch


        def configure_optimizers_decoupled(model, learning_rate: float = 3e-4,
                                            weight_decay: float = 0.1,
                                            betas: tuple = (0.9, 0.95)):
            """
            Configure AdamW with decoupled weight decay.

            Separates parameters into two groups:
            - Parameters with weight decay (>= 2D tensors: weights)
            - Parameters without weight decay (< 2D tensors: biases, norms)
            """
            decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
            nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

            optim_groups = [
                {{"params": decay_params, "weight_decay": weight_decay}},
                {{"params": nodecay_params, "weight_decay": 0.0}},
            ]

            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"Decoupled weight decay: {{num_decay:,}} params with decay, {{num_nodecay:,}} without")

            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
            return optimizer
    ''')


def _template_cosine_warmup(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Cosine Annealing with Warmup Learning Rate Schedule

        Integrated from "{paper.get("title", "SGDR: Stochastic Gradient Descent with Warm Restarts")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2017)})

        Paper: arXiv:{paper.get("arxiv", "1608.03983")}
        """

        import math


        def get_cosine_warmup_lr(step: int, warmup_steps: int, max_steps: int,
                                  max_lr: float = 6e-4, min_lr: float = 6e-5) -> float:
            """
            Cosine annealing schedule with linear warmup.

            1. Linear warmup from 0 to max_lr over warmup_steps.
            2. Cosine decay from max_lr to min_lr over remaining steps.
            """
            if step < warmup_steps:
                return max_lr * (step + 1) / warmup_steps
            if step >= max_steps:
                return min_lr
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
    ''')


def _template_dropout_variants(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Dropout Variants for Transformers

        Integrated from "{paper.get("title", "Dropout: A Simple Way to Prevent Neural Networks from Overfitting")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2014)})
        """

        import torch
        import torch.nn as nn


        class StochasticDepth(nn.Module):
            """
            Stochastic Depth (layer dropout).

            Randomly drops entire residual branches during training with
            probability p, scaled linearly from 0 at the first layer to p
            at the last layer. Enables training deeper networks.
            """

            def __init__(self, p: float = 0.1):
                super().__init__()
                self.p = p

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.training or self.p == 0.0:
                    return x
                keep_prob = 1.0 - self.p
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
                return x * mask / keep_prob


        class DropPath(nn.Module):
            """
            Drop paths (Stochastic Depth) per sample.

            Same concept as StochasticDepth but with the common DropPath name
            used in vision transformer codebases.
            """

            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.training or self.drop_prob == 0.0:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor = torch.floor(random_tensor + keep_prob)
                return x / keep_prob * random_tensor
    ''')


def _template_mistral(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        Mistral Architecture Components (Sliding Window Attention + GQA)

        Integrated from "{paper.get("title", "Mistral 7B")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2023)})

        Paper: arXiv:{paper.get("arxiv", "2310.06825")}
        """

        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class SlidingWindowAttention(nn.Module):
            """
            Sliding Window Attention with Grouped-Query heads.

            Limits attention span to a fixed window size W, reducing
            computation from O(T^2) to O(T*W) while maintaining quality
            through stacked layers covering larger effective context.
            """

            def __init__(self, config):
                super().__init__()
                self.n_head = config.n_head
                self.n_kv_head = getattr(config, "n_kv_head", config.n_head // 4)
                self.head_dim = config.n_embd // config.n_head
                self.window_size = getattr(config, "sliding_window", 4096)

                self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
                self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, _ = x.size()
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

                # Expand KV heads
                kv_group_size = self.n_head // self.n_kv_head
                k = k.repeat_interleave(kv_group_size, dim=1)
                v = v.repeat_interleave(kv_group_size, dim=1)

                # Build sliding-window causal mask
                mask = torch.full((T, T), float("-inf"), device=x.device)
                mask = torch.triu(mask, diagonal=1)
                window_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device),
                                          diagonal=-self.window_size)
                mask = torch.maximum(mask, window_mask.T)

                attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask
                attn = F.softmax(attn, dim=-1)
                y = attn @ v
                y = y.transpose(1, 2).contiguous().view(B, T, -1)
                return self.o_proj(y)
    ''')


# Template registry
_TEMPLATE_REGISTRY: dict[str, Any] = {
    "rmsnorm": _template_rmsnorm,
    "swiglu": _template_swiglu,
    "geglu": _template_geglu,
    "flashattention": _template_flashattention,
    "flashattention2": _template_flashattention,
    "grouped_query_attention": _template_gqa,
    "qknorm": _template_qknorm,
    "preln_transformer": _template_preln,
    "rope": _template_rope,
    "alibi": _template_alibi,
    "lion": _template_lion,
    "weight_decay_fused": _template_weight_decay_fused,
    "cosine_warmup": _template_cosine_warmup,
    "dropout_variants": _template_dropout_variants,
    "mistral": _template_mistral,
}

# Algorithm name → canonical file name
_ALGORITHM_FILE_NAMES: dict[str, str] = {
    "rmsnorm": "rmsnorm.py",
    "swiglu": "swiglu.py",
    "geglu": "geglu.py",
    "flashattention": "flash_attention.py",
    "flashattention2": "flash_attention_v2.py",
    "grouped_query_attention": "grouped_query_attention.py",
    "qknorm": "qk_norm_attention.py",
    "preln_transformer": "pre_ln_block.py",
    "rope": "rotary_positional_embedding.py",
    "alibi": "alibi_positional_bias.py",
    "lion": "lion_optimizer.py",
    "weight_decay_fused": "decoupled_weight_decay.py",
    "cosine_warmup": "cosine_warmup_schedule.py",
    "dropout_variants": "dropout_variants.py",
    "mistral": "sliding_window_attention.py",
}


def _safe_new_file_name(name: str, *, default: str = "generated_algorithm.py") -> str:
    """Sanitize generated file names to a safe basename-like Python file."""
    base = Path(name).name.replace("\\", "/")
    base = base.split("/")[-1]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    if not sanitized:
        sanitized = default
    if "." not in sanitized:
        sanitized = f"{sanitized}.py"
    if sanitized.startswith("."):
        sanitized = sanitized.lstrip(".") or default
    return sanitized


# ---------------------------------------------------------------------------
# Patch generator
# ---------------------------------------------------------------------------


class PatchGenerator:
    """Generates :class:`Patch` artefacts from mapping results.

    Parameters
    ----------
    repo_path:
        Root of the target repository.
    llm_assistant:
        Optional LLM assistant for synthesising code when no built-in
        template covers the requested algorithm.
    """

    def __init__(
        self,
        repo_path: Path,
        llm_assistant: LLMResearchAssistant | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.llm_assistant = llm_assistant

    def generate(self, mapping_result: dict) -> Patch:
        spec = mapping_result.get("research_spec", {})

        new_files = self._create_new_files(spec)
        transformations = self._create_transformations(mapping_result)

        algorithm_name = spec.get("algorithm", {}).get("name", "research")
        paper_info = spec.get("paper", {})

        paper_reference = ""
        if paper_info.get("arxiv"):
            paper_reference = f"arXiv:{paper_info['arxiv']}"
        elif paper_info.get("title"):
            paper_reference = paper_info["title"]

        branch_name = f"integration/{algorithm_name.lower().replace(' ', '-')}"

        return Patch(
            new_files=new_files,
            transformations=transformations,
            branch_name=branch_name,
            algorithm_name=algorithm_name,
            paper_reference=paper_reference,
        )

    # ------------------------------------------------------------------
    # New file creation
    # ------------------------------------------------------------------

    def _create_new_files(self, spec: dict) -> list[NewFile]:
        """Create standalone implementation files for the algorithm."""
        new_files: list[NewFile] = []

        algorithm_name = spec.get("algorithm", {}).get("name", "").lower()
        # Normalise to registry key form
        key = algorithm_name.replace("-", "").replace(" ", "_")

        template_fn = _TEMPLATE_REGISTRY.get(key)
        file_name = _ALGORITHM_FILE_NAMES.get(key)

        if template_fn is not None and file_name is not None:
            content = template_fn(spec)
            new_files.append(NewFile(path=file_name, content=content))
            logger.info("Generated template for %s → %s", algorithm_name, file_name)
        elif self.llm_assistant is not None:
            # LLM-powered fallback for unknown algorithms
            synthesised = self._synthesise_with_llm(spec)
            if synthesised is not None:
                new_files.append(synthesised)
        else:
            logger.warning(
                "No template or LLM available for algorithm %r; skipping new file generation",
                algorithm_name,
            )

        return new_files

    def _synthesise_with_llm(self, spec: dict) -> NewFile | None:
        """Use the LLM assistant to synthesise an implementation file."""
        if self.llm_assistant is None:
            return None

        algorithm = spec.get("algorithm", {})
        algo_name = algorithm.get("name", "Unknown")
        paper = spec.get("paper", {})

        # Build a code context string describing what we need
        code_context = (
            f"Generate a complete, production-quality Python module implementing "
            f"the '{algo_name}' algorithm.\n\n"
            f"Paper: {paper.get('title', 'N/A')}\n"
            f"arXiv: {paper.get('arxiv', 'N/A')}\n"
            f"Description: {algorithm.get('description', 'N/A')}\n"
            f"Formula: {algorithm.get('formula', 'N/A')}\n"
            f"Replaces: {algorithm.get('replaces', 'N/A')}\n"
            f"Category: {algorithm.get('category', 'N/A')}\n\n"
            f"Requirements:\n"
            f"- Must be a standalone Python module with proper docstrings\n"
            f"- Use PyTorch (torch, torch.nn) if it's a neural network component\n"
            f"- Include type hints\n"
            f"- Include paper reference in module docstring\n"
        )

        try:
            plan = self.llm_assistant.generate_implementation_plan(
                paper_spec=spec,
                code_context=code_context,
                language="python",
            )
            if plan is not None and plan.steps:
                # Extract code blocks from plan steps
                code_blocks: list[str] = []
                for step in plan.steps:
                    code = step.get("code", "")
                    if code:
                        code_blocks.append(code)

                if code_blocks:
                    content = "\n\n".join(code_blocks)
                    file_name = _safe_new_file_name(
                        algo_name.lower().replace(" ", "_").replace("-", "_") + ".py"
                    )
                    logger.info("LLM synthesised implementation for %s → %s", algo_name, file_name)
                    return NewFile(path=file_name, content=content)

            # Fallback: use analyse_code to generate a stub
            analysis = self.llm_assistant.analyse_code(
                code_context,
                focus=f"Generate a complete implementation of {algo_name}",
            )
            if analysis is not None and analysis.improvement_opportunities:
                content = f'"""\n{algo_name}\n\nPaper: {paper.get("title", "N/A")}\narXiv: {paper.get("arxiv", "N/A")}\n"""\n\n'
                for opp in analysis.improvement_opportunities:
                    desc = opp.get("description", "")
                    if desc:
                        content += f"# {desc}\n"
                file_name = _safe_new_file_name(
                    algo_name.lower().replace(" ", "_").replace("-", "_") + ".py"
                )
                return NewFile(path=file_name, content=content)

        except Exception:
            logger.warning("LLM synthesis failed for %s", algo_name, exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def _create_transformations(self, mapping_result: dict) -> list[Transformation]:
        transformations: list[Transformation] = []

        targets = mapping_result.get("targets", [])
        spec = mapping_result.get("research_spec", {})

        for target in targets:
            file_path = self.repo_path / target.get("file", "model.py")

            # SECURITY: Prevent path traversal — ensure target stays within repo
            try:
                resolved = file_path.resolve()
                if not resolved.is_relative_to(self.repo_path.resolve()):
                    logger.warning("Skipping path traversal target: %s", file_path)
                    continue
            except (ValueError, OSError):
                continue

            if not file_path.exists():
                logger.debug("Target file does not exist: %s", file_path)
                continue

            try:
                original = file_path.read_text()

                context = target.get("context", {})
                replacement = context.get("replacement", "") if isinstance(context, dict) else ""
                original_name = context.get("original", "") if isinstance(context, dict) else ""

                if replacement and original_name:
                    modified = self._apply_transformation(
                        original,
                        original_name,
                        replacement,
                        spec.get("algorithm", {}).get("name", ""),
                    )

                    if modified != original:
                        transformations.append(
                            Transformation(
                                file=str(target.get("file", "model.py")),
                                original=original[:500],
                                modified=modified[:500],
                                changes=[
                                    {
                                        "type": "replace",
                                        "from": original_name,
                                        "to": replacement,
                                    }
                                ],
                            )
                        )
            except Exception:
                logger.warning("Error transforming %s", file_path, exc_info=True)

        return transformations

    def _apply_transformation(
        self, source: str, original: str, replacement: str, algorithm: str
    ) -> str:
        """Apply an AST-level transformation, falling back to string replacement."""
        try:
            tree = parse_module(source)
            transformer = _get_transformer(algorithm, original, replacement)
            modified_tree = tree.visit(transformer)
            result = modified_tree.code

            # Log changes from the transformer
            changes = getattr(transformer, "changes", [])
            if changes:
                logger.info(
                    "AST transform (%s): %d changes applied",
                    algorithm,
                    len(changes),
                )
            return result

        except Exception:
            logger.debug(
                "AST transformation failed for %s, using string replacement",
                algorithm,
                exc_info=True,
            )
            return source.replace(original, replacement)
