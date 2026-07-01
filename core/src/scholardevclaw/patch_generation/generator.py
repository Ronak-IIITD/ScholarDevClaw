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

import ast
import json
import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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


class DropoutTransformer(CSTTransformer):
    """Replaces nn.Dropout / Dropout references with DropPath variants.

    Handles both plain ``Dropout`` names and ``nn.Dropout`` attribute-access
    patterns (the common idiom in PyTorch codebases).
    """

    def __init__(self, original: str, replacement: str) -> None:
        self.original = original
        self.replacement = replacement
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value in (self.original, "Dropout"):
            self.changes.append(
                {"type": "rename_class", "from": original_node.name.value, "to": self.replacement}
            )
            return updated_node.with_changes(name=cst.Name(self.replacement))
        return updated_node

    def leave_Attribute(  # noqa: N802
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute | cst.Name:
        # nn.Dropout → replacement (strip module prefix)
        if (
            isinstance(original_node.value, cst.Name)
            and original_node.value.value == "nn"
            and original_node.attr.value in (self.original, "Dropout")
        ):
            self.changes.append(
                {
                    "type": "replace_attribute",
                    "from": f"nn.{original_node.attr.value}",
                    "to": self.replacement,
                }
            )
            return cst.Name(self.replacement)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value in (self.original, "Dropout"):
            self.changes.append(
                {"type": "rename_reference", "from": original_node.value, "to": self.replacement}
            )
            return cst.Name(self.replacement)
        return updated_node


class MultiQueryAttentionTransformer(CSTTransformer):
    """Renames ``CausalSelfAttention`` → ``MultiQueryAttention``.

    Mirrors the GQA transformer pattern.  The template file provides the
    full MQA implementation; this transformer updates existing references.
    """

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value in ("CausalSelfAttention", "MultiHeadAttention"):
            new_name = cst.Name("MultiQueryAttention")
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": original_node.name.value,
                    "to": "MultiQueryAttention",
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
                    "to": "MultiQueryAttention",
                }
            )
            return cst.Name("MultiQueryAttention")
        return updated_node


class MistralSlidingWindowTransformer(CSTTransformer):
    """Renames MLP/feedforward references → SlidingWindowAttention variants.

    The template provides the full sliding-window + GQA implementation.
    This transformer updates existing class and reference names in the codebase.
    """

    def __init__(self) -> None:
        self.changes: list[dict] = []
        self._mlp_targets = {"MLP", "FeedForward", "feedforward"}

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        name = original_node.name.value
        if name in self._mlp_targets:
            new_name = cst.Name("SlidingWindowAttention")
            self.changes.append(
                {"type": "rename_class", "from": name, "to": "SlidingWindowAttention"}
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value in self._mlp_targets:
            self.changes.append(
                {
                    "type": "rename_reference",
                    "from": original_node.value,
                    "to": "SlidingWindowAttention",
                }
            )
            return cst.Name("SlidingWindowAttention")
        return updated_node


class KVCacheAugmentTransformer(CSTTransformer):
    """Augments attention classes by renaming to KV-cache-enabled variants.

    Renames ``CausalSelfAttention`` / ``Attention`` → ``KVCacheAttention``
    to signal that the attention should use the KV-cache implementation
    from the generated template file.
    """

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value in ("CausalSelfAttention", "Attention"):
            new_name = cst.Name("KVCache" + original_node.name.value)
            self.changes.append(
                {"type": "rename_class", "from": original_node.name.value, "to": new_name.value}
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value in ("CausalSelfAttention", "Attention"):
            replacement = "KVCache" + original_node.value
            self.changes.append(
                {"type": "rename_reference", "from": original_node.value, "to": replacement}
            )
            return cst.Name(replacement)
        return updated_node


class GradientCheckpointingTransformer(CSTTransformer):
    """Wraps Block references to signal checkpointing usage.

    Renames ``Block`` → ``CheckpointedBlock`` so the training code
    can wrap transformer blocks with gradient checkpointing.
    """

    def __init__(self) -> None:
        self.changes: list[dict] = []

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "Block":
            new_name = cst.Name("CheckpointedBlock")
            self.changes.append(
                {"type": "rename_class", "from": "Block", "to": "CheckpointedBlock"}
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        if original_node.value == "Block":
            self.changes.append(
                {"type": "rename_reference", "from": "Block", "to": "CheckpointedBlock"}
            )
            return cst.Name("CheckpointedBlock")
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
    "layernorm": lambda orig, repl: GenericRenameTransformer(orig, repl),
    "gelu": lambda orig, repl: GenericRenameTransformer(orig, repl),
    "swiglu": lambda orig, repl: SwiGLUTransformer(),
    "geglu": lambda orig, repl: GEGLUTransformer(),
    "flashattention": lambda orig, repl: FlashAttentionTransformer(),
    "flashattention2": lambda orig, repl: FlashAttentionTransformer(),
    "grouped_query_attention": lambda orig, repl: GQATransformer(),
    "qknorm": lambda orig, repl: QKNormTransformer(),
    "preln_transformer": lambda orig, repl: PreLNTransformer(),
    "rope": lambda orig, repl: RoPETransformer(),
    "alibi": lambda orig, repl: ALiBiTransformer(),
    "lora": lambda orig, repl: GenericRenameTransformer(orig, repl),
    # ---- Explicit GenericRename entries for simple renames ----
    "lion": lambda orig, repl: GenericRenameTransformer(orig, repl),
    "weight_decay_fused": lambda orig, repl: GenericRenameTransformer(orig, repl),
    "cosine_warmup": lambda orig, repl: GenericRenameTransformer(orig, repl),
    "topk_sampling": lambda orig, repl: GenericRenameTransformer(orig, repl),
    # ---- Specialized transformers ----
    "dropout_variants": lambda orig, repl: DropoutTransformer(orig, repl),
    "multiquery_attention": lambda orig, repl: MultiQueryAttentionTransformer(),
    "mistral": lambda orig, repl: MistralSlidingWindowTransformer(),
    "kv_cache": lambda orig, repl: KVCacheAugmentTransformer(),
    "gradient_checkpointing": lambda orig, repl: GradientCheckpointingTransformer(),
}


_ALGORITHM_KEY_ALIASES: dict[str, str] = {
    "rms_norm": "rmsnorm",
    "root_mean_square_layer_normalization": "rmsnorm",
    "gaussian_error_linear_units_gelus": "gelu",
    "grouped_query_attention": "grouped_query_attention",
    "groupedqueryattention": "grouped_query_attention",
    "flash_attention": "flashattention",
    "flash_attention_2": "flashattention2",
    "cosine_annealing_with_warmup": "cosine_warmup",
    "cosine_lr_schedule": "cosine_warmup",
    "low_rank_adaptation_of_large_language_models": "lora",
    "qk_norm": "qknorm",
    "qk_normalization": "qknorm",
    "query_key_normalization": "qknorm",
    "pre_ln": "preln_transformer",
    "pre_layer_normalization": "preln_transformer",
    "cosine_warmup_lr": "cosine_warmup",
    "kv_cache": "kv_cache",
    "key_value_cache": "kv_cache",
    "key_value_caching": "kv_cache",
    "multiquery_attention": "multiquery_attention",
    "multi_query_attention": "multiquery_attention",
    "mqa": "multiquery_attention",
    "gradient_checkpointing": "gradient_checkpointing",
    "activation_checkpointing": "gradient_checkpointing",
    "checkpointing": "gradient_checkpointing",
    "topk_sampling": "topk_sampling",
    "top_k_sampling": "topk_sampling",
    "nucleus_sampling": "topk_sampling",
    "top_p_sampling": "topk_sampling",
}

_ARXIV_TO_ALGORITHM_KEY: dict[str, str] = {
    "1606.08415": "gelu",
    "1607.06450": "layernorm",
    "1608.03983": "cosine_warmup",
    "1910.07467": "rmsnorm",
    "2002.05202": "swiglu",
    "2104.09864": "rope",
    "2106.09685": "lora",
    "2108.12409": "alibi",
    "2205.14135": "flashattention",
    "2305.13245": "grouped_query_attention",
    "1911.02150": "multiquery_attention",
    "2009.06732": "kv_cache",
    "1604.06174": "gradient_checkpointing",
    "1904.09751": "topk_sampling",
}


def _normalize_algorithm_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return _ALGORITHM_KEY_ALIASES.get(key, key)


def _resolve_algorithm_key(spec: dict[str, Any]) -> str:
    algorithm = spec.get("algorithm", {})
    paper = spec.get("paper", {})

    for candidate in (
        str(spec.get("_spec_key", "") or ""),
        str(algorithm.get("name", "") or ""),
        str(algorithm.get("replaces", "") or ""),
    ):
        normalized = _normalize_algorithm_key(candidate)
        if normalized in _TEMPLATE_REGISTRY or normalized in _TRANSFORMER_REGISTRY:
            return normalized

    arxiv_id = str(paper.get("arxiv", "") or "").strip()
    if arxiv_id:
        normalized_arxiv = arxiv_id.replace("arxiv:", "").strip()
        if normalized_arxiv in _ARXIV_TO_ALGORITHM_KEY:
            return _ARXIV_TO_ALGORITHM_KEY[normalized_arxiv]

    return _normalize_algorithm_key(str(algorithm.get("name", "") or ""))


def _get_transformer(algorithm: str, original: str, replacement: str) -> CSTTransformer:
    """Return the best CSTTransformer for *algorithm*.

    Falls back to :class:`GenericRenameTransformer` when no specialised
    transformer is registered.
    """
    key = _normalize_algorithm_key(algorithm)
    factory = _TRANSFORMER_REGISTRY.get(key)
    if factory is not None:
        return cast(CSTTransformer, factory(original, replacement))
    # Try matching by prefix (e.g. "flash_attention_2" matches "flashattention")
    for reg_key, factory_fn in _TRANSFORMER_REGISTRY.items():
        if key.startswith(reg_key) or reg_key.startswith(key):
            return cast(CSTTransformer, factory_fn(original, replacement))
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


def _template_layernorm(spec: dict) -> str:
    return textwrap.dedent("""\
        import math

        EXPECTED_SYMBOLS = ("LayerNorm", "layer_norm")


        def layer_norm(values, eps=1e-5):
            mean = sum(values) / max(len(values), 1)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            denom = math.sqrt(variance + eps)
            return [(value - mean) / denom for value in values]


        class LayerNorm:
            def __call__(self, values):
                return layer_norm(values)
    """)


def _template_gelu(spec: dict) -> str:
    return textwrap.dedent("""\
        import math

        EXPECTED_SYMBOLS = ("gelu",)


        def gelu(value):
            return 0.5 * value * (1.0 + math.tanh(0.7978845608 * (value + 0.044715 * value**3)))
    """)


def _template_lora(spec: dict) -> str:
    paper = spec.get("paper", {})
    return textwrap.dedent(f'''\
        """
        LoRA: Low-Rank Adaptation of Large Language Models

        Integrated from "{paper.get("title", "LoRA: Low-Rank Adaptation of Large Language Models")}"
        by {", ".join(paper.get("authors", []))} ({paper.get("year", 2021)})

        Paper: arXiv:{paper.get("arxiv", "2106.09685")}
        """

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import math

        EXPECTED_SYMBOLS = ("LoRALinear", "apply_lora")

        class LoRALinear(nn.Module):
            """
            LoRA Linear Layer.

            Wraps a base linear layer and applies a low-rank update:
            y = Wx + (BA)x = (W + BA)x
            """
            def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
                super().__init__()
                self.r = r
                self.lora_alpha = lora_alpha
                self.scaling = lora_alpha / r

                # Base frozen weight
                self.base_weight = nn.Parameter(torch.randn(out_features, in_features))

                # Low-rank matrices A and B
                self.lora_A = nn.Parameter(torch.randn(r, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, r))
                self.dropout = nn.Dropout(lora_dropout)

                # Initialize A with Kaiming-uniform and B as zeros to ensure identity start
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Main path: Wx
                result = F.linear(x, self.base_weight)

                # LoRA path: (B @ A)x
                # x must be [batch, seq, in_features]
                x_dropped = self.dropout(x)
                lora_update = (x_dropped @ self.lora_A.t()) @ self.lora_B.t()

                return result + (lora_update * self.scaling)

        def apply_lora(model: nn.Module, target_layer: str, r: int = 8, lora_alpha: float = 32.0):
            """
            Injects LoRA layers into a pre-trained model.
            Replaces existing nn.Linear layers matching target_layer name with LoRALinear.
            """
            if isinstance(model, nn.Linear) and target_layer in "":
                new_layer = LoRALinear(model.in_features, model.out_features, r=r, lora_alpha=lora_alpha)
                new_layer.base_weight.data.copy_(model.weight.data)
                return new_layer

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and target_layer in name:
                    # Extract weight and bias
                    # Replace with LoRALinear
                    new_layer = LoRALinear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha)
                    new_layer.base_weight.data.copy_(module.weight.data)

                    # Patch the module into the parent's dict
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)
            return model
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

        EXPECTED_SYMBOLS = ("FlashCausalSelfAttention",)

        class FlashCausalSelfAttention(nn.Module):
            """
            FlashAttention{version}-based causal self-attention.

            Implementation uses PyTorch's scaled_dot_product_attention (SDPA)
            which dispatches to FlashAttention or Memory-Efficient Attention kernels
            depending on hardware and tensor shapes.
            """
            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.head_dim = config.n_embd // config.n_head

                # Combined QKV projection for efficiency
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

                self.dropout_p = config.dropout
                self.resid_dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.size()

                # 1. Linear projection to Q, K, V
                qkv = self.c_attn(x) # [B, T, 3*C]
                q, k, v = qkv.split(self.n_embd, dim=2)

                # 2. Reshape for Multi-Head Attention: [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

                # 3. Efficient Attention using SDPA
                # is_causal=True enforces the causal mask automatically
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )

                # 4. Reassemble heads: [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
                y = y.transpose(1, 2).contiguous().view(B, T, C)

                # 5. Final projection and residual dropout
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


def _template_kv_cache(spec: dict) -> str:
    """KV-cache for efficient autoregressive decoding."""
    return textwrap.dedent('''\
        import torch
        import torch.nn as nn


        class KVCache:
            """
            Key-Value cache for autoregressive transformer decoding.

            Stores key and value tensors from previous timesteps so they
            do not need to be recomputed at each generation step.
            """

            def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int):
                self.cache_k = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim)
                self.cache_v = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim)
                self.seen_tokens = 0

            def update(self, batch_size: int, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                T_new = new_k.shape[-2]
                start = self.seen_tokens
                end = start + T_new
                self.cache_k[:batch_size, :, start:end] = new_k
                self.cache_v[:batch_size, :, start:end] = new_v
                self.seen_tokens = end
                return (self.cache_k[:batch_size, :, :end], self.cache_v[:batch_size, :, :end])

            def reset(self):
                self.seen_tokens = 0
    ''')


def _template_multiquery_attention(spec: dict) -> str:
    """Multi-Query Attention — single KV head shared across all query heads."""
    paper = spec.get("paper", {})
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", 2019)
    title = paper.get("title", "Fast Transformer Decoding")
    arxiv = paper.get("arxiv", "1911.02150")
    header = (
        '"""\n'
        "Multi-Query Attention\n\n"
        f'Integrated from "{title}"\n'
        f"by {authors} ({year})\n\n"
        f"Paper: arXiv:{arxiv}\n"
        "Description: Shares a single key/value head across all query heads.\n"
        '"""'
    )
    return textwrap.dedent(
        '''\
        """
        Multi-Query Attention
        Integrated from "'''
        + title
        + """"
        by """
        + authors
        + """ ("""
        + str(year)
        + """)
        Paper: arXiv:"""
        + arxiv
        + '''
        Description: Shares a single key/value head across all query heads.
        """

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import math


        class MultiQueryAttention(nn.Module):
            """
            Multi-Query Attention with a single KV head.

            All query heads share one key and one value head, reducing
            the KV-cache size by a factor of n_heads compared to MHA.
            """

            def __init__(self, config):
                super().__init__()
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.head_dim = config.n_embd // config.n_head

                self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.k_proj = nn.Linear(config.n_embd, self.head_dim, bias=config.bias)
                self.v_proj = nn.Linear(config.n_embd, self.head_dim, bias=config.bias)
                self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.shape
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)

                attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = F.softmax(attn, dim=-1)
                y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
                return self.o_proj(y)
    '''
    )


def _template_gradient_checkpointing(spec: dict) -> str:
    """Gradient checkpointing for trading compute for memory."""
    return textwrap.dedent('''\
        import torch
        import torch.nn as nn
        from typing import Optional


        class CheckpointedBlock(nn.Module):
            """
            Wraps a module with gradient checkpointing.

            During the backward pass, activations are recomputed rather than
            stored, reducing peak GPU memory at the cost of extra computation.
            """

            def __init__(self, module: nn.Module):
                super().__init__()
                self.module = module

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                return torch.utils.checkpoint.checkpoint(
                    self.module, x, *args, use_reentrant=False, **kwargs
                )


        def apply_checkpointing(model: nn.Module, segments: int = 1) -> None:
            """
            Wraps sequential blocks of a transformer model with checkpointing.

            Args:
                model: A transformer model with a 'blocks' or 'layers' attribute.
                segments: Number of checkpoint segments (1 = checkpoint entire forward).
            """
            if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
                n = len(model.blocks)
                seg_size = max(1, n // segments)
                for i in range(0, n, seg_size):
                    for j in range(i, min(i + seg_size, n)):
                        orig = model.blocks[j]
                        model.blocks[j] = CheckpointedBlock(orig)
            elif hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
                n = len(model.layers)
                seg_size = max(1, n // segments)
                for i in range(0, n, seg_size):
                    for j in range(i, min(i + seg_size, n)):
                        orig = model.layers[j]
                        model.layers[j] = CheckpointedBlock(orig)
    ''')


def _template_topk_sampling(spec: dict) -> str:
    """Top-K and nucleus (top-p) sampling for text generation."""
    return textwrap.dedent('''\
        import torch
        import torch.nn.functional as F


        def topk_filter(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
            """Keep only the top-k logits, setting the rest to -inf."""
            topk_vals, _ = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
            thresholds = topk_vals[..., -1:]
            return torch.where(logits < thresholds, float("-inf"), logits)


        def nucleus_filter(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
            """Keep the smallest set of tokens whose cumulative probability exceeds p."""
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > p
            sorted_logits[mask] = float("-inf")
            return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


        def sample_with_temperature(
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.0,
        ) -> torch.Tensor:
            """Sample from logits with temperature, top-k, and nucleus filtering.

            Args:
                logits: Raw logits tensor of shape (batch, vocab_size).
                temperature: >0. Lower values make output more deterministic.
                top_k: If >0, keep only top-k logits.
                top_p: If >0, keep tokens with cumulative probability <= top_p.

            Returns:
                Sampled token indices of shape (batch, 1).
            """
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                logits = topk_filter(logits, top_k)
            if top_p > 0:
                logits = nucleus_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
    ''')


# Template registry
_TEMPLATE_REGISTRY: dict[str, Any] = {
    "rmsnorm": _template_rmsnorm,
    "layernorm": _template_layernorm,
    "gelu": _template_gelu,
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
    "lora": _template_lora,
    "weight_decay_fused": _template_weight_decay_fused,
    "cosine_warmup": _template_cosine_warmup,
    "dropout_variants": _template_dropout_variants,
    "mistral": _template_mistral,
    "kv_cache": _template_kv_cache,
    "multiquery_attention": _template_multiquery_attention,
    "gradient_checkpointing": _template_gradient_checkpointing,
    "topk_sampling": _template_topk_sampling,
}

# Algorithm name → canonical file name
_ALGORITHM_FILE_NAMES: dict[str, str] = {
    "rmsnorm": "rmsnorm.py",
    "layernorm": "layernorm.py",
    "gelu": "gelu.py",
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
    "lora": "lora.py",
    "weight_decay_fused": "decoupled_weight_decay.py",
    "cosine_warmup": "cosine_warmup_schedule.py",
    "dropout_variants": "dropout_variants.py",
    "mistral": "sliding_window_attention.py",
    "kv_cache": "kv_cache.py",
    "multiquery_attention": "multiquery_attention.py",
    "gradient_checkpointing": "gradient_checkpointing.py",
    "topk_sampling": "topk_sampling.py",
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

        new_files = self._create_new_files(spec, mapping_result=mapping_result)
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

    def heal_patch(self, patch: Patch, validation_result: Any, mapping_result: dict) -> Patch:
        """
        Attempt to heal a failed patch by analyzing validation errors and fixing the code.

        Args:
            patch: The original failed patch
            validation_result: The validation result containing error information
            mapping_result: The mapping result used to generate the original patch

        Returns:
            A healed Patch object, or the original patch if healing fails
        """
        if self.llm_assistant is None:
            logger.warning("Cannot heal patch: no LLM assistant available")
            return patch

        errors = []
        if validation_result.error:
            errors.append(f"Error: {validation_result.error}")
        if validation_result.logs:
            # Extract test failures from logs
            log_lines = validation_result.logs.split("\n")
            for line in log_lines:
                if any(
                    keyword in line.lower()
                    for keyword in ["error", "fail", "exception", "traceback"]
                ):
                    errors.append(line.strip())

        if not errors:
            logger.info("No specific errors found to heal")
            return patch

        logger.info("Attempting to heal patch with %d error(s)", len(errors))

        # Build context for the LLM
        error_summary = "\n".join(errors[:10])  # Limit context size

        # Try to heal new files
        healed_new_files = []
        for new_file in patch.new_files:
            try:
                healed_content = self._heal_code(
                    new_file.content, error_summary, "new_file", new_file.path
                )
                healed_new_files.append(NewFile(path=new_file.path, content=healed_content))
            except Exception as e:
                logger.warning("Failed to heal new file %s: %s", new_file.path, e)
                healed_new_files.append(new_file)

        # Try to heal transformations
        healed_transformations = []
        for transformation in patch.transformations:
            try:
                healed_original = self._heal_code(
                    transformation.original, error_summary, "original", transformation.file
                )
                healed_modified = self._heal_code(
                    transformation.modified, error_summary, "modified", transformation.file
                )
                healed_transformations.append(
                    Transformation(
                        file=transformation.file,
                        original=healed_original,
                        modified=healed_modified,
                        changes=transformation.changes,
                    )
                )
            except Exception as e:
                logger.warning("Failed to heal transformation for %s: %s", transformation.file, e)
                healed_transformations.append(transformation)

        return Patch(
            new_files=healed_new_files,
            transformations=healed_transformations,
            branch_name=patch.branch_name,
            algorithm_name=patch.algorithm_name,
            paper_reference=patch.paper_reference,
        )

    def _heal_code(self, code: str, error_summary: str, code_type: str, file_path: str) -> str:
        """Use LLM to heal a piece of code given the error summary."""
        if not code or not error_summary:
            return code

        prompt = f"""You are a Python code fixing expert. The following code has validation errors.

File: {file_path}
Code type: {code_type}

Original code:
```python
{code}
```

Validation errors:
{error_summary}

Please fix the code to resolve these errors. Return ONLY the fixed code without any explanation or markdown formatting.
"""
        try:
            assistant = self.llm_assistant
            if assistant is None:
                return code
            # Use the LLM assistant to generate fixed code
            if hasattr(assistant, "generate_text"):
                fixed_code = assistant.generate_text(prompt)
            elif hasattr(assistant, "complete"):
                fixed_code = assistant.complete(prompt)
            else:
                logger.warning("LLM assistant doesn't have a known text generation method")
                return code

            # Validate the fixed code is still valid Python
            if fixed_code:
                try:
                    ast.parse(fixed_code)
                    logger.info("Successfully healed code for %s", file_path)
                    return fixed_code
                except SyntaxError as e:
                    logger.warning("Healed code has syntax errors: %s", e)
                    return code
            return code
        except Exception as e:
            logger.warning("LLM healing failed: %s", e)
            return code

    # ------------------------------------------------------------------
    # New file creation
    # ------------------------------------------------------------------

    def _create_new_files(
        self,
        spec: dict,
        *,
        mapping_result: dict[str, Any] | None = None,
    ) -> list[NewFile]:
        """Create standalone implementation files for the algorithm."""
        new_files: list[NewFile] = []

        algorithm_name = spec.get("algorithm", {}).get("name", "").lower()
        key = _resolve_algorithm_key(spec)

        template_fn = _TEMPLATE_REGISTRY.get(key)
        file_name = _ALGORITHM_FILE_NAMES.get(key)

        if template_fn is not None and file_name is not None:
            content = template_fn(spec)
            new_files.append(NewFile(path=file_name, content=content))
            logger.info("Generated template for %s → %s", algorithm_name, file_name)
        elif self.llm_assistant is not None:
            # LLM-powered fallback for unknown algorithms
            synthesised = self._synthesise_with_llm(spec, mapping_result=mapping_result)
            if synthesised is not None:
                new_files.append(synthesised)
        else:
            logger.warning(
                "No template or LLM available for algorithm %r; skipping new file generation",
                algorithm_name,
            )

        return new_files

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return stripped

    def _collect_target_file_contexts(
        self,
        mapping_result: dict[str, Any] | None,
        *,
        limit: int = 3,
    ) -> list[dict[str, str]]:
        contexts: list[dict[str, str]] = []
        if not isinstance(mapping_result, dict):
            return contexts

        seen: set[str] = set()
        for target in mapping_result.get("targets", []):
            if not isinstance(target, dict):
                continue
            relative = str(target.get("file", "") or "").strip()
            if not relative or relative in seen:
                continue
            file_path = (self.repo_path / relative).resolve()
            try:
                file_path.relative_to(self.repo_path.resolve())
            except ValueError:
                continue
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text()
            except OSError:
                continue
            seen.add(relative)
            contexts.append({"path": relative, "content": content[:8000]})
            if len(contexts) >= limit:
                break
        return contexts

    def _summarise_reference_implementations(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        repositories = spec.get("_reference_implementations", [])
        if not isinstance(repositories, list):
            return []
        summary: list[dict[str, Any]] = []
        for repo in repositories[:5]:
            if not isinstance(repo, dict):
                continue
            summary.append(
                {
                    "name": str(repo.get("name", "") or ""),
                    "owner": str(repo.get("owner", "") or ""),
                    "url": str(repo.get("url", "") or ""),
                    "description": str(repo.get("description", "") or "")[:240],
                    "stars": int(repo.get("stars", 0) or 0),
                }
            )
        return summary

    def _synthesise_with_llm(
        self,
        spec: dict,
        *,
        mapping_result: dict[str, Any] | None = None,
    ) -> NewFile | None:
        """Use the LLM assistant to synthesise an implementation file."""
        if self.llm_assistant is None:
            return None

        algorithm = spec.get("algorithm", {})
        algo_name = algorithm.get("name", "Unknown")
        paper = spec.get("paper", {})
        target_files = self._collect_target_file_contexts(mapping_result)
        reference_repos = self._summarise_reference_implementations(spec)

        prompt_payload = {
            "paper": {
                "title": paper.get("title", "N/A"),
                "arxiv": paper.get("arxiv", "N/A"),
                "authors": paper.get("authors", []),
            },
            "algorithm": {
                "name": algo_name,
                "description": algorithm.get("description", "N/A"),
                "formula": algorithm.get("formula", "N/A"),
                "replaces": algorithm.get("replaces", "N/A"),
                "category": algorithm.get("category", "N/A"),
            },
            "changes": spec.get("changes", {}),
            "reference_implementations": reference_repos,
            "target_file_contexts": target_files,
        }
        system = (
            "You are a patch synthesis engine. Produce only Python source code for the requested "
            "file. Do not include explanations, markdown fences, or commentary.\n\n"
            "Implementation Guidelines for Machine Learning:\n"
            "- Implement as a proper `torch.nn.Module` subclass when applicable.\n"
            "- Use `torch.nn.functional` for operations where appropriate (e.g., F.linear, F.scaled_dot_product_attention).\n"
            "- Handle device and dtype properly (e.g., using `x.device`, `x.dtype` for tensor creation).\n"
            "- Initialize parameters with Kaiming/Xavier initialization where relevant.\n"
            "- Include dropout layers if the original paper includes them.\n"
            "- Ensure vectorized operations and avoid Python loops over tensor dimensions.\n"
            "- Preserve the input/output interface (signature) of the original component.\n"
            "- Add docstrings explaining the implementation and citing the paper.\n"
            "- Do not introduce external dependencies beyond PyTorch and standard library."
        )
        prompt = (
            "Generate the modified Python file content for integrating this research technique.\n\n"
            f"{json.dumps(prompt_payload, indent=2)}\n\n"
            "Requirements:\n"
            "- Use the target file context as the primary integration surface.\n"
            "- Preserve existing public function and method signatures unless the paper explicitly requires new parameters.\n"
            "- Do not introduce dependencies that are not already present in the file context or standard library.\n"
            "- Use the reference implementation metadata only as an anchor, not as license-incompatible verbatim source.\n"
            "- Return only the final Python file content."
        )

        try:
            generated = None
            if hasattr(self.llm_assistant, "generate_text"):
                generated = self.llm_assistant.generate_text(
                    prompt,
                    system=system,
                    max_tokens=4096,
                    temperature=0.0,
                )
            elif hasattr(self.llm_assistant, "generate_implementation_plan"):
                plan = self.llm_assistant.generate_implementation_plan(
                    paper_spec=spec,
                    code_context=prompt,
                    language="python",
                )
                if plan is not None and getattr(plan, "steps", None):
                    code_blocks = [
                        step.get("code", "")
                        for step in plan.steps
                        if isinstance(step, dict) and step.get("code")
                    ]
                    if code_blocks:
                        generated = "\n\n".join(code_blocks)
            if generated:
                content = self._strip_code_fences(generated)
                try:
                    ast.parse(content)
                except SyntaxError:
                    logger.warning("LLM synthesis returned invalid Python for %s", algo_name)
                else:
                    file_name = _ALGORITHM_FILE_NAMES.get(
                        _resolve_algorithm_key(spec),
                        _safe_new_file_name(
                            algo_name.lower().replace(" ", "_").replace("-", "_") + ".py"
                        ),
                    )
                    logger.info("LLM synthesised implementation for %s → %s", algo_name, file_name)
                    return NewFile(path=file_name, content=content)

        except Exception:
            logger.warning("LLM synthesis failed for %s", algo_name, exc_info=True)

        return None

    def _rewrite_file_with_llm(
        self,
        *,
        source: str,
        file_path: str,
        spec: dict[str, Any],
        target: dict[str, Any],
    ) -> str | None:
        if self.llm_assistant is None:
            return None

        system = (
            "You are a patch generator. Rewrite the provided file to integrate the research "
            "technique. Return only the full modified file content.\n\n"
            "Implementation Guidelines for Machine Learning:\n"
            "- Implement as a proper `torch.nn.Module` subclass when applicable.\n"
            "- Use `torch.nn.functional` for operations where appropriate (e.g., F.linear, F.scaled_dot_product_attention).\n"
            "- Handle device and dtype properly (e.g., using `x.device`, `x.dtype` for tensor creation).\n"
            "- Initialize parameters with Kaiming/Xavier initialization where relevant.\n"
            "- Include dropout layers if the original paper includes them.\n"
            "- Ensure vectorized operations and avoid Python loops over tensor dimensions.\n"
            "- Preserve the input/output interface (signature) of the original component.\n"
            "- Add docstrings explaining the implementation and citing the paper.\n"
            "- Do not introduce external dependencies beyond PyTorch and standard library."
        )
        prompt_payload = {
            "paper": spec.get("paper", {}),
            "algorithm": spec.get("algorithm", {}),
            "changes": spec.get("changes", {}),
            "reference_implementations": self._summarise_reference_implementations(spec),
            "target": target,
        }
        prompt = (
            f"Target file: {file_path}\n"
            f"Patch request: {json.dumps(prompt_payload, indent=2)}\n\n"
            "Constraints:\n"
            "- Preserve existing public function and method signatures unless the paper requires new parameters.\n"
            "- Do not add new third-party dependencies.\n"
            "- Use the file content below as the full context.\n"
            "- Return only the modified file content.\n\n"
            f"{source}"
        )
        generated = self.llm_assistant.generate_text(
            prompt,
            system=system,
            max_tokens=4096,
            temperature=0.0,
        )
        if not generated:
            return None
        rewritten = self._strip_code_fences(generated)
        try:
            ast.parse(rewritten)
        except SyntaxError:
            return None
        return rewritten

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def _create_transformations(self, mapping_result: dict) -> list[Transformation]:
        """Create transformations for target files.

        Targets are grouped by file so multiple edits compose against one
        evolving source buffer. Independent files are processed in parallel.
        """
        targets = mapping_result.get("targets", [])
        spec = mapping_result.get("research_spec", {})

        if not targets:
            return []

        targets_by_file: dict[str, list[dict[str, Any]]] = {}
        for target in targets:
            if not isinstance(target, dict):
                continue
            file_name = str(target.get("file", "model.py"))
            targets_by_file.setdefault(file_name, []).append(target)

        target_groups = list(targets_by_file.values())
        if len(target_groups) == 1:
            return self._transform_target_group(target_groups[0], spec)

        transformations: list[Transformation] = []
        with ThreadPoolExecutor(max_workers=min(4, len(target_groups))) as executor:
            futures = {
                executor.submit(self._transform_target_group, group, spec): group
                for group in target_groups
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    transformations.extend(result)
                except Exception:
                    group = futures[future]
                    logger.warning(
                        "Error transforming %s",
                        group[0].get("file", "unknown") if group else "unknown",
                        exc_info=True,
                    )
        return transformations

    def _transform_single_target(self, target: dict, spec: dict) -> list[Transformation]:
        """Backward-compatible wrapper for one target."""
        return self._transform_target_group([target], spec)

    def _transform_target_group(
        self,
        targets: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[Transformation]:
        """Compose all mapped edits for one file into a single transformation."""
        if not targets:
            return []

        relative_path = str(targets[0].get("file", "model.py"))
        file_path = self.repo_path / relative_path

        # SECURITY: Prevent path traversal — ensure target stays within repo
        try:
            resolved = file_path.resolve()
            if not resolved.is_relative_to(self.repo_path.resolve()):
                logger.warning("Skipping path traversal target: %s", file_path)
                return []
        except (ValueError, OSError):
            return []

        if not file_path.exists():
            logger.debug("Target file does not exist: %s", file_path)
            return []

        try:
            original = file_path.read_text()
            modified = original
            applied_changes: list[dict[str, str]] = []

            for target in sorted(targets, key=lambda item: int(item.get("line", 0) or 0)):
                context = target.get("context", {})
                replacement = context.get("replacement", "") if isinstance(context, dict) else ""
                original_name = context.get("original", "") if isinstance(context, dict) else ""

                if not replacement or not original_name:
                    continue

                next_source = self._apply_transformation(
                    modified,
                    original_name,
                    replacement,
                    spec.get("algorithm", {}).get("name", ""),
                )
                if next_source == modified and self.llm_assistant is not None:
                    llm_modified = self._rewrite_file_with_llm(
                        source=modified,
                        file_path=relative_path,
                        spec=spec,
                        target=target,
                    )
                    if llm_modified:
                        next_source = llm_modified

                if next_source != modified:
                    modified = next_source
                    applied_changes.append(
                        {
                            "type": "replace",
                            "from": original_name,
                            "to": replacement,
                        }
                    )

            if modified != original:
                return [
                    Transformation(
                        file=relative_path,
                        original=original,
                        modified=modified,
                        changes=applied_changes,
                    )
                ]
        except Exception:
            logger.warning("Error transforming %s", file_path, exc_info=True)

        return []

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
