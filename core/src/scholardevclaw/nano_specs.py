"""Frozen spec loader for nanogpt-opt v1.

Loads the 10 curated specs from ``core/specs/*.json`` (frozen from
``research_intelligence/extractor.py:PAPER_SPECS``).

Scope: nanoGPT + generic PyTorch transformer only. No arXiv fetching,
no LLM synthesis, no dynamic registry in v1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# core/specs/ relative to this file: src/scholardevclaw/ -> ../../../specs/
_SPECS_DIR_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "specs",  # core/specs/ (editable install)
    Path(__file__).resolve().parent / "specs",  # packaged fallback
]

MANIFEST_NAME = "manifest.json"


def _resolve_specs_dir() -> Path | None:
    for candidate in _SPECS_DIR_CANDIDATES:
        if candidate.is_dir() and (candidate / MANIFEST_NAME).exists():
            return candidate
    return None


def list_supported_specs() -> list[str]:
    """Return the 10 frozen spec names, manifest order preserved."""
    specs_dir = _resolve_specs_dir()
    if specs_dir is None:
        # Hard fallback: must stay in sync with core/specs/manifest.json
        return [
            "rmsnorm",
            "preln_transformer",
            "qknorm",
            "swiglu",
            "flashattention2",
            "grouped_query_attention",
            "rope",
            "alibi",
            "lion",
            "cosine_warmup",
        ]
    manifest = json.loads((specs_dir / MANIFEST_NAME).read_text())
    return list(manifest["specs"])


def load_spec(name: str) -> dict[str, Any]:
    """Load one frozen spec by name. Raises KeyError on unknown name."""
    supported = list_supported_specs()
    if name not in supported:
        raise KeyError(f"Unsupported spec {name!r}. Supported: {supported}")
    specs_dir = _resolve_specs_dir()
    if specs_dir is not None:
        data = json.loads((specs_dir / f"{name}.json").read_text())
        return data[name]
    # Fallback to legacy registry when specs/ is not installed.
    from scholardevclaw.research_intelligence.extractor import PAPER_SPECS

    return PAPER_SPECS[name]


def load_all_specs() -> dict[str, dict[str, Any]]:
    """Load all 10 frozen specs as {name: spec}."""
    return {name: load_spec(name) for name in list_supported_specs()}
