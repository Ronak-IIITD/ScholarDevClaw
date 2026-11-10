"""Frozen spec loader for nanogpt-opt v1.

Loads the 10 curated specs from the packaged ``specs/*.json`` (frozen from
``research_intelligence/extractor.py:PAPER_SPECS``).

Scope: nanoGPT + generic PyTorch transformer only. No arXiv fetching,
no LLM synthesis, no dynamic registry in v1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SPECS_DIR = Path(__file__).with_name("specs")


def list_supported_specs() -> list[str]:
    """Return the 10 frozen spec names, manifest order preserved."""
    manifest = json.loads((_SPECS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return list(manifest["specs"])


def load_spec(name: str) -> dict[str, Any]:
    """Load one frozen spec by name. Raises KeyError on unknown name."""
    supported = list_supported_specs()
    if name not in supported:
        raise KeyError(f"Unsupported spec {name!r}. Supported: {supported}")
    data = json.loads((_SPECS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return data[name]


def load_all_specs() -> dict[str, dict[str, Any]]:
    """Load all 10 frozen specs as {name: spec}."""
    return {name: load_spec(name) for name in list_supported_specs()}
