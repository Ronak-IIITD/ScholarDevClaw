from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
NANOGPT_REPO = ROOT / "test_repos" / "nanogpt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def get_nanogpt_path() -> Path:
    if not NANOGPT_REPO.exists():
        pytest.skip(
            f"nanoGPT not found at {NANOGPT_REPO}. "
            "Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
        )
    return NANOGPT_REPO
