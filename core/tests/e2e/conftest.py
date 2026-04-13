from __future__ import annotations

import sys
from pathlib import Path

import pytest

CORE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CORE_ROOT.parent
SRC = CORE_ROOT / "src"
NANOGPT_REPO = REPO_ROOT / "test_repos" / "nanogpt"

# Backward-compatible alias for older imports.
ROOT = CORE_ROOT

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def get_nanogpt_path() -> Path:
    if not NANOGPT_REPO.exists():
        pytest.skip(
            f"nanoGPT not found at {NANOGPT_REPO}. "
            "Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
        )
    return NANOGPT_REPO


@pytest.fixture
def nanogpt_repo_path() -> Path:
    return get_nanogpt_path()
