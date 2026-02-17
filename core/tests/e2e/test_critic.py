from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NANOGPT_REPO = ROOT / "test_repos" / "nanogpt"


def get_nanogpt_path() -> Path:
    if not NANOGPT_REPO.exists():
        raise RuntimeError(
            f"nanoGPT not found at {NANOGPT_REPO}. "
            "Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
        )
    return NANOGPT_REPO


import pytest

from scholardevclaw.critic import run_critic


class TestE2ECritic:
    def test_critic_valid_python_passes(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [
                {
                    "path": "test_module.py",
                    "content": "import torch\n\nclass TestModule(torch.nn.Module):\n    def forward(self, x):\n        return x\n",
                }
            ],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert result.ok is True or result.payload.get("issue_count", 0) == 0

    def test_critic_detects_syntax_error(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [
                {
                    "path": "test_module.py",
                    "content": "import torch\n\nclass TestModule(torch.nn.Module:\n    def forward(self, x):\n        return x\n",
                }
            ],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert "issues" in result.payload

    def test_critic_detects_unbalanced_brackets(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [],
            "transformations": [{"file": "model.py", "modified": "def forward(x): return x + (1"}],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert result.ok is False
        assert result.payload.get("issue_count", 0) > 0

    def test_critic_detects_antipatterns(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [
                {"path": "test_module.py", "content": "for i in range(len(x)):\n    print(x[i])\n"}
            ],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert len(result.payload.get("warnings", [])) > 0

    def test_critic_detects_bare_except(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [
                {"path": "test_module.py", "content": "try:\n    x = 1\nexcept:\n    pass\n"}
            ],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        warnings = result.payload.get("warnings", [])
        antipatterns = [w for w in warnings if w.get("type") == "antipattern"]
        assert len(antipatterns) > 0

    def test_critic_detects_security_issue(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [{"path": "test_module.py", "content": "result = eval('1 + 1')\n"}],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        all_alerts = result.payload.get("issues", []) + result.payload.get("warnings", [])
        security_issues = [i for i in all_alerts if i.get("type") == "security"]
        assert len(security_issues) > 0

    def test_critic_with_spec_generates_patch(self):
        repo_path = get_nanogpt_path()
        result = run_critic(str(repo_path), spec_name="rmsnorm")

        assert result.payload is not None

    def test_critic_returns_summary(self):
        repo_path = get_nanogpt_path()

        patch_result = {
            "new_files": [{"path": "test_module.py", "content": "import torch\n"}],
            "transformations": [],
        }

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert "summary" in result.payload

    def test_critic_has_severity_counts(self):
        repo_path = get_nanogpt_path()

        patch_result = {"new_files": [], "transformations": []}

        result = run_critic(str(repo_path), patch_result=patch_result)

        assert "severity_counts" in result.payload

    def test_critic_invalid_path_returns_error(self):
        result = run_critic("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None
