"""Step 2 tests: slim nanoGPT + generic PyTorch analyzer."""

from pathlib import Path

from scholardevclaw.nano_analyze import analyze_repo

NANOGPT = Path(__file__).resolve().parents[3] / "test_repos" / "nanogpt"
# core/tests/unit/test_nano_analyze.py -> parents[3] = core/


def _nanogpt_root() -> Path:
    # Fallback: locate via repo-relative search
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "test_repos" / "nanogpt" / "model.py"
        if candidate.exists():
            return candidate.parent
    return NANOGPT


def test_nanogpt_model_detected():
    analysis = analyze_repo(_nanogpt_root())
    assert analysis.files_scanned >= 3
    for expected in ["Block", "GPT", "MLP", "CausalSelfAttention", "LayerNorm"]:
        assert expected in analysis.models, f"missing {expected}"


def test_nanogpt_self_attrs_and_calls():
    analysis = analyze_repo(_nanogpt_root())
    self_attrs = set(analysis.components.get("self_attrs", []))
    calls = set(analysis.components.get("calls", []))
    assert {"ln_1", "ln_2", "c_attn", "c_fc", "wpe"}.issubset(self_attrs)
    assert {"LayerNorm", "GELU", "Embedding", "AdamW"}.issubset(calls)


def test_nanogpt_applicable_specs():
    analysis = analyze_repo(_nanogpt_root())
    specs = set(analysis.applicable_specs)
    # Core nanoGPT wins must be present
    assert {"rmsnorm", "swiglu", "flashattention2", "rope", "lion"}.issubset(specs)
    # All suggested specs must be in the frozen 10
    from scholardevclaw.nano_specs import list_supported_specs

    assert specs.issubset(set(list_supported_specs()))


def test_generic_llama_patterns(tmp_path):
    src = """
import torch
import torch.nn as nn
class LlamaRMSNorm(nn.Module):
    pass
class LlamaAttention(nn.Module):
    def __init__(self):
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
"""
    (tmp_path / "model.py").write_text(src)
    analysis = analyze_repo(tmp_path)
    assert "qknorm" in analysis.applicable_specs
    assert "rmsnorm" not in analysis.applicable_specs  # no LayerNorm signals
