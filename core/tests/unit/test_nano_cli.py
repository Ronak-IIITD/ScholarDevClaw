"""Unit tests for the lean ``nanogpt-opt`` CLI (step 6).

Covers command routing, the honest validate scorecard (pass/fail/skipped),
dry-run safety, the write -> idempotent re-run -> revert roundtrip, and
manifest error handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholardevclaw import nano_cli

MODEL_SOURCE = '''"""mini GPT fixture"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        return self.c_proj(F.scaled_dot_product_attention(q, k, v))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd),
        ))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return x
'''

TRAIN_SOURCE = """import math


def get_lr(it, learning_rate):
    warmup_iters = 2000
    max_iters = 600000
    lr_ratio = 0.1
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return learning_rate * lr_ratio
    coeff = lr_ratio + (1 - lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * (it - warmup_iters) / (max_iters - warmup_iters))
    )
    return learning_rate * coeff
"""


@pytest.fixture()
def transformer_repo(tmp_path: Path) -> Path:
    (tmp_path / "model.py").write_text(MODEL_SOURCE)
    (tmp_path / "train.py").write_text(TRAIN_SOURCE)
    return tmp_path


def _checksums(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))}


# =========================================================================
# routing
# =========================================================================


def test_parser_routes_all_four_commands():
    parser = nano_cli.build_parser()
    args = parser.parse_args(["analyze", "some/repo", "--json"])
    assert args.command == "analyze" and args.repo == "some/repo" and args.json
    args = parser.parse_args(["suggest", "r"])
    assert args.command == "suggest"
    args = parser.parse_args(["apply", "r", "--write", "--spec", "rmsnorm"])
    assert args.command == "apply" and args.write and args.spec == "rmsnorm"
    args = parser.parse_args(["apply", "r", "--revert"])
    assert args.revert
    args = parser.parse_args(["validate", "r", "--json"])
    assert args.command == "validate" and args.json


def test_main_rejects_missing_repo(capsys):
    assert nano_cli.main(["analyze", "/definitely/not/a/repo"]) == 2
    assert "error:" in capsys.readouterr().err


# =========================================================================
# analyze / suggest
# =========================================================================


def test_analyze_reports_applicable_specs(transformer_repo, capsys):
    rc = nano_cli.main(["analyze", str(transformer_repo), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["files_scanned"] == 2
    assert "rmsnorm" in out["applicable_specs"]
    assert set(out["applicable_specs"]) <= set(out["supported_specs"])
    assert len(out["supported_specs"]) == 10


def test_analyze_unknown_spec_rejected(transformer_repo, capsys):
    with pytest.raises(SystemExit):
        nano_cli.main(["analyze", str(transformer_repo), "--spec", "made_up_spec"])


def test_suggest_maps_spec_to_targets(transformer_repo, capsys):
    rc = nano_cli.main(["suggest", str(transformer_repo), "--spec", "rmsnorm", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    suggestions = out["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["spec"] == "rmsnorm"
    assert suggestions[0]["ok"] is True
    assert suggestions[0]["targets"], "rmsnorm must map to at least one target"
    assert suggestions[0]["confidence"] >= 70
    assert suggestions[0]["targets"][0]["file"] == "model.py"


# =========================================================================
# apply / revert
# =========================================================================


def test_apply_dry_run_writes_nothing(transformer_repo, capsys):
    before = _checksums(transformer_repo)
    rc = nano_cli.main(["apply", str(transformer_repo), "--spec", "rmsnorm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run" in out
    assert _checksums(transformer_repo) == before
    assert not (transformer_repo / ".nanogpt-opt").exists()


def test_apply_write_revert_roundtrip(transformer_repo, capsys):
    before = _checksums(transformer_repo)

    rc = nano_cli.main(["apply", str(transformer_repo), "--spec", "rmsnorm", "--write"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "manifest" in out
    manifest = transformer_repo / ".nanogpt-opt" / "manifest.json"
    assert manifest.exists()
    assert "RMSNorm" in (transformer_repo / "model.py").read_text()
    assert (transformer_repo / "rmsnorm.py").exists()
    assert _checksums(transformer_repo) != before

    rc = nano_cli.main(["apply", str(transformer_repo), "--revert"])
    capsys.readouterr()
    assert rc == 0
    assert _checksums(transformer_repo) == before, "revert must restore byte-identical files"
    assert not manifest.exists()


def test_apply_second_run_is_noop(transformer_repo, capsys):
    rc = nano_cli.main(["apply", str(transformer_repo), "--spec", "rmsnorm", "--write"])
    capsys.readouterr()
    assert rc == 0

    rc = nano_cli.main(["apply", str(transformer_repo), "--spec", "rmsnorm", "--write"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to apply" in out

    rc = nano_cli.main(["apply", str(transformer_repo), "--revert"])
    capsys.readouterr()
    assert rc == 0
    assert _checksums(transformer_repo) == _checksums_base(
        transformer_repo, MODEL_SOURCE, TRAIN_SOURCE
    )


def _checksums_base(root: Path, model: str, train: str) -> dict[str, str]:
    return {"model.py": model, "train.py": train}


def test_apply_revert_without_manifest_errors(transformer_repo, capsys):
    rc = nano_cli.main(["apply", str(transformer_repo), "--revert"])
    assert rc == 1
    assert "no manifest" in capsys.readouterr().err


# =========================================================================
# validate — honest stage reporting
# =========================================================================


def test_validate_clean_repo_reports_honest_skips(tmp_path, capsys):
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    rc = nano_cli.main(["validate", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "compile    pass" in out
    assert "tests      skipped (no test files found)" in out
    assert "benchmark  skipped (no benchmark scripts found)" in out
    assert "0 failed, 2 skipped -> pass" in out


def test_validate_syntax_error_fails(tmp_path, capsys):
    (tmp_path / "broken.py").write_text("def f(:\n")
    rc = nano_cli.main(["validate", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "compile    fail" in out
    assert "broken.py" in out
    assert "-> fail" in out


def test_validate_failing_tests_fail(tmp_path, capsys):
    (tmp_path / "test_sample.py").write_text("def test_no():\n    assert False\n")
    rc = nano_cli.main(["validate", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "tests      fail" in out


def test_validate_skips_tests_when_pytest_missing(tmp_path, capsys, monkeypatch):
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: None if name == "pytest" else real_find_spec(name, *a, **k),
    )
    rc = nano_cli.main(["validate", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tests      skipped (pytest not installed)" in out


def test_validate_json_payload_shape(tmp_path, capsys):
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    rc = nano_cli.main(["validate", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["overall"] == "pass"
    assert out["counts"] == {"pass": 1, "fail": 0, "skipped": 2}
    assert [s["stage"] for s in out["stages"]] == ["compile", "tests", "benchmark"]
    assert all(s["status"] in {"pass", "fail", "skipped"} for s in out["stages"])


def test_validate_never_reports_skipped_as_pass(tmp_path, capsys):
    """A stage that did not run must never be counted as a pass."""
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    nano_cli.main(["validate", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    for stage in out["stages"]:
        if stage["stage"] in {"tests", "benchmark"}:
            assert stage["status"] == "skipped"
    assert out["counts"]["pass"] == 1  # only compile actually ran
