"""Behavior and refusal tests for the minGPT attention preview."""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import pytest

from scholardevclaw import nano_cli
from scholardevclaw.application.pipeline import run_sdpa_preview
from scholardevclaw.patch_generation.sdpa import preview_sdpa

SOURCE = b"""import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Attention starts here.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # Combine values.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
"""


def test_preview_rewrites_attention_and_preserves_surrounding_source():
    preview = preview_sdpa(SOURCE, "mingpt/model.py")
    assert [site.status for site in preview.sites] == ["supported"]
    assert preview.sites[0].line == 21
    assert preview.diff.startswith("--- a/mingpt/model.py\n+++ b/mingpt/model.py\n")
    assert b"y = F.scaled_dot_product_attention(" in preview.modified
    assert b"attn_mask=self.bias[:, :, :T, :T] != 0" in preview.modified
    assert b"self.attn_dropout.p if self.attn_dropout.training else 0.0" in preview.modified
    assert b"# Attention starts here." in preview.modified
    assert b"# Combine values." in preview.modified
    assert preview.modified.split(b"    def forward")[0] == SOURCE.split(b"    def forward")[0]
    assert (
        preview.modified.split(b"        y = y.transpose")[1]
        == SOURCE.split(b"        y = y.transpose")[1]
    )
    ast.parse(preview.modified)
    twice = preview_sdpa(preview.modified)
    assert twice.modified == preview.modified
    assert twice.diff == ""
    assert twice.sites[0].status == "already_migrated"


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    [
        (b"1.0 / math.sqrt(k.size(-1))", b"2.0 / math.sqrt(k.size(-1))", "forward differs"),
        (b"dim=-1)", b"dim=-2)", "forward differs"),
        (b"self.bias[:,:,:T,:T] == 0", b"self.bias[:,:,:T,:T] != 0", "forward differs"),
        (b"transpose(-2, -1)", b"transpose(-1, -2)", "forward differs"),
        (b"return y", b"return y, att", "forward differs"),
        (b"forward(self, x)", b"forward(self, x, mask=None)", "forward differs"),
        (b"config.attn_pdrop)", b"config.attn_pdrop, inplace=True)", "unmodified nn.Dropout"),
        (
            b"nn.Dropout(config.attn_pdrop)",
            b"CustomDropout(config.attn_pdrop)",
            "unmodified nn.Dropout",
        ),
        (
            b"import torch.nn.functional as F",
            b"import fake_functional as F",
            "binding for F.softmax",
        ),
        (b"import math", b"import fake_math as math", "binding for math.sqrt"),
        (b"import torch.nn as nn", b"import fake_nn as nn", "inheriting"),
        (b"import torch\n", b"import fake_torch as torch\n", "mask initialization"),
        (b"torch.tril", b"torch.triu", "lower-triangular mask"),
        (b"    def forward", b"    @custom_decorator\n    def forward", "forward differs"),
        (
            b"class CausalSelfAttention",
            b"@custom_decorator\nclass CausalSelfAttention",
            "inheriting",
        ),
        (
            b"        self.n_head = config.n_head",
            b"        self.attn_dropout = custom\n        self.n_head = config.n_head",
            "unmodified nn.Dropout",
        ),
        (
            b"        self.n_head = config.n_head",
            b"        self.attn_dropout.p = 0.5\n        self.n_head = config.n_head",
            "unmodified nn.Dropout",
        ),
        (
            b"        B, T, C = x.size()",
            b"        F = custom\n        B, T, C = x.size()",
            "forward differs",
        ),
        (
            b"class CausalSelfAttention",
            b"float = lambda value: 0\nclass CausalSelfAttention",
            "binding for float",
        ),
        (
            b"        B, T, C = x.size()\n        q, k, v",
            b"        B, T, C = x.size(); q, k, v",
            "one statement per line",
        ),
    ],
)
def test_unfamiliar_patterns_are_explained_without_a_diff(before, after, reason):
    assert before in SOURCE
    source = SOURCE.replace(before, after)
    preview = preview_sdpa(source)
    assert preview.modified == source
    assert not preview.diff
    assert preview.sites[0].status == "unsupported"
    assert reason in preview.sites[0].reason


@pytest.mark.parametrize("encoding", ["utf-8-sig", "latin-1"])
def test_source_encoding_and_crlf_survive(encoding):
    text = (
        SOURCE.decode()
        .replace("import math", "# coding: " + encoding + "\n# café\nimport math")
        .replace("\n", "\r\n")
    )
    source = text.encode(encoding)
    preview = preview_sdpa(source)
    assert preview.sites[0].status == "supported"
    result = preview.modified.decode(encoding)
    assert "# café\r\n" in result
    assert "\n" not in result.replace("\r\n", "")


def test_multiple_classes_and_unrelated_code_are_preserved():
    other = SOURCE.replace(b"CausalSelfAttention", b"OtherAttention")
    source = SOURCE + b"\n# Untouched module-level code\nvalue = 7\n\n" + other
    preview = preview_sdpa(source)
    assert [site.status for site in preview.sites] == ["supported", "supported"]
    assert b"# Untouched module-level code\nvalue = 7" in preview.modified


def test_parenthesized_shadowed_calls_are_refused():
    source = SOURCE.replace(b"F.softmax", b"(F . softmax)").replace(
        b"import torch.nn.functional as F", b"import custom as F"
    )
    preview = preview_sdpa(source)
    assert preview.modified == source
    assert preview.sites[0].status == "unsupported"
    assert "binding" in preview.sites[0].reason


def test_comments_inside_expressions_survive():
    source = SOURCE.replace(
        b"att = F.softmax(att, dim=-1)",
        b"att = F.softmax(\n            att, # Keep this explanation.\n            dim=-1\n        )",
    )
    preview = preview_sdpa(source)
    assert preview.sites[0].status == "supported"
    assert preview.modified.count(b"# Keep this explanation.") == 1


def test_sdpa_branch_is_skipped_in_its_entirety():
    source = SOURCE.replace(
        b"        # Attention starts here.",
        b"        if self.flash:\n"
        b"            return F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        b"        # Attention starts here.",
    )
    preview = preview_sdpa(source)
    assert preview.modified == source
    assert preview.sites[0].status == "already_migrated"


def test_cli_preview_never_executes_or_writes_target_code(tmp_path, capsys):
    source = SOURCE + b"\nraise RuntimeError('target code was executed')\n"
    model = tmp_path / "model.py"
    model.write_bytes(source)
    assert nano_cli.main(["sdpa", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["verification"] == "not_run"
    assert payload["sites"][0]["status"] == "supported"
    assert set(payload["diffs"]) == {"model.py"}
    assert model.read_bytes() == source
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.py"]


def test_scanner_prunes_environments_and_symlinks(tmp_path, capsys):
    (tmp_path / "model.py").write_bytes(SOURCE)
    for name in (".venv", "node_modules", ".git", "build"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "bad.py").write_text("not python!")
    (tmp_path / "linked.py").symlink_to(tmp_path / "model.py")
    (tmp_path / "linked_directory").symlink_to(tmp_path, target_is_directory=True)
    assert nano_cli.main(["sdpa", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_scanned"] == 1
    assert len(payload["sites"]) == 1
    assert payload["errors"] == []


def test_scan_errors_are_reported_alongside_valid_diffs(tmp_path, capsys):
    (tmp_path / "good.py").write_bytes(SOURCE)
    (tmp_path / "broken.py").write_text("def broken(:")
    assert nano_cli.main(["sdpa", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["errors"][0]["file"] == "broken.py"
    assert "good.py" in payload["diffs"]


def test_missing_repo_and_empty_repo_have_distinct_results(tmp_path, capsys):
    assert nano_cli.main(["sdpa", str(tmp_path / "missing"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["error"]
    assert nano_cli.main(["sdpa", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "no attention candidates" in output
    assert "Numerical verification is pending" in output


def test_pipeline_exposes_the_same_preview(tmp_path):
    (tmp_path / "model.py").write_bytes(SOURCE)
    result = run_sdpa_preview(str(tmp_path))
    assert result.ok
    assert result.payload["sites"][0]["status"] == "supported"
    assert "model.py" in result.payload["diffs"]


def test_emitted_code_matches_outputs_gradients_and_state_dict():
    torch = pytest.importorskip("torch")
    preview = preview_sdpa(SOURCE)
    original_namespace: dict = {}
    migrated_namespace: dict = {}
    # These are the controlled test fixture and its emitted diff.
    exec(compile(SOURCE, "<original fixture>", "exec"), original_namespace)
    exec(compile(preview.modified, "<migrated fixture>", "exec"), migrated_namespace)
    config = SimpleNamespace(n_embd=16, n_head=2, block_size=32, attn_pdrop=0.25, resid_pdrop=0.0)
    torch.manual_seed(42)
    original = original_namespace["CausalSelfAttention"](config)
    migrated = migrated_namespace["CausalSelfAttention"](config)
    migrated.load_state_dict(original.state_dict(), strict=True)
    # Exercise the dropout submodule's own mode independently of its parent.
    original.attn_dropout.eval()
    migrated.attn_dropout.eval()
    assert migrated.training
    for length in (1, 8, 32):
        x = torch.randn(2, length, config.n_embd, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        out_original = original(x)
        out_migrated = migrated(y)
        torch.testing.assert_close(out_original, out_migrated, rtol=1e-5, atol=1e-6)
        upstream = torch.randn_like(out_original)
        grad_original = torch.autograd.grad(out_original, [x, *original.parameters()], upstream)
        grad_migrated = torch.autograd.grad(out_migrated, [y, *migrated.parameters()], upstream)
        for expected, actual in zip(grad_original, grad_migrated):
            torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
