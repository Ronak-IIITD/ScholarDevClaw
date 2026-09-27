"""Step 4 tests: hardened libcst transformers — idempotent + honest + reversible.

Covers the hardening guarantees for the 10 frozen nanogpt-opt specs:
- Applying a transformer twice yields identical output and zero new changes.
- Change records only contain edits that were actually performed.
- The string fallback renames on word boundaries (no substring doubling)
  and never returns output that is syntactically worse than its input.
- Transformations are reversible byte-identically via captured originals,
  with a no-clobber guard for files edited after the patch was applied.
"""

from __future__ import annotations

import libcst as cst

from scholardevclaw.patch_generation.generator import (
    GEGLUTransformer,
    PatchGenerator,
    QKNormTransformer,
    SwiGLUTransformer,
    _get_transformer,
)

FROZEN_SPEC_KEYS = [
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

SOURCE = """import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim)
        self.mlp = MLP(dim)
    def forward(self, x):
        return nn.GELU(self.fc(x))

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)

wpe = nn.Embedding(1024, 768)
"""


def _bare_generator() -> PatchGenerator:
    """PatchGenerator without running __init__ (no repo path needed)."""
    return PatchGenerator.__new__(PatchGenerator)


# =========================================================================
# Idempotency — all 10 frozen-spec transformers
# =========================================================================


class TestIdempotency:
    def test_all_frozen_spec_transformers_are_idempotent(self):
        for key in FROZEN_SPEC_KEYS:
            first = _get_transformer(key, "LayerNorm", "RMSNorm")
            once = cst.parse_module(SOURCE).visit(first)

            second = _get_transformer(key, "LayerNorm", "RMSNorm")
            twice = cst.parse_module(once.code).visit(second)

            assert twice.code == once.code, f"{key}: second apply changed the output"
            assert second.changes == [], f"{key}: second apply recorded {second.changes}"

    def test_pipeline_transformation_is_idempotent(self, tmp_path):
        (tmp_path / "model.py").write_text(SOURCE)
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {"file": "model.py", "context": {"original": "LayerNorm", "replacement": "RMSNorm"}}
            ],
            "research_spec": {"algorithm": {"name": "RMSNorm"}, "changes": {}},
        }

        first_run = gen._create_transformations(mapping)
        assert len(first_run) == 1
        assert "RMSNorm" in first_run[0].modified
        assert first_run[0].cst_changes, "cst_changes must capture the actual CST edits"

        # Simulate applying the patch, then run the same mapping again:
        # an already-transformed repo must be a no-op.
        (tmp_path / "model.py").write_text(first_run[0].modified)
        second_run = gen._create_transformations(mapping)
        assert second_run == []


# =========================================================================
# Honest change records
# =========================================================================


class TestHonestChanges:
    def test_swiglu_performs_and_records_activation_swap(self):
        tree = cst.parse_module(SOURCE)
        transformer = SwiGLUTransformer()
        out = tree.visit(transformer)

        assert "class SwiGLU" in out.code
        assert "SiLU" in out.code
        assert "GELU" not in out.code
        swaps = [c for c in transformer.changes if c["type"] == "replace_activation"]
        assert swaps == [{"type": "replace_activation", "from": "GELU", "to": "SiLU"}]
        # References to the renamed class follow it (no dangling MLP call sites).
        assert "MLP(" not in out.code.replace("SwiGLU(", "")

    def test_geglu_does_not_claim_activation_it_cannot_make(self):
        tree = cst.parse_module(SOURCE)
        transformer = GEGLUTransformer()
        out = tree.visit(transformer)

        assert "class GEGLU" in out.code
        assert "GEGLU(" in out.code  # references renamed with the class
        # nn.GELU stays as-is: gated GELU is not an in-place rename we can do.
        assert "nn.GELU" in out.code
        assert all(c["type"] != "replace_activation" for c in transformer.changes)

    def test_qknorm_renames_class_and_references(self):
        tree = cst.parse_module(SOURCE)
        transformer = QKNormTransformer()
        out = tree.visit(transformer)

        assert "class QKNormCausalSelfAttention" in out.code
        assert "QKNormCausalSelfAttention(dim)" in out.code
        # No dangling references to the old name.
        assert "= CausalSelfAttention(" not in out.code

    def test_apply_transformation_detailed_reports_honest_changes(self):
        gen = _bare_generator()
        modified, changes = gen._apply_transformation_detailed(
            "class LayerNorm:\n    pass\n", "LayerNorm", "RMSNorm", "rmsnorm"
        )
        assert modified == "class RMSNorm:\n    pass\n"
        assert changes, "cst_changes must be reported"
        for entry in changes:
            assert set(entry) >= {"type", "from", "to"}
            assert entry["to"] == "RMSNorm"

    def test_degenerate_inputs_return_source_unchanged(self):
        gen = _bare_generator()
        source = "class LayerNorm:\n    pass\n"
        assert gen._apply_transformation_detailed(source, "LayerNorm", "LayerNorm", "rmsnorm") == (
            source,
            [],
        )
        assert gen._apply_transformation_detailed(source, "", "RMSNorm", "rmsnorm") == (source, [])


# =========================================================================
# String fallback safety
# =========================================================================


class TestStringFallbackSafety:
    def test_never_double_applies_substring(self):
        gen = _bare_generator()
        # "MyLayerNorm" contains "LayerNorm" but not as a whole word.
        source = "x = MyLayerNorm(4)\n"
        assert gen._string_replace_safe(source, "LayerNorm", "RMSNorm") == source

    def test_word_boundary_hit_and_second_run_noop(self):
        gen = _bare_generator()
        once = gen._string_replace_safe("x = LayerNorm(4)\n", "LayerNorm", "RMSNorm")
        assert once == "x = RMSNorm(4)\n"
        # Second run: "RMSNorm" has no whole-word "LayerNorm" left.
        assert gen._string_replace_safe(once, "LayerNorm", "RMSNorm") == once

    def test_preserves_legacy_parse_error_rename(self):
        gen = _bare_generator()
        broken = "class LayerNorm{:\n    pass\n"
        result = gen._string_replace_safe(broken, "LayerNorm", "RMSNorm")
        assert "RMSNorm" in result

    def test_ensure_valid_never_worsens_input(self):
        assert PatchGenerator._ensure_valid_source("x = 1\n", "x = )(") == "x = 1\n"
        assert PatchGenerator._ensure_valid_source("x = 1\n", "x = 2\n") == "x = 2\n"
        assert PatchGenerator._ensure_valid_source("x = 1\n", "x = 1\n") == "x = 1\n"


# =========================================================================
# Reversibility
# =========================================================================


class TestReversibility:
    def _apply_patch(self, tmp_path) -> tuple[list, str]:
        (tmp_path / "model.py").write_text(SOURCE)
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {"file": "model.py", "context": {"original": "LayerNorm", "replacement": "RMSNorm"}}
            ],
            "research_spec": {"algorithm": {"name": "RMSNorm"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) == 1
        (tmp_path / "model.py").write_text(transformations[0].modified)
        return transformations, SOURCE

    def test_revert_restores_original_byte_identical(self, tmp_path):
        transformations, original = self._apply_patch(tmp_path)
        restored = PatchGenerator.revert_transformations(tmp_path, transformations)
        assert restored == ["model.py"]
        assert (tmp_path / "model.py").read_text() == original

    def test_revert_refuses_to_clobber_later_edits(self, tmp_path):
        transformations, _ = self._apply_patch(tmp_path)
        target = tmp_path / "model.py"
        target.write_text(target.read_text() + "# later edit\n")

        restored = PatchGenerator.revert_transformations(tmp_path, transformations)
        assert restored == []
        assert "# later edit" in target.read_text()

    def test_revert_accepts_payload_dicts(self, tmp_path):
        (tmp_path / "model.py").write_text("changed\n")
        payload = [{"file": "model.py", "original": SOURCE, "modified": "changed\n"}]
        restored = PatchGenerator.revert_transformations(tmp_path, payload)
        assert restored == ["model.py"]
        assert (tmp_path / "model.py").read_text() == SOURCE

    def test_revert_blocks_path_traversal(self, tmp_path):
        outside = tmp_path.parent / "outside_target.py"
        outside.write_text("changed\n")
        payload = [
            {
                "file": f"../{outside.name}",
                "original": "original\n",
                "modified": "changed\n",
            }
        ]
        restored = PatchGenerator.revert_transformations(tmp_path, payload)
        assert restored == []
        assert outside.read_text() == "changed\n"
        outside.unlink()
