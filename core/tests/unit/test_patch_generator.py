"""Comprehensive tests for the patch generation engine (patch_generation/generator.py).

Covers:
  - CST Transformers: RMSNormTransformer, SwiGLUTransformer, GEGLUTransformer,
    FlashAttentionTransformer, GQATransformer, QKNormTransformer, PreLNTransformer,
    RoPETransformer, ALiBiTransformer, GenericRenameTransformer
  - _get_transformer() dispatch — exact key, prefix match, fallback
  - Template functions — verify output contains expected strings
  - PatchGenerator.generate() — with template, without template (no LLM)
  - PatchGenerator._create_new_files() — known algorithm, unknown algorithm
  - PatchGenerator._create_transformations() — existing file, non-existent file, path traversal
  - _apply_transformation() — AST success, string fallback
  - NewFile, Transformation, Patch dataclass construction
"""

from __future__ import annotations

import sys
from typing import cast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import libcst as cst

from scholardevclaw.patch_generation.generator import (
    _TEMPLATE_REGISTRY,
    ALiBiTransformer,
    FlashAttentionTransformer,
    GEGLUTransformer,
    GenericRenameTransformer,
    GQATransformer,
    NewFile,
    Patch,
    PatchGenerator,
    PreLNTransformer,
    QKNormTransformer,
    RMSNormTransformer,
    RoPETransformer,
    SwiGLUTransformer,
    Transformation,
    _get_transformer,
)

# =========================================================================
# Dataclass tests
# =========================================================================


class TestNewFile:
    def test_construction(self):
        nf = NewFile(path="rmsnorm.py", content="class RMSNorm: pass")
        assert nf.path == "rmsnorm.py"
        assert "RMSNorm" in nf.content


class TestTransformation:
    def test_construction(self):
        t = Transformation(file="model.py", original="class A:", modified="class B:", changes=[])
        assert t.file == "model.py"
        assert t.original == "class A:"

    def test_default_changes(self):
        t = Transformation(file="f.py", original="", modified="")
        assert t.changes == []


class TestPatch:
    def test_construction(self):
        p = Patch(
            new_files=[NewFile(path="f.py", content="x")],
            transformations=[],
            branch_name="integration/rmsnorm",
            algorithm_name="RMSNorm",
            paper_reference="arXiv:1910.07467",
        )
        assert p.branch_name == "integration/rmsnorm"
        assert len(p.new_files) == 1


# =========================================================================
# CST Transformer tests
# =========================================================================


class TestRMSNormTransformer:
    def test_renames_class(self):
        source = "class LayerNorm:\n    pass\n"
        tree = cst.parse_module(source)
        t = RMSNormTransformer("LayerNorm", "RMSNorm")
        modified = tree.visit(t)
        assert "class RMSNorm" in modified.code
        assert "LayerNorm" not in modified.code
        assert len(t.changes) >= 1

    def test_renames_references(self):
        source = "x = LayerNorm(dim)\n"
        tree = cst.parse_module(source)
        t = RMSNormTransformer("LayerNorm", "RMSNorm")
        modified = tree.visit(t)
        assert "RMSNorm" in modified.code
        assert len(t.changes) >= 1

    def test_leaves_unrelated_classes(self):
        source = "class Dropout:\n    pass\n"
        tree = cst.parse_module(source)
        t = RMSNormTransformer("LayerNorm", "RMSNorm")
        modified = tree.visit(t)
        assert "class Dropout" in modified.code
        assert len(t.changes) == 0


class TestSwiGLUTransformer:
    def test_renames_mlp_to_swiglu(self):
        source = "class MLP:\n    pass\n"
        tree = cst.parse_module(source)
        t = SwiGLUTransformer()
        modified = tree.visit(t)
        assert "class SwiGLU" in modified.code
        assert len(t.changes) >= 1

    def test_leaves_non_mlp(self):
        source = "class Attention:\n    pass\n"
        tree = cst.parse_module(source)
        t = SwiGLUTransformer()
        modified = tree.visit(t)
        assert "class Attention" in modified.code
        assert len(t.changes) == 0


class TestGEGLUTransformer:
    def test_renames_mlp_to_geglu(self):
        source = "class MLP:\n    pass\n"
        tree = cst.parse_module(source)
        t = GEGLUTransformer()
        modified = tree.visit(t)
        assert "class GEGLU" in modified.code


class TestFlashAttentionTransformer:
    def test_renames_attention_class(self):
        source = "class CausalSelfAttention:\n    pass\n"
        tree = cst.parse_module(source)
        t = FlashAttentionTransformer()
        modified = tree.visit(t)
        assert "FlashCausalSelfAttention" in modified.code

    def test_renames_references(self):
        source = "attn = CausalSelfAttention(config)\n"
        tree = cst.parse_module(source)
        t = FlashAttentionTransformer()
        modified = tree.visit(t)
        assert "FlashCausalSelfAttention" in modified.code


class TestGQATransformer:
    def test_renames_causal_self_attention(self):
        source = "class CausalSelfAttention:\n    pass\n"
        tree = cst.parse_module(source)
        t = GQATransformer()
        modified = tree.visit(t)
        assert "GroupedQueryAttention" in modified.code

    def test_renames_multi_head_attention(self):
        source = "class MultiHeadAttention:\n    pass\n"
        tree = cst.parse_module(source)
        t = GQATransformer()
        modified = tree.visit(t)
        assert "GroupedQueryAttention" in modified.code


class TestQKNormTransformer:
    def test_renames_attention(self):
        source = "class CausalSelfAttention:\n    pass\n"
        tree = cst.parse_module(source)
        t = QKNormTransformer()
        modified = tree.visit(t)
        assert "QKNormCausalSelfAttention" in modified.code

    def test_renames_attention_alias(self):
        source = "class Attention:\n    pass\n"
        tree = cst.parse_module(source)
        t = QKNormTransformer()
        modified = tree.visit(t)
        assert "QKNormAttention" in modified.code


class TestPreLNTransformer:
    def test_renames_block(self):
        source = "class Block:\n    pass\n"
        tree = cst.parse_module(source)
        t = PreLNTransformer()
        modified = tree.visit(t)
        assert "PreLNBlock" in modified.code

    def test_renames_block_reference(self):
        source = "b = Block(config)\n"
        tree = cst.parse_module(source)
        t = PreLNTransformer()
        modified = tree.visit(t)
        assert "PreLNBlock" in modified.code


class TestRoPETransformer:
    def test_renames_wpe(self):
        source = "x = wpe(pos)\n"
        tree = cst.parse_module(source)
        t = RoPETransformer()
        modified = tree.visit(t)
        assert "rotary_emb" in modified.code

    def test_renames_uppercase(self):
        source = "x = PositionalEncoding(dim)\n"
        tree = cst.parse_module(source)
        t = RoPETransformer()
        modified = tree.visit(t)
        assert "RotaryPositionalEmbedding" in modified.code


class TestALiBiTransformer:
    def test_renames_wpe(self):
        source = "x = wpe(pos)\n"
        tree = cst.parse_module(source)
        t = ALiBiTransformer()
        modified = tree.visit(t)
        assert "alibi_bias" in modified.code

    def test_renames_uppercase(self):
        source = "x = PositionalEncoding(dim)\n"
        tree = cst.parse_module(source)
        t = ALiBiTransformer()
        modified = tree.visit(t)
        assert "ALiBiPositionalBias" in modified.code


class TestGenericRenameTransformer:
    def test_renames_class(self):
        source = "class Foo:\n    pass\n"
        tree = cst.parse_module(source)
        t = GenericRenameTransformer("Foo", "Bar")
        modified = tree.visit(t)
        assert "class Bar" in modified.code

    def test_renames_function(self):
        source = "def compute():\n    pass\n"
        tree = cst.parse_module(source)
        t = GenericRenameTransformer("compute", "fast_compute")
        modified = tree.visit(t)
        assert "fast_compute" in modified.code

    def test_renames_references(self):
        source = "x = Foo()\n"
        tree = cst.parse_module(source)
        t = GenericRenameTransformer("Foo", "Bar")
        modified = tree.visit(t)
        assert "Bar" in modified.code
        assert "Foo" not in modified.code


# =========================================================================
# _get_transformer dispatch
# =========================================================================


class TestGetTransformer:
    def test_exact_key_rmsnorm(self):
        t = _get_transformer("rmsnorm", "LayerNorm", "RMSNorm")
        assert isinstance(t, RMSNormTransformer)

    def test_exact_key_swiglu(self):
        t = _get_transformer("swiglu", "MLP", "SwiGLU")
        assert isinstance(t, SwiGLUTransformer)

    def test_exact_key_geglu(self):
        t = _get_transformer("geglu", "MLP", "GEGLU")
        assert isinstance(t, GEGLUTransformer)

    def test_exact_key_flashattention(self):
        t = _get_transformer("flashattention", "CausalSelfAttention", "FlashCausalSelfAttention")
        assert isinstance(t, FlashAttentionTransformer)

    def test_exact_key_gqa(self):
        t = _get_transformer("grouped_query_attention", "CausalSelfAttention", "GQA")
        assert isinstance(t, GQATransformer)

    def test_exact_key_rope(self):
        t = _get_transformer("rope", "wpe", "rotary_emb")
        assert isinstance(t, RoPETransformer)

    def test_exact_key_alibi(self):
        t = _get_transformer("alibi", "wpe", "alibi_bias")
        assert isinstance(t, ALiBiTransformer)

    def test_prefix_match(self):
        """'flash_attention_2' should match 'flashattention' prefix."""
        t = _get_transformer("flashattention_v3", "x", "y")
        assert isinstance(t, FlashAttentionTransformer)

    def test_fallback_generic(self):
        t = _get_transformer("totally_unknown_algorithm", "Foo", "Bar")
        assert isinstance(t, GenericRenameTransformer)

    def test_dash_and_space_normalization(self):
        t = _get_transformer("rms-norm", "LayerNorm", "RMSNorm")
        assert isinstance(t, RMSNormTransformer)


# =========================================================================
# Template registry
# =========================================================================


class TestTemplateRegistry:
    def test_rmsnorm_template(self):
        fn = _TEMPLATE_REGISTRY["rmsnorm"]
        content = fn(
            {
                "paper": {"title": "T", "authors": ["A"], "year": 2019, "arxiv": "1910.07467"},
                "algorithm": {},
            }
        )
        assert "RMSNorm" in content
        assert "class RMSNorm" in content
        assert "torch" in content

    def test_swiglu_template(self):
        fn = _TEMPLATE_REGISTRY["swiglu"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "SwiGLU" in content

    def test_flashattention_template(self):
        fn = _TEMPLATE_REGISTRY["flashattention"]
        content = fn(
            {"paper": {"title": "T", "authors": ["A"]}, "algorithm": {"name": "FlashAttention"}}
        )
        assert "FlashCausalSelfAttention" in content

    def test_gqa_template(self):
        fn = _TEMPLATE_REGISTRY["grouped_query_attention"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "GroupedQueryAttention" in content

    def test_rope_template(self):
        fn = _TEMPLATE_REGISTRY["rope"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "RotaryPositionalEmbedding" in content

    def test_alibi_template(self):
        fn = _TEMPLATE_REGISTRY["alibi"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "ALiBiPositionalBias" in content

    def test_lion_template(self):
        fn = _TEMPLATE_REGISTRY["lion"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "Lion" in content

    def test_cosine_warmup_template(self):
        fn = _TEMPLATE_REGISTRY["cosine_warmup"]
        content = fn({"paper": {"title": "T", "authors": ["A"]}, "algorithm": {}})
        assert "cosine" in content.lower()

    def test_all_15_templates_exist(self):
        assert len(_TEMPLATE_REGISTRY) == 15


# =========================================================================
# PatchGenerator.generate()
# =========================================================================


class TestPatchGeneratorGenerate:
    def test_generate_with_known_algorithm(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [],
            "research_spec": {
                "algorithm": {"name": "RMSNorm"},
                "paper": {"title": "RMSNorm Paper", "arxiv": "1910.07467", "authors": ["A"]},
                "changes": {"target_patterns": ["LayerNorm"], "replacement": "RMSNorm"},
            },
        }
        patch = gen.generate(mapping)
        assert isinstance(patch, Patch)
        assert patch.branch_name == "integration/rmsnorm"
        assert patch.algorithm_name == "RMSNorm"
        assert patch.paper_reference == "arXiv:1910.07467"
        assert len(patch.new_files) == 1
        assert "RMSNorm" in patch.new_files[0].content

    def test_generate_unknown_algorithm_no_llm(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [],
            "research_spec": {
                "algorithm": {"name": "TotallyNovel"},
                "paper": {"title": "Novel Paper"},
                "changes": {},
            },
        }
        patch = gen.generate(mapping)
        assert isinstance(patch, Patch)
        assert len(patch.new_files) == 0  # No template, no LLM

    def test_generate_branch_name_from_algorithm(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [],
            "research_spec": {
                "algorithm": {"name": "Flash Attention 2"},
                "paper": {},
                "changes": {},
            },
        }
        patch = gen.generate(mapping)
        assert patch.branch_name == "integration/flash-attention-2"

    def test_generate_paper_reference_title_fallback(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [],
            "research_spec": {
                "algorithm": {"name": "X"},
                "paper": {"title": "My Paper Title"},
                "changes": {},
            },
        }
        patch = gen.generate(mapping)
        assert patch.paper_reference == "My Paper Title"


# =========================================================================
# PatchGenerator._create_new_files()
# =========================================================================


class TestCreateNewFiles:
    def test_known_algorithm_uses_template(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        spec = {
            "algorithm": {"name": "RMSNorm"},
            "paper": {"title": "T", "authors": ["A"], "arxiv": "1910.07467"},
        }
        files = gen._create_new_files(spec)
        assert len(files) == 1
        assert files[0].path == "rmsnorm.py"

    def test_unknown_algorithm_no_llm_empty(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        spec = {"algorithm": {"name": "UnknownAlgo"}, "paper": {}}
        files = gen._create_new_files(spec)
        assert len(files) == 0

    def test_unknown_algorithm_with_llm(self, tmp_path):
        plan = SimpleNamespace(steps=[{"code": "class Foo: pass"}])
        llm = SimpleNamespace(
            generate_implementation_plan=lambda **kw: plan,
        )
        gen = PatchGenerator(tmp_path, llm_assistant=cast(object, llm))
        spec = {"algorithm": {"name": "MyAlgo"}, "paper": {"title": "T"}}
        files = gen._create_new_files(spec)
        assert len(files) == 1
        assert "Foo" in files[0].content

    def test_unknown_algorithm_with_llm_sanitizes_filename(self, tmp_path):
        plan = SimpleNamespace(steps=[{"code": "class Foo: pass"}])
        llm = SimpleNamespace(
            generate_implementation_plan=lambda **kw: plan,
        )
        gen = PatchGenerator(tmp_path, llm_assistant=cast(object, llm))
        spec = {"algorithm": {"name": "../../evil\\name"}, "paper": {"title": "T"}}
        files = gen._create_new_files(spec)
        assert len(files) == 1
        assert "/" not in files[0].path
        assert "\\" not in files[0].path
        assert files[0].path.endswith(".py")


# =========================================================================
# PatchGenerator._create_transformations()
# =========================================================================


class TestCreateTransformations:
    def test_transforms_existing_file(self, tmp_path):
        model_file = tmp_path / "model.py"
        model_file.write_text("class LayerNorm:\n    pass\n")
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {
                    "file": "model.py",
                    "context": {"original": "LayerNorm", "replacement": "RMSNorm"},
                }
            ],
            "research_spec": {"algorithm": {"name": "RMSNorm"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) >= 1
        assert "RMSNorm" in transformations[0].modified

    def test_skips_nonexistent_file(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {
                    "file": "nonexistent.py",
                    "context": {"original": "LayerNorm", "replacement": "RMSNorm"},
                }
            ],
            "research_spec": {"algorithm": {"name": "RMSNorm"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) == 0

    def test_path_traversal_blocked(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {
                    "file": "../../../etc/passwd",
                    "context": {"original": "x", "replacement": "y"},
                }
            ],
            "research_spec": {"algorithm": {"name": "X"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) == 0

    def test_no_change_no_transformation(self, tmp_path):
        """If replacement doesn't change the source, no transformation recorded."""
        model_file = tmp_path / "model.py"
        model_file.write_text("class Dropout:\n    pass\n")
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {
                    "file": "model.py",
                    "context": {"original": "LayerNorm", "replacement": "RMSNorm"},
                }
            ],
            "research_spec": {"algorithm": {"name": "RMSNorm"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) == 0

    def test_empty_original_or_replacement_skips(self, tmp_path):
        model_file = tmp_path / "model.py"
        model_file.write_text("class LayerNorm:\n    pass\n")
        gen = PatchGenerator(tmp_path)
        mapping = {
            "targets": [
                {
                    "file": "model.py",
                    "context": {"original": "", "replacement": ""},
                }
            ],
            "research_spec": {"algorithm": {"name": "X"}, "changes": {}},
        }
        transformations = gen._create_transformations(mapping)
        assert len(transformations) == 0


# =========================================================================
# _apply_transformation
# =========================================================================


class TestApplyTransformation:
    def test_ast_transform_success(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        source = "class LayerNorm:\n    pass\n"
        result = gen._apply_transformation(source, "LayerNorm", "RMSNorm", "rmsnorm")
        assert "RMSNorm" in result
        assert "LayerNorm" not in result

    def test_string_fallback_on_parse_error(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        # Intentionally malformed Python that will fail CST parse
        source = "class LayerNorm{:\n    pass\n"
        result = gen._apply_transformation(source, "LayerNorm", "RMSNorm", "rmsnorm")
        assert "RMSNorm" in result

    def test_generic_rename_for_unknown_algo(self, tmp_path):
        gen = PatchGenerator(tmp_path)
        source = "class Foo:\n    pass\n"
        result = gen._apply_transformation(source, "Foo", "Bar", "unknown_algo")
        assert "Bar" in result
