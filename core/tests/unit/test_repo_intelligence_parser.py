"""Tests for the parser module — PyTorchRepoParser & PyTorchComponentVisitor."""

from pathlib import Path

import libcst as cst

from scholardevclaw.repo_intelligence.parser import (
    ClassInfo,
    FunctionInfo,
    ImportInfo,
    ModuleInfo,
    PyTorchComponentVisitor,
    PyTorchRepoParser,
)

# =========================================================================
# PyTorchComponentVisitor
# =========================================================================


def _parse_source(source: str):
    tree = cst.parse_module(source)
    visitor = PyTorchComponentVisitor()
    tree.visit(visitor)
    return visitor


class TestPyTorchComponentVisitor:
    def test_visit_import_simple(self):
        visitor = _parse_source("import os\nimport sys\n")
        assert any(i.name == "os" and i.alias is None for i in visitor.imports)
        assert any(i.name == "sys" and i.alias is None for i in visitor.imports)

    def test_visit_import_alias(self):
        visitor = _parse_source("import numpy as np\n")
        assert any(i.name == "numpy" and i.alias == "np" for i in visitor.imports)

    def test_visit_import_from(self):
        visitor = _parse_source("from pathlib import Path\n")
        assert any(
            i.name == "Path" and i.from_module == "pathlib" and i.alias is None
            for i in visitor.imports
        )

    def test_visit_import_from_alias(self):
        visitor = _parse_source("from collections import OrderedDict as OD\n")
        assert any(i.name == "OrderedDict" and i.alias == "OD" for i in visitor.imports)

    def test_visit_import_from_module(self):
        visitor = _parse_source("from . import local\n")
        assert any(i.name == "local" and i.from_module == "" for i in visitor.imports)

    def test_visit_class_def_basic(self):
        visitor = _parse_source("class MyClass:\n    pass\n")
        assert len(visitor.classes) == 1
        assert visitor.classes[0].name == "MyClass"
        assert visitor.classes[0].is_nn_module is False

    def test_visit_class_def_nn_module(self):
        visitor = _parse_source("import torch.nn as nn\nclass MyModel(nn.Module):\n    pass\n")
        assert len(visitor.classes) == 1
        assert visitor.classes[0].is_nn_module is True

    def test_visit_class_def_with_bases(self):
        visitor = _parse_source("class A(B, C):\n    pass\n")
        assert len(visitor.classes) == 1
        assert "B" in visitor.classes[0].bases
        assert "C" in visitor.classes[0].bases

    def test_visit_class_def_methods(self):
        source = """
class Model:
    def forward(self, x):
        return x
    def train(self):
        pass
"""
        visitor = _parse_source(source)
        assert "forward" in visitor.classes[0].methods
        assert "train" in visitor.classes[0].methods

    def test_visit_class_def_attributes_annassign(self):
        source = """
class Model:
    hidden_size: int = 768
    num_layers: int = 12
"""
        visitor = _parse_source(source)
        assert "hidden_size" in visitor.classes[0].attributes
        assert "num_layers" in visitor.classes[0].attributes

    def test_visit_class_def_attributes_assign(self):
        source = """
class Config:
    lr = 0.001
    batch_size = 32
"""
        visitor = _parse_source(source)
        assert "lr" in visitor.classes[0].attributes
        assert "batch_size" in visitor.classes[0].attributes

    def test_visit_class_def_custom_norm(self):
        visitor = _parse_source("class MyNorm:\n    pass\n")
        assert len(visitor.classes) == 1
        # "Norm" is in "MyNorm" and there's no nn. prefix in bases
        assert visitor.classes[0].is_custom_norm is True

    def test_visit_function_def(self):
        visitor = _parse_source("def foo(a, b, c): pass\n")
        assert len(visitor.functions) == 1
        assert visitor.functions[0].name == "foo"
        assert visitor.functions[0].parameters == ["a", "b", "c"]

    def test_visit_function_def_no_params(self):
        visitor = _parse_source("def bar(): pass\n")
        assert visitor.functions[0].parameters == []

    def test_empty_file(self):
        visitor = _parse_source("")
        assert len(visitor.classes) == 0
        assert len(visitor.functions) == 0
        assert len(visitor.imports) == 0


# =========================================================================
# PyTorchRepoParser — unit tests for helper methods
# =========================================================================


class TestPyTorchRepoParserHelpers:
    def test_should_ignore_git(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        assert parser._should_ignore(Path("/tmp/.git/config")) is True

    def test_should_ignore_pycache(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        assert parser._should_ignore(Path("/tmp/pkg/__pycache__/mod.py")) is True

    def test_should_ignore_hidden_file(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        assert parser._should_ignore(Path("/tmp/.hidden.py")) is True

    def test_should_ignore_ipynb_checkpoints(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        assert parser._should_ignore(Path("/tmp/.ipynb_checkpoints/foo.py")) is True

    def test_should_not_ignore_normal_file(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        assert parser._should_ignore(Path("/tmp/src/mod.py")) is False

    def test_extract_components_layers(self):
        cls = ClassInfo("Model", ["nn.Module"], 1, attributes=["layer1", "block_a"])
        mod = ModuleInfo(Path("m.py"), Path("m.py"))
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "layers" in comps

    def test_extract_components_norm(self):
        cls = ClassInfo("Model", ["nn.Module"], 1, attributes=["norm1"])
        mod = ModuleInfo(Path("m.py"), Path("m.py"))
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "normalization" in comps

    def test_extract_components_attention(self):
        cls = ClassInfo("Model", ["nn.Module"], 1, attributes=["attention"])
        mod = ModuleInfo(Path("m.py"), Path("m.py"))
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "attention" in comps

    def test_extract_components_embed(self):
        cls = ClassInfo("Model", ["nn.Module"], 1, attributes=["embedding"])
        mod = ModuleInfo(Path("m.py"), Path("m.py"))
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "embeddings" in comps

    def test_extract_components_mlp(self):
        cls = ClassInfo("Model", ["nn.Module"], 1, attributes=["mlp"])
        mod = ModuleInfo(Path("m.py"), Path("m.py"))
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "mlp" in comps

    def test_extract_components_custom_norm(self):
        cls = ClassInfo("Model", ["nn.Module"], 1)
        norm_cls = ClassInfo("RMSNorm", ["nn.Module"], 2, is_custom_norm=True)
        mod = ModuleInfo(Path("m.py"), Path("m.py"), classes=[cls, norm_cls])
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "custom_norms" in comps
        assert "RMSNorm" in comps["custom_norms"]

    def test_extract_components_transformer_import(self):
        cls = ClassInfo("Model", ["nn.Module"], 1)
        imp = ImportInfo(name="transformers")
        mod = ModuleInfo(Path("m.py"), Path("m.py"), classes=[cls], imports=[imp])
        parser = PyTorchRepoParser(Path("/tmp"))
        comps = parser._extract_components(cls, mod)
        assert "transformer" in comps

    def test_detect_models_finds_nn_module(self):
        cls = ClassInfo("GPT", ["nn.Module"], 1, is_nn_module=True)
        mod = ModuleInfo(Path("mod.py"), Path("mod.py"), classes=[cls])
        parser = PyTorchRepoParser(Path("/tmp"))
        parser.modules = [mod]
        models = parser._detect_models()
        assert len(models) == 1
        assert models[0].name == "GPT"

    def test_detect_models_empty(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        parser.modules = []
        assert parser._detect_models() == []

    def test_detect_training_loop_found(self):
        func = FunctionInfo("train", 10)
        imp = ImportInfo(name="AdamW", from_module="torch.optim")
        mod = ModuleInfo(Path("train.py"), Path("train.py"), functions=[func], imports=[imp])
        parser = PyTorchRepoParser(Path("/tmp"))
        parser.modules = [mod]
        loop = parser._detect_training_loop()
        assert loop is not None
        assert loop.optimizer == "AdamW"
        assert "train.py" in loop.file

    def test_detect_training_loop_fallback(self):
        parser = PyTorchRepoParser(Path("/tmp"))
        parser.modules = []
        loop = parser._detect_training_loop()
        assert loop is not None
        assert loop.file == "train.py"

    def test_detect_training_loop_no_torch(self):
        func = FunctionInfo("train", 10)
        imp = ImportInfo(name="json")
        mod = ModuleInfo(Path("train.py"), Path("train.py"), functions=[func], imports=[imp])
        parser = PyTorchRepoParser(Path("/tmp"))
        parser.modules = [mod]
        # Should fallback because no torch.optim import
        loop = parser._detect_training_loop()
        assert loop.file == "train.py"


# =========================================================================
# PyTorchRepoParser — integration tests with temp files
# =========================================================================


class TestPyTorchRepoParserIntegration:
    def test_parse_empty_repo(self, tmp_path: Path):
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert result.repo_name == tmp_path.name
        assert len(result.modules) == 0

    def test_parse_single_file(self, tmp_path: Path):
        src = tmp_path / "model.py"
        src.write_text(
            "import torch.nn as nn\n"
            "class MyModel(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
        )
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert len(result.modules) == 1
        assert result.modules[0].relative_path == Path("model.py")

    def test_parse_detects_nn_module(self, tmp_path: Path):
        src = tmp_path / "model.py"
        src.write_text("import torch.nn as nn\nclass MyModel(nn.Module):\n    pass\n")
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert len(result.models) == 1
        assert result.models[0].name == "MyModel"

    def test_parse_ignores_pycache(self, tmp_path: Path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("x = 1\n")
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert len(result.modules) == 0

    def test_parse_finds_test_files(self, tmp_path: Path):
        (tmp_path / "test_model.py").write_text("def test_foo(): pass\n")
        (tmp_path / "model_test.py").write_text("def test_bar(): pass\n")
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert any("test_model.py" in tf for tf in result.test_files)
        assert any("model_test.py" in tf for tf in result.test_files)

    def test_parse_syntax_error_file(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("this is not valid python {{{{\n")
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        # Syntax error file is skipped; no modules
        assert len(result.modules) == 0

    def test_parse_empty_file(self, tmp_path: Path):
        (tmp_path / "empty.py").write_text("")
        parser = PyTorchRepoParser(tmp_path)
        result = parser.parse()
        assert len(result.modules) == 1  # Empty file is valid Python
        assert len(result.modules[0].classes) == 0
