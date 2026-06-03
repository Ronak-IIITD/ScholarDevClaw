"""Tests for TreeSitterAnalyzer performance optimizations."""

from __future__ import annotations

from pathlib import Path

from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer


class TestFileCacheOptimization:
    """Tests for _file_cache and _get_files_for_language."""

    def test_file_cache_populated_after_analyze(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def foo(): pass\n")
        analyzer = TreeSitterAnalyzer(tmp_path)
        analyzer.analyze()
        assert "python" in analyzer._file_cache
        assert len(analyzer._file_cache["python"]) == 1

    def test_file_cache_returns_consistent_results(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        analyzer = TreeSitterAnalyzer(tmp_path)
        first = analyzer._get_files_for_language("python")
        second = analyzer._get_files_for_language("python")
        assert first is second  # same list object from cache

    def test_get_files_unknown_language_returns_empty(self, tmp_path: Path) -> None:
        analyzer = TreeSitterAnalyzer(tmp_path)
        result = analyzer._get_files_for_language("klingon")
        assert result == []

    def test_file_cache_excludes_ignored_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "good.py").write_text("x = 1\n")
        ignored_dir = tmp_path / "node_modules"
        ignored_dir.mkdir()
        (ignored_dir / "bad.py").write_text("y = 2\n")
        analyzer = TreeSitterAnalyzer(tmp_path)
        files = analyzer._get_files_for_language("python")
        assert all("node_modules" not in str(f) for f in files)


class TestSinglePassWalk:
    """Tests for _walk_for_elements_and_imports."""

    def test_walk_returns_both_elements_and_imports(self, tmp_path: Path) -> None:
        code = "import os\nimport sys\ndef main(): pass\nclass Foo: pass\n"
        (tmp_path / "mod.py").write_text(code)
        analyzer = TreeSitterAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert len(result.elements) >= 2  # main + Foo
        assert len(result.imports) >= 2  # import os, import sys

    def test_walk_with_no_imports(self, tmp_path: Path) -> None:
        (tmp_path / "pure.py").write_text("def a(): pass\ndef b(): pass\n")
        analyzer = TreeSitterAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert len(result.elements) >= 2
        assert len(result.imports) == 0


class TestSourceParameter:
    """Tests for optional source parameter on extract functions."""

    def test_extract_elements_with_source(self, tmp_path: Path) -> None:
        code = b"def hello(): pass\n"
        (tmp_path / "mod.py").write_bytes(code)
        analyzer = TreeSitterAnalyzer(tmp_path)
        parser = analyzer._get_parser("python")
        tree = parser.parse(code)
        elements = analyzer._extract_elements_from_tree(
            tree, tmp_path / "mod.py", "python", source=code
        )
        assert len(elements) >= 1

    def test_extract_imports_with_source(self, tmp_path: Path) -> None:
        code = b"import json\n"
        (tmp_path / "mod.py").write_bytes(code)
        analyzer = TreeSitterAnalyzer(tmp_path)
        parser = analyzer._get_parser("python")
        tree = parser.parse(code)
        imports = analyzer._extract_imports_from_tree(
            tree, tmp_path / "mod.py", "python", source=code
        )
        assert len(imports) >= 1

    def test_extract_elements_without_source(self, tmp_path: Path) -> None:
        code = b"def world(): pass\n"
        (tmp_path / "mod.py").write_bytes(code)
        analyzer = TreeSitterAnalyzer(tmp_path)
        parser = analyzer._get_parser("python")
        tree = parser.parse(code)
        elements = analyzer._extract_elements_from_tree(tree, tmp_path / "mod.py", "python")
        assert len(elements) >= 1


class TestAnalyzeEndToEnd:
    """End-to-end analysis producing correct RepoAnalysis."""

    def test_analyze_python_repo(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            "import os\nfrom pathlib import Path\ndef main(): pass\nclass App: pass\n"
        )
        analyzer = TreeSitterAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert "python" in result.languages
        assert len(result.elements) >= 2
        assert len(result.imports) >= 2

    def test_analyze_empty_repo(self, tmp_path: Path) -> None:
        analyzer = TreeSitterAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert result.languages == []
        assert result.elements == []
        assert result.imports == []
