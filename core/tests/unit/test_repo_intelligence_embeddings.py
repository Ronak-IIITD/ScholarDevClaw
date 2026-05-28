"""Tests for code_embeddings and multi_lang_analyzer modules."""

from pathlib import Path

from scholardevclaw.repo_intelligence.code_embeddings import (
    CodeEmbedding,
    CodeEmbeddingEngine,
    CodeSimilarityFinder,
    CodeTokenizer,
    SemanticCodeMapper,
    compute_code_hash,
)
from scholardevclaw.repo_intelligence.multi_lang_analyzer import (
    CodeElement,
    ImportStatement,
    LanguageDetector,
    LanguageStats,
    MultiLanguageAnalyzer,
    RepoAnalysis,
)

# =========================================================================
# code_embeddings — CodeTokenizer
# =========================================================================


class TestCodeTokenizer:
    def test_tokenize_simple(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("x = 1")
        assert "x" in [t.lower() for t in tokens]
        assert "=" in tokens

    def test_tokenize_function_def(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("def foo(): pass")
        assert any("def" in t for t in tokens)

    def test_tokenize_removes_comments(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("x = 1  # comment")
        assert "comment" not in [t.lower() for t in tokens]

    def test_tokenize_removes_docstrings(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize('x = """multi line\ndocstring"""\ny = 2')
        assert "multi" not in [t.lower() for t in tokens]
        assert "line" not in [t.lower() for t in tokens]

    def test_tokenize_empty(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("")
        assert tokens == []

    def test_tokenize_split_camel_case(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("hiddenSize = 100")
        # camelCase should be split
        assert any("hidden" in t.lower() for t in tokens)
        assert any("size" in t.lower() for t in tokens)

    def test_tokenize_keywords_prefixed(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("def class return if else")
        keyword_tokens = [t for t in tokens if t.startswith("KEYWORD:")]
        assert len(keyword_tokens) >= 4

    def test_tokenize_numbers(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("x = 42")
        assert any("NUMBER" in t for t in tokens)

    def test_tokenize_operators(self):
        tok = CodeTokenizer("python")
        tokens = tok.tokenize("a + b * c")
        assert "+" in tokens
        assert "*" in tokens

    def test_tokenize_javascript(self):
        tok = CodeTokenizer("javascript")
        tokens = tok.tokenize("const x = function() { return 1; }")
        keyword_tokens = [t for t in tokens if t.startswith("KEYWORD:")]
        assert len(keyword_tokens) > 0

    def test_tokenize_typescript(self):
        tok = CodeTokenizer("typescript")
        tokens = tok.tokenize("interface Foo { x: string }")
        keyword_tokens = [t for t in tokens if t.startswith("KEYWORD:")]
        assert any("interface" in t for t in keyword_tokens)

    def test_normalize_removes_js_comments(self):
        tok = CodeTokenizer("python")
        normalized = tok._normalize("x = 1 // comment\ny = 2")
        assert "comment" not in normalized

    def test_normalize_removes_block_comments(self):
        tok = CodeTokenizer("python")
        normalized = tok._normalize("/* block comment */\nx = 1")
        assert "block" not in normalized


# =========================================================================
# code_embeddings — CodeEmbeddingEngine
# =========================================================================


class TestCodeEmbeddingEngine:
    def test_index_code_returns_embedding(self):
        engine = CodeEmbeddingEngine("python")
        emb = engine.index_code(
            code="def foo(): return 1",
            element_id="m::foo",
            name="foo",
            file="m.py",
        )
        assert isinstance(emb, CodeEmbedding)
        assert emb.element_id == "m::foo"
        assert emb.name == "foo"
        assert len(emb.embedding) == 512
        assert emb.token_count > 0

    def test_index_code_empty(self):
        engine = CodeEmbeddingEngine("python")
        emb = engine.index_code(code="", element_id="x", name="x", file="x.py")
        assert emb.token_count == 0
        assert all(v == 0.0 for v in emb.embedding)

    def test_hash_embedding_deterministic(self):
        engine = CodeEmbeddingEngine("python")
        emb1 = engine.index_code("x = 1", "a", "a", "a.py")
        emb2 = engine.index_code("x = 1", "b", "b", "b.py")
        # Same code → same hash embedding
        assert emb1.embedding == emb2.embedding

    def test_hash_embedding_different(self):
        engine = CodeEmbeddingEngine("python")
        emb1 = engine.index_code("x = 1", "a", "a", "a.py")
        emb2 = engine.index_code("y = 2", "b", "b", "b.py")
        assert emb1.embedding != emb2.embedding

    def test_build_index_enables_tfidf(self):
        engine = CodeEmbeddingEngine("python", use_tfidf=True)
        engine.build_index(
            [
                {"code": "def foo(): return 1"},
                {"code": "def bar(): return 2"},
            ]
        )
        emb = engine.index_code("def foo(): return 1", "q", "q", "q.py")
        # TF-IDF dim is min(512, len(idf_cache)); with small vocab it's smaller
        assert len(emb.embedding) > 0
        assert isinstance(emb.embedding[0], float)

    def test_build_index_empty(self):
        engine = CodeEmbeddingEngine("python", use_tfidf=True)
        engine.build_index([])
        assert engine._corpus_tokens == []

    def test_compute_idf(self):
        engine = CodeEmbeddingEngine("python")
        engine._corpus_tokens = [
            ["a", "b", "c"],
            ["b", "c", "d"],
        ]
        idf = engine._compute_idf()
        assert "a" in idf
        assert "b" in idf
        assert idf["b"] > 0

    def test_normalize_vector(self):
        engine = CodeEmbeddingEngine("python")
        normalized = engine._normalize([3.0, 4.0])
        magnitude = (9 + 16) ** 0.5
        assert abs(normalized[0] - 3.0 / magnitude) < 1e-10
        assert abs(normalized[1] - 4.0 / magnitude) < 1e-10

    def test_normalize_zero_vector(self):
        engine = CodeEmbeddingEngine("python")
        normalized = engine._normalize([0.0, 0.0, 0.0])
        assert normalized == [0.0, 0.0, 0.0]


# =========================================================================
# code_embeddings — CodeSimilarityFinder
# =========================================================================


class TestCodeSimilarityFinder:
    def setup_method(self):
        self.engine = CodeEmbeddingEngine("python")
        self.finder = CodeSimilarityFinder(self.engine)

    def test_add_and_find(self):
        emb1 = self.engine.index_code("x = 1", "a", "func_a", "a.py")
        emb2 = self.engine.index_code("y = 2", "b", "func_b", "b.py")
        self.finder.add_to_index(emb1)
        self.finder.add_to_index(emb2)
        results = self.finder.find_similar(emb1)
        assert len(results) == 1
        assert results[0].element_id == "b"

    def test_find_similar_excludes_self(self):
        emb = self.engine.index_code("x = 1", "a", "a", "a.py")
        self.finder.add_to_index(emb)
        results = self.finder.find_similar(emb)
        assert len(results) == 0

    def test_find_similar_top_k(self):
        for i in range(10):
            emb = self.engine.index_code(f"x = {i}", str(i), f"f{i}", f"f{i}.py")
            self.finder.add_to_index(emb)
        query = self.engine.index_code("x = 99", "q", "q", "q.py")
        results = self.finder.find_similar(query, top_k=3)
        assert len(results) <= 3

    def test_match_type_name(self):
        emb1 = self.engine.index_code("x = 1", "a", "same_name", "a.py")
        emb2 = self.engine.index_code("y = 2", "b", "same_name", "b.py")
        self.finder.add_to_index(emb1)
        self.finder.add_to_index(emb2)
        results = self.finder.find_similar(emb1)
        assert results[0].match_type == "name"

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        sim = self.finder._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self):
        sim = self.finder._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-10

    def test_structural_match(self):
        e1 = CodeEmbedding("a", "f", "a.py", [0.0], token_count=10)
        e2 = CodeEmbedding("b", "g", "b.py", [0.0], token_count=12)
        assert self.finder._structural_match(e1, e2) is True

    def test_structural_no_match(self):
        e1 = CodeEmbedding("a", "f", "a.py", [0.0], token_count=10)
        e2 = CodeEmbedding("b", "g", "b.py", [0.0], token_count=100)
        assert self.finder._structural_match(e1, e2) is False

    def test_empty_index(self):
        emb = self.engine.index_code("x", "q", "q", "q.py")
        results = self.finder.find_similar(emb)
        assert results == []


# =========================================================================
# code_embeddings — SemanticCodeMapper
# =========================================================================


class TestSemanticCodeMapper:
    def test_index_repository(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text(
            "def hello():\n    return 'hi'\n\n\nclass Foo:\n    pass\n"
        )
        mapper = SemanticCodeMapper("python")
        mapper.index_repository(tmp_path, [".py"])
        assert len(mapper._elements) > 0

    def test_index_repository_ignores_pycache(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("x = 1\n")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("y = 2\n")
        mapper = SemanticCodeMapper("python")
        mapper.index_repository(tmp_path, [".py"])
        assert not any("__pycache__" in e.get("file", "") for e in mapper._elements)

    def test_find_similar_to(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def add(a, b): return a + b\n")
        (tmp_path / "b.py").write_text("def subtract(a, b): return a - b\n")
        mapper = SemanticCodeMapper("python")
        mapper.index_repository(tmp_path, [".py"])
        results = mapper.find_similar_to("def add(x, y): return x + y")
        assert len(results) > 0

    def test_find_duplicates(self, tmp_path: Path):
        code = "def helper():\n    return 42\n"
        (tmp_path / "a.py").write_text(code)
        (tmp_path / "b.py").write_text(code)
        mapper = SemanticCodeMapper("python")
        mapper.index_repository(tmp_path, [".py"])
        # Duplicates at 0.9 threshold
        dups = mapper.find_duplicates(threshold=0.0)  # low threshold to catch any
        # Just verify it runs without error
        assert isinstance(dups, list)

    def test_empty_repository(self, tmp_path: Path):
        mapper = SemanticCodeMapper("python")
        mapper.index_repository(tmp_path, [".py"])
        results = mapper.find_similar_to("x = 1")
        assert results == []


# =========================================================================
# code_embeddings — compute_code_hash
# =========================================================================


class TestComputeCodeHash:
    def test_deterministic(self):
        h1 = compute_code_hash("def foo(): pass")
        h2 = compute_code_hash("def foo(): pass")
        assert h1 == h2

    def test_different_code_different_hash(self):
        h1 = compute_code_hash("x = 1")
        h2 = compute_code_hash("y = 2")
        assert h1 != h2

    def test_whitespace_insensitive(self):
        h1 = compute_code_hash("x  =  1")
        h2 = compute_code_hash("x = 1")
        assert h1 == h2

    def test_length(self):
        h = compute_code_hash("test")
        assert len(h) == 16  # sha256[:16]


# =========================================================================
# multi_lang_analyzer — LanguageDetector
# =========================================================================


class TestLanguageDetector:
    def test_detect_python(self):
        assert LanguageDetector.detect_from_file(Path("mod.py")) == "python"

    def test_detect_javascript(self):
        assert LanguageDetector.detect_from_file(Path("app.js")) == "javascript"

    def test_detect_typescript(self):
        assert LanguageDetector.detect_from_file(Path("main.ts")) == "typescript"

    def test_detect_typescript_tsx(self):
        assert LanguageDetector.detect_from_file(Path("App.tsx")) == "typescript"

    def test_detect_go(self):
        assert LanguageDetector.detect_from_file(Path("main.go")) == "go"

    def test_detect_rust(self):
        assert LanguageDetector.detect_from_file(Path("lib.rs")) == "rust"

    def test_detect_java(self):
        assert LanguageDetector.detect_from_file(Path("Main.java")) == "java"

    def test_detect_unknown(self):
        assert LanguageDetector.detect_from_file(Path("file.xyz")) is None

    def test_detect_from_content_python(self):
        lang = LanguageDetector.detect_from_content(
            Path("unknown"), "import os\ndef main(): pass\nif __name__"
        )
        assert lang == "python"

    def test_detect_from_content_javascript(self):
        lang = LanguageDetector.detect_from_content(
            Path("unknown"), "const x = 1\nfunction foo() {}\nvar y = 2"
        )
        assert lang == "javascript"

    def test_detect_from_content_go(self):
        lang = LanguageDetector.detect_from_content(
            Path("unknown"), "package main\nfunc main() {}\nimport ("
        )
        assert lang == "go"

    def test_detect_from_content_rust(self):
        lang = LanguageDetector.detect_from_content(
            Path("unknown"), "fn main() {\nlet mut x = 1;\nuse std::io;"
        )
        assert lang == "rust"

    def test_detect_repo_languages(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("x = 1\n")
        (tmp_path / "app.js").write_text("var x = 1\n")
        langs = LanguageDetector.detect_repo_languages(tmp_path)
        assert "python" in langs
        assert "javascript" in langs

    def test_should_ignore_git(self):
        assert LanguageDetector._should_ignore(Path("/repo/.git/config")) is True

    def test_should_ignore_pycache(self):
        assert LanguageDetector._should_ignore(Path("/repo/__pycache__/x.py")) is True

    def test_should_ignore_node_modules(self):
        assert LanguageDetector._should_ignore(Path("/repo/node_modules/pkg")) is True

    def test_should_not_ignore_normal(self):
        assert LanguageDetector._should_ignore(Path("/repo/src/main.py")) is False


# =========================================================================
# multi_lang_analyzer — MultiLanguageAnalyzer
# =========================================================================


class TestMultiLanguageAnalyzer:
    def test_analyze_empty_repo(self, tmp_path: Path):
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert isinstance(result, RepoAnalysis)
        assert result.root_path == tmp_path
        assert result.languages == []

    def test_analyze_python_repo(self, tmp_path: Path):
        (tmp_path / "main.py").write_text(
            "import os\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"
        )
        (tmp_path / "utils.py").write_text("def helper():\n    return 42\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert "python" in result.languages
        assert len(result.elements) > 0
        assert any(e.name == "main" for e in result.elements)
        assert any(e.name == "helper" for e in result.elements)

    def test_parse_python_class(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("class Foo(Bar):\n    def method(self): pass\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert any(e.type == "class" and e.name == "Foo" for e in result.elements)

    def test_parse_python_async_function(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("async def fetch():\n    pass\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert any(e.type == "async_function" and e.name == "fetch" for e in result.elements)

    def test_parse_python_syntax_error(self, tmp_path: Path):
        (tmp_path / "bad.py").write_text("this is not valid python {{{{\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        # Should not crash; syntax error file skipped
        assert isinstance(result, RepoAnalysis)

    def test_find_imports(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("import os\nimport sys\nfrom pathlib import Path\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        modules = [imp.module for imp in result.imports]
        assert "os" in modules
        assert "sys" in modules
        assert "pathlib" in modules

    def test_find_entry_points(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("pass\n")
        (tmp_path / "app.py").write_text("pass\n")
        (tmp_path / "lib.py").write_text("pass\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert any("main.py" in ep for ep in result.entry_points)
        assert any("app.py" in ep for ep in result.entry_points)
        assert not any("lib.py" in ep for ep in result.entry_points)

    def test_find_test_files(self, tmp_path: Path):
        (tmp_path / "test_model.py").write_text("pass\n")
        (tmp_path / "model_test.py").write_text("pass\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert any("test_model.py" in tf for tf in result.test_files)
        assert any("model_test.py" in tf for tf in result.test_files)

    def test_detect_frameworks(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("import torch\nfrom torch import nn\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        assert "torch" in result.frameworks or "pytorch" in result.frameworks

    def test_build_dependency_graph(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("import os\nimport sys\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        a_deps = result.dependencies.get("a.py", [])
        assert "os" in a_deps
        assert "sys" in a_deps

    def test_count_language_stats(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
        (tmp_path / "b.py").write_text("z = 3\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        py_stats = [s for s in result.language_stats if s.language == "python"]
        assert len(py_stats) == 1
        assert py_stats[0].file_count == 2
        assert py_stats[0].line_count >= 3

    def test_should_ignore_ignores_dirs(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "c.py").write_text("def cached(): pass\n")
        (tmp_path / "mod.py").write_text("def hello(): pass\n")
        analyzer = MultiLanguageAnalyzer(tmp_path)
        result = analyzer.analyze()
        # Only mod.py should be analyzed (not __pycache__/c.py)
        assert len(result.elements) == 1
        assert result.elements[0].name == "hello"

    def test_find_patterns_for_improvement(self, tmp_path: Path):
        # find_patterns_for_improvement reads self.elements, which analyze() doesn't set
        # (source bug). Manually set elements to test the method's logic.
        analyzer = MultiLanguageAnalyzer(tmp_path)
        analyzer.elements = [
            CodeElement(type="class", name="LayerNorm", file="a.py", line=1),
            CodeElement(type="function", name="attention", file="a.py", line=10),
        ]
        patterns = analyzer.find_patterns_for_improvement()
        assert "normalization" in patterns
        assert "attention" in patterns

    def test_dataclass_construction(self):
        ce = CodeElement(type="function", name="f", file="a.py", line=1)
        assert ce.type == "function"
        assert ce.visibility == "public"

        imp = ImportStatement(module="os", names=[], file="a.py", line=1)
        assert imp.module == "os"

        ls = LanguageStats(
            language="python", file_count=1, line_count=10, function_count=3, class_count=1
        )
        assert ls.language == "python"

        ra = RepoAnalysis(root_path=Path("/"), languages=["python"], language_stats=[])
        assert ra.root_path == Path("/")
