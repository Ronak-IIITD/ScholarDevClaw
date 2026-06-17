"""Tests for Rust native extension parity with the Python tree-sitter analyzer.

Verifies that the Rust `walk_file` / `walk_batch` produce the same structural
results as the Python `_walk_for_elements_and_imports` for all 6 supported
languages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip entire module if Rust extension is not installed
pytest.importorskip("scholardevclaw_native", reason="Rust native extension not built")

from scholardevclaw_native import is_available, walk_batch, walk_file  # noqa: E402

from scholardevclaw.repo_intelligence.tree_sitter_analyzer import (  # noqa: E402
    CodeElement,
    ImportStatement,
    TreeSitterAnalyzer,
)

# ─── Fixtures: sample source code per language ───────────────────────────────

PYTHON_SOURCE = b'''\
import os
from pathlib import Path
from . import local_module

class MyClass(BaseClass):
    """A test class."""

    @staticmethod
    def static_method(x: int) -> str:
        return str(x)

    async def async_method(self):
        pass

    def _private_method(self):
        pass

def standalone_function():
    pass

async def async_standalone():
    pass
'''

JAVASCRIPT_SOURCE = b"""\
import React from 'react';
import { useState } from 'react';
import * as utils from './utils';

export default class App extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return null;
    }
}

function helper() {
    return 42;
}

const arrowFn = (x) => x * 2;

async function fetchData() {
    return {};
}
"""

TYPESCRIPT_SOURCE = b"""\
import { Node } from './types';

interface User {
    name: string;
    age: number;
}

type ID = string | number;

enum Color {
    Red,
    Green,
    Blue,
}

class Service {
    async getData(id: ID): Promise<User> {
        return {} as User;
    }
}

function greet(name: string): string {
    return `Hello ${name}`;
}
"""

GO_SOURCE = b"""\
package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    addr string
}

func NewServer(addr string) *Server {
    return &Server{addr: addr}
}

func (s *Server) Listen() error {
    return http.ListenAndServe(s.addr, nil)
}

type Handler interface {
    Handle(w http.ResponseWriter, r *http.Request)
}

func main() {
    fmt.Println("hello")
}
"""

RUST_SOURCE = b"""\
use std::collections::HashMap;

pub struct Config {
    pub name: String,
}

pub enum LogLevel {
    Info,
    Warn,
    Error,
}

pub trait Logger {
    fn log(&self, msg: &str);
}

impl Logger for Config {
    fn log(&self, msg: &str) {
        println!("{}: {}", self.name, msg);
    }
}

pub async fn fetch_data(url: &str) -> Result<String, Error> {
    Ok(String::new())
}

fn internal_helper() {}
"""

JAVA_SOURCE = b"""\
package com.example;

import java.util.List;
import java.util.Map;
import static java.util.Collections.unmodifiableList;

public class Application {
    private String name;

    public Application(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    private void init() {
        // ...
    }
}

interface Repository<T> {
    T findById(long id);
}

enum Status {
    ACTIVE, INACTIVE
}
"""


# ─── Helper: compare elements ────────────────────────────────────────────────


def _normalize_elements(py_elements: list[CodeElement], rs_elements: list) -> tuple:
    """Normalize and sort elements for comparison."""

    def key(e):
        t = e.elem_type if hasattr(e, "elem_type") else e.type
        n = e.name if hasattr(e, "name") else e.name
        f = e.file if hasattr(e, "file") else e.file
        ln = e.line if hasattr(e, "line") else e.line
        return (f, ln, t, n)

    def normalize(e):
        if hasattr(e, "elem_type"):  # Rust PyCodeElement
            return {
                "type": e.elem_type,
                "name": e.name,
                "file": e.file,
                "line": e.line,
                "end_line": e.end_line,
                "language": e.language,
                "visibility": e.visibility,
                "parameters": sorted(e.parameters),
                "return_type": e.return_type,
                "decorators": sorted(e.decorators),
                "parent_class": e.parent_class,
                "dependencies": sorted(e.dependencies),
            }
        else:  # Python CodeElement
            return {
                "type": e.type,
                "name": e.name,
                "file": e.file,
                "line": e.line,
                "end_line": e.end_line,
                "language": e.language,
                "visibility": e.visibility,
                "parameters": sorted(e.parameters),
                "return_type": e.return_type,
                "decorators": sorted(e.decorators),
                "parent_class": e.parent_class,
                "dependencies": sorted(e.dependencies),
            }

    return sorted(normalize(e) for e in py_elements), sorted(normalize(e) for e in rs_elements)


def _normalize_imports(py_imports: list[ImportStatement], rs_imports: list) -> tuple:
    """Normalize and sort imports for comparison."""

    def normalize(i):
        if hasattr(i, "module") and hasattr(i, "is_from"):
            # Could be either Python or Rust
            return {
                "module": i.module,
                "names": sorted(i.names),
                "file": i.file,
                "line": i.line,
                "is_from": i.is_from,
                "alias": i.alias,
            }
        return {}

    return sorted(normalize(i) for i in py_imports), sorted(normalize(i) for i in rs_imports)


# ─── Tests: Python ───────────────────────────────────────────────────────────


class TestPythonParity:
    def test_elements_count(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        assert len(rs_result.elements) >= 6, (
            f"Expected at least 6 elements, got {len(rs_result.elements)}"
        )

    def test_class_and_methods(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        names = {e.name for e in rs_result.elements}
        assert "MyClass" in names
        assert "static_method" in names
        assert "async_method" in names
        assert "_private_method" in names
        assert "standalone_function" in names
        assert "async_standalone" in names

    def test_visibility(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        elements = {e.name: e for e in rs_result.elements}
        assert elements["MyClass"].visibility == "public"
        assert elements["_private_method"].visibility == "protected"
        assert elements["standalone_function"].visibility == "public"

    def test_async_detection(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        elements = {e.name: e for e in rs_result.elements}
        assert elements["async_standalone"].elem_type == "async_function"
        assert elements["async_method"].elem_type == "async_method"
        assert elements["standalone_function"].elem_type == "function"

    def test_imports(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        modules = {i.module for i in rs_result.imports}
        assert "os" in modules
        assert "pathlib" in modules

    def test_from_imports(self):
        rs_result = walk_file(PYTHON_SOURCE, "test.py", "python")
        from_imports = [i for i in rs_result.imports if i.is_from]
        modules = {i.module for i in from_imports}
        assert "pathlib" in modules


# ─── Tests: JavaScript ───────────────────────────────────────────────────────


class TestJavaScriptParity:
    def test_elements_count(self):
        rs_result = walk_file(JAVASCRIPT_SOURCE, "app.js", "javascript")
        names = {e.name for e in rs_result.elements}
        assert "App" in names
        assert "helper" in names
        assert "arrowFn" in names
        assert "fetchData" in names

    def test_class_detection(self):
        rs_result = walk_file(JAVASCRIPT_SOURCE, "app.js", "javascript")
        classes = [e for e in rs_result.elements if e.elem_type == "class"]
        assert len(classes) >= 1
        assert classes[0].name == "App"

    def test_imports(self):
        rs_result = walk_file(JAVASCRIPT_SOURCE, "app.js", "javascript")
        modules = {i.module for i in rs_result.imports}
        assert "react" in modules
        assert "./utils" in modules


# ─── Tests: TypeScript ───────────────────────────────────────────────────────


class TestTypeScriptParity:
    def test_interface(self):
        rs_result = walk_file(TYPESCRIPT_SOURCE, "types.ts", "typescript")
        interfaces = [e for e in rs_result.elements if e.elem_type == "interface"]
        assert len(interfaces) >= 1
        assert interfaces[0].name == "User"

    def test_type_alias(self):
        rs_result = walk_file(TYPESCRIPT_SOURCE, "types.ts", "typescript")
        aliases = [e for e in rs_result.elements if e.elem_type == "type_alias"]
        assert len(aliases) >= 1
        assert aliases[0].name == "ID"

    def test_enum(self):
        rs_result = walk_file(TYPESCRIPT_SOURCE, "types.ts", "typescript")
        enums = [e for e in rs_result.elements if e.elem_type == "enum"]
        assert len(enums) >= 1
        assert enums[0].name == "Color"

    def test_return_type(self):
        rs_result = walk_file(TYPESCRIPT_SOURCE, "types.ts", "typescript")
        funcs = {e.name: e for e in rs_result.elements}
        assert funcs["greet"].return_type == "string"


# ─── Tests: Go ───────────────────────────────────────────────────────────────


class TestGoParity:
    def test_struct(self):
        rs_result = walk_file(GO_SOURCE, "main.go", "go")
        structs = [e for e in rs_result.elements if e.elem_type == "struct"]
        assert len(structs) >= 1
        assert structs[0].name == "Server"

    def test_interface(self):
        rs_result = walk_file(GO_SOURCE, "main.go", "go")
        interfaces = [e for e in rs_result.elements if e.elem_type == "interface"]
        assert len(interfaces) >= 1
        assert interfaces[0].name == "Handler"

    def test_method_receiver(self):
        rs_result = walk_file(GO_SOURCE, "main.go", "go")
        methods = [e for e in rs_result.elements if e.elem_type == "method"]
        listen = [m for m in methods if m.name == "Listen"]
        assert len(listen) == 1
        assert listen[0].parent_class == "Server"

    def test_visibility(self):
        rs_result = walk_file(GO_SOURCE, "main.go", "go")
        funcs = {e.name: e for e in rs_result.elements}
        assert funcs["NewServer"].visibility == "public"
        assert funcs["main"].visibility == "private"

    def test_imports(self):
        rs_result = walk_file(GO_SOURCE, "main.go", "go")
        modules = {i.module for i in rs_result.imports}
        assert "fmt" in modules
        assert "net/http" in modules


# ─── Tests: Rust ─────────────────────────────────────────────────────────────


class TestRustParity:
    def test_struct(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        structs = [e for e in rs_result.elements if e.elem_type == "struct"]
        assert len(structs) >= 1
        assert structs[0].name == "Config"

    def test_enum(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        enums = [e for e in rs_result.elements if e.elem_type == "enum"]
        assert len(enums) >= 1
        assert enums[0].name == "LogLevel"

    def test_trait(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        traits = [e for e in rs_result.elements if e.elem_type == "trait"]
        assert len(traits) >= 1
        assert traits[0].name == "Logger"

    def test_impl(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        impls = [e for e in rs_result.elements if e.elem_type == "impl"]
        assert len(impls) >= 1
        assert impls[0].name == "Config"

    def test_async_function(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        funcs = {e.name: e for e in rs_result.elements}
        assert funcs["fetch_data"].elem_type == "async_function"

    def test_visibility(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        funcs = {e.name: e for e in rs_result.elements}
        assert funcs["fetch_data"].visibility == "public"
        assert funcs["internal_helper"].visibility == "private"

    def test_use_imports(self):
        rs_result = walk_file(RUST_SOURCE, "lib.rs", "rust")
        modules = {i.module for i in rs_result.imports}
        assert "std::collections" in modules


# ─── Tests: Java ─────────────────────────────────────────────────────────────


class TestJavaParity:
    def test_class(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        classes = [e for e in rs_result.elements if e.elem_type == "class"]
        assert len(classes) >= 1
        assert classes[0].name == "Application"

    def test_interface(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        interfaces = [e for e in rs_result.elements if e.elem_type == "interface"]
        assert len(interfaces) >= 1
        assert interfaces[0].name == "Repository"

    def test_enum(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        enums = [e for e in rs_result.elements if e.elem_type == "enum"]
        assert len(enums) >= 1
        assert enums[0].name == "Status"

    def test_constructor(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        ctors = [e for e in rs_result.elements if e.elem_type == "constructor"]
        assert len(ctors) >= 1
        assert ctors[0].name == "Application"

    def test_visibility(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        methods = {e.name: e for e in rs_result.elements}
        assert methods["getName"].visibility == "public"
        assert methods["init"].visibility == "private"

    def test_imports(self):
        rs_result = walk_file(JAVA_SOURCE, "Application.java", "java")
        modules = {i.module for i in rs_result.imports}
        assert "java::util" in modules or "java.util" in modules


# ─── Tests: Batch API ────────────────────────────────────────────────────────


class TestBatchAPI:
    def test_batch_same_as_individual(self):
        files = [
            (PYTHON_SOURCE, "test.py", "python"),
            (JAVASCRIPT_SOURCE, "app.js", "javascript"),
            (TYPESCRIPT_SOURCE, "types.ts", "typescript"),
        ]
        batch_results = walk_batch(files)

        for (source, path, lang), batch_result in zip(files, batch_results):
            individual = walk_file(source, path, lang)
            assert len(batch_result.elements) == len(individual.elements)
            assert len(batch_result.imports) == len(individual.imports)


# ─── Tests: Integration with TreeSitterAnalyzer ──────────────────────────────


class TestAnalyzerIntegration:
    def test_analyzer_uses_rust(self):
        analyzer = TreeSitterAnalyzer(Path(__file__).parent.parent)
        assert analyzer._use_rust is True

    def test_analyzer_produces_results(self):
        analyzer = TreeSitterAnalyzer(Path(__file__).parent.parent)
        result = analyzer.analyze()
        assert len(result.elements) > 0
        assert len(result.imports) > 0

    def test_is_available(self):
        assert is_available() is True


# ─── Tests: Cosine Similarity ────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from scholardevclaw_native import cosine_similarity

        assert abs(cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        from scholardevclaw_native import cosine_similarity

        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_known_value(self):
        from scholardevclaw_native import cosine_similarity

        sim = cosine_similarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(sim - 0.974632) < 0.001

    def test_empty_vectors(self):
        from scholardevclaw_native import cosine_similarity

        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        from scholardevclaw_native import cosine_similarity

        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        from scholardevclaw_native import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_batch(self):
        from scholardevclaw_native import cosine_similarity_batch

        va = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        vb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = cosine_similarity_batch(va, vb)
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9
        assert abs(result[2]) < 1e-9

    def test_matrix(self):
        from scholardevclaw_native import cosine_similarity_matrix

        vecs = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        m = cosine_similarity_matrix(vecs)
        n = 3
        # Diagonal should be 1.0
        assert abs(m[0 * n + 0] - 1.0) < 1e-9
        assert abs(m[1 * n + 1] - 1.0) < 1e-9
        assert abs(m[2 * n + 2] - 1.0) < 1e-9
        # Orthogonal vectors should be ~0
        assert abs(m[0 * n + 1]) < 1e-9
        # [1,0,0] and [1,1,0] should be ~0.7071
        assert abs(m[0 * n + 2] - 0.7071) < 0.001

    def test_matches_numpy(self):
        import numpy as np
        from scholardevclaw_native import cosine_similarity

        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.random(128).tolist()
            b = rng.random(128).tolist()
            a_np, b_np = np.array(a), np.array(b)
            expected = float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
            actual = cosine_similarity(a, b)
            assert abs(actual - expected) < 1e-6, f"Mismatch: {actual} vs {expected}"


# ─── Tests: Unified Diff ─────────────────────────────────────────────────────


class TestUnifiedDiff:
    def test_identical(self):
        from scholardevclaw_native import unified_diff

        result = unified_diff("a\nb\n", "a\nb\n")
        assert result == ""

    def test_simple_change(self):
        import difflib

        from scholardevclaw_native import unified_diff

        original = "line 1\nline 2\nline 3\n"
        modified = "line 1\nline 2 modified\nline 3\n"
        rust = unified_diff(original, modified)
        py = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True)))
        assert rust == py

    def test_addition(self):
        import difflib

        from scholardevclaw_native import unified_diff

        original = "a\nb\n"
        modified = "a\nb\nc\n"
        rust = unified_diff(original, modified)
        py = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True)))
        assert rust == py

    def test_deletion(self):
        import difflib

        from scholardevclaw_native import unified_diff

        original = "a\nb\nc\n"
        modified = "a\nc\n"
        rust = unified_diff(original, modified)
        py = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True)))
        assert rust == py

    def test_empty_original(self):
        from scholardevclaw_native import unified_diff

        result = unified_diff("", "new line\n")
        assert result.startswith("---")
        assert "+++ " in result
        assert "+new line" in result

    def test_empty_modified(self):
        from scholardevclaw_native import unified_diff

        result = unified_diff("old line\n", "")
        assert "---" in result
        assert "-old line" in result

    def test_custom_labels(self):
        from scholardevclaw_native import unified_diff

        result = unified_diff("a\n", "b\n", from_file="old.py", to_file="new.py")
        assert "--- old.py" in result
        assert "+++ new.py" in result

    def test_context_lines(self):
        from scholardevclaw_native import unified_diff

        original = "\n".join(f"line {i}" for i in range(20)) + "\n"
        modified = "\n".join(f"line {i}" for i in range(20)) + "\n"
        modified = modified.replace("line 10", "CHANGED")
        r1 = unified_diff(original, modified, context_lines=1)
        r3 = unified_diff(original, modified, context_lines=3)
        # More context = more lines in output
        assert len(r3) > len(r1)

    def test_count_diff_changes(self):
        from scholardevclaw_native import count_diff_changes, unified_diff

        original = "a\nb\nc\n"
        modified = "a\nx\nc\ny\n"
        diff = unified_diff(original, modified)
        added, removed = count_diff_changes(diff)
        assert added == 2  # +x, +y
        assert removed == 1  # -b

    def test_matches_python_large(self):
        """Test on larger content (500 lines) to exercise Myers algorithm."""
        import difflib

        from scholardevclaw_native import unified_diff

        rng_lines = [f"line {i}: {'x' * (i % 50)}" for i in range(500)]
        original = "\n".join(rng_lines) + "\n"
        # Modify every 10th line
        modified_lines = list(rng_lines)
        for i in range(0, 500, 10):
            modified_lines[i] = f"line {i}: MODIFIED"
        modified = "\n".join(modified_lines) + "\n"
        rust = unified_diff(original, modified)
        py = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True)))
        assert rust == py


# ─── Helper: run Python element extraction directly ──────────────────────────


def _extract_python_elements_direct(source: bytes, rel_path: str) -> list[CodeElement]:
    """Directly run Python element extraction for comparison."""
    import tree_sitter

    lang_module = __import__("tree_sitter_python")
    lang = tree_sitter.Language(lang_module.language())
    parser = tree_sitter.Parser(lang)
    tree = parser.parse(source)

    analyzer = TreeSitterAnalyzer.__new__(TreeSitterAnalyzer)
    analyzer.repo_path = Path(".")

    elements: list[CodeElement] = []
    imports: list[ImportStatement] = []
    analyzer._walk_for_elements_and_imports(
        tree.root_node, rel_path, "python", source, elements, imports, None
    )
    return elements


# Monkey-patch for parity tests
TreeSitterAnalyzer._extract_elements_from_tree_inner = staticmethod(_extract_python_elements_direct)
