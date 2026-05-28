"""Tests for detector, call_graph, and dependency_graph submodules."""

from pathlib import Path

from scholardevclaw.repo_intelligence.call_graph import (
    CallGraph,
    CallGraphAnalyzer,
)
from scholardevclaw.repo_intelligence.dependency_graph import (
    DependencyAnalyzer,
    DependencyGraph,
)
from scholardevclaw.repo_intelligence.detector import (
    PyTorchComponentDetector,
)

# =========================================================================
# detector — PyTorchComponentDetector
# =========================================================================


class _MockClass:
    """Minimal mock class object with the attributes detector uses."""

    def __init__(self, name: str, line_number: int = 1, is_nn_module: bool = False):
        self.name = name
        self.line_number = line_number
        self.is_nn_module = is_nn_module


class _MockModule:
    """Minimal mock module object with the attributes detector uses."""

    def __init__(self, relative_path: Path | str, classes: list | None = None):
        self.relative_path = (
            relative_path if isinstance(relative_path, Path) else Path(relative_path)
        )
        self.path = self.relative_path  # detector accesses module.path.read_text()
        self.classes = classes or []


class TestPyTorchComponentDetector:
    def setup_method(self):
        self.detector = PyTorchComponentDetector()

    def test_detect_modules_finds_nn_module(self):
        cls = _MockClass("LinearModel", is_nn_module=True)
        mod = _MockModule(Path("models/lm.py"), classes=[cls])
        result = self.detector.detect_modules([mod])
        assert len(result) == 1
        assert result[0].name == "LinearModel"
        assert result[0].component_type == "model"
        assert "lm.py" in result[0].file

    def test_detect_modules_skips_non_nn(self):
        nn_cls = _MockClass("GoodModel", is_nn_module=True)
        plain_cls = _MockClass("Helper")
        mod = _MockModule(Path("m.py"), classes=[nn_cls, plain_cls])
        result = self.detector.detect_modules([mod])
        assert len(result) == 1
        assert result[0].name == "GoodModel"

    def test_detect_modules_empty_modules(self):
        assert self.detector.detect_modules([]) == []

    def test_detect_modules_multiple_modules(self):
        c1 = _MockClass("A", is_nn_module=True)
        c2 = _MockClass("B", is_nn_module=True)
        m1 = _MockModule(Path("a.py"), classes=[c1])
        m2 = _MockModule(Path("b.py"), classes=[c2])
        result = self.detector.detect_modules([m1, m2])
        assert len(result) == 2

    def test_detect_layer_norms_finds_layer_norm(self):
        cls = _MockClass("LayerNorm")
        mod = _MockModule(Path("norms.py"), classes=[cls])
        result = self.detector.detect_layer_norms([mod])
        assert len(result) == 1
        assert result[0].name == "LayerNorm"
        assert result[0].component_type == "normalization"

    def test_detect_layer_norms_finds_lowercase(self):
        cls = _MockClass("layer_norm_custom")
        mod = _MockModule(Path("layers.py"), classes=[cls])
        result = self.detector.detect_layer_norms([mod])
        assert len(result) == 1
        assert result[0].name == "layer_norm_custom"

    def test_detect_layer_norms_skips_other(self):
        cls = _MockClass("BatchNorm")
        mod = _MockModule(Path("layers.py"), classes=[cls])
        result = self.detector.detect_layer_norms([mod])
        assert len(result) == 0

    def test_detect_attention_finds_attention(self):
        cls = _MockClass("MultiHeadAttention")
        mod = _MockModule(Path("attn.py"), classes=[cls])
        result = self.detector.detect_attention([mod])
        assert len(result) == 1
        assert result[0].component_type == "attention"

    def test_detect_attention_finds_lowercase(self):
        cls = _MockClass("cross_attention")
        mod = _MockModule(Path("attn.py"), classes=[cls])
        result = self.detector.detect_attention([mod])
        assert len(result) == 1

    def test_detect_attention_empty(self):
        cls = _MockClass("Linear")
        mod = _MockModule(Path("layers.py"), classes=[cls])
        result = self.detector.detect_attention([mod])
        assert len(result) == 0


# =========================================================================
# call_graph — CallGraph
# =========================================================================


class TestCallGraph:
    def setup_method(self):
        self.graph = CallGraph(root_path=Path("/fake"))

    def _add_func(self, name: str, file: str = "mod.py", **kw) -> str:
        return self.graph.add_function(
            name=name,
            file=Path(file),
            line=1,
            **kw,
        ).full_name

    def test_add_function_creates_node(self):
        node = self.graph.add_function("foo", Path("mod.py"), line=10)
        assert node.name == "foo"
        assert node.full_name == "mod.py::foo"
        assert node.line == 10
        assert node.end_line == 0

    def test_add_function_duplicate_updates(self):
        self.graph.add_function("foo", Path("mod.py"), line=10)
        node = self.graph.add_function("foo", Path("mod.py"), line=20, end_line=30)
        assert node.line == 20
        assert node.end_line == 30
        assert len(self.graph.functions) == 1

    def test_add_call_creates_relationship(self):
        caller = self._add_func("caller")
        callee = self._add_func("callee")
        self.graph.add_call(caller, callee)
        assert callee in self.graph.functions[caller].calls
        assert caller in self.graph.functions[callee].called_by

    def test_add_call_creates_placeholder_callee(self):
        caller = self._add_func("caller")
        self.graph.add_call(caller, "unknown::ghost")
        assert "unknown::ghost" in self.graph.functions
        assert self.graph.functions["unknown::ghost"].line == 0

    def test_get_calls_missing(self):
        assert self.graph.get_calls("nonexistent") == set()

    def test_get_called_by_missing(self):
        assert self.graph.get_called_by("nonexistent") == set()

    def test_find_call_chain_direct(self):
        a = self._add_func("a")
        b = self._add_func("b")
        self.graph.add_call(a, b)
        chain = self.graph.find_call_chain(a, b)
        assert chain is not None
        assert chain.length == 1
        assert chain.path == [a, b]

    def test_find_call_chain_indirect(self):
        a = self._add_func("a")
        b = self._add_func("b")
        c = self._add_func("c")
        self.graph.add_call(a, b)
        self.graph.add_call(b, c)
        chain = self.graph.find_call_chain(a, c)
        assert chain is not None
        assert chain.length == 2

    def test_find_call_chain_same(self):
        a = self._add_func("a")
        chain = self.graph.find_call_chain(a, a)
        assert chain is not None
        assert chain.length == 0
        assert chain.path == [a]

    def test_find_call_chain_no_path(self):
        a = self._add_func("a")
        b = self._add_func("b")
        assert self.graph.find_call_chain(a, b) is None

    def test_find_call_chain_missing(self):
        assert self.graph.find_call_chain("a", "b") is None

    def test_find_all_callers_transitive(self):
        a = self._add_func("a")
        b = self._add_func("b")
        c = self._add_func("c")
        self.graph.add_call(a, b)
        self.graph.add_call(b, c)
        callers = self.graph.find_all_callers(c)
        assert "mod.py::a" in callers
        assert "mod.py::b" in callers

    def test_find_all_callers_missing(self):
        assert self.graph.find_all_callers("nonexistent") == set()

    def test_find_callees_transitive(self):
        a = self._add_func("a")
        b = self._add_func("b")
        c = self._add_func("c")
        self.graph.add_call(a, b)
        self.graph.add_call(b, c)
        callees = self.graph.find_callees(a)
        assert "mod.py::b" in callees
        assert "mod.py::c" in callees

    def test_find_callees_missing(self):
        assert self.graph.find_callees("nonexistent") == set()

    def test_get_call_depth(self):
        a = self._add_func("a")
        b = self._add_func("b")
        c = self._add_func("c")
        self.graph.add_call(a, b)
        self.graph.add_call(b, c)
        assert self.graph.get_call_depth(a) == 2

    def test_get_call_depth_zero(self):
        a = self._add_func("a")
        assert self.graph.get_call_depth(a) == 0

    def test_get_call_depth_missing(self):
        assert self.graph.get_call_depth("nope") == 0

    def test_get_impact_score(self):
        a = self._add_func("a")
        b = self._add_func("b")
        c = self._add_func("c")
        self.graph.add_call(b, a)
        self.graph.add_call(c, a)
        score = self.graph.get_impact_score(a)
        assert score > 0

    def test_get_impact_score_missing(self):
        assert self.graph.get_impact_score("nope") == 0.0

    def test_find_external_calls(self):
        a = self._add_func("a", file="a.py")
        b = self._add_func("b", file="b.py")
        self.graph.add_call(a, b)
        external = self.graph.find_external_calls(a)
        assert b in external

    def test_find_external_calls_same_file(self):
        a = self._add_func("a", file="mod.py")
        b = self._add_func("b", file="mod.py")
        self.graph.add_call(a, b)
        external = self.graph.find_external_calls(a)
        assert len(external) == 0

    def test_find_external_calls_missing(self):
        assert self.graph.find_external_calls("nope") == set()

    def test_to_dict(self):
        a = self._add_func("foo")
        d = self.graph.to_dict()
        assert "functions" in d
        assert a in d["functions"]
        assert d["functions"][a]["name"] == "foo"

    def test_build_from_directory_nonexistent(self):
        g = CallGraph()
        g.build_from_directory(Path("/nonexistent_dir_xyz"))
        assert len(g.functions) == 0


class TestCallGraphAnalyzer:
    def setup_method(self):
        self.graph = CallGraph(root_path=Path("/fake"))
        self.analyzer = CallGraphAnalyzer(self.graph)

    def _add_func(self, name: str, file: str = "mod.py") -> str:
        return self.graph.add_function(name=name, file=Path(file), line=1).full_name

    def test_analyze_function_missing(self):
        assert self.analyzer.analyze_function("nope") == {}

    def test_analyze_function_basic(self):
        fn = self._add_func("foo")
        result = self.analyzer.analyze_function(fn)
        assert result["function"] == fn
        assert result["direct_calls"] == 0
        assert result["is_method"] is False

    def test_find_critical_functions(self):
        a = self._add_func("a")
        for _ in range(10):
            c = self._add_func(f"caller_{_}")
            self.graph.add_call(c, a)
        critical = self.analyzer.find_critical_functions(threshold=3.0)
        assert any("a" in name for name, _ in critical)

    def test_find_leaf_functions(self):
        a = self._add_func("a")
        b = self._add_func("b")
        self.graph.add_call(a, b)
        leafs = self.analyzer.find_leaf_functions()
        assert b in leafs
        assert a not in leafs

    def test_find_entry_points(self):
        a = self._add_func("a")
        b = self._add_func("b")
        self.graph.add_call(a, b)
        entries = self.analyzer.find_entry_points()
        assert a in entries  # a has no callers
        assert b not in entries  # b is called by a

    def test_suggest_refactoring_missing(self):
        assert self.analyzer.suggest_refactoring("nope") == []

    def test_suggest_refactoring_many_calls(self):
        fn = self._add_func("busy")
        for _ in range(20):
            callee = self._add_func(f"callee_{_}")
            self.graph.add_call(fn, callee)
        suggestions = self.analyzer.suggest_refactoring(fn)
        assert any("calls" in s for s in suggestions)

    def test_suggest_refactoring_deep_depth(self):
        root_fn = self._add_func("root")
        prev = root_fn
        for i in range(15):
            cur = self._add_func(f"deep_{i}")
            self.graph.add_call(prev, cur)
            prev = cur
        suggestions = self.analyzer.suggest_refactoring(root_fn)
        assert any("depth" in s for s in suggestions)

    def test_suggest_refactoring_many_callers(self):
        fn = self._add_func("popular")
        for _ in range(12):
            caller = self._add_func(f"caller_{_}")
            self.graph.add_call(caller, fn)
        suggestions = self.analyzer.suggest_refactoring(fn)
        assert any("called by" in s for s in suggestions)


# =========================================================================
# dependency_graph — DependencyGraph
# =========================================================================


class TestDependencyGraph:
    def setup_method(self):
        self.graph = DependencyGraph(root_path=Path("/fake"))

    def test_add_module_creates_node(self):
        node = self.graph.add_module(Path("mymod.py"))
        assert node.name == "mymod"
        assert node.path == "mymod.py"

    def test_add_module_duplicate_returns_same(self):
        n1 = self.graph.add_module(Path("m.py"))
        n2 = self.graph.add_module(Path("m.py"))
        assert n1 is n2

    def test_add_module_outside_root(self):
        n = self.graph.add_module(Path("/outside/pkg/mod.py"))
        # The absolute path's parts include "/" which becomes part of the name
        assert ".outside.pkg.mod" in n.name

    def test_add_import_creates_relationship(self):
        importer = Path("/fake/pkg/a.py")
        self.graph.add_import(importer, "b", [])
        assert self.graph.modules["pkg.a"].name == "pkg.a"
        assert "b" in self.graph.modules["pkg.a"].imports

    def test_add_import_creates_target(self):
        self.graph.add_import(Path("a.py"), "unknown_mod", [])
        assert "unknown_mod" in self.graph.modules

    def test_path_to_module_name(self):
        assert self.graph._path_to_module_name(Path("pkg/sub/mod.py")) == "pkg.sub.mod"

    def test_path_to_module_name_init(self):
        assert self.graph._path_to_module_name(Path("pkg/__init__.py")) == "pkg"

    def test_is_package_init_file(self):
        assert self.graph._is_package(Path("__init__.py"))

    def test_get_imports_missing(self):
        assert self.graph.get_imports("nope") == set()

    def test_get_imported_by_missing(self):
        assert self.graph.get_imported_by("nope") == set()

    def test_find_shortest_path_direct(self):
        self.graph.add_import(Path("a.py"), "b", [])
        # Also ensure b exists
        self.graph._ensure_module("b")
        path = self.graph.find_shortest_path("a", "b")
        assert path is not None
        assert path.length == 1

    def test_find_shortest_path_same(self):
        self.graph._ensure_module("a")
        path = self.graph.find_shortest_path("a", "a")
        assert path is not None
        assert path.length == 0

    def test_find_shortest_path_missing(self):
        assert self.graph.find_shortest_path("a", "b") is None

    def test_find_shortest_path_no_route(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        assert self.graph.find_shortest_path("a", "b") is None

    def test_find_all_dependencies(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("c")
        deps = self.graph.find_all_dependencies("a")
        assert "b" in deps
        assert "c" in deps

    def test_find_all_dependents(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["b"].imported_by.add("a")
        self.graph.modules["c"].imported_by.add("b")
        # a -> b -> c
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("c")
        deps = self.graph.find_all_dependents("c")
        assert "b" in deps
        assert "a" in deps

    def test_find_circular_dependencies(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("c")
        self.graph.modules["c"].imports.add("a")
        cycles = self.graph.find_circular_dependencies()
        assert any("a" in c and "b" in c and "c" in c for c in cycles)

    def test_get_dependency_depth(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("c")
        depth = self.graph.get_dependency_depth("a")
        assert depth == 2

    def test_get_dependency_depth_zero(self):
        self.graph._ensure_module("a")
        assert self.graph.get_dependency_depth("a") == 0

    def test_get_dependency_depth_missing(self):
        assert self.graph.get_dependency_depth("nope") == 0

    def test_get_impact_score(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["a"].imported_by.add("b")
        self.graph.modules["a"].imported_by.add("c")
        score = self.graph.get_impact_score("a")
        assert score > 0

    def test_get_impact_score_missing(self):
        assert self.graph.get_impact_score("nope") == 0.0

    def test_get_package_structure(self):
        self.graph._ensure_module("pkg.a")
        self.graph._ensure_module("pkg.b")
        self.graph._ensure_module("standalone")
        packages = self.graph.get_package_structure()
        assert "pkg" in packages
        assert "pkg.a" in packages["pkg"]
        assert "standalone" not in packages

    def test_to_dict(self):
        self.graph._ensure_module("mod")
        d = self.graph.to_dict()
        assert "modules" in d
        assert "mod" in d["modules"]

    def test_build_from_directory_nonexistent(self):
        g = DependencyGraph()
        g.build_from_directory(Path("/nonexistent_dir_xyz"))
        assert len(g.modules) == 0


class TestDependencyAnalyzer:
    def setup_method(self):
        self.graph = DependencyGraph(root_path=Path("/fake"))
        self.analyzer = DependencyAnalyzer(self.graph)

    def test_analyze_impact_missing(self):
        assert self.analyzer.analyze_impact("nope") == {}

    def test_analyze_impact_basic(self):
        self.graph._ensure_module("a")
        result = self.analyzer.analyze_impact("a")
        assert result["module"] == "a"
        assert result["direct_imports"] == 0

    def test_find_critical_modules(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph.modules["a"].imported_by.add("x")
        self.graph.modules["a"].imported_by.add("y")
        self.graph.modules["a"].imported_by.add("z")
        critical = self.analyzer.find_critical_modules(threshold=1.0)
        assert any(name == "a" for name, _ in critical)

    def test_suggest_refactoring_missing(self):
        assert self.analyzer.suggest_refactoring("nope") == []

    def test_suggest_refactoring_many_imports(self):
        self.graph._ensure_module("a")
        for i in range(15):
            self.graph.modules["a"].imports.add(f"dep_{i}")
        suggestions = self.analyzer.suggest_refactoring("a")
        assert any("imports" in s for s in suggestions)

    def test_suggest_refactoring_many_imported_by(self):
        self.graph._ensure_module("a")
        for i in range(15):
            self.graph._ensure_module(f"caller_{i}")
            self.graph.modules[f"caller_{i}"].imports.add("a")
            self.graph.modules["a"].imported_by.add(f"caller_{i}")
        suggestions = self.analyzer.suggest_refactoring("a")
        assert any("imported by" in s for s in suggestions)

    def test_suggest_refactoring_circular(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("a")
        suggestions = self.analyzer.suggest_refactoring("a")
        assert any("circular" in s for s in suggestions)

    def test_get_dependency_order(self):
        self.graph._ensure_module("a")
        self.graph._ensure_module("b")
        self.graph._ensure_module("c")
        self.graph.modules["a"].imports.add("b")
        self.graph.modules["b"].imports.add("c")
        order = self.analyzer.get_dependency_order()
        # c should come before b before a
        assert order.index("c") < order.index("b")
        assert order.index("b") < order.index("a")
