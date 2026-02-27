"""
Dependency graph analysis for understanding import relationships.

Analyzes:
- Import statements and module dependencies
- Circular dependency detection
- Dependency depth and impact analysis
- Package structure inference
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModuleNode:
    """A node in the dependency graph representing a module"""

    path: str
    name: str
    imports: set[str] = field(default_factory=set)
    imported_by: set[str] = field(default_factory=set)
    is_package: bool = False
    package_path: str = ""


@dataclass
class DependencyChain:
    """A chain of dependencies between two modules"""

    from_module: str
    to_module: str
    path: list[str]
    length: int


class DependencyGraph:
    """Graph of module dependencies"""

    def __init__(self, root_path: Path | None = None):
        self.root_path = root_path or Path.cwd()
        self.modules: dict[str, ModuleNode] = {}
        self._package_cache: dict[str, str] = {}

    def add_module(self, module_path: Path) -> ModuleNode:
        """Add a module to the graph"""
        try:
            relative = module_path.relative_to(self.root_path)
        except ValueError:
            relative = module_path

        module_name = self._path_to_module_name(relative)
        module_path_str = str(relative)

        if module_name in self.modules:
            return self.modules[module_name]

        node = ModuleNode(
            path=module_path_str,
            name=module_name,
            is_package=self._is_package(module_path),
        )
        self.modules[module_name] = node
        return node

    def add_import(self, importer_path: Path, imported_module: str, imported_names: list[str]):
        """Add an import relationship"""
        importer_node = self.add_module(importer_path)
        importer_node.imports.add(imported_module)

        if imported_module in self.modules:
            self.modules[imported_module].imported_by.add(importer_node.name)
        else:
            node = ModuleNode(path=imported_module, name=imported_module)
            node.imported_by.add(importer_node.name)
            self.modules[imported_module] = node

    def _path_to_module_name(self, path: Path) -> str:
        """Convert file path to module name"""
        parts = list(path.parts)
        if path.suffix == ".py":
            parts[-1] = path.stem
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _is_package(self, path: Path) -> bool:
        """Check if path is a package (has __init__.py)"""
        return path.name == "__init__.py" or (path / "__init__.py").exists()

    def build_from_directory(self, directory: Path):
        """Build dependency graph from a directory"""
        self.root_path = directory.resolve()

        for py_file in directory.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            self._analyze_file(py_file)

    def _analyze_file(self, file_path: Path):
        """Analyze a Python file for imports"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, OSError):
            return

        importer_node = self.add_module(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    importer_node.imports.add(module)
                    self._ensure_module(module)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    importer_node.imports.add(module)
                    self._ensure_module(module)

    def _ensure_module(self, module_name: str):
        """Ensure a module exists in the graph"""
        if module_name not in self.modules:
            self.modules[module_name] = ModuleNode(
                path=module_name,
                name=module_name,
            )

    def get_imports(self, module_name: str) -> set[str]:
        """Get direct imports of a module"""
        if module_name not in self.modules:
            return set()
        return self.modules[module_name].imports

    def get_imported_by(self, module_name: str) -> set[str]:
        """Get modules that import this module"""
        if module_name not in self.modules:
            return set()
        return self.modules[module_name].imported_by

    def find_shortest_path(self, from_module: str, to_module: str) -> DependencyChain | None:
        """Find shortest dependency path between two modules"""
        if from_module not in self.modules or to_module not in self.modules:
            return None

        if from_module == to_module:
            return DependencyChain(from_module, to_module, [from_module], 0)

        from collections import deque

        queue = deque([(from_module, [from_module])])
        visited = {from_module}

        while queue:
            current, path = queue.popleft()

            for neighbor in self.modules[current].imports:
                if neighbor == to_module:
                    return DependencyChain(
                        from_module,
                        to_module,
                        path + [neighbor],
                        len(path),
                    )

                if neighbor not in visited and neighbor in self.modules:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_all_dependencies(self, module_name: str) -> set[str]:
        """Find all transitive dependencies"""
        if module_name not in self.modules:
            return set()

        result = set()
        to_visit = list(self.modules[module_name].imports)

        while to_visit:
            dep = to_visit.pop()
            if dep not in result and dep in self.modules:
                result.add(dep)
                to_visit.extend(self.modules[dep].imports)

        return result

    def find_all_dependents(self, module_name: str) -> set[str]:
        """Find all transitive dependents (modules that depend on this)"""
        if module_name not in self.modules:
            return set()

        result = set()
        to_visit = list(self.modules[module_name].imported_by)

        while to_visit:
            dep = to_visit.pop()
            if dep not in result and dep in self.modules:
                result.add(dep)
                to_visit.extend(self.modules[dep].imported_by)

        return result

    def find_circular_dependencies(self) -> list[list[str]]:
        """Find circular dependencies in the graph"""
        cycles = []

        def dfs(module: str, path: list[str], visited: set[str]) -> bool:
            visited.add(module)
            path.append(module)

            for neighbor in self.modules.get(module, ModuleNode("", "")).imports:
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
                elif neighbor not in visited:
                    if dfs(neighbor, path.copy(), visited):
                        return True

            return False

        for module in self.modules:
            dfs(module, [], set())

        return cycles

    def get_dependency_depth(self, module_name: str) -> int:
        """Get the maximum depth of dependencies"""
        if module_name not in self.modules:
            return 0

        max_depth = 0
        to_visit = [(m, 1) for m in self.modules[module_name].imports]
        visited = {module_name}

        while to_visit:
            dep, depth = to_visit.pop()
            if dep not in visited:
                visited.add(dep)
                max_depth = max(max_depth, depth)
                if dep in self.modules:
                    to_visit.extend((m, depth + 1) for m in self.modules[dep].imports)

        return max_depth

    def get_impact_score(self, module_name: str) -> float:
        """Calculate impact score based on dependents and depth"""
        if module_name not in self.modules:
            return 0.0

        direct_dependents = len(self.modules[module_name].imported_by)
        transitive_dependents = len(self.find_all_dependents(module_name))

        score = direct_dependents + (transitive_dependents * 0.5)
        return score

    def get_package_structure(self) -> dict[str, list[str]]:
        """Infer package structure from module names"""
        packages: dict[str, list[str]] = defaultdict(list)

        for module_name in self.modules:
            parts = module_name.split(".")
            if len(parts) > 1:
                packages[parts[0]].append(module_name)

        return dict(packages)

    def to_dict(self) -> dict:
        """Export graph to dictionary"""
        return {
            "modules": {
                name: {
                    "path": node.path,
                    "imports": list(node.imports),
                    "imported_by": list(node.imported_by),
                    "is_package": node.is_package,
                }
                for name, node in self.modules.items()
            }
        }


class DependencyAnalyzer:
    """Analyze dependency patterns and provide insights"""

    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def analyze_impact(self, module_name: str) -> dict:
        """Get detailed impact analysis for a module"""
        if module_name not in self.graph.modules:
            return {}

        node = self.graph.modules[module_name]
        direct_imports = len(node.imports)
        direct_imported_by = len(node.imported_by)
        transitive_deps = len(self.graph.find_all_dependencies(module_name))
        transitive_dependents = len(self.graph.find_all_dependents(module_name))
        depth = self.graph.get_dependency_depth(module_name)
        impact = self.graph.get_impact_score(module_name)

        return {
            "module": module_name,
            "direct_imports": direct_imports,
            "direct_imported_by": direct_imported_by,
            "transitive_dependencies": transitive_deps,
            "transitive_dependents": transitive_dependents,
            "dependency_depth": depth,
            "impact_score": impact,
            "is_package": node.is_package,
        }

    def find_critical_modules(self, threshold: float = 5.0) -> list[tuple[str, float]]:
        """Find modules with high impact scores"""
        scores = []
        for module_name in self.graph.modules:
            score = self.graph.get_impact_score(module_name)
            if score >= threshold:
                scores.append((module_name, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def suggest_refactoring(self, module_name: str) -> list[str]:
        """Suggest refactoring based on dependency analysis"""
        suggestions = []

        if module_name not in self.graph.modules:
            return suggestions

        node = self.graph.modules[module_name]

        if len(node.imports) > 10:
            suggestions.append(
                f"Module imports {len(node.imports)} dependencies - consider splitting"
            )

        if len(node.imported_by) > 10:
            suggestions.append(
                f"Module is imported by {len(node.imported_by)} modules - changes will have wide impact"
            )

        transitive_deps = self.graph.find_all_dependencies(module_name)
        if len(transitive_deps) > 20:
            suggestions.append(
                f"Module has {len(transitive_deps)} transitive dependencies - consider reducing coupling"
            )

        cycles = self.graph.find_circular_dependencies()
        for cycle in cycles:
            if module_name in cycle:
                suggestions.append(f"Part of circular dependency: {' -> '.join(cycle)}")

        return suggestions

    def get_dependency_order(self) -> list[str]:
        """Get modules in dependency order (leaf first)"""
        sorted_modules = []
        visited = set()

        def visit(module: str):
            if module in visited:
                return
            visited.add(module)

            if module in self.graph.modules:
                for dep in self.graph.modules[module].imports:
                    visit(dep)

            sorted_modules.append(module)

        for module in self.graph.modules:
            visit(module)

        return sorted_modules
