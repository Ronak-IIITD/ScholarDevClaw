"""
Call graph analysis for understanding function/method relationships.

Analyzes:
- Function calls within and across files
- Method resolution order (MRO)
- Call chains and impact analysis
- Static and dynamic call sites
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FunctionNode:
    """A node representing a function or method"""

    name: str
    full_name: str  # module.class.method or module.function
    file: str
    line: int
    end_line: int = 0
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    calls: set[str] = field(default_factory=set)
    called_by: set[str] = field(default_factory=set)
    is_method: bool = False
    is_static: bool = False
    is_classmethod: bool = False


@dataclass
class CallChain:
    """A chain of function calls"""

    from_function: str
    to_function: str
    path: list[str]
    length: int


class CallGraph:
    """Graph of function/method calls"""

    def __init__(self, root_path: Path | None = None):
        self.root_path = root_path or Path.cwd()
        self.functions: dict[str, FunctionNode] = {}
        self._file_functions: dict[str, list[str]] = defaultdict(list)

    def add_function(
        self,
        name: str,
        file: Path,
        line: int,
        end_line: int = 0,
        parameters: list[str] | None = None,
        full_name: str | None = None,
        is_method: bool = False,
        is_static: bool = False,
        is_classmethod: bool = False,
    ) -> FunctionNode:
        """Add a function to the graph"""
        try:
            relative = file.relative_to(self.root_path)
        except ValueError:
            relative = file

        file_str = str(relative)
        full = full_name or f"{file_str}::{name}"

        if full in self.functions:
            node = self.functions[full]
            node.line = line
            node.end_line = end_line
            return node

        node = FunctionNode(
            name=name,
            full_name=full,
            file=file_str,
            line=line,
            end_line=end_line,
            parameters=parameters or [],
            is_method=is_method,
            is_static=is_static,
            is_classmethod=is_classmethod,
        )
        self.functions[full] = node
        self._file_functions[file_str].append(full)
        return node

    def add_call(self, caller: str, callee: str):
        """Add a call relationship"""
        if caller in self.functions:
            self.functions[caller].calls.add(callee)

        if callee in self.functions:
            self.functions[callee].called_by.add(caller)
        else:
            node = FunctionNode(
                name=callee.split("::")[-1],
                full_name=callee,
                file="",
                line=0,
            )
            node.called_by.add(caller)
            self.functions[callee] = node

    def build_from_directory(self, directory: Path, languages: list[str] | None = None):
        """Build call graph from a directory"""
        self.root_path = directory.resolve()
        languages = languages or [".py"]

        for ext in languages:
            for file_path in directory.rglob(f"*{ext}"):
                if "__pycache__" in file_path.parts:
                    continue
                self._analyze_file(file_path)

    def _analyze_file(self, file_path: Path):
        """Analyze a Python file for functions and calls"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, OSError):
            return

        current_class = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        self._add_function(item, file_path, current_class)

            elif isinstance(node, ast.FunctionDef):
                self._add_function(node, file_path, None)

        self._analyze_calls(tree, file_path)

    def _add_function(self, node: ast.FunctionDef, file_path: Path, class_name: str | None):
        """Add a function node from AST"""
        params = [arg.arg for arg in node.args.args]
        full_name = (
            f"{file_path.relative_to(self.root_path)}::{class_name}.{node.name}"
            if class_name
            else f"{file_path.relative_to(self.root_path)}::{node.name}"
        )

        is_static = any(isinstance(d, ast.StaticMethod) for d in node.decorator_list)
        is_classmethod = any(isinstance(d, ast.ClassMethod) for d in node.decorator_list)

        self.add_function(
            name=node.name,
            file=file_path,
            line=node.lineno or 0,
            end_line=node.end_lineno or 0,
            parameters=params,
            full_name=full_name,
            is_method=class_name is not None,
            is_static=is_static,
            is_classmethod=is_classmethod,
        )

    def _analyze_calls(self, tree: ast.AST, file_path: Path):
        """Analyze calls in the AST"""
        current_class = None

        class CallVisitor(ast.NodeVisitor):
            def __init__(inner_self, outer):
                inner_self.outer = outer
                inner_self.current_class = None

            def visit_ClassDef(inner_self, node: ast.ClassDef):
                old_class = inner_self.current_class
                inner_self.current_class = node.name
                inner_self.generic_visit(node)
                inner_self.current_class = old_class

            def visit_FunctionDef(inner_self, node: ast.FunctionDef):
                inner_self.generic_visit(node)

            def visit_Call(inner_self, node: ast.Call):
                caller_name = inner_self._get_current_function_name()
                if caller_name and inner_self.outer.functions.get(caller_name):
                    callee = inner_self._get_callee_name(node)
                    if callee:
                        inner_self.outer.add_call(caller_name, callee)
                inner_self.generic_visit(node)

            def _get_current_function_name(inner_self) -> str | None:
                return None

            def _get_callee_name(inner_self, node: ast.Call) -> str | None:
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        return node.func.attr
                return None

        visitor = CallVisitor(self)
        visitor.visit(tree)

    def get_calls(self, function_name: str) -> set[str]:
        """Get functions called by this function"""
        if function_name not in self.functions:
            return set()
        return self.functions[function_name].calls

    def get_called_by(self, function_name: str) -> set[str]:
        """Get functions that call this function"""
        if function_name not in self.functions:
            return set()
        return self.functions[function_name].called_by

    def find_call_chain(self, from_func: str, to_func: str) -> CallChain | None:
        """Find shortest call chain between two functions"""
        if from_func not in self.functions or to_func not in self.functions:
            return None

        if from_func == to_func:
            return CallChain(from_func, to_func, [from_func], 0)

        from collections import deque

        queue = deque([(from_func, [from_func])])
        visited = {from_func}

        while queue:
            current, path = queue.popleft()

            for callee in self.functions[current].calls:
                if callee == to_func:
                    return CallChain(from_func, to_func, path + [callee], len(path))

                if callee not in visited and callee in self.functions:
                    visited.add(callee)
                    queue.append((callee, path + [callee]))

        return None

    def find_all_callers(self, function_name: str) -> set[str]:
        """Find all transitive callers"""
        if function_name not in self.functions:
            return set()

        result = set()
        to_visit = list(self.functions[function_name].called_by)

        while to_visit:
            caller = to_visit.pop()
            if caller not in result and caller in self.functions:
                result.add(caller)
                to_visit.extend(self.functions[caller].called_by)

        return result

    def find_callees(self, function_name: str) -> set[str]:
        """Find all transitive callees"""
        if function_name not in self.functions:
            return set()

        result = set()
        to_visit = list(self.functions[function_name].calls)

        while to_visit:
            callee = to_visit.pop()
            if callee not in result and callee in self.functions:
                result.add(callee)
                to_visit.extend(self.functions[callee].calls)

        return result

    def get_call_depth(self, function_name: str) -> int:
        """Get maximum call depth"""
        if function_name not in self.functions:
            return 0

        max_depth = 0
        to_visit = [(f, 1) for f in self.functions[function_name].calls]
        visited = {function_name}

        while to_visit:
            func, depth = to_visit.pop()
            if func not in visited:
                visited.add(func)
                max_depth = max(max_depth, depth)
                if func in self.functions:
                    to_visit.extend((f, depth + 1) for f in self.functions[func].calls)

        return max_depth

    def get_impact_score(self, function_name: str) -> float:
        """Calculate impact score based on callers"""
        if function_name not in self.functions:
            return 0.0

        direct_callers = len(self.functions[function_name].called_by)
        transitive_callers = len(self.find_all_callers(function_name))

        return direct_callers + (transitive_callers * 0.5)

    def find_external_calls(self, function_name: str) -> set[str]:
        """Find calls to functions in other files"""
        if function_name not in self.functions:
            return set()

        caller_file = self.functions[function_name].file
        external = set()

        for callee in self.functions[function_name].calls:
            if callee in self.functions:
                callee_file = self.functions[callee].file
                if callee_file and callee_file != caller_file:
                    external.add(callee)

        return external

    def to_dict(self) -> dict:
        """Export graph to dictionary"""
        return {
            "functions": {
                name: {
                    "name": node.name,
                    "full_name": node.full_name,
                    "file": node.file,
                    "line": node.line,
                    "end_line": node.end_line,
                    "parameters": node.parameters,
                    "calls": list(node.calls),
                    "called_by": list(node.called_by),
                    "is_method": node.is_method,
                }
                for name, node in self.functions.items()
            }
        }


class CallGraphAnalyzer:
    """Analyze call graph patterns"""

    def __init__(self, graph: CallGraph):
        self.graph = graph

    def analyze_function(self, function_name: str) -> dict:
        """Get detailed analysis of a function"""
        if function_name not in self.graph.functions:
            return {}

        node = self.graph.functions[function_name]
        direct_calls = len(node.calls)
        direct_callers = len(node.called_by)
        transitive_callees = len(self.graph.find_callees(function_name))
        transitive_callers = len(self.graph.find_all_callers(function_name))
        depth = self.graph.get_call_depth(function_name)
        impact = self.graph.get_impact_score(function_name)
        external = self.graph.find_external_calls(function_name)

        return {
            "function": function_name,
            "name": node.name,
            "file": node.file,
            "line": node.line,
            "direct_calls": direct_calls,
            "direct_callers": direct_callers,
            "transitive_callees": transitive_callees,
            "transitive_callers": transitive_callers,
            "call_depth": depth,
            "impact_score": impact,
            "external_calls": list(external),
            "is_method": node.is_method,
        }

    def find_critical_functions(self, threshold: float = 5.0) -> list[tuple[str, float]]:
        """Find functions with high impact scores"""
        scores = []
        for func_name in self.graph.functions:
            score = self.graph.get_impact_score(func_name)
            if score >= threshold:
                scores.append((func_name, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def find_leaf_functions(self) -> list[str]:
        """Find functions that don't call other functions"""
        leafs = []
        for func_name, node in self.graph.functions.items():
            if not node.calls:
                leafs.append(func_name)
        return leafs

    def find_entry_points(self) -> list[str]:
        """Find functions that are called but don't have callers"""
        entry_points = []
        for func_name, node in self.graph.functions.items():
            if not node.called_by:
                entry_points.append(func_name)
        return entry_points

    def suggest_refactoring(self, function_name: str) -> list[str]:
        """Suggest refactoring based on call analysis"""
        suggestions = []

        if function_name not in self.graph.functions:
            return suggestions

        node = self.graph.functions[function_name]

        if len(node.calls) > 15:
            suggestions.append(
                f"Function calls {len(node.calls)} others - consider extracting logic"
            )

        if len(node.called_by) > 10:
            suggestions.append(
                f"Function is called by {len(node.called_by)} functions - changes will have wide impact"
            )

        transitive_callees = self.graph.find_callees(function_name)
        if len(transitive_callees) > 30:
            suggestions.append(
                f"Function has {len(transitive_callees)} transitive callees - consider reducing coupling"
            )

        depth = self.graph.get_call_depth(function_name)
        if depth > 10:
            suggestions.append(f"Call chain depth is {depth} - consider simplifying logic flow")

        return suggestions
