"""
Cross-file refactoring support.

Provides:
- Coordinated changes across multiple files
- Refactoring pattern detection
- Safe refactoring with dependency awareness
- Import/update management
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RefactorChange:
    """A single refactoring change"""

    file: str
    old_start_line: int
    old_end_line: int
    new_code: str
    change_type: str  # replace, insert, delete, update_import
    description: str


@dataclass
class RefactoringPlan:
    """A complete refactoring plan across files"""

    name: str
    changes: list[RefactorChange] = field(default_factory=list)
    new_files: list[tuple[str, str]] = field(default_factory=list)  # (path, content)
    deleted_files: list[str] = field(default_factory=list)
    dependencies_to_add: dict[str, list[str]] = field(default_factory=dict)
    dependencies_to_remove: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class RefactoringResult:
    """Result of applying a refactoring"""

    success: bool
    files_modified: list[str]
    files_created: list[str]
    files_deleted: list[str]
    errors: list[str]


class CrossFileRefactorer:
    """Handle coordinated refactoring across multiple files"""

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def plan_extract_method(
        self,
        source_file: str,
        class_name: str,
        method_name: str,
        new_class_name: str,
    ) -> RefactoringPlan:
        """Plan extraction of a method to a new class"""
        plan = RefactoringPlan(name=f"Extract {method_name} to {new_class_name}")

        file_path = self.root_path / source_file
        if not file_path.exists():
            return plan

        content = file_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        method_source = ast.get_source_segment(content, item)
                        if method_source:
                            new_class_code = self._generate_new_class(
                                new_class_name, method_name, method_source
                            )
                            plan.changes.append(
                                RefactorChange(
                                    file=source_file,
                                    old_start_line=item.lineno or 0,
                                    old_end_line=item.end_lineno or 0,
                                    new_code=f"# Method {method_name} moved to {new_class_name}",
                                    change_type="replace",
                                    description=f"Extract method {method_name} to new class",
                                )
                            )
                            plan.new_files.append(
                                (
                                    f"{new_class_name.lower()}.py",
                                    new_class_code,
                                )
                            )

        return plan

    def _generate_new_class(self, class_name: str, method_name: str, method_code: str) -> str:
        """Generate code for new class"""
        return f"""class {class_name}:
    {method_code}


"""

    def plan_move_function(
        self,
        source_file: str,
        function_name: str,
        target_file: str,
    ) -> RefactoringPlan:
        """Plan moving a function to another file"""
        plan = RefactoringPlan(name=f"Move {function_name} to {target_file}")

        source_path = self.root_path / source_file
        if not source_path.exists():
            return plan

        content = source_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                func_source = ast.get_source_segment(content, node)
                if func_source:
                    plan.new_files.append((target_file, func_source + "\n\n"))
                    plan.changes.append(
                        RefactorChange(
                            file=source_file,
                            old_start_line=node.lineno or 0,
                            old_end_line=node.end_lineno or 0,
                            new_code=f"# Function {function_name} moved to {target_file}",
                            change_type="delete",
                            description=f"Move function {function_name}",
                        )
                    )
                    plan.dependencies_to_add[target_file] = self._get_imports(content, node)

        return plan

    def _get_imports(self, content: str, node: ast.AST) -> list[str]:
        """Get imports used by a node"""
        imports = []
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                if self._is_imported(item.id, content):
                    imports.append(item.id)
        return list(set(imports))

    def _is_imported(self, name: str, content: str) -> bool:
        """Check if a name is imported"""
        for line in content.split("\n"):
            if f"import {name}" in line or f"from . import {name}" in line:
                return True
        return False

    def plan_rename_across_files(
        self,
        old_name: str,
        new_name: str,
        file_pattern: str = "*.py",
    ) -> RefactoringPlan:
        """Plan renaming across multiple files"""
        plan = RefactoringPlan(name=f"Rename {old_name} to {new_name}")

        for file_path in self.root_path.rglob(file_pattern):
            if "__pycache__" in file_path.parts:
                continue

            content = file_path.read_text()
            new_content = content

            if old_name in content:
                new_content = re.sub(rf"\b{old_name}\b", new_name, content)

            if new_content != content:
                rel_path = str(file_path.relative_to(self.root_path))
                plan.changes.append(
                    RefactorChange(
                        file=rel_path,
                        old_start_line=1,
                        old_end_line=len(content.split("\n")),
                        new_code=new_content,
                        change_type="replace",
                        description=f"Rename {old_name} to {new_name}",
                    )
                )

        return plan

    def plan_inline_function(
        self,
        file: str,
        function_name: str,
    ) -> RefactoringPlan:
        """Plan inlining a function at call sites"""
        plan = RefactoringPlan(name=f"Inline {function_name}")

        file_path = self.root_path / file
        if not file_path.exists():
            return plan

        content = file_path.read_text()
        tree = ast.parse(content)

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                func_node = node
                break

        if not func_node:
            return plan

        func_source = ast.get_source_segment(content, func_node)
        if not func_source:
            return plan

        new_content = content
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == function_name:
                    inlined = self._inline_call(node, func_source)
                    new_content = new_content.replace(
                        ast.get_source_segment(content, node) or "",
                        inlined,
                    )

        rel_path = str(file_path.relative_to(self.root_path))
        plan.changes.append(
            RefactorChange(
                file=rel_path,
                old_start_line=1,
                old_end_line=len(content.split("\n")),
                new_code=new_content,
                change_type="replace",
                description=f"Inline function {function_name}",
            )
        )

        return plan

    def _inline_call(self, call_node: ast.Call, func_source: str) -> str:
        """Inline a function call"""
        return f"# Inlined: {func_source[:50]}..."

    def apply_plan(self, plan: RefactoringPlan, dry_run: bool = True) -> RefactoringResult:
        """Apply a refactoring plan"""
        errors = []
        files_modified = []
        files_created = []
        files_deleted = []

        if dry_run:
            return RefactoringResult(
                success=True,
                files_modified=[c.file for c in plan.changes],
                files_created=[f[0] for f in plan.new_files],
                files_deleted=plan.deleted_files,
                errors=[],
            )

        for change in plan.changes:
            try:
                file_path = self.root_path / change.file
                if change.change_type == "replace":
                    content = file_path.read_text()
                    lines = content.split("\n")
                    new_lines = lines[: change.old_start_line - 1]
                    new_lines.extend(change.new_code.split("\n"))
                    new_lines.extend(lines[change.old_end_line :])
                    file_path.write_text("\n".join(new_lines))
                    files_modified.append(change.file)

                elif change.change_type == "delete":
                    content = file_path.read_text()
                    lines = content.split("\n")
                    new_lines = lines[: change.old_start_line - 1]
                    new_lines.extend(lines[change.old_end_line :])
                    file_path.write_text("\n".join(new_lines))
                    files_modified.append(change.file)

            except Exception as e:
                errors.append(f"Error modifying {change.file}: {e}")

        for path, content in plan.new_files:
            try:
                file_path = self.root_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                files_created.append(path)
            except Exception as e:
                errors.append(f"Error creating {path}: {e}")

        for path in plan.deleted_files:
            try:
                file_path = self.root_path / path
                if file_path.exists():
                    file_path.unlink()
                    files_deleted.append(path)
            except Exception as e:
                errors.append(f"Error deleting {path}: {e}")

        return RefactoringResult(
            success=len(errors) == 0,
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            errors=errors,
        )


class RefactoringAssistant:
    """AI-assisted refactoring suggestions"""

    def __init__(self, dependency_graph, call_graph):
        self.dep_graph = dependency_graph
        self.call_graph = call_graph

    def suggest_extractions(self, file: str) -> list[dict]:
        """Suggest methods/functions that could be extracted"""
        suggestions = []

        if not self.call_graph:
            return suggestions

        file_funcs = [
            (name, node) for name, node in self.call_graph.functions.items() if node.file == file
        ]

        for name, node in file_funcs:
            if len(node.calls) > 5:
                suggestions.append(
                    {
                        "function": name,
                        "reason": f"Calls {len(node.calls)} other functions",
                        "suggestion": "Consider extracting complex logic to separate methods",
                    }
                )

            if self.call_graph.get_call_depth(name) > 5:
                suggestions.append(
                    {
                        "function": name,
                        "reason": f"Call chain depth is {self.call_graph.get_call_depth(name)}",
                        "suggestion": "Consider breaking into smaller functions",
                    }
                )

        return suggestions

    def suggest_moves(self, module_name: str) -> list[dict]:
        """Suggest moving functions to other modules"""
        suggestions = []

        if not self.dep_graph:
            return suggestions

        imports = self.dep_graph.find_all_dependencies(module_name)
        if len(imports) > 10:
            suggestions.append(
                {
                    "module": module_name,
                    "reason": f"Depends on {len(imports)} other modules",
                    "suggestion": "Consider splitting into smaller, focused modules",
                }
            )

        return suggestions

    def analyze_circular_deps(self) -> list[list[str]]:
        """Analyze circular dependencies"""
        if not self.dep_graph:
            return []
        return self.dep_graph.find_circular_dependencies()

    def get_refactoring_plan(self, refactor_type: str, **kwargs) -> RefactoringPlan:
        """Get a refactoring plan based on type"""
        from .dependency_graph import DependencyGraph
        from .call_graph import CallGraph

        if refactor_type == "extract_method":
            return RefactoringPlan(name="Extract method")
        elif refactor_type == "move_function":
            return RefactoringPlan(name="Move function")
        elif refactor_type == "inline":
            return RefactoringPlan(name="Inline")

        return RefactoringPlan(name="Unknown")
