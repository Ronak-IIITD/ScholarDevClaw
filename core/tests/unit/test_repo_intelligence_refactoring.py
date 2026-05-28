"""Tests for the refactoring module — CrossFileRefactorer & RefactoringAssistant."""

from pathlib import Path

import pytest

from scholardevclaw.repo_intelligence.call_graph import CallGraph, FunctionNode
from scholardevclaw.repo_intelligence.dependency_graph import DependencyGraph, ModuleNode
from scholardevclaw.repo_intelligence.refactoring import (
    CrossFileRefactorer,
    RefactorChange,
    RefactoringAssistant,
    RefactoringPlan,
    RefactoringResult,
)


# =========================================================================
# Dataclasses
# =========================================================================


class TestRefactorChange:
    def test_construction(self):
        c = RefactorChange(
            file="a.py",
            old_start_line=1,
            old_end_line=5,
            new_code="x = 1",
            change_type="replace",
            description="test",
        )
        assert c.file == "a.py"
        assert c.change_type == "replace"

    def test_all_change_types(self):
        for ct in ("replace", "insert", "delete", "update_import"):
            c = RefactorChange("f.py", 1, 1, "", ct, "")
            assert c.change_type == ct


class TestRefactoringPlan:
    def test_default_construction(self):
        plan = RefactoringPlan(name="test")
        assert plan.name == "test"
        assert plan.changes == []
        assert plan.new_files == []
        assert plan.deleted_files == []
        assert plan.dependencies_to_add == {}
        assert plan.dependencies_to_remove == {}

    def test_with_data(self):
        c = RefactorChange("f.py", 1, 2, "new", "replace", "desc")
        plan = RefactoringPlan(
            name="plan1",
            changes=[c],
            new_files=[("new.py", "code")],
            deleted_files=["old.py"],
            dependencies_to_add={"new.py": ["os"]},
            dependencies_to_remove={"old.py": ["sys"]},
        )
        assert len(plan.changes) == 1
        assert plan.new_files[0][0] == "new.py"


class TestRefactoringResult:
    def test_construction(self):
        r = RefactoringResult(
            success=True,
            files_modified=["a.py"],
            files_created=["b.py"],
            files_deleted=["c.py"],
            errors=[],
        )
        assert r.success is True
        assert len(r.files_modified) == 1


# =========================================================================
# CrossFileRefactorer
# =========================================================================


class TestCrossFileRefactorer:
    def test_plan_extract_method(self, tmp_path: Path):
        src = tmp_path / "models.py"
        src.write_text(
            "class MyModel:\n"
            "    def forward(self, x):\n"
            "        return x + 1\n"
            "\n"
            "    def predict(self, x):\n"
            "        return self.forward(x)\n"
        )
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_extract_method(
            source_file="models.py",
            class_name="MyModel",
            method_name="forward",
            new_class_name="ForwardHelper",
        )
        assert plan.name == "Extract forward to ForwardHelper"
        assert len(plan.changes) == 1
        assert plan.changes[0].change_type == "replace"
        assert len(plan.new_files) == 1
        assert plan.new_files[0][0] == "forwardhelper.py"

    def test_plan_extract_method_missing_file(self, tmp_path: Path):
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_extract_method(
            source_file="nonexistent.py",
            class_name="Foo",
            method_name="bar",
            new_class_name="BarHelper",
        )
        assert plan.changes == []

    def test_plan_extract_method_missing_class(self, tmp_path: Path):
        src = tmp_path / "mod.py"
        src.write_text("class Foo:\n    def bar(self): pass\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_extract_method(
            source_file="mod.py",
            class_name="Nonexistent",
            method_name="bar",
            new_class_name="Helper",
        )
        assert plan.changes == []

    def test_plan_move_function(self, tmp_path: Path):
        src = tmp_path / "utils.py"
        src.write_text("import os\nimport sys\n\ndef helper():\n    return os.getcwd()\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_move_function(
            source_file="utils.py",
            function_name="helper",
            target_file="helpers.py",
        )
        assert plan.name == "Move helper to helpers.py"
        assert len(plan.changes) == 1
        assert plan.changes[0].change_type == "delete"
        assert len(plan.new_files) == 1
        assert plan.new_files[0][0] == "helpers.py"

    def test_plan_move_function_missing_file(self, tmp_path: Path):
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_move_function(
            source_file="nope.py",
            function_name="fn",
            target_file="target.py",
        )
        assert plan.changes == []

    def test_plan_move_function_missing_func(self, tmp_path: Path):
        src = tmp_path / "m.py"
        src.write_text("def existing(): pass\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_move_function(
            source_file="m.py",
            function_name="missing",
            target_file="t.py",
        )
        assert plan.changes == []

    def test_plan_rename_across_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("old_name = 1\nother = old_name\n")
        (tmp_path / "b.py").write_text("def old_name(): pass\n")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("old_name = 2\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_rename_across_files("old_name", "new_name")
        assert plan.name == "Rename old_name to new_name"
        assert len(plan.changes) == 2  # a.py and b.py, not __pycache__

    def test_plan_rename_no_matches(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_rename_across_files("nonexistent", "new")
        assert plan.changes == []

    def test_plan_inline_function(self, tmp_path: Path):
        src = tmp_path / "code.py"
        src.write_text("def double(n):\n    return n * 2\n\nresult = double(5)\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_inline_function("code.py", "double")
        assert plan.name == "Inline double"
        assert len(plan.changes) == 1
        assert plan.changes[0].change_type == "replace"

    def test_plan_inline_missing_file(self, tmp_path: Path):
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_inline_function("nope.py", "fn")
        assert plan.changes == []

    def test_plan_inline_missing_function(self, tmp_path: Path):
        src = tmp_path / "m.py"
        src.write_text("def existing(): pass\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = refactorer.plan_inline_function("m.py", "nonexistent")
        assert plan.changes == []

    def test_apply_plan_dry_run(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n")
        refactorer = CrossFileRefactorer(tmp_path)
        change = RefactorChange(
            file="a.py",
            old_start_line=1,
            old_end_line=1,
            new_code="x = 2",
            change_type="replace",
            description="test",
        )
        plan = RefactoringPlan(
            name="test",
            changes=[change],
            new_files=[("b.py", "y = 1")],
            deleted_files=["c.py"],
        )
        result = refactorer.apply_plan(plan, dry_run=True)
        assert result.success is True
        assert "a.py" in result.files_modified
        assert "b.py" in result.files_created
        # File should NOT actually be modified in dry run
        assert tmp_path.joinpath("a.py").read_text() == "x = 1\n"
        assert not tmp_path.joinpath("b.py").exists()

    def test_apply_plan_real_replace(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
        refactorer = CrossFileRefactorer(tmp_path)
        change = RefactorChange(
            file="a.py",
            old_start_line=2,
            old_end_line=2,
            new_code="NEW",
            change_type="replace",
            description="test",
        )
        plan = RefactoringPlan(name="test", changes=[change])
        result = refactorer.apply_plan(plan, dry_run=False)
        assert result.success is True
        assert "a.py" in result.files_modified
        content = tmp_path.joinpath("a.py").read_text()
        assert "NEW" in content

    def test_apply_plan_real_delete(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
        refactorer = CrossFileRefactorer(tmp_path)
        change = RefactorChange(
            file="a.py",
            old_start_line=2,
            old_end_line=3,
            new_code="",
            change_type="delete",
            description="test",
        )
        plan = RefactoringPlan(name="test", changes=[change])
        result = refactorer.apply_plan(plan, dry_run=False)
        assert result.success is True
        content = tmp_path.joinpath("a.py").read_text()
        assert "line1" in content
        assert "line2" not in content

    def test_apply_plan_real_create_files(self, tmp_path: Path):
        refactorer = CrossFileRefactorer(tmp_path)
        plan = RefactoringPlan(
            name="test",
            new_files=[("sub/new.py", "x = 42")],
        )
        result = refactorer.apply_plan(plan, dry_run=False)
        assert result.success is True
        assert "sub/new.py" in result.files_created
        assert tmp_path.joinpath("sub/new.py").read_text() == "x = 42"

    def test_apply_plan_real_delete_file(self, tmp_path: Path):
        (tmp_path / "old.py").write_text("x = 1\n")
        refactorer = CrossFileRefactorer(tmp_path)
        plan = RefactoringPlan(name="test", deleted_files=["old.py"])
        result = refactorer.apply_plan(plan, dry_run=False)
        assert result.success is True
        assert "old.py" in result.files_deleted
        assert not tmp_path.joinpath("old.py").exists()

    def test_apply_plan_error_handling(self, tmp_path: Path):
        refactorer = CrossFileRefactorer(tmp_path)
        # Try to modify a nonexistent file
        change = RefactorChange(
            file="nonexistent.py",
            old_start_line=1,
            old_end_line=1,
            new_code="x",
            change_type="replace",
            description="test",
        )
        plan = RefactoringPlan(name="test", changes=[change])
        result = refactorer.apply_plan(plan, dry_run=False)
        assert result.success is False
        assert len(result.errors) > 0


# =========================================================================
# RefactoringAssistant
# =========================================================================


class TestRefactoringAssistant:
    def setup_method(self):
        self.graph = CallGraph(root_path=Path("/fake"))

    def _add_func(self, name: str, file: str = "mod.py") -> str:
        return self.graph.add_function(name=name, file=Path(file), line=1).full_name

    def test_suggest_extractions_many_calls(self):
        fn = self._add_func("busy")
        for i in range(10):
            callee = self._add_func(f"callee_{i}")
            self.graph.add_call(fn, callee)
        assistant = RefactoringAssistant(None, self.graph)
        suggestions = assistant.suggest_extractions("mod.py")
        assert len(suggestions) > 0
        assert any("calls" in s["reason"].lower() for s in suggestions)

    def test_suggest_extractions_deep_chain(self):
        root = self._add_func("root")
        prev = root
        for i in range(10):
            cur = self._add_func(f"deep_{i}")
            self.graph.add_call(prev, cur)
            prev = cur
        assistant = RefactoringAssistant(None, self.graph)
        suggestions = assistant.suggest_extractions("mod.py")
        assert len(suggestions) > 0
        assert any("depth" in s["reason"].lower() for s in suggestions)

    def test_suggest_extractions_no_call_graph(self):
        assistant = RefactoringAssistant(None, None)
        assert assistant.suggest_extractions("mod.py") == []

    def test_suggest_extractions_nothing_to_suggest(self):
        self._add_func("simple")
        assistant = RefactoringAssistant(None, self.graph)
        suggestions = assistant.suggest_extractions("mod.py")
        assert suggestions == []

    def test_suggest_moves_many_deps(self):
        dep_graph = DependencyGraph(root_path=Path("/fake"))
        dep_graph._ensure_module("mod")
        for i in range(15):
            dep_graph._ensure_module(f"dep_{i}")
            dep_graph.modules["mod"].imports.add(f"dep_{i}")
        assistant = RefactoringAssistant(dep_graph, None)
        suggestions = assistant.suggest_moves("mod")
        assert len(suggestions) == 1
        assert "depends on" in suggestions[0]["reason"].lower()

    def test_suggest_moves_few_deps(self):
        dep_graph = DependencyGraph(root_path=Path("/fake"))
        dep_graph._ensure_module("mod")
        dep_graph.modules["mod"].imports.add("dep1")
        assistant = RefactoringAssistant(dep_graph, None)
        suggestions = assistant.suggest_moves("mod")
        assert suggestions == []

    def test_suggest_moves_no_dep_graph(self):
        assistant = RefactoringAssistant(None, None)
        assert assistant.suggest_moves("mod") == []

    def test_analyze_circular_deps(self):
        dep_graph = DependencyGraph(root_path=Path("/fake"))
        dep_graph._ensure_module("a")
        dep_graph._ensure_module("b")
        dep_graph.modules["a"].imports.add("b")
        dep_graph.modules["b"].imports.add("a")
        assistant = RefactoringAssistant(dep_graph, None)
        cycles = assistant.analyze_circular_deps()
        assert len(cycles) > 0

    def test_analyze_circular_deps_none(self):
        dep_graph = DependencyGraph(root_path=Path("/fake"))
        dep_graph._ensure_module("a")
        assistant = RefactoringAssistant(dep_graph, None)
        assert assistant.analyze_circular_deps() == []

    def test_analyze_circular_deps_no_graph(self):
        assistant = RefactoringAssistant(None, None)
        assert assistant.analyze_circular_deps() == []

    def test_get_refactoring_plan_extract(self):
        assistant = RefactoringAssistant(None, None)
        plan = assistant.get_refactoring_plan("extract_method")
        assert plan.name == "Extract method"
        assert isinstance(plan, RefactoringPlan)

    def test_get_refactoring_plan_move(self):
        assistant = RefactoringAssistant(None, None)
        plan = assistant.get_refactoring_plan("move_function")
        assert plan.name == "Move function"

    def test_get_refactoring_plan_inline(self):
        assistant = RefactoringAssistant(None, None)
        plan = assistant.get_refactoring_plan("inline")
        assert plan.name == "Inline"

    def test_get_refactoring_plan_unknown(self):
        assistant = RefactoringAssistant(None, None)
        plan = assistant.get_refactoring_plan("something_else")
        assert plan.name == "Unknown"
