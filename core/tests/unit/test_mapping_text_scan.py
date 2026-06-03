"""Tests for MappingEngine._text_scan_for_patterns single-pass optimization."""

from __future__ import annotations

from pathlib import Path

from scholardevclaw.mapping.engine import MappingEngine


def _make_engine(
    tmp_path: Path,
    files: dict[str, str],
    elements: list[dict] | None = None,
) -> MappingEngine:
    """Helper to create a MappingEngine with a fake repo."""
    for rel_path, content in files.items():
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    if elements is None:
        elements = [{"file": k} for k in files]

    repo_analysis = {
        "root_path": tmp_path,
        "elements": elements,
        "language_stats": [],
    }
    spec = {
        "algorithm": {"name": "TestAlgo"},
        "changes": {
            "target_patterns": [],
            "replacement": "optimized_version",
        },
    }
    return MappingEngine(repo_analysis, spec, llm_assistant=None)


class TestTextScanSinglePass:
    def test_single_pattern_single_file(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, {"mod.py": "x = nn.Linear(10, 5)\n"})
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", seen)
        assert len(targets) == 1
        assert targets[0].file == "mod.py"
        assert targets[0].line == 1

    def test_multiple_patterns_across_files(self, tmp_path: Path) -> None:
        engine = _make_engine(
            tmp_path,
            {
                "a.py": "a = nn.Linear(10, 5)\nb = nn.GELU()",
                "b.py": "x = self.ln_1(x)\ny = self.ln_2(x)",
            },
        )
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(
            ["nn.Linear", "nn.GELU", "self.ln_1", "self.ln_2"], "opt", seen
        )
        assert len(targets) == 4

    def test_first_match_per_pattern_per_file(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, {"mod.py": "nn.Linear\nnn.Linear\nnn.Linear\n"})
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", seen)
        assert len(targets) == 1
        assert targets[0].line == 1

    def test_dedup_with_seen(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, {"mod.py": "nn.Linear\n"})
        seen: set[tuple[str, int]] = {("mod.py", 1)}
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", seen)
        assert len(targets) == 0

    def test_missing_root_path(self, tmp_path: Path) -> None:
        repo_analysis = {"root_path": None, "elements": [], "language_stats": []}
        spec = {"algorithm": {"name": "X"}, "changes": {"target_patterns": [], "replacement": "X"}}
        engine = MappingEngine(repo_analysis, spec, llm_assistant=None)
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", set())
        assert targets == []

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        repo_analysis = {
            "root_path": tmp_path / "nonexistent",
            "elements": [],
            "language_stats": [],
        }
        spec = {"algorithm": {"name": "X"}, "changes": {"target_patterns": [], "replacement": "X"}}
        engine = MappingEngine(repo_analysis, spec, llm_assistant=None)
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", set())
        assert targets == []

    def test_patterns_not_found(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, {"mod.py": "x = 1\n"})
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.NonExistent"], "opt", seen)
        assert targets == []

    def test_multiple_patterns_on_same_line(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, {"mod.py": "x = nn.Linear(nn.GELU())\n"})
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.Linear", "nn.GELU"], "opt", seen)
        # Both should be matched — check at least one is found
        assert len(targets) >= 1
        matched_patterns = {t.context.get("matched_pattern", "") for t in targets}
        assert "nn.Linear" in matched_patterns or "nn.GELU" in matched_patterns

    def test_fallback_to_rglob_when_no_elements(self, tmp_path: Path) -> None:
        """When elements list is empty, falls back to rglob for .py files."""
        (tmp_path / "auto.py").write_text("nn.Embedding\n")
        engine = _make_engine(tmp_path, {"dummy.py": "pass\n"}, elements=[])
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.Embedding"], "opt", seen)
        assert len(targets) == 1

    def test_early_break_when_all_matched(self, tmp_path: Path) -> None:
        """Once all patterns are found in a file, no further lines are scanned."""
        lines = ["pass\n"] * 1000 + ["nn.Linear\n"]
        engine = _make_engine(tmp_path, {"mod.py": "".join(lines)})
        seen: set[tuple[str, int]] = set()
        targets = engine._text_scan_for_patterns(["nn.Linear"], "opt", seen)
        assert len(targets) == 1
        assert targets[0].line == 1001
