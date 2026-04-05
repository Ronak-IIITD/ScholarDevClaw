from __future__ import annotations

from pathlib import Path

from scholardevclaw.repo_intelligence.tree_sitter_analyzer import (
    CodeElement,
    ImportStatement,
    TreeSitterAnalyzer,
)


def _analyzer() -> TreeSitterAnalyzer:
    return TreeSitterAnalyzer(Path("."))


def test_detect_frameworks_ignores_symbol_name_false_positives() -> None:
    analyzer = _analyzer()

    elements = [
        CodeElement(type="class", name="PyTorchRepoParser", file="a.py", line=1),
        CodeElement(type="function", name="get_next_run", file="b.py", line=2),
    ]
    imports: list[ImportStatement] = []

    frameworks = analyzer._detect_frameworks(elements, imports)

    assert "torch" not in frameworks
    assert "nextjs" not in frameworks


def test_detect_frameworks_requires_import_module_evidence() -> None:
    analyzer = _analyzer()

    imports = [
        ImportStatement(module="fastapi", names=["FastAPI"], file="api.py", line=1),
        ImportStatement(module="numpy", names=["array"], file="math.py", line=3),
        ImportStatement(module="torch.nn", names=["Module"], file="ml.py", line=4),
    ]

    frameworks = analyzer._detect_frameworks([], imports)

    assert "fastapi" in frameworks
    assert "numpy" in frameworks
    assert "torch" in frameworks


def test_detect_frameworks_handles_js_path_imports() -> None:
    analyzer = _analyzer()

    imports = [
        ImportStatement(module="next/navigation", names=["useRouter"], file="app.tsx", line=1),
        ImportStatement(module="react", names=["useState"], file="app.tsx", line=2),
    ]

    frameworks = analyzer._detect_frameworks([], imports)

    assert "nextjs" in frameworks
    assert "react" in frameworks
