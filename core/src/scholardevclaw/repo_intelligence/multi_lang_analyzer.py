"""
Multi-language code analyzer using tree-sitter

This module provides AST parsing capabilities for multiple programming languages,
enabling ScholarDevClaw to analyze any codebase regardless of language.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import os


# Language configurations
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx", ".mts"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "csharp": [".cs"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "markdown": [".md"],
    "sql": [".sql"],
    "shell": [".sh", ".bash", ".zsh"],
}


@dataclass
class CodeElement:
    """Represents a code element (function, class, etc.)"""

    type: str  # function, class, method, import, variable, etc.
    name: str
    file: str
    line: int
    end_line: int = 0
    language: str = ""
    visibility: str = "public"  # public, private, protected
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ImportStatement:
    """Represents an import statement"""

    module: str
    names: List[str]  # imported names
    file: str
    line: int
    is_from: bool = False
    alias: Optional[str] = None


@dataclass
class LanguageStats:
    """Statistics about a language in the codebase"""

    language: str
    file_count: int
    line_count: int
    function_count: int
    class_count: int


@dataclass
class RepoAnalysis:
    """Complete repository analysis"""

    root_path: Path
    languages: List[str]
    language_stats: List[LanguageStats]
    elements: List[CodeElement] = field(default_factory=list)
    imports: List[ImportStatement] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    frameworks: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)


class LanguageDetector:
    """Detects programming languages in a repository"""

    @staticmethod
    def detect_from_file(file_path: Path) -> Optional[str]:
        """Detect language from file extension"""
        ext = file_path.suffix.lower()

        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return lang

        return None

    @staticmethod
    def detect_from_content(file_path: Path, content: str) -> Optional[str]:
        """Detect language from file content (heuristic)"""
        first_lines = content.split("\n")[:10]
        combined = " ".join(first_lines).lower()

        # Language-specific patterns
        patterns = {
            "python": ["import ", "from ", "def ", "class ", "if __name__"],
            "javascript": ["const ", "let ", "var ", "function ", "=>", "require("],
            "typescript": ["interface ", "type ", ": string", ": number", "export "],
            "go": ["package ", "func ", "import (", "go "],
            "rust": ["fn ", "let mut", "impl ", "use ", "pub fn"],
            "java": ["public class", "private ", "import java.", "void "],
        }

        for lang, pattern_list in patterns.items():
            matches = sum(1 for p in pattern_list if p in combined)
            if matches >= 2:
                return lang

        return LanguageDetector.detect_from_file(file_path)

    @staticmethod
    def detect_repo_languages(repo_path: Path) -> List[str]:
        """Detect all languages in a repository"""
        languages: Set[str] = set()

        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not LanguageDetector._should_ignore(file_path):
                lang = LanguageDetector.detect_from_file(file_path)
                if lang:
                    languages.add(lang)

        return sorted(list(languages))

    @staticmethod
    def _should_ignore(path: Path) -> bool:
        """Check if path should be ignored"""
        ignore_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "target",
            ".idea",
            ".vscode",
            "vendor",
            "packages",
            ".cache",
            "coverage",
            ".pytest_cache",
        }

        ignore_patterns = {
            ".min.js",
            ".bundle.js",
            "generated",
            "node_modules",
            "__pycache__",
            ".pyc",
            ".pyo",
        }

        parts = path.parts

        # Check directory ignore
        if any(d in ignore_dirs for d in parts):
            return True

        # Check file ignore
        if any(p in str(path) for p in ignore_patterns):
            return True

        return False


class MultiLanguageAnalyzer:
    """Analyzes codebases in multiple languages"""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.detector = LanguageDetector()

    def analyze(self) -> RepoAnalysis:
        """Perform comprehensive repository analysis"""

        # Detect languages
        languages = self.detector.detect_repo_languages(self.repo_path)

        # Count files and lines per language
        language_stats = self._count_language_stats(languages)

        # Find code elements
        elements = self._find_code_elements()

        # Find imports
        imports = self._find_imports()

        # Find entry points
        entry_points = self._find_entry_points()

        # Detect frameworks
        frameworks = self._detect_frameworks(imports, elements)

        # Find test files
        test_files = self._find_test_files()

        return RepoAnalysis(
            root_path=self.repo_path,
            languages=languages,
            language_stats=language_stats,
            elements=elements,
            imports=imports,
            entry_points=entry_points,
            frameworks=frameworks,
            test_files=test_files,
            dependencies=self._build_dependency_graph(imports),
        )

    def _count_language_stats(self, languages: List[str]) -> List[LanguageStats]:
        """Count files and lines per language"""
        stats = []

        for lang in languages:
            extensions = LANGUAGE_EXTENSIONS.get(lang, [])
            file_count = 0
            line_count = 0

            for ext in extensions:
                for file_path in self.repo_path.rglob(f"*{ext}"):
                    if not self.detector._should_ignore(file_path):
                        file_count += 1
                        try:
                            line_count += len(file_path.read_text().splitlines())
                        except:
                            pass

            if file_count > 0:
                stats.append(
                    LanguageStats(
                        language=lang,
                        file_count=file_count,
                        line_count=line_count,
                        function_count=0,
                        class_count=0,
                    )
                )

        return stats

    def _find_code_elements(self) -> List[CodeElement]:
        """Find all code elements (functions, classes, etc.)"""
        elements = []

        for file_path in self.repo_path.rglob("*.py"):
            if self.detector._should_ignore(file_path):
                continue

            try:
                content = file_path.read_text()
                elements.extend(self._parse_python(file_path, content))
            except:
                pass

        return elements

    def _parse_python(self, file_path: Path, content: str) -> List[CodeElement]:
        """Parse Python file using simple regex patterns"""
        import re

        elements = []
        lines = content.split("\n")

        # Find classes
        class_pattern = re.compile(r"^class\s+(\w+)(?:\(([^)]+)\))?:")

        # Find functions
        func_pattern = re.compile(r"^def\s+(\w+)\s*\(([^)]*)\):")

        # Find async functions
        async_func_pattern = re.compile(r"^async\s+def\s+(\w+)\s*\(([^)]*)\):")

        for i, line in enumerate(lines, 1):
            # Check for class
            class_match = class_pattern.match(line.strip())
            if class_match:
                elements.append(
                    CodeElement(
                        type="class",
                        name=class_match.group(1),
                        file=str(file_path.relative_to(self.repo_path)),
                        line=i,
                        language="python",
                    )
                )
                continue

            # Check for function
            func_match = func_pattern.match(line.strip())
            if func_match:
                elements.append(
                    CodeElement(
                        type="function",
                        name=func_match.group(1),
                        file=str(file_path.relative_to(self.repo_path)),
                        line=i,
                        language="python",
                        parameters=[p.strip() for p in func_match.group(2).split(",") if p.strip()],
                    )
                )
                continue

            # Check for async function
            async_match = async_func_pattern.match(line.strip())
            if async_match:
                elements.append(
                    CodeElement(
                        type="async_function",
                        name=async_match.group(1),
                        file=str(file_path.relative_to(self.repo_path)),
                        line=i,
                        language="python",
                        parameters=[
                            p.strip() for p in async_match.group(2).split(",") if p.strip()
                        ],
                    )
                )

        return elements

    def _find_imports(self) -> List[ImportStatement]:
        """Find all import statements"""
        imports = []

        for file_path in self.repo_path.rglob("*.py"):
            if self.detector._should_ignore(file_path):
                continue

            try:
                content = file_path.read_text()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    line = line.strip()

                    if line.startswith("import "):
                        module = line.replace("import ", "").split(" as ")[0].strip()
                        imports.append(
                            ImportStatement(
                                module=module,
                                names=[],
                                file=str(file_path.relative_to(self.repo_path)),
                                line=i,
                                is_from=False,
                            )
                        )

                    elif line.startswith("from "):
                        parts = line.split(" import ")
                        if len(parts) == 2:
                            module = parts[0].replace("from ", "").strip()
                            names_str = parts[1].strip()
                            names = [n.strip() for n in names_str.split(",")]
                            imports.append(
                                ImportStatement(
                                    module=module,
                                    names=names,
                                    file=str(file_path.relative_to(self.repo_path)),
                                    line=i,
                                    is_from=True,
                                )
                            )
            except:
                pass

        return imports

    def _find_entry_points(self) -> List[str]:
        """Find entry points (main files)"""
        entry_points = []

        candidates = [
            "main.py",
            "app.py",
            "server.py",
            "index.py",
            "run.py",
            "serve.py",
            "cli.py",
            "__main__.py",
            "index.ts",
            "main.ts",
            "app.ts",
        ]

        for candidate in candidates:
            for file_path in self.repo_path.rglob(candidate):
                if not self.detector._should_ignore(file_path):
                    entry_points.append(str(file_path.relative_to(self.repo_path)))

        return entry_points

    def _detect_frameworks(
        self, imports: List[ImportStatement], elements: List[CodeElement]
    ) -> List[str]:
        """Detect frameworks used in the codebase"""
        frameworks = []

        # Common framework patterns
        framework_patterns = {
            "django": ["django", "from django"],
            "flask": ["flask", "from flask"],
            "fastapi": ["fastapi", "from fastapi"],
            "express": ["express"],
            "react": ["react"],
            "vue": ["vue"],
            "angular": ["@angular"],
            "nextjs": ["next"],
            "flask": ["flask"],
            "torch": ["torch", "torch.nn"],
            "tensorflow": ["tensorflow", "tf."],
            "pytorch": ["torch"],
            "transformers": ["transformers"],
        }

        all_imports = " ".join(imp.module.lower() for imp in imports)

        for framework, patterns in framework_patterns.items():
            if any(p in all_imports for p in patterns):
                frameworks.append(framework)

        return frameworks

    def _find_test_files(self) -> List[str]:
        """Find test files"""
        test_files = []

        patterns = [
            "test_*.py",
            "*_test.py",
            "tests/",
            "__tests__/",
            "*.test.ts",
            "*.spec.ts",
            "test/",
            "tests/",
        ]

        for pattern in patterns:
            for file_path in self.repo_path.rglob(pattern):
                if not self.detector._should_ignore(file_path):
                    test_files.append(str(file_path.relative_to(self.repo_path)))

        return test_files

    def _build_dependency_graph(self, imports: List[ImportStatement]) -> Dict[str, List[str]]:
        """Build dependency graph"""
        graph: Dict[str, List[str]] = {}

        for imp in imports:
            if imp.file not in graph:
                graph[imp.file] = []
            graph[imp.file].append(imp.module)

        return graph

    def find_patterns_for_improvement(self) -> Dict[str, List[str]]:
        """Find code patterns that could be improved with research papers"""
        patterns: Dict[str, List[str]] = {
            "normalization": [],
            "attention": [],
            "activation": [],
            "optimization": [],
            "architecture": [],
        }

        for element in self.elements:
            name_lower = element.name.lower()
            file_lower = element.file.lower()

            # Find normalization patterns
            if any(p in name_lower for p in ["norm", "normalize", "batch"]):
                patterns["normalization"].append(f"{element.file}:{element.line}")

            # Find attention patterns
            if any(p in name_lower for p in ["attention", "attn", "self_attn"]):
                patterns["attention"].append(f"{element.file}:{element.line}")

            # Find MLP/feedforward
            if any(p in name_lower for p in ["mlp", "feedforward", "ffn", "dense"]):
                patterns["activation"].append(f"{element.file}:{element.line}")

            # Find optimizers
            if any(p in name_lower for p in ["optimizer", "adam", "sgd", "sgdr"]):
                patterns["optimization"].append(f"{element.file}:{element.line}")

        return {k: v for k, v in patterns.items() if v}
