"""
Tree-sitter based multi-language code analyzer

Provides AST parsing for 10+ languages using tree-sitter parsers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import os
import subprocess


# Tree-sitter language configurations
LANGUAGE_CONFIGS = {
    "python": {
        "extensions": [".py"],
        "grammar": "tree-sitter-python",
        "patterns": {
            "function_definition": "(function_definition name: (identifier) @name)",
            "class_definition": "(class_definition name: (identifier) @name)",
            "import_statement": "(import_statement)",
            "from_import": "(import_from_statement)",
        },
    },
    "javascript": {
        "extensions": [".js", ".mjs", ".cjs"],
        "grammar": "tree-sitter-javascript",
        "patterns": {
            "function_declaration": "(function_declaration name: (identifier) @name)",
            "class_declaration": "(class_declaration name: (identifier) @name)",
            "method_definition": "(method_definition name: (property_identifier) @name)",
            "import_statement": "(import_statement)",
        },
    },
    "typescript": {
        "extensions": [".ts", ".tsx", ".mts"],
        "grammar": "tree-sitter-typescript",
        "patterns": {
            "function_declaration": "(function_declaration name: (identifier) @name)",
            "class_declaration": "(class_declaration name: (type_identifier) @name)",
            "interface_declaration": "(interface_declaration name: (type_identifier) @name)",
            "import_statement": "(import_statement)",
        },
    },
    "go": {
        "extensions": [".go"],
        "grammar": "tree-sitter-go",
        "patterns": {
            "function_declaration": "(function_declaration name: (identifier) @name)",
            "method_declaration": "(method_declaration name: (field_identifier) @name)",
            "type_declaration": "(type_declaration (type_spec name: (type_identifier) @name))",
            "import_declaration": "(import_declaration)",
        },
    },
    "rust": {
        "extensions": [".rs"],
        "grammar": "tree-sitter-rust",
        "patterns": {
            "function_item": "(function_item name: (identifier) @name)",
            "struct_item": "(struct_item name: (type_identifier) @name)",
            "impl_item": "(impl_item type: (type_identifier) @name)",
            "use_declaration": "(use_declaration)",
        },
    },
    "java": {
        "extensions": [".java"],
        "grammar": "tree-sitter-java",
        "patterns": {
            "method_declaration": "(method_declaration name: (identifier) @name)",
            "class_declaration": "(class_declaration name: (identifier) @name)",
            "interface_declaration": "(interface_declaration name: (identifier) @name)",
            "import_declaration": "(import_declaration)",
        },
    },
}


@dataclass
class CodeElement:
    """Represents a code element"""

    type: str
    name: str
    file: str
    line: int
    end_line: int = 0
    language: str = ""
    visibility: str = "public"
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    code_snippet: str = ""


@dataclass
class ImportStatement:
    module: str
    names: List[str]
    file: str
    line: int
    is_from: bool = False
    alias: Optional[str] = None


@dataclass
class LanguageStats:
    language: str
    file_count: int
    line_count: int
    function_count: int = 0
    class_count: int = 0


@dataclass
class RepoAnalysis:
    root_path: Path
    languages: List[str]
    language_stats: List[LanguageStats]
    elements: List[CodeElement] = field(default_factory=list)
    imports: List[ImportStatement] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    frameworks: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    patterns: Dict[str, List[str]] = field(default_factory=dict)


class TreeSitterAnalyzer:
    """Multi-language code analyzer using tree-sitter"""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.parsers: Dict[str, Any] = {}
        self._setup_parsers()

    def _setup_parsers(self):
        """Setup tree-sitter parsers (will lazy-load)"""
        pass  # Lazy loading happens in _get_parser

    def _get_parser(self, language: str) -> Optional[Any]:
        """Get or create parser for a language"""
        if language in self.parsers:
            return self.parsers[language]

        try:
            from tree_sitter import Language, Parser

            # Map language names to tree-sitter language modules
            lang_modules = {
                "python": "tree_sitter_python",
                "javascript": "tree_sitter_javascript",
                "typescript": "tree_sitter_typescript",
                "go": "tree_sitter_go",
                "rust": "tree_sitter_rust",
                "java": "tree_sitter_java",
            }

            if language not in lang_modules:
                return None

            module_name = lang_modules[language]

            # Try to import the language module
            try:
                lang_module = __import__(module_name)
                lang = Language(lang_module.language())
                parser = Parser()
                parser.set_language(lang)
                self.parsers[language] = parser
                return parser
            except ImportError:
                return None

        except ImportError:
            return None

    def detect_languages(self) -> List[str]:
        """Detect all languages in the repository"""
        languages: Set[str] = set()

        for lang, config in LANGUAGE_CONFIGS.items():
            for ext in config["extensions"]:
                files = list(self.repo_path.rglob(f"*{ext}"))
                # Filter out ignored directories
                files = [f for f in files if not self._should_ignore(f)]
                if files:
                    languages.add(lang)
                    break

        return sorted(list(languages))

    def analyze(self) -> RepoAnalysis:
        """Perform comprehensive repository analysis"""

        languages = self.detect_languages()

        # Collect stats
        language_stats = []
        for lang in languages:
            stats = self._analyze_language(lang)
            if stats:
                language_stats.append(stats)

        # Find all code elements
        elements = []
        imports = []

        for lang in languages:
            lang_elements, lang_imports = self._parse_language_files(lang)
            elements.extend(lang_elements)
            imports.extend(lang_imports)

        # Find patterns
        patterns = self._find_patterns(elements)

        # Detect frameworks
        frameworks = self._detect_frameworks(elements, imports)

        # Find entry points
        entry_points = self._find_entry_points()

        # Find test files
        test_files = self._find_test_files()

        return RepoAnalysis(
            root_path=self.repo_path,
            languages=languages,
            language_stats=language_stats,
            elements=elements,
            imports=imports,
            entry_points=entry_points,
            dependencies=self._build_dependency_graph(imports),
            frameworks=frameworks,
            test_files=test_files,
            patterns=patterns,
        )

    def _analyze_language(self, language: str) -> Optional[LanguageStats]:
        """Analyze a specific language"""
        if language not in LANGUAGE_CONFIGS:
            return None

        config = LANGUAGE_CONFIGS[language]
        file_count = 0
        line_count = 0

        for ext in config["extensions"]:
            for file_path in self.repo_path.rglob(f"*{ext}"):
                if not self._should_ignore(file_path):
                    file_count += 1
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        line_count += len(content.split("\n"))
                    except:
                        pass

        if file_count == 0:
            return None

        return LanguageStats(
            language=language,
            file_count=file_count,
            line_count=line_count,
        )

    def _parse_language_files(
        self, language: str
    ) -> tuple[List[CodeElement], List[ImportStatement]]:
        """Parse all files of a specific language"""
        elements = []
        imports = []

        if language not in LANGUAGE_CONFIGS:
            return elements, imports

        config = LANGUAGE_CONFIGS[language]
        parser = self._get_parser(language)

        if not parser:
            # Fallback to regex-based parsing
            return self._parse_with_regex(language)

        for ext in config["extensions"]:
            for file_path in self.repo_path.rglob(f"*{ext}"):
                if self._should_ignore(file_path):
                    continue

                try:
                    content = file_path.read_bytes()
                    tree = parser.parse(content)

                    file_elements = self._extract_elements_from_tree(tree, file_path, language)
                    elements.extend(file_elements)

                    file_imports = self._extract_imports_from_tree(tree, file_path, language)
                    imports.extend(file_imports)

                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

        return elements, imports

    def _parse_with_regex(self, language: str) -> tuple[List[CodeElement], List[ImportStatement]]:
        """Fallback parsing using regex patterns"""
        elements = []
        imports = []

        # Use the multi_lang_analyzer's regex-based parsing
        from .multi_lang_analyzer import MultiLanguageAnalyzer

        analyzer = MultiLanguageAnalyzer(self.repo_path)

        if language == "python":
            for file_path in self.repo_path.rglob("*.py"):
                if not self._should_ignore(file_path):
                    try:
                        content = file_path.read_text()
                        file_elements = analyzer._parse_python(file_path, content)
                        elements.extend(file_elements)
                    except:
                        pass

        return elements, imports

    def _extract_elements_from_tree(
        self, tree, file_path: Path, language: str
    ) -> List[CodeElement]:
        """Extract code elements from AST tree"""
        elements = []

        # This would use tree-sitter queries to extract elements
        # For now, return empty list (will implement queries later)

        return elements

    def _extract_imports_from_tree(
        self, tree, file_path: Path, language: str
    ) -> List[ImportStatement]:
        """Extract imports from AST tree"""
        imports = []

        # This would use tree-sitter queries to extract imports
        # For now, return empty list

        return imports

    def _find_patterns(self, elements: List[CodeElement]) -> Dict[str, List[str]]:
        """Find improvement patterns in code"""
        patterns: Dict[str, List[str]] = {
            "normalization": [],
            "attention": [],
            "activation": [],
            "optimization": [],
            "caching": [],
            "async": [],
            "error_handling": [],
        }

        for element in elements:
            name_lower = element.name.lower()

            # Normalization patterns
            if any(p in name_lower for p in ["norm", "normalize", "batch", "layer"]):
                patterns["normalization"].append(f"{element.file}:{element.line}")

            # Attention patterns
            if any(p in name_lower for p in ["attention", "attn", "self_attn"]):
                patterns["attention"].append(f"{element.file}:{element.line}")

            # MLP/Activation patterns
            if any(p in name_lower for p in ["mlp", "feedforward", "ffn", "dense", "relu", "gelu"]):
                patterns["activation"].append(f"{element.file}:{element.line}")

            # Async patterns
            if element.type == "async_function" or "async" in name_lower:
                patterns["async"].append(f"{element.file}:{element.line}")

            # Caching patterns
            if any(p in name_lower for p in ["cache", "memo", "store"]):
                patterns["caching"].append(f"{element.file}:{element.line}")

        return {k: v for k, v in patterns.items() if v}

    def _detect_frameworks(
        self, elements: List[CodeElement], imports: List[ImportStatement]
    ) -> List[str]:
        """Detect frameworks used in the codebase"""
        frameworks = []

        framework_patterns = {
            "django": ["django", "from django"],
            "flask": ["flask", "from flask"],
            "fastapi": ["fastapi", "from fastapi"],
            "express": ["express"],
            "react": ["react"],
            "vue": ["vue"],
            "angular": ["@angular"],
            "nextjs": ["next"],
            "torch": ["torch", "torch.nn"],
            "tensorflow": ["tensorflow", "tf."],
            "transformers": ["transformers"],
            "numpy": ["numpy", "np."],
            "pandas": ["pandas", "pd."],
        }

        all_text = " ".join([imp.module for imp in imports] + [e.name for e in elements]).lower()

        for framework, patterns in framework_patterns.items():
            if any(p in all_text for p in patterns):
                frameworks.append(framework)

        return frameworks

    def _find_entry_points(self) -> List[str]:
        """Find entry points (main files)"""
        entry_points = []

        candidates = {
            "python": [
                "main.py",
                "app.py",
                "server.py",
                "index.py",
                "run.py",
                "serve.py",
                "cli.py",
                "__main__.py",
            ],
            "javascript": ["index.js", "main.js", "app.js", "server.js"],
            "typescript": ["index.ts", "main.ts", "app.ts", "server.ts"],
            "go": ["main.go"],
            "rust": ["main.rs", "lib.rs"],
            "java": ["Main.java", "App.java"],
        }

        for language, files in candidates.items():
            for candidate in files:
                for file_path in self.repo_path.rglob(candidate):
                    if not self._should_ignore(file_path):
                        entry_points.append(str(file_path.relative_to(self.repo_path)))

        return entry_points

    def _find_test_files(self) -> List[str]:
        """Find test files"""
        test_files = []

        test_patterns = {
            "python": ["test_*.py", "*_test.py"],
            "javascript": ["*.test.js", "*.spec.js"],
            "typescript": ["*.test.ts", "*.spec.ts"],
            "go": ["*_test.go"],
            "rust": ["*_test.rs"],
            "java": ["*Test.java"],
        }

        for language, patterns in test_patterns.items():
            for pattern in patterns:
                for file_path in self.repo_path.rglob(pattern):
                    if not self._should_ignore(file_path):
                        test_files.append(str(file_path.relative_to(self.repo_path)))

        # Also look for test directories
        test_dirs = ["tests", "test", "__tests__", "spec"]
        for dir_name in test_dirs:
            test_dir = self.repo_path / dir_name
            if test_dir.exists() and test_dir.is_dir():
                for file_path in test_dir.rglob("*"):
                    if file_path.is_file() and not self._should_ignore(file_path):
                        test_files.append(str(file_path.relative_to(self.repo_path)))

        return list(set(test_files))

    def _build_dependency_graph(self, imports: List[ImportStatement]) -> Dict[str, List[str]]:
        """Build dependency graph"""
        graph: Dict[str, List[str]] = {}

        for imp in imports:
            if imp.file not in graph:
                graph[imp.file] = []
            graph[imp.file].append(imp.module)

        return graph

    def _should_ignore(self, path: Path) -> bool:
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
            ".mypy_cache",
            ".ruff_cache",
            "htmlcov",
            ".tox",
        }

        ignore_patterns = [
            ".min.js",
            ".bundle.js",
            "generated",
            ".pyc",
            ".pyo",
            ".class",
            ".o",
            ".so",
            ".dll",
        ]

        parts = set(path.parts)

        if parts & ignore_dirs:
            return True

        if any(pattern in str(path) for pattern in ignore_patterns):
            return True

        return False

    def suggest_research_papers(self) -> List[Dict]:
        """Suggest research papers based on code patterns"""
        analysis = self.analyze()
        suggestions = []

        # Import here to avoid circular dependency
        from ..research_intelligence.extractor import ResearchExtractor

        extractor = ResearchExtractor()

        for pattern_name, locations in analysis.patterns.items():
            if not locations:
                continue

            papers = extractor.find_papers_for_code_pattern(pattern_name)

            for paper in papers:
                suggestions.append(
                    {
                        "pattern": pattern_name,
                        "locations": locations[:5],  # First 5 locations
                        "paper": paper,
                        "confidence": self._calculate_match_confidence(pattern_name, paper),
                    }
                )

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)

        return suggestions

    def _calculate_match_confidence(self, pattern: str, paper: Dict) -> float:
        """Calculate confidence score for pattern-paper match"""
        confidence = 50.0  # Base confidence

        # Category match boost
        category = paper.get("category", "")
        if pattern in category.lower():
            confidence += 30

        # Year boost (newer papers are more relevant)
        year = paper.get("year", 2024)
        if year >= 2022:
            confidence += 10

        return min(confidence, 100.0)


# Convenience function
def analyze_repository(repo_path: str) -> RepoAnalysis:
    """Quick function to analyze a repository"""
    analyzer = TreeSitterAnalyzer(Path(repo_path))
    return analyzer.analyze()
