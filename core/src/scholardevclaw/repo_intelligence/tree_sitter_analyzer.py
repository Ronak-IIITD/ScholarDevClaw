"""
Tree-sitter based multi-language code analyzer

Provides AST parsing for 10+ languages using tree-sitter parsers.
Uses real AST walking to extract functions, classes, methods, imports,
decorators, parameters, return types, and visibility across Python,
JavaScript, TypeScript, Go, Rust, and Java.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
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
        """Get or create parser for a language.

        Supports tree-sitter 0.25+ API (Parser(lang) constructor).
        Handles the TypeScript module's separate language_typescript() entrypoint.
        """
        if language in self.parsers:
            return self.parsers[language]

        try:
            from tree_sitter import Language, Parser

            # Map language names to (module_name, factory_function_name) pairs.
            # TypeScript is special: its module exposes language_typescript() not language().
            lang_modules: Dict[str, Tuple[str, str]] = {
                "python": ("tree_sitter_python", "language"),
                "javascript": ("tree_sitter_javascript", "language"),
                "typescript": ("tree_sitter_typescript", "language_typescript"),
                "go": ("tree_sitter_go", "language"),
                "rust": ("tree_sitter_rust", "language"),
                "java": ("tree_sitter_java", "language"),
            }

            if language not in lang_modules:
                return None

            module_name, factory_name = lang_modules[language]

            try:
                lang_module = __import__(module_name)
                factory = getattr(lang_module, factory_name)
                lang = Language(factory())
                # tree-sitter 0.25+: pass language to Parser constructor
                parser = Parser(lang)
                self.parsers[language] = parser
                return parser
            except (ImportError, AttributeError):
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
        """Extract code elements from AST tree using real tree-sitter walking.

        Handles functions, classes, methods, interfaces, structs, impls, and
        type declarations across Python, JavaScript, TypeScript, Go, Rust, Java.
        Extracts parameters, return types, decorators, visibility, and parent class.
        """
        elements: List[CodeElement] = []
        rel_path = str(file_path.relative_to(self.repo_path))

        try:
            source = file_path.read_bytes()
        except Exception:
            return elements

        root = tree.root_node
        self._walk_for_elements(root, rel_path, language, source, elements, parent_class=None)
        return elements

    def _walk_for_elements(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Recursively walk tree-sitter nodes and extract code elements."""
        handler = self._ELEMENT_HANDLERS.get(language)
        if handler:
            handler(self, node, rel_path, language, source, elements, parent_class)

        # Recurse into children, updating parent_class context for class bodies
        for child in node.children:
            new_parent = parent_class

            # When entering a class body, set parent_class to the class name
            if language == "python" and node.type == "class_definition":
                name_node = self._child_by_field(node, "name")
                if name_node:
                    new_parent = self._node_text(name_node, source)
            elif language in ("javascript", "typescript") and node.type in (
                "class_declaration",
                "class",
            ):
                name_node = self._child_by_field(node, "name")
                if name_node:
                    new_parent = self._node_text(name_node, source)
            elif language == "java" and node.type == "class_declaration":
                name_node = self._child_by_field(node, "name")
                if name_node:
                    new_parent = self._node_text(name_node, source)
            elif language == "rust" and node.type == "impl_item":
                type_node = self._child_by_field(node, "type")
                if type_node:
                    new_parent = self._node_text(type_node, source)

            self._walk_for_elements(child, rel_path, language, source, elements, new_parent)

    # ---------- Python element extraction ----------

    def _extract_python_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract Python functions, classes, and decorated definitions."""
        if node.type == "class_definition":
            self._extract_python_class(node, rel_path, source, elements)
        elif node.type == "function_definition":
            self._extract_python_function(node, rel_path, source, elements, parent_class)
        elif node.type == "decorated_definition":
            # The decorators are children of decorated_definition;
            # the actual class/function is also a child. We collect decorators
            # and pass them down (the class/function child will be visited by recursion).
            pass  # Handled via recursion + decorator collection in the child handlers

    def _extract_python_class(
        self,
        node: Any,
        rel_path: str,
        source: bytes,
        elements: List[CodeElement],
    ) -> None:
        name_node = self._child_by_field(node, "name")
        if not name_node:
            return

        name = self._node_text(name_node, source)
        decorators = self._get_python_decorators(node, source)

        # Base classes from argument_list/superclasses
        bases: List[str] = []
        superclasses = self._child_by_field(node, "superclasses")
        if superclasses:
            for child in superclasses.children:
                if child.type == "identifier":
                    bases.append(self._node_text(child, source))
                elif child.type == "attribute":
                    bases.append(self._node_text(child, source))
        else:
            # Check for argument_list pattern
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            bases.append(self._node_text(arg, source))
                        elif arg.type == "attribute":
                            bases.append(self._node_text(arg, source))

        elements.append(
            CodeElement(
                type="class",
                name=name,
                file=rel_path,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language="python",
                visibility="public" if not name.startswith("_") else "private",
                decorators=decorators,
                dependencies=bases,
            )
        )

    def _extract_python_function(
        self,
        node: Any,
        rel_path: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        name_node = self._child_by_field(node, "name")
        if not name_node:
            return

        name = self._node_text(name_node, source)
        is_async = any(c.type == "async" for c in node.children)
        decorators = self._get_python_decorators(node, source)

        # Parameters
        params: List[str] = []
        params_node = self._child_by_field(node, "parameters")
        if params_node:
            for child in params_node.children:
                if child.type in (
                    "identifier",
                    "typed_parameter",
                    "default_parameter",
                    "typed_default_parameter",
                    "list_splat_pattern",
                    "dictionary_splat_pattern",
                ):
                    params.append(self._node_text(child, source))

        # Return type
        return_type = ""
        return_node = self._child_by_field(node, "return_type")
        if return_node:
            return_type = self._node_text(return_node, source)
        else:
            # Look for -> type pattern in children
            found_arrow = False
            for child in node.children:
                if found_arrow and child.type == "type":
                    return_type = self._node_text(child, source)
                    break
                if child.type == "->" or self._node_text(child, source) == "->":
                    found_arrow = True

        # Determine visibility
        if name.startswith("__") and name.endswith("__"):
            visibility = "public"  # dunder methods are public
        elif name.startswith("__"):
            visibility = "private"
        elif name.startswith("_"):
            visibility = "protected"
        else:
            visibility = "public"

        # Element type
        if parent_class:
            elem_type = "async_method" if is_async else "method"
        else:
            elem_type = "async_function" if is_async else "function"

        elements.append(
            CodeElement(
                type=elem_type,
                name=name,
                file=rel_path,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language="python",
                visibility=visibility,
                parameters=params,
                return_type=return_type,
                decorators=decorators,
                parent_class=parent_class,
            )
        )

    def _get_python_decorators(self, node: Any, source: bytes) -> List[str]:
        """Collect decorators from a decorated_definition parent."""
        decorators: List[str] = []
        parent = node.parent
        if parent and parent.type == "decorated_definition":
            for child in parent.children:
                if child.type == "decorator":
                    # Skip the '@' symbol, get the rest
                    parts = []
                    for deco_child in child.children:
                        if deco_child.type != "@":
                            parts.append(self._node_text(deco_child, source))
                    if parts:
                        decorators.append("".join(parts))
        return decorators

    # ---------- JavaScript element extraction ----------

    def _extract_js_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract JavaScript functions, classes, methods, and arrow functions."""
        if node.type == "class_declaration":
            self._extract_js_class(node, rel_path, language, source, elements)
        elif node.type == "function_declaration":
            self._extract_js_function(node, rel_path, language, source, elements)
        elif node.type == "method_definition":
            self._extract_js_method(node, rel_path, language, source, elements, parent_class)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            # Check for arrow functions: const foo = (x) => ...
            self._extract_js_arrow_functions(node, rel_path, language, source, elements)

    def _extract_js_class(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
    ) -> None:
        name_node = self._child_by_field(node, "name")
        if not name_node:
            return

        name = self._node_text(name_node, source)

        # Heritage (extends)
        bases: List[str] = []
        for child in node.children:
            if child.type == "class_heritage":
                for hc in child.children:
                    if hc.type == "identifier":
                        bases.append(self._node_text(hc, source))

        elements.append(
            CodeElement(
                type="class",
                name=name,
                file=rel_path,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
                dependencies=bases,
            )
        )

    def _extract_js_function(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
    ) -> None:
        name_node = self._child_by_field(node, "name")
        if not name_node:
            return

        name = self._node_text(name_node, source)
        is_async = any(c.type == "async" for c in node.children)
        params = self._get_js_params(node, source)
        return_type = self._get_ts_return_type(node, source) if language == "typescript" else ""

        elements.append(
            CodeElement(
                type="async_function" if is_async else "function",
                name=name,
                file=rel_path,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
                parameters=params,
                return_type=return_type,
            )
        )

    def _extract_js_method(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        name_node = self._child_by_field(node, "name")
        if not name_node:
            return

        name = self._node_text(name_node, source)
        is_async = any(c.type == "async" for c in node.children)
        is_static = any(c.type == "static" for c in node.children)
        is_getter = any(c.type == "get" for c in node.children)
        is_setter = any(c.type == "set" for c in node.children)

        params = self._get_js_params(node, source)
        return_type = self._get_ts_return_type(node, source) if language == "typescript" else ""

        elem_type = "async_method" if is_async else "method"
        if is_getter:
            elem_type = "getter"
        elif is_setter:
            elem_type = "setter"

        visibility = "private" if name.startswith("#") else "public"

        elements.append(
            CodeElement(
                type=elem_type,
                name=name,
                file=rel_path,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
                visibility=visibility,
                parameters=params,
                return_type=return_type,
                parent_class=parent_class,
            )
        )

    def _extract_js_arrow_functions(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
    ) -> None:
        """Extract named arrow functions from const/let declarations."""
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = self._child_by_field(child, "name")
                value_node = self._child_by_field(child, "value")
                if name_node and value_node and value_node.type == "arrow_function":
                    name = self._node_text(name_node, source)
                    is_async = any(c.type == "async" for c in value_node.children)
                    params = self._get_js_params(value_node, source)
                    return_type = (
                        self._get_ts_return_type(value_node, source)
                        if language == "typescript"
                        else ""
                    )

                    elements.append(
                        CodeElement(
                            type="async_function" if is_async else "function",
                            name=name,
                            file=rel_path,
                            line=value_node.start_point[0] + 1,
                            end_line=value_node.end_point[0] + 1,
                            language=language,
                            parameters=params,
                            return_type=return_type,
                        )
                    )

    def _get_js_params(self, node: Any, source: bytes) -> List[str]:
        """Extract parameter names from JS/TS formal_parameters."""
        params: List[str] = []
        for child in node.children:
            if child.type == "formal_parameters":
                for p in child.children:
                    if p.type in (
                        "identifier",
                        "required_parameter",
                        "optional_parameter",
                        "rest_pattern",
                        "assignment_pattern",
                    ):
                        params.append(self._node_text(p, source))
                break
        return params

    def _get_ts_return_type(self, node: Any, source: bytes) -> str:
        """Extract return type annotation from TypeScript node."""
        for child in node.children:
            if child.type == "type_annotation":
                # Skip the colon, return the type text
                for tc in child.children:
                    if tc.type != ":":
                        return self._node_text(tc, source)
        return ""

    # ---------- TypeScript element extraction ----------

    def _extract_ts_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract TypeScript elements — delegates to JS handler plus interfaces/type aliases."""
        # Handle TS-specific nodes
        if node.type == "interface_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                elements.append(
                    CodeElement(
                        type="interface",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="typescript",
                    )
                )
        elif node.type == "type_alias_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                elements.append(
                    CodeElement(
                        type="type_alias",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="typescript",
                    )
                )
        elif node.type == "enum_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                elements.append(
                    CodeElement(
                        type="enum",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="typescript",
                    )
                )
        else:
            # Delegate common JS/TS nodes to JS handler
            self._extract_js_element(node, rel_path, language, source, elements, parent_class)

    # ---------- Go element extraction ----------

    def _extract_go_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract Go functions, methods, types, structs, and interfaces."""
        if node.type == "function_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                name = self._node_text(name_node, source)
                params = self._get_go_params(node, source)
                return_type = self._get_go_return_type(node, source)
                visibility = "public" if name[0:1].isupper() else "private"

                elements.append(
                    CodeElement(
                        type="function",
                        name=name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="go",
                        visibility=visibility,
                        parameters=params,
                        return_type=return_type,
                    )
                )

        elif node.type == "method_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                name = self._node_text(name_node, source)
                params = self._get_go_params(node, source)
                return_type = self._get_go_return_type(node, source)
                visibility = "public" if name[0:1].isupper() else "private"

                # Receiver type is the parent struct
                receiver = None
                for child in node.children:
                    if child.type == "parameter_list":
                        # First parameter_list is the receiver
                        for pc in child.children:
                            if pc.type == "parameter_declaration":
                                type_node = None
                                for tc in pc.children:
                                    if tc.type in ("type_identifier", "pointer_type"):
                                        type_node = tc
                                if type_node:
                                    receiver = self._node_text(type_node, source).lstrip("*")
                        break

                elements.append(
                    CodeElement(
                        type="method",
                        name=name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="go",
                        visibility=visibility,
                        parameters=params,
                        return_type=return_type,
                        parent_class=receiver,
                    )
                )

        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = self._child_by_field(child, "name")
                    type_node = self._child_by_field(child, "type")
                    if name_node:
                        name = self._node_text(name_node, source)
                        visibility = "public" if name[0:1].isupper() else "private"
                        elem_type = "struct"
                        if type_node and type_node.type == "interface_type":
                            elem_type = "interface"
                        elif type_node and type_node.type == "struct_type":
                            elem_type = "struct"
                        else:
                            elem_type = "type_alias"

                        elements.append(
                            CodeElement(
                                type=elem_type,
                                name=name,
                                file=rel_path,
                                line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                language="go",
                                visibility=visibility,
                            )
                        )

    def _get_go_params(self, node: Any, source: bytes) -> List[str]:
        """Extract Go function parameters."""
        params: List[str] = []
        param_lists_seen = 0
        for child in node.children:
            if child.type == "parameter_list":
                param_lists_seen += 1
                # For method_declaration, first param_list is receiver, second is params
                if node.type == "method_declaration" and param_lists_seen == 1:
                    continue
                for pc in child.children:
                    if pc.type == "parameter_declaration":
                        params.append(self._node_text(pc, source))
                if node.type != "method_declaration":
                    break
        return params

    def _get_go_return_type(self, node: Any, source: bytes) -> str:
        """Extract Go return type."""
        for child in node.children:
            if (
                child.type
                in ("type_identifier", "pointer_type", "qualified_type", "parameter_list")
                and child != node.children[0]
            ):
                # The last type-like node before the block is the return type
                # But we need to be careful not to confuse it with parameter lists
                pass
        # Simpler approach: look for result node
        result = self._child_by_field(node, "result")
        if result:
            return self._node_text(result, source)
        return ""

    # ---------- Rust element extraction ----------

    def _extract_rust_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract Rust functions, structs, enums, traits, impls."""
        if node.type == "function_item":
            name_node = self._child_by_field(node, "name")
            if name_node:
                name = self._node_text(name_node, source)
                is_pub = any(c.type == "visibility_modifier" for c in node.children)
                is_async = any(c.type == "async" for c in node.children)
                params = self._get_rust_params(node, source)
                return_type = self._get_rust_return_type(node, source)

                elem_type = "async_function" if is_async else "function"
                if parent_class:
                    elem_type = "async_method" if is_async else "method"

                elements.append(
                    CodeElement(
                        type=elem_type,
                        name=name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        visibility="public" if is_pub else "private",
                        parameters=params,
                        return_type=return_type,
                        parent_class=parent_class,
                    )
                )

        elif node.type == "struct_item":
            name_node = self._child_by_field(node, "name")
            if name_node:
                is_pub = any(c.type == "visibility_modifier" for c in node.children)
                elements.append(
                    CodeElement(
                        type="struct",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        visibility="public" if is_pub else "private",
                    )
                )

        elif node.type == "enum_item":
            name_node = self._child_by_field(node, "name")
            if name_node:
                is_pub = any(c.type == "visibility_modifier" for c in node.children)
                elements.append(
                    CodeElement(
                        type="enum",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        visibility="public" if is_pub else "private",
                    )
                )

        elif node.type == "trait_item":
            name_node = self._child_by_field(node, "name")
            if name_node:
                is_pub = any(c.type == "visibility_modifier" for c in node.children)
                elements.append(
                    CodeElement(
                        type="trait",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        visibility="public" if is_pub else "private",
                    )
                )

        elif node.type == "impl_item":
            type_node = self._child_by_field(node, "type")
            if type_node:
                # Check for trait impl: impl Trait for Type
                trait_node = self._child_by_field(node, "trait")
                trait_name = self._node_text(trait_node, source) if trait_node else None
                type_name = self._node_text(type_node, source)

                elements.append(
                    CodeElement(
                        type="impl",
                        name=type_name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        dependencies=[trait_name] if trait_name else [],
                    )
                )

    def _get_rust_params(self, node: Any, source: bytes) -> List[str]:
        """Extract Rust function parameters."""
        params: List[str] = []
        parameters = self._child_by_field(node, "parameters")
        if parameters:
            for child in parameters.children:
                if child.type == "parameter":
                    params.append(self._node_text(child, source))
                elif child.type == "self_parameter":
                    params.append(self._node_text(child, source))
        return params

    def _get_rust_return_type(self, node: Any, source: bytes) -> str:
        """Extract Rust return type."""
        return_type = self._child_by_field(node, "return_type")
        if return_type:
            return self._node_text(return_type, source)
        return ""

    # ---------- Java element extraction ----------

    def _extract_java_element(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        elements: List[CodeElement],
        parent_class: Optional[str],
    ) -> None:
        """Extract Java classes, interfaces, methods, and constructors."""
        if node.type == "class_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                name = self._node_text(name_node, source)
                modifiers = self._get_java_modifiers(node)

                # Superclass
                bases: List[str] = []
                superclass = self._child_by_field(node, "superclass")
                if superclass:
                    bases.append(self._node_text(superclass, source))
                interfaces = self._child_by_field(node, "interfaces")
                if interfaces:
                    for child in interfaces.children:
                        if child.type == "type_identifier":
                            bases.append(self._node_text(child, source))

                elements.append(
                    CodeElement(
                        type="class",
                        name=name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="java",
                        visibility=self._java_visibility(modifiers),
                        dependencies=bases,
                    )
                )

        elif node.type == "interface_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                modifiers = self._get_java_modifiers(node)
                elements.append(
                    CodeElement(
                        type="interface",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="java",
                        visibility=self._java_visibility(modifiers),
                    )
                )

        elif node.type == "method_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                name = self._node_text(name_node, source)
                modifiers = self._get_java_modifiers(node)
                params = self._get_java_params(node, source)
                return_type = ""
                type_node = self._child_by_field(node, "type")
                if type_node:
                    return_type = self._node_text(type_node, source)

                elements.append(
                    CodeElement(
                        type="method",
                        name=name,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="java",
                        visibility=self._java_visibility(modifiers),
                        parameters=params,
                        return_type=return_type,
                        parent_class=parent_class,
                    )
                )

        elif node.type == "constructor_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                modifiers = self._get_java_modifiers(node)
                params = self._get_java_params(node, source)

                elements.append(
                    CodeElement(
                        type="constructor",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="java",
                        visibility=self._java_visibility(modifiers),
                        parameters=params,
                        parent_class=parent_class,
                    )
                )

        elif node.type == "enum_declaration":
            name_node = self._child_by_field(node, "name")
            if name_node:
                modifiers = self._get_java_modifiers(node)
                elements.append(
                    CodeElement(
                        type="enum",
                        name=self._node_text(name_node, source),
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="java",
                        visibility=self._java_visibility(modifiers),
                    )
                )

    def _get_java_modifiers(self, node: Any) -> List[str]:
        """Get Java access modifiers."""
        modifiers: List[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    modifiers.append(self._node_text_fast(mod))
        return modifiers

    def _java_visibility(self, modifiers: List[str]) -> str:
        """Determine Java visibility from modifiers."""
        if "public" in modifiers:
            return "public"
        elif "private" in modifiers:
            return "private"
        elif "protected" in modifiers:
            return "protected"
        return "package"  # default Java visibility

    def _get_java_params(self, node: Any, source: bytes) -> List[str]:
        """Extract Java method parameters."""
        params: List[str] = []
        for child in node.children:
            if child.type == "formal_parameters":
                for p in child.children:
                    if p.type == "formal_parameter":
                        params.append(self._node_text(p, source))
                    elif p.type == "spread_parameter":
                        params.append(self._node_text(p, source))
                break
        return params

    # ---------- Handler dispatch table ----------

    _ELEMENT_HANDLERS: Dict[str, Callable] = {
        "python": _extract_python_element,
        "javascript": _extract_js_element,
        "typescript": _extract_ts_element,
        "go": _extract_go_element,
        "rust": _extract_rust_element,
        "java": _extract_java_element,
    }

    # ---------- Import extraction ----------

    def _extract_imports_from_tree(
        self, tree, file_path: Path, language: str
    ) -> List[ImportStatement]:
        """Extract import statements from AST tree using real tree-sitter walking.

        Handles Python import/from-import, JS/TS ESM imports and require(),
        Go import declarations, Rust use declarations, and Java import declarations.
        """
        imports: List[ImportStatement] = []
        rel_path = str(file_path.relative_to(self.repo_path))

        try:
            source = file_path.read_bytes()
        except Exception:
            return imports

        root = tree.root_node
        self._walk_for_imports(root, rel_path, language, source, imports)
        return imports

    def _walk_for_imports(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Recursively walk tree-sitter nodes and extract import statements."""
        handler = self._IMPORT_HANDLERS.get(language)
        if handler:
            handler(self, node, rel_path, language, source, imports)

        for child in node.children:
            self._walk_for_imports(child, rel_path, language, source, imports)

    # ---------- Python imports ----------

    def _extract_python_import(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Extract Python import and from-import statements."""
        if node.type == "import_statement":
            # import os / import sys as system / import os, sys
            for child in node.children:
                if child.type == "dotted_name":
                    module = self._node_text(child, source)
                    imports.append(
                        ImportStatement(
                            module=module,
                            names=[],
                            file=rel_path,
                            line=node.start_point[0] + 1,
                            is_from=False,
                        )
                    )
                elif child.type == "aliased_import":
                    dotted = None
                    alias = None
                    for ac in child.children:
                        if ac.type == "dotted_name":
                            dotted = self._node_text(ac, source)
                        elif ac.type == "identifier" and ac != child.children[0]:
                            alias = self._node_text(ac, source)
                    if dotted:
                        imports.append(
                            ImportStatement(
                                module=dotted,
                                names=[],
                                file=rel_path,
                                line=node.start_point[0] + 1,
                                is_from=False,
                                alias=alias,
                            )
                        )

        elif node.type == "import_from_statement":
            # from pathlib import Path / from . import local / from ..relative import something
            module_parts: List[str] = []
            names: List[str] = []
            found_import_keyword = False

            for child in node.children:
                if child.type == "from":
                    continue
                elif child.type == "import":
                    found_import_keyword = True
                    continue
                elif child.type == "relative_import":
                    # Dots + optional module
                    prefix = ""
                    for rc in child.children:
                        if rc.type == "import_prefix":
                            prefix = self._node_text(rc, source)
                        elif rc.type == "dotted_name":
                            prefix += self._node_text(rc, source)
                    module_parts.append(prefix)
                elif child.type == "dotted_name":
                    if not found_import_keyword:
                        module_parts.append(self._node_text(child, source))
                    else:
                        names.append(self._node_text(child, source))
                elif child.type == "wildcard_import":
                    names.append("*")

            module = "".join(module_parts)
            if module or names:
                imports.append(
                    ImportStatement(
                        module=module,
                        names=names,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        is_from=True,
                    )
                )

    # ---------- JavaScript/TypeScript imports ----------

    def _extract_js_import(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Extract JS/TS import statements and require() calls."""
        if node.type == "import_statement":
            module = ""
            names: List[str] = []
            alias: Optional[str] = None

            # Find the source string
            for child in node.children:
                if child.type == "string":
                    module = self._extract_string_content(child, source)
                elif child.type == "import_clause":
                    for ic in child.children:
                        if ic.type == "named_imports":
                            for spec in ic.children:
                                if spec.type == "import_specifier":
                                    name_node = self._child_by_field(spec, "name")
                                    if name_node:
                                        names.append(self._node_text(name_node, source))
                                    else:
                                        # Fallback: first identifier
                                        for sc in spec.children:
                                            if sc.type == "identifier":
                                                names.append(self._node_text(sc, source))
                                                break
                        elif ic.type == "identifier":
                            names.append(self._node_text(ic, source))
                        elif ic.type == "namespace_import":
                            # import * as utils
                            for nc in ic.children:
                                if nc.type == "identifier":
                                    alias = self._node_text(nc, source)

            if module:
                imports.append(
                    ImportStatement(
                        module=module,
                        names=names,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        is_from=True,
                        alias=alias,
                    )
                )

        elif node.type == "export_statement":
            # export { foo } from 'bar'
            module = ""
            names: List[str] = []  # type: ignore[no-redef]
            for child in node.children:
                if child.type == "string":
                    module = self._extract_string_content(child, source)
                elif child.type == "export_clause":
                    for spec in child.children:
                        if spec.type == "export_specifier":
                            for sc in spec.children:
                                if sc.type == "identifier":
                                    names.append(self._node_text(sc, source))
                                    break
            if module:
                imports.append(
                    ImportStatement(
                        module=module,
                        names=names,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        is_from=True,
                    )
                )

    # ---------- Go imports ----------

    def _extract_go_import(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Extract Go import declarations."""
        if node.type == "import_declaration":
            for child in node.children:
                if child.type == "import_spec":
                    path_node = self._child_by_field(child, "path")
                    name_node = self._child_by_field(child, "name")
                    if path_node:
                        module = self._extract_string_content(path_node, source)
                        alias = self._node_text(name_node, source) if name_node else None
                        imports.append(
                            ImportStatement(
                                module=module,
                                names=[],
                                file=rel_path,
                                line=child.start_point[0] + 1,
                                is_from=False,
                                alias=alias,
                            )
                        )
                elif child.type == "import_spec_list":
                    for spec in child.children:
                        if spec.type == "import_spec":
                            path_node = self._child_by_field(spec, "path")
                            name_node = self._child_by_field(spec, "name")
                            if path_node:
                                module = self._extract_string_content(path_node, source)
                                alias = self._node_text(name_node, source) if name_node else None
                                imports.append(
                                    ImportStatement(
                                        module=module,
                                        names=[],
                                        file=rel_path,
                                        line=spec.start_point[0] + 1,
                                        is_from=False,
                                        alias=alias,
                                    )
                                )
                elif child.type == "interpreted_string_literal":
                    # Single import: import "fmt"
                    module = self._extract_string_content(child, source)
                    imports.append(
                        ImportStatement(
                            module=module,
                            names=[],
                            file=rel_path,
                            line=child.start_point[0] + 1,
                            is_from=False,
                        )
                    )

    # ---------- Rust imports ----------

    def _extract_rust_import(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Extract Rust use declarations."""
        if node.type == "use_declaration":
            # Extract the full use path
            for child in node.children:
                if child.type in (
                    "use_as_clause",
                    "scoped_use_list",
                    "use_wildcard",
                    "scoped_identifier",
                    "identifier",
                    "use_list",
                ):
                    path_text = self._node_text(child, source)
                    # Parse into module and names
                    if "::" in path_text:
                        parts = path_text.rsplit("::", 1)
                        module = parts[0]
                        name_part = parts[1]
                        if name_part.startswith("{") and name_part.endswith("}"):
                            names = [n.strip() for n in name_part[1:-1].split(",")]
                        elif name_part == "*":
                            names = ["*"]
                        else:
                            names = [name_part]
                    else:
                        module = path_text
                        names = []

                    imports.append(
                        ImportStatement(
                            module=module,
                            names=names,
                            file=rel_path,
                            line=node.start_point[0] + 1,
                            is_from=False,
                        )
                    )
                    break

    # ---------- Java imports ----------

    def _extract_java_import(
        self,
        node: Any,
        rel_path: str,
        language: str,
        source: bytes,
        imports: List[ImportStatement],
    ) -> None:
        """Extract Java import declarations."""
        if node.type == "import_declaration":
            # Get full import path
            for child in node.children:
                if child.type == "scoped_identifier":
                    full_path = self._node_text(child, source)
                    # Split into module and imported name
                    if "." in full_path:
                        parts = full_path.rsplit(".", 1)
                        module = parts[0]
                        name = parts[1]
                    else:
                        module = full_path
                        name = ""

                    is_static = any(c.type == "static" for c in node.children)
                    names = [name] if name else []

                    imports.append(
                        ImportStatement(
                            module=module,
                            names=names,
                            file=rel_path,
                            line=node.start_point[0] + 1,
                            is_from=is_static,
                        )
                    )
                    break
                elif child.type == "asterisk":
                    # import foo.bar.*
                    pass  # handled by scoped_identifier already

    # ---------- Import handler dispatch table ----------

    _IMPORT_HANDLERS: Dict[str, Callable] = {
        "python": _extract_python_import,
        "javascript": _extract_js_import,
        "typescript": _extract_js_import,
        "go": _extract_go_import,
        "rust": _extract_rust_import,
        "java": _extract_java_import,
    }

    # ---------- Helper methods ----------

    @staticmethod
    def _node_text(node: Any, source: bytes) -> str:
        """Extract text content of a tree-sitter node."""
        try:
            return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _node_text_fast(node: Any) -> str:
        """Extract text from node using .text attribute (for small nodes)."""
        try:
            if hasattr(node, "text") and node.text:
                return (
                    node.text.decode("utf-8", errors="replace")
                    if isinstance(node.text, bytes)
                    else str(node.text)
                )
        except Exception:
            pass
        return ""

    @staticmethod
    def _child_by_field(node: Any, field_name: str) -> Optional[Any]:
        """Get a child node by field name, handling API differences."""
        try:
            result = node.child_by_field_name(field_name)
            return result
        except Exception:
            return None

    @staticmethod
    def _extract_string_content(node: Any, source: bytes) -> str:
        """Extract the content from a string literal node, stripping quotes."""
        text = TreeSitterAnalyzer._node_text(node, source)
        # Strip surrounding quotes
        if len(text) >= 2 and text[0] in ('"', "'", "`") and text[-1] in ('"', "'", "`"):
            return text[1:-1]
        # Handle string fragments inside string nodes
        for child in node.children:
            if child.type == "string_fragment" or child.type == "string_content":
                return TreeSitterAnalyzer._node_text(child, source)
        return text

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

    def suggest_research_papers(self, llm_assistant: "Any | None" = None) -> List[Dict]:
        """Suggest research papers based on code patterns.

        Args:
            llm_assistant: Optional LLMResearchAssistant instance. When provided,
                enables dynamic spec discovery via arXiv search and LLM-powered
                extraction, significantly expanding the set of relevant papers.
        """
        analysis = self.analyze()
        suggestions = []

        # Import here to avoid circular dependency
        from ..research_intelligence.extractor import ResearchExtractor

        extractor = ResearchExtractor(llm_assistant=llm_assistant)

        # When an LLM assistant is available, dynamically discover new specs
        # based on the patterns and frameworks found in the repo.
        if llm_assistant is not None:
            try:
                extractor.discover_specs_for_repo(
                    patterns=dict(analysis.patterns),
                    frameworks=list(analysis.frameworks),
                )
            except Exception:
                # Discovery is best-effort; don't block suggestions on failure.
                pass

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
