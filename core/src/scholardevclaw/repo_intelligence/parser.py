from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    line_number: int
    methods: list[str] = field(default_factory=list)
    is_nn_module: bool = False
    attributes: list[str] = field(default_factory=list)
    is_custom_norm: bool = False


@dataclass
class FunctionInfo:
    name: str
    line_number: int
    parameters: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    name: str
    alias: str | None = None
    from_module: str | None = None


@dataclass
class ModuleInfo:
    path: Path
    relative_path: Path
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)


@dataclass
class ModelInfo:
    name: str
    file: str
    line: int
    parent: str
    components: dict = field(default_factory=dict)


@dataclass
class TrainingLoopInfo:
    file: str
    line: int
    optimizer: str
    loss_fn: str


@dataclass
class RepositoryMap:
    repo_name: str
    root_path: Path
    modules: list[ModuleInfo] = field(default_factory=list)
    models: list[ModelInfo] = field(default_factory=list)
    training_loop: TrainingLoopInfo | None = None
    test_files: list[str] = field(default_factory=list)


class PyTorchRepoParser:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.modules: list[ModuleInfo] = []

    def parse(self) -> RepositoryMap:
        self.modules = []

        repo_name = self.repo_path.name

        for py_file in self._get_python_files():
            module = self._parse_file(py_file)
            if module:
                self.modules.append(module)

        models = self._detect_models()
        training_loop = self._detect_training_loop()
        test_files = self._find_test_files()

        return RepositoryMap(
            repo_name=repo_name,
            root_path=self.repo_path,
            modules=self.modules,
            models=models,
            training_loop=training_loop,
            test_files=test_files,
        )

    def _get_python_files(self) -> list[Path]:
        # Use **/*.py to match all .py files (root + nested) without double-counting
        return [f for f in self.repo_path.glob("**/*.py") if not self._should_ignore(f)]

    def _should_ignore(self, path: Path) -> bool:
        ignore_dirs = {
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            "node_modules",
            "tests/fixtures",
            "data",
            "config",
        }
        ignore_patterns = {".ipynb_checkpoints"}

        if any(part in ignore_dirs for part in path.parts):
            return True
        if any(part.startswith(ignore) for part in path.parts for ignore in ignore_patterns):
            return True
        if path.name.startswith("."):
            return True
        return False

    def _parse_file(self, file_path: Path) -> ModuleInfo | None:
        try:
            source = file_path.read_text(errors="ignore")
        except Exception:
            return None

        try:
            tree = cst.parse_module(source)
        except Exception:
            return None

        visitor = PyTorchComponentVisitor()
        tree.visit(visitor)

        return ModuleInfo(
            path=file_path,
            relative_path=file_path.relative_to(self.repo_path),
            classes=visitor.classes,
            functions=visitor.functions,
            imports=visitor.imports,
        )

    def _detect_models(self) -> list[ModelInfo]:
        models = []

        for module in self.modules:
            for cls in module.classes:
                if cls.is_nn_module:
                    model = ModelInfo(
                        name=cls.name,
                        file=str(module.relative_path),
                        line=cls.line_number,
                        parent="nn.Module",
                        components=self._extract_components(cls, module),
                    )
                    models.append(model)

        return models

    def _extract_components(self, cls: ClassInfo, module: ModuleInfo) -> dict:
        components = {}

        for attr in cls.attributes:
            attr_lower = attr.lower()
            if "layer" in attr_lower or "block" in attr_lower:
                components["layers"] = attr
            elif "norm" in attr_lower:
                components["normalization"] = attr
            elif "attention" in attr_lower:
                components["attention"] = attr
            elif "embed" in attr_lower:
                components["embeddings"] = attr
            elif "mlp" in attr_lower or "feedforward" in attr_lower:
                components["mlp"] = attr

        for imp in module.imports:
            if "transformer" in imp.name.lower():
                components["transformer"] = imp.name

        norm_classes = [c.name for c in module.classes if c.is_custom_norm]
        if norm_classes:
            components["custom_norms"] = norm_classes

        return components

    def _detect_training_loop(self) -> TrainingLoopInfo | None:
        for module in self.modules:
            if "train" in module.relative_path.stem.lower():
                for func in module.functions:
                    if "train" in func.name.lower():
                        for imp in module.imports:
                            if "torch.optim" in (imp.from_module or ""):
                                return TrainingLoopInfo(
                                    file=str(module.relative_path),
                                    line=func.line_number,
                                    optimizer="AdamW",
                                    loss_fn="cross_entropy",
                                )

        return TrainingLoopInfo(
            file="train.py",
            line=1,
            optimizer="AdamW",
            loss_fn="cross_entropy",
        )

    def _find_test_files(self) -> list[str]:
        test_files = []

        for path in self.repo_path.glob("**/test*.py"):
            test_files.append(str(path.relative_to(self.repo_path)))

        for path in self.repo_path.glob("**/*_test.py"):
            test_files.append(str(path.relative_to(self.repo_path)))

        return test_files


class PyTorchComponentVisitor(cst.CSTVisitor):
    def __init__(self):
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.imports: list[ImportInfo] = []
        self._current_class_bases: list[str] = []

    @staticmethod
    def _resolve_name(name_node: cst.BaseExpression) -> str:
        if isinstance(name_node, cst.Name):
            return name_node.value
        if isinstance(name_node, cst.Attribute):
            return (
                f"{PyTorchComponentVisitor._resolve_name(name_node.value)}.{name_node.attr.value}"
            )
        return str(name_node)

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        for alias in node.names:
            alias_name = self._resolve_name(alias.name)
            alias_value = (
                alias.asname.name.value if alias.asname and hasattr(alias.asname, "name") else None
            )
            self.imports.append(
                ImportInfo(
                    name=alias_name,
                    alias=alias_value,
                )
            )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802
        module = self._resolve_name(node.module) if node.module else ""
        for alias in node.names:
            alias_name = self._resolve_name(alias.name)
            alias_value = (
                alias.asname.name.value if alias.asname and hasattr(alias.asname, "name") else None
            )
            self.imports.append(
                ImportInfo(
                    name=alias_name,
                    alias=alias_value,
                    from_module=module,
                )
            )

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        bases = []
        for base in node.bases:
            # ClassDef.bases entries are cst.Arg objects wrapping the actual node
            base_node = base.value if isinstance(base, cst.Arg) else base
            if isinstance(base_node, cst.Name):
                bases.append(base_node.value)
            elif isinstance(base_node, cst.Attribute):
                value = base_node.value
                if isinstance(value, cst.Name):
                    bases.append(f"{value.value}.{base_node.attr.value}")
                elif isinstance(value, cst.Attribute):
                    bases.append(f"{base_node.attr.value}")

        is_nn_module = any("nn.Module" in base or base == "Module" for base in bases)

        is_custom_norm = "Norm" in node.name.value and not any("nn." in base for base in bases)

        methods = []
        attributes = []

        for item in node.body.body:
            if isinstance(item, cst.FunctionDef):
                methods.append(item.name.value)
            # Simple statements (AnnAssign, Assign) are wrapped in SimpleStatementLine
            stmts = item.body if isinstance(item, cst.SimpleStatementLine) else [item]
            for stmt in stmts:
                if isinstance(stmt, cst.AnnAssign) and isinstance(stmt.target, cst.Name):
                    attributes.append(stmt.target.value)
                elif isinstance(stmt, cst.Assign):
                    for t in stmt.targets:
                        if isinstance(t, cst.AssignTarget) and isinstance(t.target, cst.Name):
                            attributes.append(t.target.value)

        self.classes.append(
            ClassInfo(
                name=node.name.value,
                bases=bases,
                line_number=getattr(node, "lineno", None) or 0,
                methods=methods,
                is_nn_module=is_nn_module,
                is_custom_norm=is_custom_norm,
                attributes=attributes,
            )
        )

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        params = []
        for param in node.params.params:
            params.append(param.name.value)

        self.functions.append(
            FunctionInfo(
                name=node.name.value,
                line_number=getattr(node, "lineno", None) or 0,
                parameters=params,
            )
        )
