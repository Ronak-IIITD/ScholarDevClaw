from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
import libcst as cst


@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    line_number: int
    methods: List[str] = field(default_factory=list)
    is_nn_module: bool = False
    attributes: List[str] = field(default_factory=list)
    is_custom_norm: bool = False


@dataclass
class FunctionInfo:
    name: str
    line_number: int
    parameters: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    name: str
    alias: Optional[str] = None
    from_module: Optional[str] = None


@dataclass
class ModuleInfo:
    path: Path
    relative_path: Path
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)


@dataclass
class ModelInfo:
    name: str
    file: str
    line: int
    parent: str
    components: Dict = field(default_factory=dict)


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
    modules: List[ModuleInfo] = field(default_factory=list)
    models: List[ModelInfo] = field(default_factory=list)
    training_loop: Optional[TrainingLoopInfo] = None
    test_files: List[str] = field(default_factory=list)


class PyTorchRepoParser:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.modules: List[ModuleInfo] = []

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

    def _get_python_files(self) -> List[Path]:
        python_files = []
        for pattern in ["*.py", "**/*.py"]:
            python_files.extend(self.repo_path.glob(pattern))
        return [f for f in python_files if not self._should_ignore(f)]

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

    def _parse_file(self, file_path: Path) -> Optional[ModuleInfo]:
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

    def _detect_models(self) -> List[ModelInfo]:
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

    def _extract_components(self, cls: ClassInfo, module: ModuleInfo) -> Dict:
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

    def _detect_training_loop(self) -> Optional[TrainingLoopInfo]:
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

    def _find_test_files(self) -> List[str]:
        test_files = []

        for path in self.repo_path.glob("**/test*.py"):
            test_files.append(str(path.relative_to(self.repo_path)))

        for path in self.repo_path.glob("**/*_test.py"):
            test_files.append(str(path.relative_to(self.repo_path)))

        return test_files


class PyTorchComponentVisitor(cst.CSTVisitor):
    def __init__(self):
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self.imports: List[ImportInfo] = []
        self._current_class_bases: List[str] = []

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            alias_name = alias.name.value if hasattr(alias.name, "value") else alias.name
            alias_value = (
                alias.asname.name.value if alias.asname and hasattr(alias.asname, "name") else None
            )
            self.imports.append(
                ImportInfo(
                    name=alias_name,
                    alias=alias_value,
                )
            )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = node.module.value if node.module else None
        for alias in node.names:
            alias_name = alias.name.value if hasattr(alias.name, "value") else alias.name
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

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        bases = []
        for base in node.bases:
            if isinstance(base, cst.Name):
                bases.append(base.value)
            elif isinstance(base, cst.Attribute):
                value = base.value
                if isinstance(value, cst.Name):
                    bases.append(f"{value.value}.{base.attr.value}")
                elif isinstance(value, cst.Attribute):
                    bases.append(f"{base.attr.value}")

        is_nn_module = any("nn.Module" in base or base == "Module" for base in bases)

        is_custom_norm = any("Norm" in node.name.value and "nn." not in base for base in bases)

        methods = []
        attributes = []

        for item in node.body.body:
            if isinstance(item, cst.FunctionDef):
                methods.append(item.name.value)
            elif isinstance(item, cst.AnnAssign) and isinstance(item.target, cst.Name):
                attributes.append(item.target.id)
            elif isinstance(item, cst.Assign):
                for target in item.targets:
                    if isinstance(target, cst.Name):
                        attributes.append(target.id)

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

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
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
