"""Slim Python-only analyzer for nanogpt-opt v1.

Wraps :class:`PyTorchRepoParser` (libcst, Python-only) with an AST
signal scan that understands nanoGPT-style + generic PyTorch
transformer layouts:

- nanoGPT: ``Block, GPT, MLP, CausalSelfAttention, LayerNorm`` with
  ``self.ln_1/ln_2, c_attn/c_proj/c_fc, wte/wpe, GELU, configure_optimizers``
- generic: ``q_proj/k_proj/v_proj/o_proj, LlamaRMSNorm, SwiGLU`` etc.

Deliberately Python-only. No tree-sitter multi-lang, no embeddings,
no call/dependency graphs in v1. Additive module — legacy
``TreeSitterAnalyzer`` / pipeline paths are untouched.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from scholardevclaw.nano_specs import list_supported_specs
from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser

_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "data",
    "config",
    ".hypothesis",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

_NANOGPT_CLASS_SIGNALS = {
    "LayerNorm",
    "CausalSelfAttention",
    "MultiHeadAttention",
    "MLP",
    "Block",
    "GPT",
    "GPTConfig",
    "LlamaRMSNorm",
    "SwiGLU",
}

_SELF_ATTR_SIGNALS = {
    "ln_1",
    "ln_2",
    "c_attn",
    "c_proj",
    "c_fc",
    "wpe",
    "wte",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "mlp",
    "attn",
}

_CALL_SIGNALS = {
    "LayerNorm",
    "GELU",
    "Embedding",
    "Linear",
    "Dropout",
    "Adam",
    "AdamW",
    "StepLR",
    "CosineAnnealingLR",
}


@dataclass
class NanoSignal:
    file: str
    line: int
    kind: str  # class | self_attr | call | function | import
    detail: str


@dataclass
class NanoAnalysis:
    repo: str
    root: Path
    files_scanned: int
    models: list[str] = field(default_factory=list)
    components: dict[str, list[str]] = field(default_factory=dict)
    signals: list[NanoSignal] = field(default_factory=list)
    applicable_specs: list[str] = field(default_factory=list)

    def has(self, component: str) -> bool:
        return bool(self.components.get(component))


def _should_ignore(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return True
    if any(part in _IGNORE_DIRS for part in rel_parts):
        return True
    if path.name.startswith("."):
        return True
    return False


def _collect_file_signals(rel: str, source: str) -> list[NanoSignal]:
    """AST scan for class / self-attr / call signals. Never raises."""
    signals: list[NanoSignal] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return signals
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in _NANOGPT_CLASS_SIGNALS:
            signals.append(NanoSignal(rel, node.lineno, "class", node.name))
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "configure_optimizers",
            "get_lr",
            "forward",
        }:
            signals.append(NanoSignal(rel, node.lineno, "function", node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if "torch" in alias.name or "transformers" in alias.name:
                    signals.append(NanoSignal(rel, node.lineno, "import", alias.name))
    # self.* attributes assigned in __init__ (parser.py misses these)
    try:
        tree2 = ast.parse(source)
    except SyntaxError:
        return signals
    for node in ast.walk(tree2):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            is_init = item.name == "__init__"
            for sub in ast.walk(item):
                if (
                    is_init
                    and isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                ):
                    if sub.value.id == "self" and sub.attr in _SELF_ATTR_SIGNALS:
                        signals.append(
                            NanoSignal(rel, getattr(sub, "lineno", 0), "self_attr", sub.attr)
                        )
                if isinstance(sub, ast.Call):
                    func = sub.func
                    name = ""
                    if isinstance(func, ast.Attribute):
                        name = func.attr
                    elif isinstance(func, ast.Name):
                        name = func.id
                    if name in _CALL_SIGNALS:
                        signals.append(
                            NanoSignal(rel, getattr(sub, "lineno", 0), "call", name)
                        )
                    # keyword patterns: wte/wpe inside nn.ModuleDict(...),
                    # q_proj=... in generic configs, etc.
                    for kw in sub.keywords:
                        if kw.arg in _SELF_ATTR_SIGNALS:
                            signals.append(
                                NanoSignal(
                                    rel, getattr(sub, "lineno", 0), "self_attr", kw.arg
                                )
                            )
    return signals


def _derive_applicable_specs(details: set[str]) -> list[str]:
    """Conservative high-precision mapping from signal details to frozen specs."""
    out: list[str] = []

    def any_of(*names: str) -> bool:
        return any(n in details for n in names)

    if any_of("LayerNorm", "ln_1", "ln_2"):
        out.append("rmsnorm")
    if "Block" in details:
        out.append("preln_transformer")
    if any_of("CausalSelfAttention", "c_attn", "q_proj", "k_proj"):
        out.append("qknorm")
    if any_of("MLP", "GELU", "c_fc"):
        out.append("swiglu")
    if any_of("CausalSelfAttention", "c_attn"):
        out.append("flashattention2")
    if any_of("CausalSelfAttention", "c_attn", "MultiHeadAttention"):
        out.append("grouped_query_attention")
    if any_of("wpe", "wte", "Embedding"):
        out.extend(["rope", "alibi"])
    if any_of("Adam", "AdamW", "configure_optimizers"):
        out.append("lion")
    if any_of("configure_optimizers", "get_lr", "StepLR", "train"):
        out.append("cosine_warmup")

    supported = set(list_supported_specs())
    seen: set[str] = set()
    ordered = [s for s in out if s in supported and not (s in seen or seen.add(s))]
    return ordered


def analyze_repo(repo_path: str | Path) -> NanoAnalysis:
    """Analyze a Python transformer repo. Never raises on bad files."""
    root = Path(repo_path).resolve()
    # Reuse legacy parser for models/files (compat), signals come from AST scan.
    legacy_models: list[str] = []
    try:
        repo_map = PyTorchRepoParser(root).parse()
        legacy_models = [m.name for m in repo_map.models]
    except Exception:
        legacy_models = []

    py_files = [
        p for p in root.glob("**/*.py") if not _should_ignore(p, root) and p.is_file()
    ]
    all_signals: list[NanoSignal] = []
    details: set[str] = set()
    for path in sorted(py_files):
        rel = str(path.relative_to(root))
        try:
            source = path.read_text(errors="ignore")
        except OSError:
            continue
        if "train" in path.stem.lower():
            details.add("train")
        for sig in _collect_file_signals(rel, source):
            all_signals.append(sig)
            details.add(sig.detail)

    components: dict[str, list[str]] = {}
    for sig in all_signals:
        bucket = {
            "class": "classes",
            "self_attr": "self_attrs",
            "call": "calls",
            "function": "functions",
            "import": "imports",
        }[sig.kind]
        components.setdefault(bucket, [])
        if sig.detail not in components[bucket]:
            components[bucket].append(sig.detail)

    models = sorted(set(legacy_models) | {s.detail for s in all_signals if s.kind == "class"})
    return NanoAnalysis(
        repo=root.name,
        root=root,
        files_scanned=len(py_files),
        models=models,
        components=components,
        signals=all_signals,
        applicable_specs=_derive_applicable_specs(details),
    )
