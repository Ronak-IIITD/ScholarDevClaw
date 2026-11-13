"""Preview the minGPT causal-attention migration without executing repository code."""

from __future__ import annotations

import ast
import difflib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import libcst as cst
import libcst.matchers as m
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    QualifiedName,
    QualifiedNameProvider,
    QualifiedNameSource,
)

# Match the complete method so altered scaling, masks, layouts, uses of the
# attention weights, and control flow cannot silently enter this migration.
_FORWARD = ast.parse("""
def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y
""").body[0]
assert isinstance(_FORWARD, ast.FunctionDef)
_FORWARD_STATEMENT_COUNT = len(_FORWARD.body)
_DROPOUT = ast.parse("self.attn_dropout = nn.Dropout(config.attn_pdrop)").body[0]
_MASK = ast.parse("""
self.register_buffer(
    "bias", torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size)
)
""").body[0]
_REPLACEMENT = """y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=self.bias[:, :, :T, :T] != 0,
    dropout_p=self.attn_dropout.p if self.attn_dropout.training else 0.0,
)
"""
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".nanogpt-opt",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}


@dataclass
class SdpaSite:
    file: str
    class_name: str
    line: int
    status: str
    reason: str


@dataclass
class SdpaPreview:
    modified: bytes
    sites: list[SdpaSite]
    diff: str


class _SdpaTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        QualifiedNameProvider,
        ParentNodeProvider,
    )

    def __init__(self, module: cst.Module, filename: str):
        self.module = module
        self.filename = filename
        self.sites: list[SdpaSite] = []

    def _is_import(self, node: cst.CSTNode, name: str) -> bool:
        return self.get_metadata(QualifiedNameProvider, node, set()) == {
            QualifiedName(name, QualifiedNameSource.IMPORT)
        }

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802
        # Each top-level class is considered once. Nested definitions are outside
        # this migration's supported layout.
        return False

    def leave_ClassDef(  # noqa: N802
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if not isinstance(original_node.body, cst.IndentedBlock):
            return updated_node
        methods = [
            node
            for node in original_node.body.body
            if isinstance(node, cst.FunctionDef) and node.name.value == "forward"
        ]
        if not methods:
            return updated_node
        forward = methods[0]
        calls = m.findall(forward, m.Call())
        sdpa_calls = [
            call
            for call in calls
            if (
                isinstance(call.func, cst.Attribute)
                and call.func.attr.value == "scaled_dot_product_attention"
            )
            or self._is_import(call.func, "torch.nn.functional.scaled_dot_product_attention")
        ]
        already_sdpa = any(
            self._is_import(call.func, "torch.nn.functional.scaled_dot_product_attention")
            for call in sdpa_calls
        )
        tree = ast.parse(self.module.code_for_node(original_node)).body[0]
        assert isinstance(tree, ast.ClassDef)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        )
        is_attention = (
            "attention" in tree.name.lower()
            or any(isinstance(node, ast.MatMult) for node in ast.walk(function))
            and any(
                isinstance(call.func, cst.Attribute) and call.func.attr.value == "softmax"
                for call in calls
            )
        )
        if not sdpa_calls and not is_attention:
            return updated_node
        site = SdpaSite(
            self.filename,
            tree.name,
            self.get_metadata(PositionProvider, forward).start.line,
            "unsupported",
            "",
        )
        self.sites.append(site)
        if already_sdpa:
            site.status = "already_migrated"
            site.reason = "forward already contains SDPA"
            return updated_node
        if sdpa_calls:
            site.reason = "an SDPA-like call has an ambiguous or shadowed binding"
            return updated_node
        if len(methods) != 1 or ast.dump(function) != ast.dump(_FORWARD):
            site.reason = "forward differs from the supported minGPT attention pattern"
            return updated_node
        if (
            original_node.decorators
            or original_node.keywords
            or not isinstance(self.get_metadata(ParentNodeProvider, original_node), cst.Module)
            or len(original_node.bases) != 1
            or original_node.bases[0].star
            or not self._is_import(original_node.bases[0].value, "torch.nn.Module")
        ):
            site.reason = "expected a top-level class directly inheriting from torch.nn.Module"
            return updated_node
        bindings = {
            "F.softmax": "torch.nn.functional.softmax",
            "math.sqrt": "math.sqrt",
            "float": "builtins.float",
        }
        for call in calls:
            expected = bindings.get(get_full_name_for_node(call.func) or "")
            if expected:
                actual = self.get_metadata(QualifiedNameProvider, call.func, set())
                source = (
                    QualifiedNameSource.BUILTIN
                    if expected == "builtins.float"
                    else QualifiedNameSource.IMPORT
                )
                if actual != {QualifiedName(expected, source)}:
                    site.reason = (
                        f"ambiguous or shadowed binding for {self.module.code_for_node(call.func)}"
                    )
                    return updated_node
        initializers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (node.attr == "attn_dropout" or ast.unparse(node.value) == "self.attn_dropout")
        ]
        if (
            len(initializers) != 1
            or len(writes) != 1
            or not any(ast.dump(node) == ast.dump(_DROPOUT) for node in initializers[0].body)
        ):
            site.reason = "expected one unmodified nn.Dropout(config.attn_pdrop) attention module"
            return updated_node
        if not any(ast.dump(node) == ast.dump(_MASK) for node in initializers[0].body):
            site.reason = "expected the standard minGPT lower-triangular mask buffer"
            return updated_node
        init_bindings = {
            "nn.Dropout": "torch.nn.Dropout",
            "torch.tril": "torch.tril",
            "torch.ones": "torch.ones",
        }
        for call in m.findall(original_node, m.Call()):
            expected = init_bindings.get(get_full_name_for_node(call.func) or "")
            if expected and not self._is_import(call.func, expected):
                site.reason = "ambiguous or shadowed binding for dropout or mask initialization"
                return updated_node

        assert isinstance(forward.body, cst.IndentedBlock)
        # Semicolon-separated statements have the same AST but a different CST.
        if len(forward.body.body) != _FORWARD_STATEMENT_COUNT:
            site.reason = "expected one statement per line in forward"
            return updated_node
        removed = forward.body.body[5:10]
        if any(not isinstance(node, cst.SimpleStatementLine) for node in removed):
            site.reason = "expected a straight-line attention block"
            return updated_node
        comments = []
        for node in removed:
            assert isinstance(node, cst.SimpleStatementLine)
            comments.extend(node.leading_lines)
            leading_comments = {id(line.comment) for line in node.leading_lines}
            for comment in m.findall(node, m.Comment()):
                if id(comment) not in leading_comments:
                    comments.append(cst.EmptyLine(comment=comment))
        replacement = cst.parse_statement(
            _REPLACEMENT, config=self.module.config_for_parsing
        ).with_changes(leading_lines=comments)
        migrated = forward.with_changes(
            body=forward.body.with_changes(
                body=[*forward.body.body[:5], replacement, *forward.body.body[10:]]
            )
        )
        site.status = "supported"
        site.reason = "replace manual attention with SDPA using the existing mask and dropout mode"
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=[migrated if node is forward else node for node in original_node.body.body]
            )
        )


def preview_sdpa(source: bytes, filename: str = "model.py") -> SdpaPreview:
    """Return a source-preserving preview. No target code is imported or run."""
    wrapper = MetadataWrapper(cst.parse_module(source))
    transformer = _SdpaTransformer(wrapper.module, filename)
    modified = wrapper.visit(transformer).bytes
    encoding = wrapper.module.encoding
    diff = "".join(
        difflib.unified_diff(
            source.decode(encoding).splitlines(keepends=True),
            modified.decode(encoding).splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    return SdpaPreview(modified, transformer.sites, diff)


def preview_sdpa_repo(root: Path) -> dict[str, Any]:
    """Scan regular Python files, pruning environments and symbolic links."""
    sites: list[dict[str, Any]] = []
    diffs: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    scanned = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in _IGNORED_DIRS
            and not name.startswith(".")
            and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix != ".py" or path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            scanned += 1
            try:
                preview = preview_sdpa(path.read_bytes(), rel)
            except (cst.ParserSyntaxError, SyntaxError, UnicodeError, LookupError, OSError) as exc:
                errors.append({"file": rel, "error": str(exc)})
                continue
            sites.extend(asdict(site) for site in preview.sites)
            if preview.diff:
                diffs[rel] = preview.diff
    return {
        "repo": str(root),
        "dry_run": True,
        "verification": "not_run",
        "files_scanned": scanned,
        "sites": sites,
        "diffs": diffs,
        "errors": errors,
    }
