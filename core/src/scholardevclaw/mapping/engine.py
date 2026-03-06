"""
Mapping engine — maps research specs to real code locations.

Uses AST-extracted elements from tree-sitter analysis to find exact code
locations (classes, functions, methods, imports) that match a research spec's
target_patterns.  Provides three match tiers:

1. **Exact match** — element name equals a target pattern
2. **Fuzzy match** — substring, case-insensitive, or code-snippet matching
3. **Import match** — imported module/names matching target patterns

When an ``llm_assistant`` is provided, an optional fourth tier performs
LLM-powered semantic matching for cases where naming conventions differ
from the target patterns.

All public dataclasses (``InsertionPoint``, ``MappingResult``, etc.) and
the ``MappingEngine`` constructor signature are fully backward-compatible
with existing callers in ``pipeline.py``, ``cli.py``, ``server.py``,
``experiment/__init__.py``, and ``critic/__init__.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes — fully backward-compatible
# ---------------------------------------------------------------------------


@dataclass
class InsertionPoint:
    file: str
    line: int
    current_code: str
    replacement_required: bool
    context: Dict = field(default_factory=dict)


@dataclass
class CompatibilityIssue:
    location: str
    issue: str
    severity: str


@dataclass
class ValidationResult:
    passed: bool
    issues: List[CompatibilityIssue] = field(default_factory=list)


@dataclass
class MappingResult:
    targets: List[InsertionPoint]
    strategy: str
    confidence: int
    research_spec: Dict


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

# Patterns that indicate a code snippet contains a specific construct
# (used for fuzzy matching when element names don't directly match).
_CODE_PATTERN_ALIASES: Dict[str, List[str]] = {
    "layernorm": ["LayerNorm", "layer_norm", "nn.LayerNorm"],
    "nn.layernorm": ["LayerNorm", "layer_norm", "nn.LayerNorm"],
    "nn.gelu": ["GELU", "gelu", "nn.GELU"],
    "gelu": ["GELU", "gelu", "nn.GELU"],
    "self.ln_1": ["ln_1", "self.ln_1", "layer_norm_1", "norm1"],
    "self.ln_2": ["ln_2", "self.ln_2", "layer_norm_2", "norm2"],
    "self.c_attn": ["c_attn", "self.c_attn", "qkv_proj"],
    "self.q_proj": ["q_proj", "self.q_proj", "query_proj"],
    "self.k_proj": ["k_proj", "self.k_proj", "key_proj"],
    "self.wpe": ["wpe", "position_embedding", "pos_embed"],
    "nn.embedding": ["nn.Embedding", "Embedding"],
    "nn.dropout": ["nn.Dropout", "Dropout"],
    "self.drop": ["self.drop", "dropout", "self.attn_dropout", "self.resid_dropout"],
    "self.flash": ["flash", "flash_attn", "flash_attention"],
    "self.gelu = nn.gelu()": ["GELU", "gelu", "nn.GELU"],
    "self.mlp": ["mlp", "self.mlp", "feedforward", "ffn"],
}


def _normalise(s: str) -> str:
    """Lower-case and strip common prefixes like 'class ', 'self.'."""
    return s.lower().replace("class ", "").replace("self.", "").strip()


def _exact_match(element_name: str, pattern: str) -> bool:
    """Return *True* when *element_name* exactly equals *pattern* (case-sensitive)."""
    return element_name == pattern


def _fuzzy_match(element_name: str, pattern: str) -> bool:
    """Return *True* when *element_name* matches *pattern* via case-insensitive
    substring or known alias expansion."""
    norm_name = _normalise(element_name)
    norm_pat = _normalise(pattern)

    # Direct substring (case-insensitive)
    if norm_pat in norm_name or norm_name in norm_pat:
        return True

    # Alias expansion — check if any alias of the pattern matches the name
    aliases = _CODE_PATTERN_ALIASES.get(norm_pat, [])
    for alias in aliases:
        if _normalise(alias) in norm_name or norm_name in _normalise(alias):
            return True

    return False


def _snippet_match(code_snippet: str, pattern: str) -> bool:
    """Return *True* when *pattern* appears inside a code snippet."""
    if not code_snippet:
        return False
    return pattern.lower() in code_snippet.lower()


def _import_matches(module: str, names: Sequence[str], pattern: str) -> bool:
    """Return *True* when an import statement references *pattern*."""
    norm_pat = pattern.lower()
    if norm_pat in module.lower():
        return True
    for name in names:
        if norm_pat in name.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Element accessor helpers — handle both CodeElement dataclass instances
# and raw dicts (from API server's request.repoAnalysis).
# ---------------------------------------------------------------------------


def _el_attr(element: Any, key: str, default: Any = "") -> Any:
    if isinstance(element, dict):
        return element.get(key, default)
    return getattr(element, key, default)


# ---------------------------------------------------------------------------
# Mapping engine
# ---------------------------------------------------------------------------


class MappingEngine:
    """Maps a research spec onto real code locations in a repository.

    Parameters
    ----------
    repo_analysis : dict
        The ``__dict__`` of a ``RepoAnalysis`` dataclass (from tree-sitter
        analyzer).  Expected keys: ``elements``, ``imports``, ``patterns``,
        ``frameworks``.  Missing keys are tolerated gracefully.
    research_spec : dict
        A spec dict from ``PAPER_SPECS`` or ``ResearchExtractor.get_spec()``.
        Expected shape: ``{"changes": {"target_patterns": [...], "replacement": ...}, ...}``.
    llm_assistant : optional
        An ``LLMResearchAssistant`` instance for semantic matching fallback.
        When ``None``, LLM matching is skipped.
    """

    def __init__(
        self,
        repo_analysis: Dict,
        research_spec: Dict,
        *,
        llm_assistant: Any = None,
    ):
        self.repo_analysis = repo_analysis
        self.research_spec = research_spec
        self.llm_assistant = llm_assistant

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self) -> MappingResult:
        targets = self._find_target_locations()
        validation = self._validate_compatibility(targets)
        strategy = self._select_strategy(targets, validation)
        confidence = self._calculate_confidence(validation, targets)

        return MappingResult(
            targets=targets,
            strategy=strategy,
            confidence=confidence,
            research_spec=self.research_spec,
        )

    # ------------------------------------------------------------------
    # Target location finding — the core rewrite
    # ------------------------------------------------------------------

    def _find_target_locations(self) -> List[InsertionPoint]:
        """Find code locations that match the spec's ``target_patterns``.

        Search order:
        1. Exact element name matches (highest confidence)
        2. Fuzzy element name / snippet matches
        3. Import-statement matches (for patterns like ``nn.LayerNorm``)
        4. LLM semantic matching (if assistant available, no matches yet)
        5. Graceful empty return (never fabricates locations)
        """
        changes = self.research_spec.get("changes", {})
        target_patterns: List[str] = changes.get("target_patterns", [])
        replacement: str = changes.get("replacement", "")

        if not target_patterns:
            return []

        elements = self.repo_analysis.get("elements", [])
        imports = self.repo_analysis.get("imports", [])

        # Separate patterns into class-like (start with uppercase or "class ")
        # and reference-like (e.g. "self.ln_1", "nn.GELU").
        class_patterns: List[str] = []
        ref_patterns: List[str] = []
        for pat in target_patterns:
            stripped = pat.replace("class ", "").strip()
            if stripped and stripped[0].isupper() and "." not in pat:
                class_patterns.append(pat)
            else:
                ref_patterns.append(pat)

        seen: set[tuple[str, int]] = set()  # (file, line) dedup
        targets: List[InsertionPoint] = []

        # ---- Tier 1: Exact matches on elements ----
        for element in elements:
            el_name = _el_attr(element, "name", "")
            el_type = _el_attr(element, "type", "")
            el_file = _el_attr(element, "file", "")
            el_line = _el_attr(element, "line", 0)
            el_parent = _el_attr(element, "parent_class")

            for pattern in target_patterns:
                if _exact_match(el_name, pattern) or _exact_match(f"class {el_name}", pattern):
                    key = (el_file, el_line)
                    if key not in seen:
                        seen.add(key)
                        targets.append(
                            self._make_insertion_point(
                                file=el_file,
                                line=el_line,
                                current_code=el_name,
                                replacement=replacement,
                                component_type=el_type,
                                match_tier="exact",
                                pattern=pattern,
                                parent_class=el_parent,
                            )
                        )

        # ---- Tier 2: Fuzzy matches on elements ----
        for element in elements:
            el_name = _el_attr(element, "name", "")
            el_type = _el_attr(element, "type", "")
            el_file = _el_attr(element, "file", "")
            el_line = _el_attr(element, "line", 0)
            el_parent = _el_attr(element, "parent_class")
            el_snippet = _el_attr(element, "code_snippet", "")

            key = (el_file, el_line)
            if key in seen:
                continue

            for pattern in target_patterns:
                if _fuzzy_match(el_name, pattern):
                    seen.add(key)
                    targets.append(
                        self._make_insertion_point(
                            file=el_file,
                            line=el_line,
                            current_code=el_name,
                            replacement=replacement,
                            component_type=el_type,
                            match_tier="fuzzy_name",
                            pattern=pattern,
                            parent_class=el_parent,
                        )
                    )
                    break
                if el_snippet and _snippet_match(el_snippet, pattern):
                    seen.add(key)
                    targets.append(
                        self._make_insertion_point(
                            file=el_file,
                            line=el_line,
                            current_code=el_name,
                            replacement=replacement,
                            component_type=el_type,
                            match_tier="fuzzy_snippet",
                            pattern=pattern,
                            parent_class=el_parent,
                        )
                    )
                    break

        # ---- Tier 3: Import matches ----
        for imp in imports:
            imp_module = _el_attr(imp, "module", "")
            imp_names = _el_attr(imp, "names", [])
            imp_file = _el_attr(imp, "file", "")
            imp_line = _el_attr(imp, "line", 0)

            key = (imp_file, imp_line)
            if key in seen:
                continue

            for pattern in target_patterns:
                if _import_matches(imp_module, imp_names, pattern):
                    seen.add(key)
                    targets.append(
                        self._make_insertion_point(
                            file=imp_file,
                            line=imp_line,
                            current_code=f"import {imp_module}"
                            + (f" ({', '.join(imp_names)})" if imp_names else ""),
                            replacement=replacement,
                            component_type="import",
                            match_tier="import",
                            pattern=pattern,
                        )
                    )
                    break

        # ---- Tier 3.5: Text scan for usage patterns (self.*, nn.*) ----
        # Some target patterns (e.g. "self.wpe", "nn.GELU") are usage
        # patterns inside method bodies that don't appear as element names
        # or imports.  Do a lightweight text scan when we have a root_path.
        usage_patterns = [
            p
            for p in target_patterns
            if "." in p and p not in {t.context.get("matched_pattern") for t in targets}
        ]
        if usage_patterns:
            text_targets = self._text_scan_for_patterns(usage_patterns, replacement, seen)
            targets.extend(text_targets)

        # ---- Tier 4: Legacy architecture.models path (backward compat) ----
        # Some callers (tests, API) may provide an ``architecture`` dict with
        # ``models[].components`` instead of real AST elements.  Check this
        # path so those callers keep working.
        if not targets:
            legacy_targets = self._search_legacy_architecture(target_patterns, replacement)
            targets.extend(legacy_targets)

        # ---- Tier 5: LLM semantic matching (optional) ----
        if not targets and self.llm_assistant is not None:
            llm_targets = self._llm_semantic_match(elements, target_patterns, replacement)
            targets.extend(llm_targets)

        if targets:
            logger.info(
                "Mapping found %d target(s) across %d unique file(s)",
                len(targets),
                len({t.file for t in targets}),
            )
        else:
            logger.warning(
                "Mapping found no targets for patterns: %s",
                target_patterns,
            )

        return targets

    # ------------------------------------------------------------------
    # Text scan for usage patterns (self.*, nn.*, etc.)
    # ------------------------------------------------------------------

    def _text_scan_for_patterns(
        self,
        patterns: List[str],
        replacement: str,
        seen: set,
    ) -> List[InsertionPoint]:
        """Scan source files for usage patterns that appear inside method bodies.

        This covers patterns like ``self.wpe``, ``nn.GELU``, ``nn.Embedding``
        that aren't element names or imports but do appear in source code.
        Uses the ``root_path`` from repo_analysis if available.
        """
        targets: List[InsertionPoint] = []
        root_path = self.repo_analysis.get("root_path")
        if root_path is None:
            return targets

        root = Path(root_path) if not isinstance(root_path, Path) else root_path
        if not root.is_dir():
            return targets

        # Determine which files to scan from elements
        element_files: set[str] = set()
        for el in self.repo_analysis.get("elements", []):
            f = _el_attr(el, "file", "")
            if f:
                element_files.add(f)

        # If no element files, scan .py files in root
        if not element_files:
            for py_file in root.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    try:
                        rel = str(py_file.relative_to(root))
                        element_files.add(rel)
                    except ValueError:
                        pass

        for rel_file in element_files:
            full_path = root / rel_file
            if not full_path.is_file():
                continue

            try:
                content = full_path.read_text(errors="replace")
            except OSError:
                continue

            lines = content.split("\n")
            for pattern in patterns:
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        key = (rel_file, i)
                        if key in seen:
                            continue
                        seen.add(key)
                        targets.append(
                            self._make_insertion_point(
                                file=rel_file,
                                line=i,
                                current_code=line.strip(),
                                replacement=replacement,
                                component_type="usage",
                                match_tier="text_scan",
                                pattern=pattern,
                            )
                        )
                        # Only report the first occurrence per pattern per file
                        break

        return targets

    # ------------------------------------------------------------------
    # LLM semantic matching
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Legacy architecture.models backward compatibility
    # ------------------------------------------------------------------

    def _search_legacy_architecture(
        self, target_patterns: List[str], replacement: str
    ) -> List[InsertionPoint]:
        """Search the old-style ``architecture.models[].components`` dict.

        This preserves backward compatibility with callers that provide an
        ``architecture`` key instead of real ``elements``.
        """
        targets: List[InsertionPoint] = []
        models = self.repo_analysis.get("architecture", {}).get("models", [])

        for model in models:
            model_file = model.get("file", "")
            components = model.get("components", {})

            # Check individual component fields
            for comp_key, comp_val in components.items():
                if not isinstance(comp_val, str):
                    # Could be a list (e.g. custom_norms)
                    if isinstance(comp_val, list):
                        for item in comp_val:
                            if not isinstance(item, str):
                                continue
                            for pattern in target_patterns:
                                if pattern.lower() in item.lower():
                                    targets.append(
                                        self._make_insertion_point(
                                            file=model_file,
                                            line=1,
                                            current_code=item,
                                            replacement=replacement,
                                            component_type=comp_key,
                                            match_tier="legacy_architecture",
                                            pattern=pattern,
                                        )
                                    )
                    continue

                for pattern in target_patterns:
                    if pattern.lower() in comp_val.lower():
                        targets.append(
                            self._make_insertion_point(
                                file=model_file,
                                line=1,
                                current_code=comp_val,
                                replacement=replacement,
                                component_type=comp_key,
                                match_tier="legacy_architecture",
                                pattern=pattern,
                            )
                        )

        return targets

    # ------------------------------------------------------------------
    # LLM semantic matching
    # ------------------------------------------------------------------

    def _llm_semantic_match(
        self,
        elements: List[Any],
        target_patterns: List[str],
        replacement: str,
    ) -> List[InsertionPoint]:
        """Use LLM to semantically match elements to target patterns.

        Sends the element list and target patterns to the LLM and asks it to
        identify which elements are semantically related to the patterns even
        if the names don't overlap lexically.
        """
        targets: List[InsertionPoint] = []
        if not elements:
            return targets

        try:
            # Build a concise element summary for the LLM
            element_lines: List[str] = []
            for i, el in enumerate(elements):
                el_name = _el_attr(el, "name", "")
                el_type = _el_attr(el, "type", "")
                el_file = _el_attr(el, "file", "")
                el_line = _el_attr(el, "line", 0)
                el_parent = _el_attr(el, "parent_class")
                parent_str = f" (in {el_parent})" if el_parent else ""
                element_lines.append(f"{i}: {el_type} {el_name}{parent_str} at {el_file}:{el_line}")

            element_summary = "\n".join(element_lines[:200])  # cap to avoid token blow-up

            spec_name = self.research_spec.get("algorithm", {}).get("name", replacement)
            spec_replaces = self.research_spec.get("algorithm", {}).get("replaces", "")
            spec_category = self.research_spec.get("algorithm", {}).get("category", "")

            prompt = (
                "You are a code mapping assistant. Given a list of code elements from a "
                "repository and a research spec, identify which elements should be modified.\n\n"
                f"Research spec: {spec_name}\n"
                f"Replaces: {spec_replaces}\n"
                f"Category: {spec_category}\n"
                f"Target patterns: {target_patterns}\n"
                f"Replacement: {replacement}\n\n"
                f"Code elements:\n{element_summary}\n\n"
                "Return a JSON array of objects with keys: "
                '"index" (element index from the list above), '
                '"reason" (short explanation of why this element matches).\n'
                "Only include elements that are semantically related to the spec. "
                "Return an empty array [] if nothing matches."
            )

            # Use analyse_code which returns a CodeAnalysis with raw_response
            analysis = self.llm_assistant.analyse_code(prompt)
            raw = getattr(analysis, "raw_response", "")

            # Extract JSON array from the response
            matches = self._parse_llm_matches(raw)

            for match in matches:
                idx = match.get("index")
                reason = match.get("reason", "LLM semantic match")
                if idx is None or not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= len(elements):
                    continue

                el = elements[idx]
                targets.append(
                    self._make_insertion_point(
                        file=_el_attr(el, "file", ""),
                        line=_el_attr(el, "line", 0),
                        current_code=_el_attr(el, "name", ""),
                        replacement=replacement,
                        component_type=_el_attr(el, "type", ""),
                        match_tier="llm_semantic",
                        pattern=", ".join(target_patterns),
                        parent_class=_el_attr(el, "parent_class"),
                        extra_context={"llm_reason": reason},
                    )
                )

            if targets:
                logger.info("LLM semantic matching found %d target(s)", len(targets))

        except Exception as exc:
            logger.warning("LLM semantic matching failed: %s", exc)

        return targets

    @staticmethod
    def _parse_llm_matches(raw: str) -> List[Dict[str, Any]]:
        """Extract a JSON array from LLM output, tolerating fenced blocks."""
        import json

        if not raw:
            return []

        # Try to find a JSON array in the response
        # Handle ```json ... ``` fenced blocks
        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find a bare JSON array
        bracket = re.search(r"\[.*?\]", raw, re.DOTALL)
        if bracket:
            try:
                return json.loads(bracket.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    # ------------------------------------------------------------------
    # Helper to build InsertionPoint with rich context
    # ------------------------------------------------------------------

    @staticmethod
    def _make_insertion_point(
        *,
        file: str,
        line: int,
        current_code: str,
        replacement: str,
        component_type: str,
        match_tier: str,
        pattern: str,
        parent_class: Optional[str] = None,
        extra_context: Optional[Dict] = None,
    ) -> InsertionPoint:
        context: Dict[str, Any] = {
            "component_type": component_type,
            "replacement": replacement,
            "original": current_code,
            "match_tier": match_tier,
            "matched_pattern": pattern,
        }
        if parent_class:
            context["parent_class"] = parent_class
        if extra_context:
            context.update(extra_context)

        return InsertionPoint(
            file=file,
            line=line,
            current_code=current_code,
            replacement_required=True,
            context=context,
        )

    # ------------------------------------------------------------------
    # Compatibility validation
    # ------------------------------------------------------------------

    def _validate_compatibility(self, targets: List[InsertionPoint]) -> ValidationResult:
        issues: List[CompatibilityIssue] = []

        for target in targets:
            original = target.context.get("original", "")
            replacement = target.context.get("replacement", "")

            # Same-name replacement is a no-op
            if original and replacement and original == replacement:
                issues.append(
                    CompatibilityIssue(
                        location=f"{target.file}:{target.line}",
                        issue=f"Same replacement: {original} -> {replacement}",
                        severity="error",
                    )
                )

            # Warn if replacing inside a test file
            if target.file and (
                target.file.startswith("test_")
                or "/tests/" in target.file
                or "/test/" in target.file
            ):
                issues.append(
                    CompatibilityIssue(
                        location=f"{target.file}:{target.line}",
                        issue=f"Target is in a test file: {target.file}",
                        severity="warning",
                    )
                )

        return ValidationResult(
            passed=all(i.severity != "error" for i in issues),
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, targets: List[InsertionPoint], validation: ValidationResult) -> str:
        if not targets:
            return "none"

        if not validation.passed:
            return "manual_review"

        changes = self.research_spec.get("changes", {})
        change_type = changes.get("type", "replace")
        return change_type

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self, validation: ValidationResult, targets: List[InsertionPoint]
    ) -> int:
        """Calculate a 0-100 confidence score.

        Scoring:
        - Base: 30
        - +20 if any targets found
        - +10 per exact-match target (up to +30)
        - +10 for fuzzy or import matches
        - +5 for LLM semantic matches
        - +10 if validation passed
        - +10 if spec has a code template
        - -10 per validation error
        """
        confidence = 30

        if targets:
            confidence += 20

            exact_count = sum(1 for t in targets if t.context.get("match_tier") == "exact")
            fuzzy_count = sum(
                1 for t in targets if t.context.get("match_tier", "").startswith("fuzzy")
            )
            import_count = sum(1 for t in targets if t.context.get("match_tier") == "import")
            llm_count = sum(1 for t in targets if t.context.get("match_tier") == "llm_semantic")

            confidence += min(exact_count * 10, 30)
            if fuzzy_count > 0:
                confidence += 10
            if import_count > 0:
                confidence += 5
            if llm_count > 0:
                confidence += 5

        if validation.passed:
            confidence += 10
        else:
            error_count = sum(1 for i in validation.issues if i.severity == "error")
            confidence -= error_count * 10

        if self.research_spec.get("implementation", {}).get("code_template"):
            confidence += 10

        return max(0, min(confidence, 100))


# ---------------------------------------------------------------------------
# Standalone repo-pattern search (unchanged public API)
# ---------------------------------------------------------------------------


def analyze_repo_for_pattern(repo_path: str, pattern: str) -> List[Dict]:
    """Scan Python files in *repo_path* for lines containing *pattern*."""
    results: List[Dict] = []
    path = Path(repo_path)

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if pattern.lower() in line.lower():
                    results.append(
                        {
                            "file": str(py_file.relative_to(path)),
                            "line": i,
                            "content": line.strip(),
                        }
                    )
        except Exception:
            continue

    return results
