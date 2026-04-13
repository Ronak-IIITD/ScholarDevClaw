"""Comprehensive tests for the mapping engine (mapping/engine.py).

Covers:
  - Helper functions (_normalise, _exact_match, _fuzzy_match, _snippet_match, _import_matches, _el_attr)
  - MappingEngine.map() orchestration
  - _find_target_locations() — exact, fuzzy, import, text_scan, legacy_architecture, LLM tiers
  - _text_scan_for_patterns() — real temp files
  - _search_legacy_architecture() — string values, list values, no matches
  - _llm_semantic_match() — success, failure, empty elements, invalid indices
  - _parse_llm_matches() — empty, fenced JSON, bare JSON, invalid JSON
  - _validate_compatibility() — same-name replacement, test file warning, clean targets
  - _select_strategy() — no targets, validation failed, normal
  - _calculate_confidence() — various tier combinations, code template boost
  - analyze_repo_for_pattern() — with temp files
  - InsertionPoint, MappingResult dataclass construction
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholardevclaw.mapping.engine import (
    CompatibilityIssue,
    InsertionPoint,
    MappingEngine,
    MappingResult,
    ValidationResult,
    _el_attr,
    _exact_match,
    _fuzzy_match,
    _import_matches,
    _normalise,
    _snippet_match,
    analyze_repo_for_pattern,
)

# =========================================================================
# Helper builders
# =========================================================================


def _make_element(
    *,
    name: str = "LayerNorm",
    type: str = "class",
    file: str = "model.py",
    line: int = 10,
    parent_class: str | None = None,
    code_snippet: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        type=type,
        file=file,
        line=line,
        parent_class=parent_class,
        code_snippet=code_snippet,
    )


def _make_import(
    *,
    module: str = "torch.nn",
    names: list[str] | None = None,
    file: str = "model.py",
    line: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(module=module, names=names or [], file=file, line=line)


def _make_spec(
    *,
    target_patterns: list[str] | None = None,
    replacement: str = "RMSNorm",
    algorithm_name: str = "RMSNorm",
    change_type: str = "replace",
    code_template: str = "",
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "algorithm": {"name": algorithm_name},
        "changes": {
            "target_patterns": target_patterns or ["LayerNorm"],
            "replacement": replacement,
            "type": change_type,
        },
    }
    if code_template:
        spec["implementation"] = {"code_template": code_template}
    return spec


def _make_engine(
    *,
    elements: list | None = None,
    imports: list | None = None,
    patterns: dict | None = None,
    architecture: dict | None = None,
    root_path: str | None = None,
    spec: dict | None = None,
    llm_assistant: Any = None,
) -> MappingEngine:
    repo_analysis: dict[str, Any] = {
        "elements": elements or [],
        "imports": imports or [],
        "patterns": patterns or {},
        "frameworks": [],
    }
    if architecture is not None:
        repo_analysis["architecture"] = architecture
    if root_path is not None:
        repo_analysis["root_path"] = root_path
    return MappingEngine(
        repo_analysis,
        spec or _make_spec(),
        llm_assistant=llm_assistant,
    )


# =========================================================================
# _normalise
# =========================================================================


class TestNormalise:
    def test_lowercase(self):
        assert _normalise("LayerNorm") == "layernorm"

    def test_strip_class_prefix(self):
        assert _normalise("class MyNorm") == "mynorm"

    def test_strip_self_prefix(self):
        assert _normalise("self.ln_1") == "ln_1"

    def test_strip_whitespace(self):
        assert _normalise("  hello  ") == "hello"

    def test_combined(self):
        # "class " replacement is global, so "myclass " -> "my  " after removing
        # both the leading "class " and the embedded "class " in "myclass ".
        assert _normalise("class self.MyClass  ") == "my"

    def test_combined_no_class_in_name(self):
        assert _normalise("class self.MyWidget  ") == "mywidget"


# =========================================================================
# _exact_match
# =========================================================================


class TestExactMatch:
    def test_same_string_matches(self):
        assert _exact_match("LayerNorm", "LayerNorm") is True

    def test_case_differs_no_match(self):
        assert _exact_match("layernorm", "LayerNorm") is False

    def test_substring_no_match(self):
        assert _exact_match("Layer", "LayerNorm") is False

    def test_empty_strings_match(self):
        assert _exact_match("", "") is True


# =========================================================================
# _fuzzy_match
# =========================================================================


class TestFuzzyMatch:
    def test_case_insensitive_substring(self):
        assert _fuzzy_match("LayerNorm", "layernorm") is True

    def test_pattern_in_name(self):
        assert _fuzzy_match("MyLayerNorm", "layernorm") is True

    def test_name_in_pattern(self):
        assert _fuzzy_match("ln", "ln_1") is True

    def test_alias_expansion(self):
        # "nn.layernorm" has alias "LayerNorm" which normalizes to "layernorm"
        assert _fuzzy_match("LayerNorm", "nn.layernorm") is True

    def test_no_match(self):
        assert _fuzzy_match("Dropout", "LayerNorm") is False

    def test_empty_pattern(self):
        assert _fuzzy_match("LayerNorm", "") is True  # empty in anything

    def test_self_prefix_alias(self):
        # "self.ln_1" has aliases ["ln_1", "self.ln_1", ...]
        assert _fuzzy_match("ln_1", "self.ln_1") is True


# =========================================================================
# _snippet_match
# =========================================================================


class TestSnippetMatch:
    def test_pattern_in_snippet(self):
        assert _snippet_match("x = nn.LayerNorm(dim)", "LayerNorm") is True

    def test_case_insensitive(self):
        assert _snippet_match("x = nn.LayerNorm(dim)", "layernorm") is True

    def test_no_match(self):
        assert _snippet_match("x = nn.Dropout(0.1)", "LayerNorm") is False

    def test_empty_snippet(self):
        assert _snippet_match("", "LayerNorm") is False

    def test_none_like_empty(self):
        assert _snippet_match("", "anything") is False


# =========================================================================
# _import_matches
# =========================================================================


class TestImportMatches:
    def test_module_contains_pattern(self):
        assert _import_matches("torch.nn", [], "nn") is True

    def test_name_contains_pattern(self):
        assert _import_matches("torch", ["LayerNorm"], "LayerNorm") is True

    def test_case_insensitive(self):
        assert _import_matches("torch.nn", ["layernorm"], "LayerNorm") is True

    def test_no_match(self):
        assert _import_matches("torch.optim", ["Adam"], "LayerNorm") is False

    def test_empty_names(self):
        assert _import_matches("random", [], "LayerNorm") is False


# =========================================================================
# _el_attr
# =========================================================================


class TestElAttr:
    def test_dict_key(self):
        assert _el_attr({"name": "foo"}, "name") == "foo"

    def test_dict_missing_key_default(self):
        assert _el_attr({}, "name", "default") == "default"

    def test_object_attr(self):
        obj = SimpleNamespace(name="bar")
        assert _el_attr(obj, "name") == "bar"

    def test_object_missing_attr_default(self):
        obj = SimpleNamespace()
        assert _el_attr(obj, "name", "fallback") == "fallback"


# =========================================================================
# InsertionPoint dataclass
# =========================================================================


class TestInsertionPoint:
    def test_construction(self):
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="LayerNorm",
            replacement_required=True,
            context={"match_tier": "exact"},
        )
        assert ip.file == "model.py"
        assert ip.line == 10
        assert ip.replacement_required is True
        assert ip.context["match_tier"] == "exact"

    def test_default_context(self):
        ip = InsertionPoint(file="f.py", line=1, current_code="x", replacement_required=False)
        assert ip.context == {}


# =========================================================================
# MappingResult dataclass
# =========================================================================


class TestMappingResult:
    def test_construction(self):
        mr = MappingResult(targets=[], strategy="replace", confidence=80, research_spec={})
        assert mr.targets == []
        assert mr.strategy == "replace"
        assert mr.confidence == 80
        assert mr.confidence_breakdown == {}


# =========================================================================
# MappingEngine._find_target_locations — Tier 1: Exact matches
# =========================================================================


class TestFindTargetLocationsExact:
    def test_exact_element_name(self):
        el = _make_element(name="LayerNorm")
        engine = _make_engine(elements=[el])
        result = engine.map()
        assert len(result.targets) >= 1
        exact = [t for t in result.targets if t.context.get("match_tier") == "exact"]
        assert len(exact) == 1
        assert exact[0].file == "model.py"
        assert exact[0].line == 10

    def test_exact_class_prefix(self):
        el = _make_element(name="LayerNorm")
        spec = _make_spec(target_patterns=["class LayerNorm"])
        engine = _make_engine(elements=[el], spec=spec)
        result = engine.map()
        exact = [t for t in result.targets if t.context.get("match_tier") == "exact"]
        assert len(exact) == 1

    def test_no_match_returns_empty(self):
        el = _make_element(name="Dropout")
        spec = _make_spec(target_patterns=["LayerNorm"])
        engine = _make_engine(elements=[el], spec=spec)
        result = engine.map()
        exact = [t for t in result.targets if t.context.get("match_tier") == "exact"]
        assert len(exact) == 0


# =========================================================================
# MappingEngine._find_target_locations — Tier 2: Fuzzy matches
# =========================================================================


class TestFindTargetLocationsFuzzy:
    def test_fuzzy_name_match(self):
        el = _make_element(name="layer_norm_1")
        spec = _make_spec(target_patterns=["LayerNorm"])
        engine = _make_engine(elements=[el], spec=spec)
        result = engine.map()
        fuzzy = [t for t in result.targets if t.context.get("match_tier", "").startswith("fuzzy")]
        assert len(fuzzy) >= 1

    def test_fuzzy_snippet_match(self):
        el = _make_element(name="forward", code_snippet="self.norm = nn.LayerNorm(dim)")
        spec = _make_spec(target_patterns=["LayerNorm"])
        engine = _make_engine(elements=[el], spec=spec)
        result = engine.map()
        snippet = [t for t in result.targets if t.context.get("match_tier") == "fuzzy_snippet"]
        assert len(snippet) >= 1

    def test_dedup_already_exact(self):
        """Exact match should prevent duplicate fuzzy match at same location."""
        el = _make_element(name="LayerNorm", file="model.py", line=10)
        engine = _make_engine(elements=[el])
        result = engine.map()
        matches_at_10 = [t for t in result.targets if t.file == "model.py" and t.line == 10]
        assert len(matches_at_10) == 1


# =========================================================================
# MappingEngine._find_target_locations — Tier 3: Import matches
# =========================================================================


class TestFindTargetLocationsImport:
    def test_import_match(self):
        imp = _make_import(module="torch.nn", names=["LayerNorm"], line=2)
        spec = _make_spec(target_patterns=["LayerNorm"])
        engine = _make_engine(imports=[imp], spec=spec)
        result = engine.map()
        imp_matches = [t for t in result.targets if t.context.get("match_tier") == "import"]
        assert len(imp_matches) >= 1
        assert imp_matches[0].line == 2


# =========================================================================
# MappingEngine._text_scan_for_patterns
# =========================================================================


class TestTextScanForPatterns:
    def test_scan_finds_usage_pattern(self, tmp_path):
        py_file = tmp_path / "model.py"
        py_file.write_text("x = self.wpe(positions)\ny = self.ln_1(x)\n")
        el = _make_element(name="GPT", file="model.py", line=1)
        spec = _make_spec(target_patterns=["self.wpe"], replacement="rotary_emb")
        engine = _make_engine(
            elements=[el],
            root_path=str(tmp_path),
            spec=spec,
        )
        result = engine.map()
        text_scan = [t for t in result.targets if t.context.get("match_tier") == "text_scan"]
        assert len(text_scan) >= 1

    def test_scan_no_root_path(self):
        spec = _make_spec(target_patterns=["self.wpe"])
        engine = _make_engine(spec=spec)
        # No root_path => text scan returns empty
        targets = engine._text_scan_for_patterns(["self.wpe"], "rotary_emb", set())
        assert targets == []

    def test_scan_nonexistent_root(self, tmp_path):
        engine = _make_engine(root_path=str(tmp_path / "nonexistent"), spec=_make_spec())
        targets = engine._text_scan_for_patterns(["self.wpe"], "rotary_emb", set())
        assert targets == []


# =========================================================================
# MappingEngine._search_legacy_architecture
# =========================================================================


class TestSearchLegacyArchitecture:
    def test_string_component_match(self):
        arch = {
            "models": [
                {
                    "file": "model.py",
                    "components": {"normalization": "LayerNorm"},
                }
            ]
        }
        engine = _make_engine(architecture=arch)
        result = engine.map()
        legacy = [t for t in result.targets if t.context.get("match_tier") == "legacy_architecture"]
        assert len(legacy) >= 1
        assert legacy[0].current_code == "LayerNorm"

    def test_list_component_match(self):
        arch = {
            "models": [
                {
                    "file": "model.py",
                    "components": {"norms": ["LayerNorm", "GroupNorm"]},
                }
            ]
        }
        engine = _make_engine(architecture=arch)
        result = engine.map()
        legacy = [t for t in result.targets if t.context.get("match_tier") == "legacy_architecture"]
        assert len(legacy) >= 1

    def test_no_match_in_legacy(self):
        arch = {
            "models": [
                {
                    "file": "model.py",
                    "components": {"activation": "GELU"},
                }
            ]
        }
        spec = _make_spec(target_patterns=["LayerNorm"])
        engine = _make_engine(architecture=arch, spec=spec)
        legacy = engine._search_legacy_architecture(["LayerNorm"], "RMSNorm")
        assert len(legacy) == 0

    def test_empty_architecture(self):
        engine = _make_engine()
        legacy = engine._search_legacy_architecture(["LayerNorm"], "RMSNorm")
        assert legacy == []


# =========================================================================
# MappingEngine._llm_semantic_match
# =========================================================================


class TestLLMSemanticMatch:
    def test_llm_match_success(self):
        el = _make_element(name="CustomNorm", file="model.py", line=5)
        llm = SimpleNamespace(
            analyse_code=lambda prompt, **kw: SimpleNamespace(
                raw_response='[{"index": 0, "reason": "normalisation component"}]'
            )
        )
        engine = _make_engine(elements=[el], llm_assistant=llm)
        targets = engine._llm_semantic_match([el], ["LayerNorm"], "RMSNorm")
        assert len(targets) == 1
        assert targets[0].context["match_tier"] == "llm_semantic"
        assert targets[0].context["llm_reason"] == "normalisation component"

    def test_llm_match_empty_elements(self):
        llm = SimpleNamespace(analyse_code=lambda p, **kw: None)
        engine = _make_engine(llm_assistant=llm)
        targets = engine._llm_semantic_match([], ["LayerNorm"], "RMSNorm")
        assert targets == []

    def test_llm_match_invalid_index(self):
        el = _make_element(name="CustomNorm")
        llm = SimpleNamespace(
            analyse_code=lambda prompt, **kw: SimpleNamespace(
                raw_response='[{"index": 99, "reason": "bad index"}]'
            )
        )
        engine = _make_engine(elements=[el], llm_assistant=llm)
        targets = engine._llm_semantic_match([el], ["LayerNorm"], "RMSNorm")
        assert targets == []

    def test_llm_match_exception_handled(self):
        el = _make_element(name="CustomNorm")
        llm = SimpleNamespace(
            analyse_code=lambda prompt, **kw: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        engine = _make_engine(elements=[el], llm_assistant=llm)
        targets = engine._llm_semantic_match([el], ["LayerNorm"], "RMSNorm")
        assert targets == []

    def test_llm_not_used_when_targets_found(self):
        """LLM tier is only used when no targets found from prior tiers."""
        el = _make_element(name="LayerNorm")
        call_count = [0]

        def fake_analyse(prompt, **kw):
            call_count[0] += 1
            return SimpleNamespace(raw_response="[]")

        llm = SimpleNamespace(analyse_code=fake_analyse)
        engine = _make_engine(elements=[el], llm_assistant=llm)
        result = engine.map()
        # Exact match found => LLM should NOT be called
        assert call_count[0] == 0
        assert len(result.targets) >= 1


# =========================================================================
# MappingEngine._parse_llm_matches
# =========================================================================


class TestParseLLMMatches:
    def test_empty_string(self):
        assert MappingEngine._parse_llm_matches("") == []

    def test_fenced_json(self):
        raw = '```json\n[{"index": 0, "reason": "match"}]\n```'
        result = MappingEngine._parse_llm_matches(raw)
        assert len(result) == 1
        assert result[0]["index"] == 0

    def test_bare_json_array(self):
        raw = 'Some text before [{"index": 1, "reason": "test"}] and after'
        result = MappingEngine._parse_llm_matches(raw)
        assert len(result) == 1
        assert result[0]["index"] == 1

    def test_invalid_json(self):
        raw = "```json\n{not valid json}\n```"
        result = MappingEngine._parse_llm_matches(raw)
        assert result == []

    def test_no_json_at_all(self):
        raw = "There are no matches found in the codebase."
        result = MappingEngine._parse_llm_matches(raw)
        assert result == []

    def test_empty_array(self):
        raw = "[]"
        result = MappingEngine._parse_llm_matches(raw)
        assert result == []


# =========================================================================
# _validate_compatibility
# =========================================================================


class TestValidateCompatibility:
    def test_same_name_replacement_error(self):
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="LayerNorm",
            replacement_required=True,
            context={"original": "LayerNorm", "replacement": "LayerNorm"},
        )
        engine = _make_engine()
        result = engine._validate_compatibility([ip])
        assert result.passed is False
        assert any(i.severity == "error" for i in result.issues)

    def test_test_file_warning(self):
        ip = InsertionPoint(
            file="test_model.py",
            line=10,
            current_code="LayerNorm",
            replacement_required=True,
            context={"original": "LayerNorm", "replacement": "RMSNorm"},
        )
        engine = _make_engine()
        result = engine._validate_compatibility([ip])
        assert result.passed is True  # warnings don't block
        assert any(i.severity == "warning" for i in result.issues)

    def test_tests_subdir_warning(self):
        ip = InsertionPoint(
            file="src/tests/model.py",
            line=10,
            current_code="LayerNorm",
            replacement_required=True,
            context={"original": "LayerNorm", "replacement": "RMSNorm"},
        )
        engine = _make_engine()
        result = engine._validate_compatibility([ip])
        assert any(i.severity == "warning" for i in result.issues)

    def test_clean_targets_pass(self):
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="LayerNorm",
            replacement_required=True,
            context={"original": "LayerNorm", "replacement": "RMSNorm"},
        )
        engine = _make_engine()
        result = engine._validate_compatibility([ip])
        assert result.passed is True
        assert len(result.issues) == 0

    def test_empty_targets(self):
        engine = _make_engine()
        result = engine._validate_compatibility([])
        assert result.passed is True
        assert len(result.issues) == 0


# =========================================================================
# _select_strategy
# =========================================================================


class TestSelectStrategy:
    def test_no_targets_returns_none(self):
        engine = _make_engine()
        validation = ValidationResult(passed=True)
        assert engine._select_strategy([], validation) == "none"

    def test_validation_failed_returns_manual_review(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="x",
            replacement_required=True,
            context={},
        )
        validation = ValidationResult(passed=False)
        assert engine._select_strategy([ip], validation) == "manual_review"

    def test_normal_returns_change_type(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="x",
            replacement_required=True,
            context={},
        )
        validation = ValidationResult(passed=True)
        assert engine._select_strategy([ip], validation) == "replace"

    def test_custom_change_type(self):
        spec = _make_spec(change_type="augment")
        engine = _make_engine(spec=spec)
        ip = InsertionPoint(
            file="model.py",
            line=10,
            current_code="x",
            replacement_required=True,
            context={},
        )
        validation = ValidationResult(passed=True)
        assert engine._select_strategy([ip], validation) == "augment"


# =========================================================================
# _calculate_confidence
# =========================================================================


class TestCalculateConfidence:
    def test_confidence_breakdown_total_matches_calculate_confidence(self):
        engine = _make_engine()
        ips = [
            InsertionPoint(
                file="f.py",
                line=1,
                current_code="x",
                replacement_required=True,
                context={"match_tier": "exact"},
            )
        ]
        validation = ValidationResult(passed=True)
        breakdown = engine._calculate_confidence_breakdown(validation, ips)
        score = engine._calculate_confidence(validation, ips)
        assert breakdown
        assert breakdown["total"] == score

    def test_no_targets_base_plus_validation(self):
        engine = _make_engine()
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [])
        assert score == 40  # base 30 + 10 validation

    def test_exact_match_boost(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="f.py",
            line=1,
            current_code="x",
            replacement_required=True,
            context={"match_tier": "exact"},
        )
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [ip])
        # base 30 + 20 (has targets) + 10 (1 exact) + 10 (validation) = 70
        assert score == 70

    def test_fuzzy_match_boost(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="f.py",
            line=1,
            current_code="x",
            replacement_required=True,
            context={"match_tier": "fuzzy_name"},
        )
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [ip])
        # base 30 + 20 + 10 (fuzzy) + 10 (validation) = 70
        assert score == 70

    def test_import_match_boost(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="f.py",
            line=1,
            current_code="x",
            replacement_required=True,
            context={"match_tier": "import"},
        )
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [ip])
        # base 30 + 20 + 5 (import) + 10 (validation) = 65
        assert score == 65

    def test_llm_match_boost(self):
        engine = _make_engine()
        ip = InsertionPoint(
            file="f.py",
            line=1,
            current_code="x",
            replacement_required=True,
            context={"match_tier": "llm_semantic"},
        )
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [ip])
        # base 30 + 20 + 5 (llm) + 10 (validation) = 65
        assert score == 65

    def test_code_template_boost(self):
        spec = _make_spec(code_template="class RMSNorm: pass")
        engine = _make_engine(spec=spec)
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, [])
        # base 30 + 10 (validation) + 10 (template) = 50
        assert score == 50

    def test_validation_failure_penalty(self):
        engine = _make_engine()
        issues = [
            CompatibilityIssue(location="f.py:1", issue="err", severity="error"),
            CompatibilityIssue(location="f.py:2", issue="err2", severity="error"),
        ]
        validation = ValidationResult(passed=False, issues=issues)
        score = engine._calculate_confidence(validation, [])
        # base 30 - 20 (2 errors) = 10
        assert score == 10

    def test_exact_match_cap_at_30(self):
        engine = _make_engine()
        ips = [
            InsertionPoint(
                file="f.py",
                line=i,
                current_code="x",
                replacement_required=True,
                context={"match_tier": "exact"},
            )
            for i in range(5)
        ]
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, ips)
        # base 30 + 20 + 30 (cap) + 10 (validation) = 90
        assert score == 90

    def test_confidence_capped_at_100(self):
        spec = _make_spec(code_template="template")
        engine = _make_engine(spec=spec)
        ips = [
            InsertionPoint(
                file="f.py",
                line=i,
                current_code="x",
                replacement_required=True,
                context={"match_tier": "exact"},
            )
            for i in range(5)
        ]
        validation = ValidationResult(passed=True)
        score = engine._calculate_confidence(validation, ips)
        # base 30 + 20 + 30 + 10 + 10 = 100
        assert score == 100

    def test_confidence_floored_at_0(self):
        engine = _make_engine()
        issues = [
            CompatibilityIssue(location=f"f.py:{i}", issue="err", severity="error")
            for i in range(10)
        ]
        validation = ValidationResult(passed=False, issues=issues)
        score = engine._calculate_confidence(validation, [])
        assert score == 0


# =========================================================================
# MappingEngine.map() — full orchestration
# =========================================================================


class TestMapOrchestration:
    def test_map_returns_mapping_result(self):
        el = _make_element(name="LayerNorm")
        engine = _make_engine(elements=[el])
        result = engine.map()
        assert isinstance(result, MappingResult)
        assert len(result.targets) >= 1
        assert result.strategy == "replace"
        assert 0 <= result.confidence <= 100
        assert result.research_spec is not None

    def test_map_no_elements_no_targets(self):
        spec = _make_spec(target_patterns=["NonExistent"])
        engine = _make_engine(spec=spec)
        result = engine.map()
        assert len(result.targets) == 0
        assert result.strategy == "none"

    def test_map_empty_target_patterns(self):
        spec = _make_spec(target_patterns=[])
        engine = _make_engine(spec=spec)
        result = engine.map()
        assert len(result.targets) == 0

    def test_multiple_elements_dedup(self):
        """Same file+line should only appear once."""
        el1 = _make_element(name="LayerNorm", file="model.py", line=10)
        el2 = _make_element(name="LayerNorm", file="model.py", line=10)
        engine = _make_engine(elements=[el1, el2])
        result = engine.map()
        lines = [(t.file, t.line) for t in result.targets]
        assert len(set(lines)) == len(lines)

    def test_map_with_dict_elements(self):
        """Elements can be dicts, not just SimpleNamespace objects."""
        el = {"name": "LayerNorm", "type": "class", "file": "model.py", "line": 10}
        engine = _make_engine(elements=[el])
        result = engine.map()
        assert len(result.targets) >= 1

    def test_map_returns_confidence_breakdown_with_matching_total(self):
        el = _make_element(name="LayerNorm")
        engine = _make_engine(elements=[el])
        result = engine.map()
        assert result.confidence_breakdown
        assert result.confidence_breakdown["total"] == result.confidence


# =========================================================================
# analyze_repo_for_pattern
# =========================================================================


class TestAnalyzeRepoForPattern:
    def test_finds_pattern_in_file(self, tmp_path):
        py_file = tmp_path / "model.py"
        py_file.write_text("class LayerNorm:\n    pass\n")
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert len(results) >= 1
        assert results[0]["file"] == "model.py"
        assert results[0]["line"] == 1
        assert "LayerNorm" in results[0]["content"]

    def test_case_insensitive(self, tmp_path):
        py_file = tmp_path / "model.py"
        py_file.write_text("class layernorm:\n    pass\n")
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert len(results) >= 1

    def test_no_match(self, tmp_path):
        py_file = tmp_path / "model.py"
        py_file.write_text("class Dropout:\n    pass\n")
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert len(results) == 0

    def test_skips_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.py").write_text("LayerNorm")
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert len(results) == 0

    def test_empty_dir(self, tmp_path):
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert results == []

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("LayerNorm\n")
        (tmp_path / "b.py").write_text("LayerNorm\n")
        results = analyze_repo_for_pattern(str(tmp_path), "LayerNorm")
        assert len(results) == 2
