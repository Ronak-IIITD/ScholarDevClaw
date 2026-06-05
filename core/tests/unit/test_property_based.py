"""Property-based tests for scholardevclaw.utils.validation and error_codes.

These tests use Hypothesis to verify invariants that should hold for any
input drawn from the relevant domain, not just for the specific examples
checked by the regular unit tests.
"""

from __future__ import annotations

import json
import re

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scholardevclaw.utils.error_codes import (
    ERROR_CODE_MAP,
    AppException,
    ErrorCategory,
    ErrorCode,
    ErrorCodes,
    ErrorSeverity,
    create_error,
    get_error,
)
from scholardevclaw.utils.validation import (
    CompositeValidator,
    EnumValidator,
    InputValidator,
    RangeValidator,
    RequiredValidator,
    StringValidator,
    validate_repo_path,
    validate_spec_name,
)

# Strategies --------------------------------------------------------------

# Strings that are valid spec names (ASCII alphanumeric + - and _).
# The spec_name regex is ASCII-only, so the strategy must be too.
spec_name_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "Nd"),
        whitelist_characters="-_",
        max_codepoint=0x7F,  # ASCII only
    ),
    min_size=1,
    max_size=100,
)

# Strings that are NOT valid spec names (contain at least one illegal char).
invalid_spec_name_chars = st.characters(
    blacklist_categories=("L", "Nd", "Cs", "Cc"),
    blacklist_characters="-_",
)

non_empty_text = st.text(min_size=1)
short_text = st.text(max_size=200)


# Tests for RequiredValidator --------------------------------------------


class TestRequiredValidator:
    @given(value=st.none() | st.just("") | st.just("   ") | st.just("\t\n"))
    def test_none_or_whitespace_is_invalid(self, value):
        result = RequiredValidator().validate(value)
        assert not result.valid
        assert result.errors

    @given(value=st.text(min_size=1).filter(lambda s: s.strip()))
    def test_nonempty_string_is_valid(self, value):
        result = RequiredValidator().validate(value)
        assert result.valid
        assert result.errors == []
        assert result.sanitized_value == value

    @given(value=st.integers() | st.floats() | st.lists(st.integers()) | st.booleans())
    def test_other_types_pass_through(self, value):
        # RequiredValidator only treats None and empty/whitespace strings as
        # missing. Anything else is considered "present".
        result = RequiredValidator().validate(value)
        assert result.valid
        assert result.sanitized_value == value


# Tests for StringValidator ----------------------------------------------


class TestStringValidator:
    @given(
        s=st.text(min_size=5, max_size=50),
        min_length=st.integers(min_value=1, max_value=3),
        max_length=st.integers(min_value=51, max_value=200),
    )
    def test_string_in_range_is_valid(self, s, min_length, max_length):
        v = StringValidator(min_length=min_length, max_length=max_length)
        result = v.validate(s)
        assert result.valid
        assert result.errors == []

    @given(s=st.text(min_size=11, max_size=50), max_length=st.integers(min_value=1, max_value=10))
    def test_string_above_max_is_invalid(self, s, max_length):
        v = StringValidator(max_length=max_length)
        result = v.validate(s)
        assert not result.valid
        assert any("at most" in e for e in result.errors)

    @given(
        s=st.text(min_size=0, max_size=5),
        min_length=st.integers(min_value=6, max_value=20),
    )
    def test_string_below_min_is_invalid(self, s, min_length):
        v = StringValidator(min_length=min_length)
        result = v.validate(s)
        assert not result.valid
        assert any("at least" in e for e in result.errors)

    @given(s=non_empty_text)
    def test_pattern_matching_digits(self, s):
        pattern = r"^[a-zA-Z0-9_-]+$"
        v = StringValidator(pattern=pattern)
        result = v.validate(s)
        # Validation result matches direct regex match.
        assert result.valid == bool(re.match(pattern, s))

    @given(s=st.text())
    def test_allowed_chars_invariant(self, s):
        allowed = "abc123"
        v = StringValidator(allowed_chars=allowed)
        result = v.validate(s)
        if result.valid:
            # If the validator accepted the string, every char must be allowed.
            assert set(s).issubset(set(allowed))
        else:
            # If rejected, it must be because of invalid chars.
            assert any("Invalid characters" in e for e in result.errors)


# Tests for RangeValidator -----------------------------------------------


class TestRangeValidator:
    @given(
        value=st.integers(min_value=0, max_value=100),
        min_value=st.integers(min_value=-10, max_value=0),
        max_value=st.integers(min_value=101, max_value=200),
    )
    def test_value_in_range_is_valid(self, value, min_value, max_value):
        v = RangeValidator(min_value=min_value, max_value=max_value)
        result = v.validate(value)
        assert result.valid
        assert result.sanitized_value == value

    @given(
        min_value=st.integers(min_value=10, max_value=20),
        max_value=st.integers(min_value=21, max_value=50),
    )
    def test_value_below_min_is_invalid(self, min_value, max_value):
        v = RangeValidator(min_value=min_value, max_value=max_value)
        result = v.validate(min_value - 1)
        assert not result.valid

    @given(
        min_value=st.integers(min_value=0, max_value=10),
        max_value=st.integers(min_value=11, max_value=20),
    )
    def test_value_above_max_is_invalid(self, min_value, max_value):
        v = RangeValidator(min_value=min_value, max_value=max_value)
        result = v.validate(max_value + 1)
        assert not result.valid

    @given(
        value=st.integers(min_value=0, max_value=100),
        min_value=st.integers(min_value=0, max_value=100),
        max_value=st.integers(min_value=0, max_value=100),
    )
    def test_integer_only_rejects_floats(self, value, min_value, max_value):
        # Make sure the bounds are valid (min <= value <= max)
        if min_value > value or max_value < value:
            return
        v = RangeValidator(min_value=min_value, max_value=max_value, integer_only=True)
        # A whole-number float should be rejected because boolean integer_only
        # uses isinstance(value, int).
        result = v.validate(float(value))
        assert not result.valid
        assert any("integer" in e for e in result.errors)


# Tests for EnumValidator ------------------------------------------------


class TestEnumValidator:
    @given(allowed=st.lists(st.integers(), min_size=1, max_size=20), value=st.integers())
    def test_value_must_be_in_allowed(self, allowed, value):
        v = EnumValidator(allowed)
        result = v.validate(value)
        assert result.valid == (value in allowed)

    @given(
        allowed=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10),
        value=st.text(min_size=1, max_size=10),
    )
    def test_case_insensitive_matching(self, allowed, value):
        v = EnumValidator(allowed, case_insensitive=True)
        result = v.validate(value)
        expected = value.lower() in [a.lower() for a in allowed]
        assert result.valid == expected


# Tests for CompositeValidator -------------------------------------------


class TestCompositeValidator:
    @given(
        s=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "Nd"),
                whitelist_characters="-_",
                max_codepoint=0x7F,
            ),
            min_size=1,
            max_size=50,
        )
    )
    def test_required_and_string_chained(self, s):
        v = CompositeValidator(
            [
                RequiredValidator(),
                StringValidator(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$"),
            ]
        )
        result = v.validate(s)
        assert result.valid

    @given(
        s=st.text(min_size=200, max_size=500),
    )
    def test_too_long_string_fails_chain(self, s):
        v = CompositeValidator(
            [
                RequiredValidator(),
                StringValidator(max_length=100),
            ]
        )
        result = v.validate(s)
        assert not result.valid
        # Error from the second validator is propagated.
        assert result.errors

    def test_empty_input_short_circuits(self):
        v = CompositeValidator(
            [
                RequiredValidator("required"),
                StringValidator(min_length=5),
            ],
            stop_on_first_error=True,
        )
        result = v.validate("")
        assert not result.valid
        # Only the first error should be reported.
        assert result.errors == ["required"]


# Tests for InputValidator -----------------------------------------------


class TestInputValidator:
    @given(s=spec_name_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_spec_name_round_trip(self, s):
        validator = InputValidator().field(
            "name", StringValidator(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
        )
        result = validator.validate({"name": s})
        assert result.valid
        assert result.sanitized_value == {"name": s}

    def test_unknown_field_emits_warning(self):
        validator = InputValidator().field("name", RequiredValidator())
        result = validator.validate({"name": "ok", "extra": 1})
        assert result.valid
        assert any("Unknown field" in w for w in result.warnings)


# Tests for the public helpers -------------------------------------------


class TestPublicValidators:
    @given(s=spec_name_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_spec_name_accepts_well_formed(self, s):
        result = validate_spec_name(s)
        assert result.valid
        assert result.sanitized_value == s

    @given(s=st.text(min_size=101, max_size=300))
    def test_validate_spec_name_rejects_too_long(self, s):
        result = validate_spec_name(s)
        assert not result.valid

    def test_validate_repo_path_requires_existing_dir(self, tmp_path):
        existing = str(tmp_path)
        missing = str(tmp_path / "does-not-exist")
        assert validate_repo_path(existing).valid
        assert not validate_repo_path(missing).valid

    def test_validate_repo_path_empty_is_invalid(self):
        result = validate_repo_path("")
        assert not result.valid

    def test_validate_spec_name_empty_is_invalid(self):
        # validate_spec_name returns a result rather than raising; verify
        # the empty case is reported as invalid.
        result = validate_spec_name("")
        assert not result.valid
        assert result.errors


# Tests for error codes ---------------------------------------------------


class TestErrorCode:
    @given(
        code=st.text(min_size=3, max_size=10),
        message=st.text(min_size=1, max_size=200),
    )
    def test_to_dict_is_json_round_trip(self, code, message):
        ec = ErrorCode(
            code=code,
            category=ErrorCategory.VALIDATION,
            message=message,
            severity=ErrorSeverity.ERROR,
            http_status=400,
        )
        d = ec.to_dict()
        # Must be JSON-serializable (i.e. no datetime, no custom objects).
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded == d

    def test_to_dict_preserves_all_fields(self):
        ec = ErrorCodes.VAL001
        d = ec.to_dict()
        assert d["code"] == "VAL001"
        assert d["category"] == ErrorCategory.VALIDATION.value
        assert d["severity"] == ErrorSeverity.WARNING.value
        assert d["http_status"] == 400
        assert d["message"]

    def test_str_includes_code(self):
        ec = ErrorCodes.NET001
        s = str(ec)
        assert "NET001" in s
        assert ec.message in s


class TestErrorCodeMap:
    def test_all_attributes_have_string_codes(self):
        # Every ErrorCode defined as a class attribute should appear in the
        # error code map (the map is the source of truth for lookup).
        seen_codes = set()
        for name in dir(ErrorCodes):
            value = getattr(ErrorCodes, name)
            if isinstance(value, ErrorCode):
                seen_codes.add(value.code)
        assert seen_codes == set(ERROR_CODE_MAP.keys())

    @given(code=st.sampled_from(sorted(ERROR_CODE_MAP.keys())))
    def test_get_error_returns_mapped_code(self, code):
        ec = get_error(code)
        assert ec is not None
        assert ec.code == code

    @given(code=st.text(min_size=1, max_size=20).filter(lambda c: c not in ERROR_CODE_MAP))
    def test_get_error_returns_none_for_unknown(self, code):
        assert get_error(code) is None


class TestAppException:
    @given(code=st.sampled_from(sorted(ERROR_CODE_MAP.keys())))
    def test_known_code_preserved(self, code):
        exc = AppException(code)
        assert exc.error_code.code == code

    @given(code=st.text(min_size=1, max_size=20).filter(lambda c: c not in ERROR_CODE_MAP))
    def test_unknown_code_falls_back_to_sys001(self, code):
        exc = AppException(code)
        assert exc.error_code.code == ErrorCodes.SYS001.code

    @given(
        code=st.sampled_from(sorted(ERROR_CODE_MAP.keys())),
        details=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.integers() | st.text(max_size=20),
            max_size=5,
        ),
    )
    def test_to_dict_includes_details(self, code, details):
        exc = AppException(code, details=details)
        d = exc.to_dict()
        for key, value in details.items():
            assert d["details"][key] == value

    def test_cause_included_in_to_dict(self):
        cause = ValueError("boom")
        exc = AppException("VAL001", cause=cause)
        d = exc.to_dict()
        assert d["cause"] == "boom"


class TestCreateError:
    @given(
        code=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Nd"),
                whitelist_characters="_",
            ),
            min_size=3,
            max_size=20,
        ),
        message=st.text(min_size=1, max_size=100),
    )
    def test_create_error_preserves_inputs(self, code, message):
        ec = create_error(
            code=code,
            message=message,
            category=ErrorCategory.VALIDATION,
        )
        assert ec.code == code
        assert ec.message == message
        assert ec.category == ErrorCategory.VALIDATION
