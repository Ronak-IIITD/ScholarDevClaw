"""Tests for utils/validation.py"""

import pytest

from scholardevclaw.utils.validation import (
    CompositeValidator,
    EnumValidator,
    InputValidator,
    PathValidator,
    RangeValidator,
    RequiredValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    Validator,
    validate_output_dir,
    validate_repo_path,
    validate_spec_name,
)


class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.warnings == []
        assert r.sanitized_value is None

    def test_with_fields(self):
        r = ValidationResult(
            valid=False,
            errors=["err1"],
            warnings=["warn1"],
            sanitized_value="val",
        )
        assert r.valid is False
        assert r.errors == ["err1"]
        assert r.warnings == ["warn1"]
        assert r.sanitized_value == "val"


class TestValidationError:
    def test_message_format(self):
        err = ValidationError(["err1", "err2"])
        assert "err1" in str(err)
        assert "err2" in str(err)

    def test_errors_attr(self):
        err = ValidationError(["only one"])
        assert err.errors == ["only one"]


class TestValidator:
    def test_base_raises_not_implemented(self):
        v = Validator()
        with pytest.raises(NotImplementedError):
            v.validate("anything")


class TestRequiredValidator:
    def test_none_value(self):
        v = RequiredValidator()
        result = v.validate(None)
        assert result.valid is False
        assert result.errors == ["This field is required"]

    def test_empty_string(self):
        v = RequiredValidator()
        result = v.validate("")
        assert result.valid is False

    def test_whitespace_string(self):
        v = RequiredValidator()
        result = v.validate("   ")
        assert result.valid is False

    def test_valid_string(self):
        v = RequiredValidator()
        result = v.validate("hello")
        assert result.valid is True
        assert result.sanitized_value == "hello"

    def test_valid_number(self):
        v = RequiredValidator()
        result = v.validate(0)
        assert result.valid is True

    def test_custom_message(self):
        v = RequiredValidator(message="Custom required")
        result = v.validate(None)
        assert result.errors == ["Custom required"]


class TestPathValidator:
    def test_valid_path(self, tmp_path):
        p = tmp_path / "existing_dir"
        p.mkdir()
        v = PathValidator(must_exist=True, must_be_dir=True)
        result = v.validate(str(p))
        assert result.valid is True

    def test_not_existing(self):
        v = PathValidator(must_exist=True)
        result = v.validate("/nonexistent/path/12345")
        assert result.valid is False

    def test_not_a_string(self):
        v = PathValidator()
        result = v.validate(123)
        assert result.valid is False
        assert "must be a string" in result.errors[0]

    def test_absolute_not_allowed(self):
        v = PathValidator(allow_absolute=False, allow_relative=True)
        result = v.validate("/tmp")
        assert result.valid is False

    def test_absolute_not_allowed_blocks_resolved_path(self):
        v = PathValidator(allow_absolute=False, allow_relative=True, must_exist=False)
        result = v.validate("/tmp")
        assert result.valid is False
        assert "Absolute paths" in result.errors[0]

    def test_must_be_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        v = PathValidator(must_exist=True, must_be_file=True)
        result = v.validate(str(f))
        assert result.valid is True

    def test_must_be_file_but_is_dir(self, tmp_path):
        v = PathValidator(must_exist=True, must_be_file=True)
        result = v.validate(str(tmp_path))
        assert result.valid is False

    def test_must_be_dir_but_is_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        v = PathValidator(must_exist=True, must_be_dir=True)
        result = v.validate(str(f))
        assert result.valid is False

    def test_sanitized_is_absolute(self, tmp_path):
        v = PathValidator(must_exist=False, must_be_dir=True)
        result = v.validate(str(tmp_path))
        assert result.sanitized_value is not None


class TestStringValidator:
    def test_valid_string(self):
        v = StringValidator()
        result = v.validate("hello")
        assert result.valid is True

    def test_not_a_string(self):
        v = StringValidator()
        result = v.validate(42)
        assert result.valid is False

    def test_min_length(self):
        v = StringValidator(min_length=3)
        assert v.validate("ab").valid is False
        assert v.validate("abc").valid is True

    def test_max_length(self):
        v = StringValidator(max_length=3)
        assert v.validate("abcd").valid is False
        assert v.validate("abc").valid is True

    def test_pattern(self):
        v = StringValidator(pattern=r"^[a-z]+$")
        assert v.validate("abc").valid is True
        assert v.validate("ABC").valid is False

    def test_allowed_chars(self):
        v = StringValidator(allowed_chars="abc")
        assert v.validate("abc").valid is True
        assert v.validate("xyz").valid is False

    def test_sanitized_stripped(self):
        v = StringValidator()
        result = v.validate("  hello  ")
        assert result.sanitized_value == "hello"

    def test_all_validators_combined(self):
        v = StringValidator(min_length=2, max_length=5, pattern=r"^[a-z]+$")
        assert v.validate("ab").valid is True
        assert v.validate("abcdef").valid is False
        assert v.validate("A").valid is False


class TestEnumValidator:
    def test_valid_value(self):
        v = EnumValidator(allowed=["a", "b", "c"])
        assert v.validate("a").valid is True
        assert v.validate("d").valid is False

    def test_case_insensitive(self):
        v = EnumValidator(allowed=["Red", "Green"], case_insensitive=True)
        assert v.validate("red").valid is True
        assert v.validate("GREEN").valid is True
        assert v.validate("blue").valid is False

    def test_error_message(self):
        v = EnumValidator(allowed=["x", "y"])
        result = v.validate("z")
        assert "x" in result.errors[0]
        assert "y" in result.errors[0]

    def test_sanitized_value(self):
        v = EnumValidator(allowed=["val"])
        result = v.validate("val")
        assert result.sanitized_value == "val"


class TestRangeValidator:
    def test_valid_integer(self):
        v = RangeValidator(min_value=0, max_value=10)
        assert v.validate(5).valid is True

    def test_below_min(self):
        v = RangeValidator(min_value=5)
        assert v.validate(3).valid is False

    def test_above_max(self):
        v = RangeValidator(max_value=10)
        assert v.validate(15).valid is False

    def test_not_a_number(self):
        v = RangeValidator()
        result = v.validate("not a number")
        assert result.valid is False

    def test_integer_only(self):
        v = RangeValidator(integer_only=True)
        assert v.validate(5).valid is True
        assert v.validate(5.5).valid is False

    def test_sanitized(self):
        v = RangeValidator(min_value=0)
        result = v.validate(42)
        assert result.sanitized_value == 42


class TestCompositeValidator:
    def test_all_pass(self):
        cv = CompositeValidator([RequiredValidator(), StringValidator(min_length=2)])
        result = cv.validate("hello")
        assert result.valid is True

    def test_stop_on_first_error(self):
        cv = CompositeValidator(
            [RequiredValidator(), StringValidator(min_length=100)],
            stop_on_first_error=True,
        )
        result = cv.validate(None)
        assert result.valid is False
        assert len(result.errors) == 1

    def test_continue_on_error(self):
        first = RequiredValidator()
        second = StringValidator(min_length=100)
        cv = CompositeValidator([first, second], stop_on_first_error=False)
        result = cv.validate("")
        assert result.valid is False
        assert len(result.errors) >= 1


class TestInputValidator:
    def test_valid_data(self):
        iv = InputValidator()
        iv.field("name", RequiredValidator()).field("age", RangeValidator(min_value=0))
        result = iv.validate({"name": "alice", "age": 30})
        assert result.valid is True

    def test_invalid_data(self):
        iv = InputValidator()
        iv.field("name", RequiredValidator())
        result = iv.validate({"name": ""})
        assert result.valid is False

    def test_unknown_field_warning(self):
        iv = InputValidator()
        iv.field("known", RequiredValidator())
        result = iv.validate({"known": "val", "unknown": "x"})
        assert "unknown" in result.warnings[0]

    def test_missing_field(self):
        iv = InputValidator()
        iv.field("required_field", RequiredValidator())
        result = iv.validate({})
        assert result.valid is False


class TestValidateRepoPath:
    def test_valid_path(self, tmp_path):
        result = validate_repo_path(str(tmp_path))
        assert result.valid is True

    def test_none_path(self):
        result = validate_repo_path(None)
        assert result.valid is False

    def test_nonexistent_path(self):
        result = validate_repo_path("/nonexistent/path/for/test")
        assert result.valid is False


class TestValidateSpecName:
    def test_valid_name(self):
        result = validate_spec_name("rmsnorm")
        assert result.valid is True

    def test_invalid_characters(self):
        result = validate_spec_name("hello world")
        assert result.valid is False

    def test_empty_name(self):
        result = validate_spec_name("")
        assert result.valid is False

    def test_name_with_hyphen(self):
        result = validate_spec_name("my-spec")
        assert result.valid is True

    def test_name_with_underscore(self):
        result = validate_spec_name("my_spec")
        assert result.valid is True


class TestValidateOutputDir:
    def test_valid_dir(self, tmp_path):
        result = validate_output_dir(str(tmp_path))
        assert result.valid is True

    def test_none(self):
        result = validate_output_dir(None)
        assert result.valid is True
        assert result.sanitized_value is None

    def test_nonexistent_allowed(self, tmp_path):
        p = tmp_path / "new_dir"
        result = validate_output_dir(str(p))
        assert result.valid is True
