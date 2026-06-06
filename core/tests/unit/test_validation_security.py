"""Tests for the validation runner security module.

Covers:
    - SecurityCheckResult dataclass behaviour
    - Regex-based detection of destructive operations and sandbox escape attempts
    - AST-based detection of dangerous calls / imports / attribute access
    - Categorized security check (destructive vs escape)
    - Backward-compatible :func:`_comprehensive_security_check`
    - Integration with the validation runner's destructive / escape helpers
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholardevclaw.validation.runner import (  # noqa: E402
    _is_sandbox_escape,
    _is_script_destructive,
)
from scholardevclaw.validation.security import (  # noqa: E402
    SecurityCheckResult,
    _categorized_security_check,
    _comprehensive_security_check,
    _is_sandbox_escape_ast,
    _is_script_destructive_ast,
)

# =========================================================================
# SecurityCheckResult
# =========================================================================


class TestSecurityCheckResult:
    def test_default_is_safe(self):
        result = SecurityCheckResult()
        assert result.is_safe is True
        assert result.destructive_issues == []
        assert result.escape_issues == []
        assert result.all_issues == []

    def test_destructive_makes_unsafe(self):
        result = SecurityCheckResult(destructive_issues=["bad"])
        assert result.is_safe is False

    def test_escape_makes_unsafe(self):
        result = SecurityCheckResult(escape_issues=["bad"])
        assert result.is_safe is False

    def test_all_issues_combines(self):
        result = SecurityCheckResult(
            destructive_issues=["a", "b"],
            escape_issues=["c"],
        )
        assert result.all_issues == ["a", "b", "c"]
        assert result.is_safe is False


# =========================================================================
# _is_script_destructive_ast
# =========================================================================


class TestIsScriptDestructiveAst:
    def test_clean_script(self):
        is_destructive, reasons = _is_script_destructive_ast(
            "import json\nprint(json.dumps({'ok': 1}))\n"
        )
        assert is_destructive is False
        assert reasons == []

    def test_os_system_call(self):
        is_destructive, reasons = _is_script_destructive_ast("import os\nos.system('ls')\n")
        assert is_destructive is True
        assert any("Dangerous attribute access" in r for r in reasons)

    def test_subprocess_run(self):
        is_destructive, reasons = _is_script_destructive_ast(
            "import subprocess\nsubprocess.run(['ls'])\n"
        )
        assert is_destructive is True
        assert any("Dangerous attribute access" in r for r in reasons)

    def test_import_os(self):
        is_destructive, reasons = _is_script_destructive_ast("import os\n")
        assert is_destructive is True
        assert any("Dangerous import" in r for r in reasons)

    def test_import_from_subprocess(self):
        is_destructive, reasons = _is_script_destructive_ast("from subprocess import run\n")
        assert is_destructive is True
        assert any("Dangerous import from" in r for r in reasons)

    def test_syntax_error_blocks(self):
        is_destructive, reasons = _is_script_destructive_ast("def broken(:\n")
        assert is_destructive is True
        assert "Syntax error" in reasons[0]

    def test_safe_import(self):
        is_destructive, reasons = _is_script_destructive_ast(
            "import json\nimport math\nimport pathlib\n"
        )
        assert is_destructive is False
        assert reasons == []

    def test_safe_calls(self):
        is_destructive, reasons = _is_script_destructive_ast(
            "import json\ndata = json.loads('{}')\nprint(len(data))\n"
        )
        assert is_destructive is False
        assert reasons == []


# =========================================================================
# _is_sandbox_escape_ast
# =========================================================================


class TestIsSandboxEscapeAst:
    def test_clean_script(self):
        is_escape, reasons = _is_sandbox_escape_ast("x = 1 + 2\n")
        assert is_escape is False
        assert reasons == []

    def test_eval_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("eval('1+1')\n")
        assert is_escape is True
        assert any("Code execution function" in r for r in reasons)

    def test_exec_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("exec('pass')\n")
        assert is_escape is True
        assert any("Code execution function" in r for r in reasons)

    def test_compile_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("compile('1', '<x>', 'eval')\n")
        assert is_escape is True
        assert any("Code execution function" in r for r in reasons)

    def test_globals_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("globals()['__builtins__']\n")
        assert is_escape is True
        assert any("Dangerous builtin call" in r for r in reasons)

    def test_locals_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("locals()\n")
        assert is_escape is True

    def test_vars_call(self):
        is_escape, reasons = _is_sandbox_escape_ast("vars()\n")
        assert is_escape is True

    def test_dunder_attribute(self):
        is_escape, reasons = _is_sandbox_escape_ast("x.__class__\n")
        assert is_escape is True
        assert any("Dangerous dunder attribute" in r for r in reasons)

    def test_dunder_subclasses(self):
        is_escape, reasons = _is_sandbox_escape_ast("x.__subclasses__()\n")
        assert is_escape is True
        assert any("Dangerous dunder attribute" in r for r in reasons)

    def test_dunder_globals(self):
        is_escape, reasons = _is_sandbox_escape_ast("fn.__globals__\n")
        assert is_escape is True

    def test_direct_builtins_name(self):
        is_escape, reasons = _is_sandbox_escape_ast("x = __builtins__\n")
        assert is_escape is True
        assert any("Direct __builtins__" in r for r in reasons)

    def test_syntax_error_blocks(self):
        is_escape, reasons = _is_sandbox_escape_ast("def :\n")
        assert is_escape is True
        assert "Syntax error" in reasons[0]


# =========================================================================
# _categorized_security_check
# =========================================================================


class TestCategorizedSecurityCheck:
    def test_clean_script(self):
        result = _categorized_security_check("import json\nprint(json.dumps({}))\n")
        assert result.is_safe is True
        assert result.destructive_issues == []
        assert result.escape_issues == []

    def test_destructive_only(self):
        result = _categorized_security_check("import os\nos.remove('x')\n")
        assert result.destructive_issues
        assert result.escape_issues == []
        assert result.is_safe is False

    def test_escape_only(self):
        result = _categorized_security_check("eval('1+1')\n")
        assert result.escape_issues
        assert result.is_safe is False

    def test_combined(self):
        result = _categorized_security_check("import os\nos.remove('x')\neval('1+1')\n")
        assert result.destructive_issues
        assert result.escape_issues
        assert result.is_safe is False

    def test_syntax_error_is_unsafe(self):
        result = _categorized_security_check("def :\n")
        assert result.is_safe is False

    def test_regex_destructive_rm_rf(self):
        result = _categorized_security_check("import os; os.system('rm -rf /tmp/x')\n")
        assert result.destructive_issues
        assert result.is_safe is False

    def test_regex_escape_import(self):
        result = _categorized_security_check("__import__('os')\n")
        assert result.escape_issues
        assert result.is_safe is False

    def test_regex_destructive_subprocess_run(self):
        result = _categorized_security_check("import subprocess\nsubprocess.run(['ls'])\n")
        # subprocess.run is in BOTH destructive patterns and AST check
        assert any("Destructive pattern" in i for i in result.destructive_issues)

    def test_regex_escape_dunder_access(self):
        # obfuscated access via .__class__ chain
        result = _categorized_security_check("x.__class__.__mro__\n")
        assert result.escape_issues

    def test_open_etc_passwd(self):
        result = _categorized_security_check("open('/etc/passwd').read()\n")
        assert result.destructive_issues
        assert result.is_safe is False

    def test_network_socket(self):
        result = _categorized_security_check("import socket\ns = socket.socket()\n")
        assert result.destructive_issues
        assert result.is_safe is False


# =========================================================================
# _comprehensive_security_check (backward-compatible wrapper)
# =========================================================================


class TestComprehensiveSecurityCheck:
    def test_clean_script(self):
        safe, issues = _comprehensive_security_check("x = 1\n")
        assert safe is True
        assert issues == []

    def test_unsafe_script(self):
        safe, issues = _comprehensive_security_check("import os\nos.remove('x')\n")
        assert safe is False
        assert issues

    def test_returns_combined_issues(self):
        safe, issues = _comprehensive_security_check("import os\nos.remove('x')\neval('1')\n")
        assert safe is False
        # Both kinds of issues should appear
        assert any("Dangerous" in i or "pattern" in i for i in issues)

    def test_signature_matches_documentation(self):
        # Smoke test the documented return shape
        result = _comprehensive_security_check("x = 1\n")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# =========================================================================
# Runner integration helpers
# =========================================================================


class TestIsScriptDestructiveIntegration:
    def test_clean_script_passes(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_script_destructive("x = 1\n") is False

    def test_destructive_script_blocked(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_script_destructive("import os\nos.remove('x')\n") is True

    def test_yolo_mode_disables_check(self, monkeypatch):
        monkeypatch.setenv("SCHOLARDEVCLAW_YOLO_MODE", "true")
        assert _is_script_destructive("import os\nos.remove('x')\n") is False

    def test_yolo_mode_truthy_values(self, monkeypatch):
        for value in ("1", "yes", "TRUE", "Yes"):
            monkeypatch.setenv("SCHOLARDEVCLAW_YOLO_MODE", value)
            assert _is_script_destructive("os.remove('x')") is False

    def test_escape_pattern_does_not_trigger_destructive(self, monkeypatch):
        # eval is escape, not destructive
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_script_destructive("eval('1+1')\n") is False


class TestIsSandboxEscapeIntegration:
    def test_clean_script_passes(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_sandbox_escape("x = 1\n") is False

    def test_escape_attempt_blocked(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_sandbox_escape("eval('1+1')\n") is True

    def test_yolo_mode_disables_check(self, monkeypatch):
        monkeypatch.setenv("SCHOLARDEVCLAW_YOLO_MODE", "true")
        assert _is_sandbox_escape("eval('1+1')\n") is False

    def test_destructive_pattern_does_not_trigger_escape(self, monkeypatch):
        # os.remove is destructive, not escape
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_sandbox_escape("import os\nos.remove('x')\n") is False

    def test_dunder_attribute_blocked(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_sandbox_escape("x.__class__\n") is True

    def test_compile_blocked(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_YOLO_MODE", raising=False)
        assert _is_sandbox_escape("compile('x', '<s>', 'exec')\n") is True
