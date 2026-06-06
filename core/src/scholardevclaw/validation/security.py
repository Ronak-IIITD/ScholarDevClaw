"""Validation runner hardening - improved sandbox escape detection and error handling.

Provides:
    - Regex-based detection of destructive operations and sandbox escape attempts.
    - AST-based analysis to catch patterns that evade regex (e.g. dynamic attribute
      lookups, obfuscated ``getattr`` / ``__import__`` chains).
    - A categorized ``SecurityCheckResult`` so the validation runner can
      distinguish between destructive operations and sandbox escape attempts.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# Sandbox-escape patterns are intentionally focused on the *mechanisms* used
# to bypass a Python sandbox (dynamic code execution, attribute / builtin
# introspection, dunder access).  File-system and network access patterns
# live in ``_DESTRUCTIVE_PATTERNS_EXTENDED`` so each issue belongs to
# exactly one bucket.
_SANDBOX_ESCAPE_PATTERNS_EXTENDED = [
    # Dynamic code execution
    r"__import__\s*\(",
    r"importlib\.",
    r"sys\.modules",
    r"getattr\s*\(\s*builtins",
    r"__class__|__mro__|__subclasses__|__globals__|__builtins__",
    r"compile\s*\(",
    r"exec\s*\(",
    r"eval\s*\(",
    r"setattr\s*\(",
    r"del\s+os\b|del\s+sys\b|del\s+builtins\b",
    # Additional dangerous patterns
    r"__import__\s*\(\s*['\"]os['\"]",
    r"__import__\s*\(\s*['\"]subprocess['\"]",
    r"__import__\s*\(\s*['\"]sys['\"]",
    r"__import__\s*\(\s*['\"]builtins['\"]",
    r"getattr\s*\(\s*__import__",
    r"__builtins__\s*\[",
    r"globals\s*\(\s*\)\s*\[",
    r"locals\s*\(\s*\)\s*\[",
    r"vars\s*\(\s*\)\s*\[",
    # Code injection
    r"exec\s*\(\s*compile",
    r"eval\s*\(\s*compile",
    r"ast\.parse\s*\(",
    r"code\.compile_command",
    # Module manipulation
    r"sys\.modules\s*\[",
    r"importlib\.import_module",
    r"importlib\.reload",
    r"pkgutil\.",
    r"runpy\.",
    # Dangerous builtins
    r"__import__\s*\(\s*['\"]importlib['\"]",
    r"__import__\s*\(\s*['\"]pkgutil['\"]",
    r"__import__\s*\(\s*['\"]runpy['\"]",
    # Low-level escape hatches
    r"pty\.",
    r"ctypes\.",
    r"fcntl\.",
    r"resource\.",
]

# Destructive operation patterns - extended
_DESTRUCTIVE_PATTERNS_EXTENDED = [
    r"rm\s+-rf\s+/",
    r"dd\s+if=.*of=/dev/sd[^ ]",
    r"curl\s+[^|]*\|\s*bash",
    r"wget\s+[^|]*\|\s*bash",
    r":\(\)\{.*\|\:\)",
    r"os\.system\s*\(",
    r"subprocess\.(call|run|Popen|check_output|check_call)\s*\(",
    r"socket\.",
    r"urllib\.|requests\.",
    r"open\s*\(['\"]/(?:etc|proc|sys|dev|run)/",
    # File / filesystem destruction
    r"os\.remove\s*\(",
    r"os\.unlink\s*\(",
    r"os\.rmdir\s*\(",
    r"shutil\.rmtree\s*\(",
    r"pathlib\.Path\.unlink\s*\(",
    r"pathlib\.Path\.rmdir\s*\(",
    r"pathlib\.Path\s*\(\s*['\"]/",
    r"os\.(?:path|walk|listdir|scandir|remove|rmdir|mkdir|rename|replace)",
    r"shutil\.(?:rmtree|copy|move|copy2)",
    r"open\s*\(\s*['\"]/(?:etc|proc|sys|dev|run|root|home|tmp)/",
    # Process / environment
    r"os\.environ",
    r"os\.popen\s*\(",
    r"tempfile\.mkdtemp\s*\(",
    r"tempfile\.mkstemp\s*\(",
    r"commands\.(?:getstatusoutput|getoutput)",
    r"threading\.(?:Thread|Timer|Lock|RLock|Condition|Semaphore|Event|Barrier)",
    r"multiprocessing\.(?:Process|Pool|Queue|Pipe|Manager|Value|Array)",
    r"concurrent\.futures\.(?:ThreadPoolExecutor|ProcessPoolExecutor)",
    r"asyncio\.(?:run|create_task|gather|wait|as_completed)",
    # Network clients
    r"urllib\.(?:request|parse|response)",
    r"requests\.(?:get|post|put|delete|patch|head|options|request)",
    r"http\.client",
    r"ftplib\.",
    r"telnetlib\.",
    r"smtplib\.",
    r"poplib\.",
    r"imaplib\.",
    r"nntplib\.",
]

# AST-based dangerous node types, split into "destructive" (data / state
# modification) and "escape" (sandbox bypass / arbitrary code execution)
# categories.  These are disjoint so a single script element belongs to
# exactly one bucket.
_DANGEROUS_AST_NODES: dict[str, dict[str, set[str]]] = {
    "Call": {
        "destructive": {
            "open",
        },
        "escape": {
            "eval",
            "exec",
            "compile",
            "__import__",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "input",
            "exit",
            "quit",
        },
    },
    "Attribute": {
        "destructive": {
            "system",
            "popen",
            "popen2",
            "popen3",
            "popen4",
            "remove",
            "unlink",
            "rmdir",
            "mkdir",
            "rename",
            "replace",
            "rmtree",
            "copy",
            "move",
            "copy2",
            "run",
            "Popen",
            "call",
            "check_output",
            "check_call",
            "getstatusoutput",
            "getoutput",
            "mkdtemp",
            "mkstemp",
            "Thread",
            "Timer",
            "Process",
            "Pool",
            "ThreadPoolExecutor",
            "ProcessPoolExecutor",
            "create_task",
            "gather",
            "wait",
            "as_completed",
        },
        "escape": {
            "import_module",
            "reload",
            "parse",
            "compile_command",
        },
    },
    "Import": {
        "destructive": {
            "os",
            "sys",
            "subprocess",
            "socket",
            "urllib",
            "requests",
            "threading",
            "multiprocessing",
            "concurrent",
            "asyncio",
            "ctypes",
            "pty",
            "tempfile",
            "shutil",
            "http",
            "ftplib",
            "telnetlib",
            "smtplib",
            "poplib",
            "imaplib",
            "nntplib",
        },
        "escape": {
            "importlib",
            "pkgutil",
            "runpy",
            "code",
            "ast",
        },
    },
    "ImportFrom": {
        "destructive": {
            "os",
            "sys",
            "subprocess",
            "socket",
            "urllib",
            "requests",
            "threading",
            "multiprocessing",
            "concurrent",
            "asyncio",
            "ctypes",
            "pty",
            "tempfile",
            "shutil",
            "http",
            "ftplib",
            "telnetlib",
            "smtplib",
            "poplib",
            "imaplib",
            "nntplib",
        },
        "escape": {
            "importlib",
            "pkgutil",
            "runpy",
            "code",
            "ast",
        },
    },
}


@dataclass
class SecurityCheckResult:
    """Categorized result of a script security check.

    Attributes:
        is_safe: True when neither destructive nor escape issues were found.
        destructive_issues: List of reasons the script would perform a
            destructive operation (e.g. ``os.remove``).
        escape_issues: List of reasons the script would escape the sandbox
            (e.g. ``eval``, ``__import__`` of ``os``).
        all_issues: Convenience accessor returning both lists combined.
    """

    destructive_issues: list[str] = field(default_factory=list)
    escape_issues: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.destructive_issues and not self.escape_issues

    @property
    def all_issues(self) -> list[str]:
        return [*self.destructive_issues, *self.escape_issues]


def _is_script_destructive_ast(script: str) -> tuple[bool, list[str]]:
    """Check if script contains destructive operations using AST analysis.

    Returns (is_destructive, list_of_reasons).
    """
    reasons = []
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # If we can't parse it, be conservative
        return True, ["Syntax error - cannot analyze"]

    destructive_calls = _DANGEROUS_AST_NODES["Call"]["destructive"]
    destructive_attrs = _DANGEROUS_AST_NODES["Attribute"]["destructive"]
    destructive_imports = _DANGEROUS_AST_NODES["Import"]["destructive"]
    destructive_import_froms = _DANGEROUS_AST_NODES["ImportFrom"]["destructive"]

    for node in ast.walk(tree):
        # Check dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in destructive_calls:
                    reasons.append(f"Dangerous function call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in destructive_attrs:
                    reasons.append(f"Dangerous attribute access: {node.func.attr}")

        # Check dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in destructive_imports:
                    reasons.append(f"Dangerous import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in destructive_import_froms:
                reasons.append(f"Dangerous import from: {node.module}")

    return len(reasons) > 0, reasons


def _is_sandbox_escape_ast(script: str) -> tuple[bool, list[str]]:
    """Check if script contains sandbox escape attempts using AST analysis.

    Returns (is_escape_attempt, list_of_reasons).
    """
    reasons = []
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return True, ["Syntax error - cannot analyze"]

    escape_calls = _DANGEROUS_AST_NODES["Call"]["escape"]
    escape_attrs = _DANGEROUS_AST_NODES["Attribute"]["escape"]
    escape_imports = _DANGEROUS_AST_NODES["Import"]["escape"]
    escape_import_froms = _DANGEROUS_AST_NODES["ImportFrom"]["escape"]

    for node in ast.walk(tree):
        # Check for __builtins__ access
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            reasons.append("Direct __builtins__ access")

        # Check for __class__, __mro__, __subclasses__, __globals__, __builtins__ attributes
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            if attr_name in (
                "__class__",
                "__mro__",
                "__subclasses__",
                "__globals__",
                "__builtins__",
            ):
                reasons.append(f"Dangerous dunder attribute: {attr_name}")

        # Check for globals()/locals()/vars() calls and code-execution calls
        if isinstance(node, ast.Call):
            call_func = node.func
            if isinstance(call_func, ast.Name):
                func_id = call_func.id
                if func_id in ("globals", "locals", "vars"):
                    reasons.append(f"Dangerous builtin call: {func_id}()")
                elif func_id in escape_calls:
                    reasons.append(f"Code execution function: {func_id}()")

        # Escape-level attribute access (e.g. importlib.import_module, ast.parse)
        # Note: function calls are caught by the ast.Call branch above; this
        # branch only flags attribute access that is not itself a Call node
        # (for example reading ``importlib.import_module`` as a value).
        if isinstance(node, ast.Attribute) and not isinstance(node, ast.Call):
            attr_name = node.attr
            if attr_name in escape_attrs:
                reasons.append(f"Dangerous escape attribute: {attr_name}")

        # Escape-level import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in escape_imports:
                    reasons.append(f"Dangerous escape import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in escape_import_froms:
                reasons.append(f"Dangerous escape import from: {node.module}")

    return len(reasons) > 0, reasons


def _categorized_security_check(script: str) -> SecurityCheckResult:
    """Run a categorized security check on *script*.

    Returns a :class:`SecurityCheckResult` with destructive and escape
    issues bucketed separately so callers can react to each kind
    independently.
    """
    result = SecurityCheckResult()

    for pattern in _DESTRUCTIVE_PATTERNS_EXTENDED:
        if re.search(pattern, script, re.IGNORECASE):
            result.destructive_issues.append(f"Destructive pattern: {pattern}")

    for pattern in _SANDBOX_ESCAPE_PATTERNS_EXTENDED:
        if re.search(pattern, script, re.IGNORECASE):
            result.escape_issues.append(f"Sandbox escape pattern: {pattern}")

    _, destructive_reasons = _is_script_destructive_ast(script)
    result.destructive_issues.extend(destructive_reasons)

    _, escape_reasons = _is_sandbox_escape_ast(script)
    result.escape_issues.extend(escape_reasons)

    return result


def _comprehensive_security_check(script: str) -> tuple[bool, list[str]]:
    """Run comprehensive security checks on a script.

    Returns (is_safe, list_of_issues).

    Kept for backward compatibility. New code should prefer
    :func:`_categorized_security_check` which returns a structured
    :class:`SecurityCheckResult` instead.
    """
    result = _categorized_security_check(script)
    return result.is_safe, result.all_issues


# Export for use in validation runner
__all__ = [
    "SecurityCheckResult",
    "_categorized_security_check",
    "_comprehensive_security_check",
    "_is_script_destructive_ast",
    "_is_sandbox_escape_ast",
    "_DESTRUCTIVE_PATTERNS_EXTENDED",
    "_SANDBOX_ESCAPE_PATTERNS_EXTENDED",
    "_DANGEROUS_AST_NODES",
]
