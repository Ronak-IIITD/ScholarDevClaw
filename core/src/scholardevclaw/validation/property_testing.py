"""
Property-based testing integration using Hypothesis.

Provides:
- Automatic test generation from function signatures
- Data generation strategies for common types
- Property-based testing for ScholarDevClaw functions
- Integration with existing test suites
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, get_type_hints


try:
    from hypothesis import given, settings, Phase, Verbosity
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


T = TypeVar("T")


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test run"""

    test_name: str
    passed: bool
    examples_generated: int
    failures: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)


@dataclass
class PropertyTestConfig:
    """Configuration for property-based testing"""

    max_examples: int = 100
    min_examples: int = 10
    deadline: int = 2000  # ms
    phases: list[str] = field(default_factory=lambda: ["generate", "shrink"])
    verbosity: str = "quiet"
    derandomize: bool = False


class TypeToStrategy:
    """Convert Python types to Hypothesis strategies"""

    @staticmethod
    def from_type(typ: type) -> Any:
        """Convert a type to a Hypothesis strategy"""
        if not HAS_HYPOTHESIS:
            raise ImportError("hypothesis is required. Install with: pip install hypothesis")

        origin = getattr(typ, "__origin__", None)

        if origin is list:
            return st.lists(TypeToStrategy.from_type(typ.__args__[0]))
        elif origin is dict:
            return st.dictionaries(
                TypeToStrategy.from_type(typ.__args__[0]),
                TypeToStrategy.from_type(typ.__args__[1]),
            )
        elif origin is tuple:
            return st.tuples(*[TypeToStrategy.from_type(t) for t in typ.__args__])
        elif origin is set:
            return st.sets(TypeToStrategy.from_type(typ.__args__[0]))
        elif origin is type(None) or typ is type(None):
            return st.none()
        else:
            return _simple_type_to_strategy(typ)

    @staticmethod
    def from_annotation(annotation: str) -> Any:
        """Convert type annotation string to strategy"""
        if not HAS_HYPOTHESIS:
            raise ImportError("hypothesis is required")

        annotation = annotation.strip()

        if "str" in annotation:
            return st.text(min_size=0, max_size=1000)
        elif "int" in annotation:
            if "positive" in annotation.lower():
                return st.integers(min_value=1)
            elif "negative" in annotation.lower():
                return st.integers(max_value=-1)
            else:
                return st.integers(min_value=-1000000, max_value=1000000)
        elif "float" in annotation:
            return st.floats(min_value=-1e6, max_value=1e6)
        elif "bool" in annotation:
            return st.booleans()
        elif "bytes" in annotation:
            return st.binary(max_size=10000)
        elif "Path" in annotation:
            return st.from_type(Path)
        elif "list" in annotation or "List" in annotation:
            return st.lists(st.text())
        elif "dict" in annotation or "Dict" in annotation:
            return st.dictionaries(st.text(), st.text())
        else:
            return st.text()


def _simple_type_to_strategy(typ: type) -> Any:
    """Convert simple types to strategies"""
    if typ is str:
        return st.text(min_size=0, max_size=1000)
    elif typ is int:
        return st.integers(min_value=-1000000, max_value=1000000)
    elif typ is float:
        return st.floats(min_value=-1e6, max_value=1e6)
    elif typ is bool:
        return st.booleans()
    elif typ is bytes:
        return st.binary(max_size=10000)
    elif typ is type(None) or typ is None:
        return st.none()
    elif typ is Path:
        return st.text(
            alphabet=st.characters(categories=["Ll", "Lu", "Nd", "Ps", "Pe"]), min_size=1
        ).map(Path)
    else:
        return st.sampled_from([typ()])


class PropertyTestGenerator:
    """Generate property-based tests automatically"""

    def __init__(self, config: PropertyTestConfig | None = None):
        self.config = config or PropertyTestConfig()
        self._tests: dict[str, Callable] = {}

    def add_property(
        self,
        name: str,
        func: Callable,
        preconditions: list[Callable] | None = None,
    ):
        """Add a property to test"""
        self._tests[name] = {
            "func": func,
            "preconditions": preconditions or [],
        }

    def generate_test(self, name: str) -> str:
        """Generate a Hypothesis test function as string"""
        test_def = self._tests.get(name)
        if not test_def:
            return ""

        func = test_def["func"]
        hints = get_type_hints(func)

        strategies = []
        for param_name, param_type in hints.items():
            if param_name == "return":
                continue
            strategy = TypeToStrategy.from_type(param_type)
            strategies.append(f"    {param_name} = {repr(strategy)}")

        params = ", ".join(hints.keys() - {"return"})

        return f'''
@given({", ".join([s.split(" = ")[0] for s in strategies])})
@settings(max_examples={self.config.max_examples}, deadline={self.config.deadline})
def test_{name}({params}):
    """Auto-generated property test for {name}"""
    # Preconditions
    # Test body
    result = {func.__name__}({params})
    # Assertions would go here
'''

    def run_test(self, name: str) -> HypothesisTestResult:
        """Run a single property test"""
        if name not in self._tests:
            return HypothesisTestResult(
                test_name=name,
                passed=False,
                examples_generated=0,
                failures=0,
                duration_seconds=0,
                errors=[f"Test {name} not found"],
            )

        if not HAS_HYPOTHESIS:
            return HypothesisTestResult(
                test_name=name,
                passed=False,
                examples_generated=0,
                failures=0,
                duration_seconds=0,
                errors=["Hypothesis not installed"],
            )

        import time

        start = time.time()
        test_def = self._tests[name]
        func = test_def["func"]

        try:
            hints = get_type_hints(func)
            params = [p for p in hints.keys() if p != "return"]

            strategy_dict = {}
            for param in params:
                param_type = hints.get(param, str)
                strategy_dict[param] = TypeToStrategy.from_type(param_type)

            @given(**strategy_dict)
            @settings(
                max_examples=self.config.max_examples,
                deadline=self.config.deadline,
                derandomize=self.config.derandomize,
            )
            def test_wrapper(**kwargs):
                for precondition in test_def["preconditions"]:
                    precondition(**kwargs)
                return func(**kwargs)

            test_wrapper()

            duration = time.time() - start
            return HypothesisTestResult(
                test_name=name,
                passed=True,
                examples_generated=self.config.max_examples,
                failures=0,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return HypothesisTestResult(
                test_name=name,
                passed=False,
                examples_generated=0,
                failures=1,
                duration_seconds=duration,
                errors=[str(e)],
            )

    def run_all(self) -> list[HypothesisTestResult]:
        """Run all property tests"""
        results = []
        for name in self._tests:
            results.append(self.run_test(name))
        return results


def create_property_test(
    func: Callable,
    strategy_config: dict[str, Any] | None = None,
    config: PropertyTestConfig | None = None,
) -> Callable:
    """Decorator to convert a function to a property test"""
    if not HAS_HYPOTHESIS:
        raise ImportError("hypothesis is required")

    config = config or PropertyTestConfig()
    strategy_config = strategy_config or {}

    hints = get_type_hints(func)
    params = [p for p in hints.keys() if p != "return"]

    strategies = {}
    for param in params:
        if param in strategy_config:
            strategies[param] = strategy_config[param]
        else:
            strategies[param] = TypeToStrategy.from_type(hints[param])

    @given(**strategies)
    @settings(
        max_examples=config.max_examples,
        deadline=config.deadline,
        derandomize=config.derandomize,
    )
    def test_wrapper(**kwargs):
        return func(**kwargs)

    return test_wrapper


class HypothesisTestRunner:
    """Run Hypothesis tests for a module or project"""

    def __init__(self, root_path: Path | None = None):
        self.root_path = root_path or Path.cwd()

    def discover_tests(self, pattern: str = "test_*.py") -> list[str]:
        """Discover test files in the project"""
        tests = []
        for py_file in self.root_path.rglob(pattern):
            if "__pycache__" in py_file.parts:
                continue
            tests.append(str(py_file.relative_to(self.root_path)))
        return tests

    def run_module_tests(
        self,
        module_path: str,
        config: PropertyTestConfig | None = None,
    ) -> HypothesisTestResult:
        """Run Hypothesis tests for a specific module"""
        config = config or PropertyTestConfig()

        if not HAS_HYPOTHESIS:
            return HypothesisTestResult(
                test_name=module_path,
                passed=False,
                examples_generated=0,
                failures=0,
                duration_seconds=0,
                errors=["Hypothesis not installed"],
            )

        import time

        start = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            module_path,
            "-v",
            "--hypothesis-show-statistics",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            duration = time.time() - start

            passed = result.returncode == 0
            output = result.stdout + result.stderr

            examples = 0
            if "hypothesis" in output.lower():
                import re

                match = re.search(r"(\d+) examples", output)
                if match:
                    examples = int(match.group(1))

            failures = 0
            if "hypothesis" in output.lower():
                import re

                match = re.search(r"(\d+) failed", output)
                if match:
                    failures = int(match.group(1))

            return HypothesisTestResult(
                test_name=module_path,
                passed=passed,
                examples_generated=examples,
                failures=failures,
                duration_seconds=duration,
                errors=[] if passed else [output[-500:]],
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return HypothesisTestResult(
                test_name=module_path,
                passed=False,
                examples_generated=0,
                failures=1,
                duration_seconds=duration,
                errors=["Test timed out after 5 minutes"],
            )
        except Exception as e:
            duration = time.time() - start
            return HypothesisTestResult(
                test_name=module_path,
                passed=False,
                examples_generated=0,
                failures=1,
                duration_seconds=duration,
                errors=[str(e)],
            )


def quickcheck(func: Callable, **strategies: Any) -> bool:
    """Quick property check - run a function with generated inputs"""
    if not HAS_HYPOTHESIS:
        raise ImportError("hypothesis is required")

    @given(**strategies)
    @settings(max_examples=100, deadline=2000)
    def test_wrapper(**kwargs):
        return func(**kwargs)

    try:
        test_wrapper()
        return True
    except Exception:
        return False
