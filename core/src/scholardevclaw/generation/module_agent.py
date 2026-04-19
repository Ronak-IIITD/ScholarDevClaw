from __future__ import annotations

import ast
from typing import Any

from scholardevclaw.generation.models import ModuleResult
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding

MODULE_SYSTEM_PROMPT = """You are an expert Python developer implementing a
research paper. Write production-quality, fully typed Python code.
Rules:
- Use type hints everywhere (Python 3.11+).
- Every public function has a docstring.
- Imports are at the top, stdlib → third-party → local.
- No placeholder comments like "# TODO: implement". Write real code.
- Match the tech stack specified. If PyTorch: use nn.Module, DataLoader etc.
- Code must be importable without errors.
- Return ONLY the Python code. No markdown fences. No explanation.
"""


TEST_SYSTEM_PROMPT = """You are an expert in pytest. Write comprehensive tests
for the given Python module. Use pytest fixtures. Mock external dependencies.
Include at least one happy-path test and one edge-case test per public function.
Return ONLY the pytest code. No markdown. No explanation.
"""


def _extract_tokens_used(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0

    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(total_tokens, int):
        return total_tokens

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens

    return 0


class ModuleAgent:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        knowledge_base: Any | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.knowledge_base = knowledge_base

    async def generate(
        self,
        module: CodeModule,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        context_modules: dict[str, str],
        max_attempts: int = 3,
        error_context: str | None = None,
    ) -> ModuleResult:
        attempts = 0
        errors: list[str] = []
        code = ""
        test_code = ""
        tokens_used = 0

        while attempts < max_attempts:
            attempts += 1
            prompt = self._build_module_prompt(
                module=module,
                plan=plan,
                understanding=understanding,
                context_modules=context_modules,
                prior_errors=errors,
                error_context=error_context,
            )
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=MODULE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            tokens_used += _extract_tokens_used(response)

            if not response.content:
                errors = [
                    "What failed: module code generation. "
                    "Why: model returned an empty response content block. "
                    "Fix: retry generation with the same module context."
                ]
                continue

            block_text = self._extract_first_text_block(response.content)
            code = block_text.strip() if isinstance(block_text, str) else ""

            validation_errors = self._validate_generated_module(module, plan, code)
            if not validation_errors:
                errors = []
                break
            errors = validation_errors

        test_errors: list[str] = []
        test_prior_errors: list[str] = []
        test_attempts = 0
        while test_attempts < max_attempts:
            test_attempts += 1
            generated_test_code, test_response = await self._generate_tests(
                module,
                code,
                understanding,
                prior_errors=test_prior_errors,
            )
            tokens_used += _extract_tokens_used(test_response)
            test_code = generated_test_code

            syntax_errors = self._check_syntax(test_code)
            if not syntax_errors:
                test_errors = []
                break

            test_errors = [f"Test file: {error}" for error in syntax_errors]
            test_prior_errors = test_errors

        final_errors = [*errors, *test_errors]

        return ModuleResult(
            module_id=module.id,
            file_path=module.file_path,
            test_file_path=module.test_file_path,
            code=code,
            test_code=test_code,
            generation_attempts=attempts,
            final_errors=final_errors,
            tokens_used=tokens_used,
        )

    def _build_module_prompt(
        self,
        module: CodeModule,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        context_modules: dict[str, str],
        prior_errors: list[str],
        error_context: str | None = None,
    ) -> str:
        context_parts: list[str] = []
        for dep_id in module.depends_on:
            if dep_id in context_modules:
                context_parts.append(f"# Already implemented: {dep_id}")
                context_parts.append(context_modules[dep_id][:2000])

        if self.knowledge_base is not None:
            similar_snippets: list[str] = []
            try:
                similar = self.knowledge_base.retrieve_similar_implementations(
                    module.description or module.name,
                    plan.tech_stack,
                    n=2,
                )
                if isinstance(similar, list):
                    similar_snippets = [
                        snippet
                        for snippet in similar
                        if isinstance(snippet, str) and snippet.strip()
                    ]
            except Exception:
                similar_snippets = []

            if similar_snippets:
                context_parts.append("")
                context_parts.append("# Similar implementations from knowledge base:")
                for snippet in similar_snippets:
                    context_parts.append(snippet[:1000])
                    context_parts.append("---")

        error_block = ""
        if prior_errors:
            error_block = "\nPrevious attempt had these errors — fix them:\n" + "\n".join(
                prior_errors
            )

        report_block = ""
        if error_context:
            report_block = (
                "\nSandbox/test error context from previous execution (if relevant):\n"
                f"{error_context[:3000]}"
            )

        context_block = "\n".join(context_parts)

        return f"""Paper: {understanding.paper_title}
Core algorithm: {understanding.core_algorithm_description}
Tech stack: {plan.tech_stack}

Implement this module:
Module: {module.name} ({module.id})
File: {module.file_path}
Description: {module.description}
Estimated lines: {module.estimated_lines}
{error_block}
{report_block}

Context from dependencies:
{context_block}

Write complete, production-quality Python code for {module.file_path}.
"""

    def _check_syntax(self, code: str) -> list[str]:
        try:
            ast.parse(code)
            return []
        except SyntaxError as exc:
            return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]

    def _validate_generated_module(
        self,
        module: CodeModule,
        plan: ImplementationPlan,
        code: str,
    ) -> list[str]:
        syntax_errors = self._check_syntax(code)
        if syntax_errors:
            return syntax_errors

        tree = ast.parse(code)
        known_src_modules = self._collect_known_src_modules(plan, module)

        errors: list[str] = []
        errors.extend(self._check_src_import_consistency(tree, known_src_modules))
        errors.extend(self._check_public_function_type_hints(tree))
        return errors

    def _collect_known_src_modules(
        self,
        plan: ImplementationPlan,
        module: CodeModule,
    ) -> set[str]:
        known_modules: set[str] = set()

        for planned_module in plan.modules:
            module_path = self._file_path_to_module_path(planned_module.file_path)
            if module_path and module_path.startswith("src."):
                known_modules.add(module_path)

        current_module_path = self._file_path_to_module_path(module.file_path)
        if current_module_path and current_module_path.startswith("src."):
            known_modules.add(current_module_path)

        return known_modules

    def _file_path_to_module_path(self, file_path: str) -> str | None:
        normalized = file_path.strip().replace("\\", "/")
        if not normalized.endswith(".py"):
            return None

        module_path = normalized.removesuffix(".py").lstrip("./").replace("/", ".")
        if module_path.endswith(".__init__"):
            module_path = module_path.removesuffix(".__init__")
        return module_path or None

    def _check_src_import_consistency(
        self,
        tree: ast.Module,
        known_src_modules: set[str],
    ) -> list[str]:
        errors: list[str] = []
        expected_modules = sorted(known_src_modules)
        expected_preview = ", ".join(expected_modules[:8]) if expected_modules else "none"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if not imported.name.startswith("src."):
                        continue
                    if imported.name not in known_src_modules:
                        errors.append(
                            f"Line {node.lineno}: Unknown local import '{imported.name}'. "
                            f"Expected planned src modules: {expected_preview}."
                        )

            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or not node.module or not node.module.startswith("src."):
                continue
            if node.module in known_src_modules:
                continue

            unresolved_names: list[str] = []
            for imported in node.names:
                if imported.name == "*":
                    unresolved_names.append("*")
                    continue

                candidate_module = f"{node.module}.{imported.name}"
                if candidate_module not in known_src_modules:
                    unresolved_names.append(imported.name)

            if unresolved_names:
                imported_text = ", ".join(unresolved_names)
                errors.append(
                    f"Line {node.lineno}: Unknown local import target '{node.module}' "
                    f"for [{imported_text}]. Expected planned src modules: {expected_preview}."
                )

        return errors

    def _check_public_function_type_hints(self, tree: ast.Module) -> list[str]:
        errors: list[str] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue

            missing_annotations = self._collect_missing_arg_annotations(node)
            if missing_annotations:
                missing_text = ", ".join(missing_annotations)
                errors.append(
                    f"Line {node.lineno}: Public function '{node.name}' is missing "
                    f"type annotations for parameters: {missing_text}."
                )

            if node.returns is None:
                errors.append(
                    f"Line {node.lineno}: Public function '{node.name}' is missing "
                    "a return type annotation."
                )

        return errors

    def _collect_missing_arg_annotations(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        missing: list[str] = []
        all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in all_args:
            if arg.annotation is None:
                missing.append(arg.arg)

        if node.args.vararg is not None and node.args.vararg.annotation is None:
            missing.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg is not None and node.args.kwarg.annotation is None:
            missing.append(f"**{node.args.kwarg.arg}")

        return missing

    async def _generate_tests(
        self,
        module: CodeModule,
        code: str,
        understanding: PaperUnderstanding,
        prior_errors: list[str] | None = None,
    ) -> tuple[str, Any]:
        error_block = ""
        if prior_errors:
            error_block = (
                "\nPrevious test generation attempt had syntax issues — fix them:\n"
                + "\n".join(prior_errors)
            )

        prompt = f"""Module to test: {module.name} ({module.file_path})
Paper context: {understanding.one_line_summary}
Code:
{code[:6000]}
{error_block}

Write pytest tests for this module."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=TEST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.content:
            return "", response

        block_text = self._extract_first_text_block(response.content)
        return (block_text.strip() if isinstance(block_text, str) else "", response)

    @staticmethod
    def _extract_first_text_block(content_blocks: Any) -> str | None:
        if not isinstance(content_blocks, list):
            return None
        for block in content_blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None
