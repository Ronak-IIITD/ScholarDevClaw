# Contributing to ScholarDevClaw

ScholarDevClaw is a research-to-code system spanning a Python execution plane, a TypeScript control plane, and a React dashboard. This guide covers the local workflow expected for pull requests.

## Before You Start

- Read [AGENTS.md](./AGENTS.md) for repo layout and architectural boundaries.
- Check [UPDATES.md](./UPDATES.md) to avoid duplicating in-flight work.
- Search existing issues and discussions before opening a new one.

## Local Setup

### Core

```bash
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml,dev,tui]"
```

### Agent

```bash
cd agent
bun install
bun run build
bun run test
```

### Dashboard

```bash
cd web
npm install
npm run build
```

## Test Matrix

Run the narrowest checks that prove your change, then broaden before opening a PR.

```bash
cd core && pytest -q
cd core && ruff check src tests benchmarks
cd core && mypy src/scholardevclaw/cli.py src/scholardevclaw/api/server.py src/scholardevclaw/application/pipeline.py --ignore-missing-imports --follow-imports=skip
cd agent && bun run build && bun run test
cd web && npm run build
```

If you touched launch/demo behavior, also run:

```bash
cd core && python -m scholardevclaw.cli demo --spec rmsnorm --skip-validate
cd core && python -m benchmarks.runner
```

## Branches and PRs

- Branch from `main` with a scoped name such as `feature/<topic>` or `fix/<topic>`.
- Keep commits intentional and descriptive.
- Update docs when contracts, commands, or UX change.
- Add or update tests for every behavior change.
- Update [UPDATES.md](./UPDATES.md) when the task materially changes shipped behavior.

Use the pull request template in [.github/PULL_REQUEST_TEMPLATE/pull_request_template.md](./.github/PULL_REQUEST_TEMPLATE/pull_request_template.md).

## Adding a New Paper Spec

The easiest high-value contribution is extending the built-in research spec registry.

1. Add the spec to [core/src/scholardevclaw/research_intelligence/extractor.py](./core/src/scholardevclaw/research_intelligence/extractor.py).
2. Include `paper`, `algorithm`, `implementation`, `changes`, and `validation` sections.
3. Add a template or transformation support path in [core/src/scholardevclaw/patch_generation/generator.py](./core/src/scholardevclaw/patch_generation/generator.py) when the spec should emit runnable artifacts.
4. Add or update tests in `core/tests/unit/`.
5. If the spec should participate in the eval harness, update `core/benchmarks/`.

Good paper-spec PRs include:

- a real arXiv ID
- concrete target patterns
- a clear replacement or augmentation strategy
- validation expectations

## Adding a New tree-sitter Language

1. Add the parser dependency in [core/pyproject.toml](./core/pyproject.toml).
2. Register it in [core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py](./core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py).
3. Extend language detection, file-extension mapping, and parsing tests.
4. Add fixtures or representative unit coverage showing model/function/import extraction for the new language.

Do not weaken existing parsers or remove multi-language support to land a new language.

## Coding Rules

### Python

- Prefer explicit dataclasses and narrow functions.
- Reuse `application/pipeline.py` for shared workflow behavior.
- Keep filesystem and subprocess handling path-safe and bounded.

### TypeScript

- Keep strict ESM imports with `.js` suffixes.
- Preserve bridge payload compatibility between `agent/` and `core/`.

### Frontend

- Reuse the existing dashboard API contracts instead of creating parallel ones.
- Keep the WebSocket pipeline flow observable and deterministic.

## Issue and PR Templates

Repository templates live in:

- [.github/ISSUE_TEMPLATE/bug_report.md](./.github/ISSUE_TEMPLATE/bug_report.md)
- [.github/ISSUE_TEMPLATE/feature_request.md](./.github/ISSUE_TEMPLATE/feature_request.md)
- [.github/ISSUE_TEMPLATE/new_paper_spec.md](./.github/ISSUE_TEMPLATE/new_paper_spec.md)
- [.github/PULL_REQUEST_TEMPLATE/pull_request_template.md](./.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)

Use the paper-spec template when proposing new research integrations.

## Review Expectations

A PR is ready for review when:

- the relevant tests pass locally
- the scope is tight
- behavior changes are described clearly
- new prompts or extraction logic return structured output
- generated artifacts and validation results are reproducible

## Security

- Never commit secrets, tokens, or credentials.
- Respect `SCHOLARDEVCLAW_ALLOWED_REPO_DIRS` and other confinement controls.
- Treat repository inputs and outbound network lookups as untrusted.

## Getting Help

- Open a bug report or feature request if behavior is broken or missing.
- Use the paper-spec issue template for new algorithm requests.
- Link benchmark evidence, repo examples, or failing traces whenever possible.
