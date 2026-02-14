# ScholarDevClaw Agent Handbook

Scope: This file governs the repository root. For anything inside agent/workspace/, follow the closer policy file at agent/workspace/AGENTS.md.

This document is written for autonomous coding agents and human contributors who need fast, accurate context.

## 1) Mission and Product Shape

ScholarDevClaw is a research-to-code integration system:
- analyzes an existing repository,
- finds research improvements,
- maps them onto the codebase,
- generates patch artifacts,
- validates expected impact,
- and reports outcomes.

Current user interfaces:
- Primary CLI (Python core).
- Optional Textual TUI (Python core) via the tui extra.
- FastAPI endpoints for programmatic integration.
- TypeScript orchestrator agent for control-plane automation.

## 2) High-Level Architecture

Monorepo components:
- agent/: TypeScript control-plane orchestration (OpenClaw-facing runner, heartbeats, phase execution).
- core/: Python execution-plane engine (analysis, research, mapping, patch generation, validation, API, CLI, TUI).
- convex/: persistence/state integration for lifecycle tracking and approvals.
- docker/: local multi-service compose setup.
- test_repos/: safe benchmark/demo repos (for example nanoGPT).
- vendor/openclaw/: vendored integration dependency; treat as third-party code.

End-to-end phase model (conceptual):
1. Repo Intelligence
2. Research Intelligence
3. Mapping
4. Patch Generation
5. Validation
6. Report

## 3) Source-of-Truth Entrypoints

Core Python:
- CLI entry: core/src/scholardevclaw/cli.py
- API server: core/src/scholardevclaw/api/server.py
- Shared workflow layer: core/src/scholardevclaw/application/pipeline.py
- TUI app: core/src/scholardevclaw/tui/app.py

Agent TypeScript:
- Process entry: agent/src/index.ts
- Orchestrator: agent/src/orchestrator.ts
- Python subprocess bridge: agent/src/bridges/python-subprocess.ts
- Python HTTP bridge: agent/src/bridges/python-http.ts

State integration:
- agent/src/api/convex.ts
- convex/integrations.ts
- convex/integrations-mutations.ts

## 4) Current CLI/TUI Reality (Important)

CLI subcommands currently include:
- analyze
- search
- suggest
- map
- generate
- validate
- integrate
- specs
- demo
- tui

TUI status (implemented):
- wizard-style workflow launcher for analyze/suggest/search/specs/map/generate/validate/integrate,
- action-aware input enabling/disabling,
- live log streaming from pipeline callbacks,
- run status feedback,
- run history pane with quick rerun by run id,
- agent launcher controls (start/stop + log stream) for the TypeScript agent process.

TUI installation is optional:
- pip install -e ".[tui]"

## 5) Coding Standards and Design Rules

### TypeScript (agent/)
- Strict ESM + .js import suffixes in TS files.
- Keep phases explicit and typed; favor structured phase result objects.
- Preserve bridge contracts (subprocess/HTTP payload shapes).

### Python (core/)
- Prefer dataclasses and explicit dictionaries between stages.
- Keep functions narrow and composable; avoid hidden side effects.
- Use the application/pipeline seam for cross-interface reuse (CLI/TUI/API).
- Keep error handling explicit with actionable messages.

### Quality settings
- Source of truth: core/pyproject.toml
  - Python >= 3.10
  - Ruff configured
  - Mypy configured
  - pytest configured with src on pythonpath

## 6) Build, Run, and Test Commands

### Core setup
- cd core
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -e ".[arxiv,ml,dev,tui]"

### Core runtime
- scholardevclaw --help
- scholardevclaw tui
- uvicorn scholardevclaw.api.server:app --reload

### Core tests
- cd core && pytest
- Focused: cd core && pytest tests/unit/test_pipeline.py -q

### Agent setup
- cd agent && bun install
- cd agent && bun run build
- cd agent && bun run test
- cd agent && bun run dev

### Optional local stack
- docker compose -f docker/docker-compose.yml up -d

## 7) Modification Strategy for Agents

When implementing features/fixes:
- Prefer targeted edits over broad rewrites.
- Preserve existing interfaces unless task explicitly requires contract changes.
- Reuse pipeline.py for behavior shared by CLI and TUI.
- Keep TypeScript orchestration boundaries intact (do not collapse control-plane into core).
- Do not change vendor/openclaw unless explicitly requested.

When touching UI behavior (TUI):
- Keep interactions predictable and keyboard-friendly.
- Maintain action-specific input relevance.
- Stream progress/logs for long operations.
- Keep history and rerun semantics deterministic.

## 8) API and Integration Contracts

Known bridge surfaces:
- Agent -> Core subprocess: structured JSON-like payload exchange.
- Agent -> Core HTTP: FastAPI endpoints in core API server.

If changing payload schema or endpoint behavior:
- update both bridge layer and server/client expectations,
- keep backward-compatible fields where possible,
- document migration notes in commit/PR text.

## 9) Security and Safety

Secrets policy:
- Never hardcode tokens, keys, or credentials.
- Use environment variables (for example GITHUB_TOKEN, ANTHROPIC_API_KEY, CONVEX_URL).

Execution boundaries:
- Treat repository paths and subprocess calls as untrusted inputs.
- Validate paths before file operations and process execution.
- Keep retries bounded and observable.
- Keep approval gates before destructive operations and PR automation.

## 10) Git and Branching Discipline

Recommended workflow:
- Work in feature branches; avoid direct long-lived development on main.
- Keep commits scoped and descriptive.
- Push frequently after passing quick validation.

Typical branch names:
- feature/<topic>
- integration/<paper-name>
- fix/<bug>

## 11) Quick File Map by Concern

Repo analysis:
- core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py

Research extraction/search:
- core/src/scholardevclaw/research_intelligence/extractor.py
- core/src/scholardevclaw/research_intelligence/web_research.py

Mapping and patch generation:
- core/src/scholardevclaw/mapping/engine.py
- core/src/scholardevclaw/patch_generation/generator.py

Validation:
- core/src/scholardevclaw/validation/runner.py

Unified workflows:
- core/src/scholardevclaw/application/pipeline.py

Interfaces:
- core/src/scholardevclaw/cli.py
- core/src/scholardevclaw/tui/app.py
- core/src/scholardevclaw/api/server.py

## 12) Agent Checklist Before Finishing a Task

Before finalizing changes:
1. Run targeted tests for changed area.
2. Run at least syntax/compile checks for touched modules.
3. Smoke-check CLI/TUI imports when UI/entrypoint changes are involved.
4. Verify git status is clean after commit/push.
5. Summarize exactly what changed and where.

## 13) What Good Contributions Look Like Here

High-quality changes in this repo usually:
- improve shared seams (pipeline) instead of duplicating logic,
- make autonomous behavior more observable (status/log/history),
- keep control-plane and execution-plane responsibilities clean,
- avoid speculative complexity,
- and preserve developer trust through predictable outputs and careful validation.
