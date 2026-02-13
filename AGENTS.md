# Project Guidelines

Scope: This file governs the repository root. For code under `agent/workspace/`, follow `agent/workspace/AGENTS.md` (closest file wins).

## Code Style
- TypeScript (`agent/`) uses strict ESM and `.js` import suffixes from TS files; follow patterns in `agent/src/orchestrator.ts`, `agent/src/phases/types.ts`, and `agent/src/bridges/python-subprocess.ts`.
- Keep phase modules small and explicit: each phase returns structured success/error payloads (`PhaseResult<T>` style).
- Python (`core/`) favors dataclasses and explicit dictionaries between pipeline stages; mirror `core/src/scholardevclaw/mapping/engine.py` and `core/src/scholardevclaw/patch_generation/generator.py`.
- Python quality settings are source of truth in `core/pyproject.toml` (`ruff`, `mypy`, Python >=3.10, line length 100).

## Architecture
- `agent/`: TypeScript orchestrator and phase runners (control-plane).
- `core/`: Python analysis/research/mapping/patch/validation engine (execution-plane).
- `convex/`: integration lifecycle persistence (status, phase outputs, retries, approvals).
- `docker/`: local multi-service runtime (`docker-compose.yml`).
- `test_repos/`: benchmark/demo repositories (for safe integration tests).
- End-to-end flow is phase-based (1→6): repo intelligence → research intelligence → mapping → patch generation → validation → report.

## Build and Test
- Agent install/build/test:
  - `cd agent && bun install`
  - `cd agent && bun run build`
  - `cd agent && bun run test`
  - `cd agent && bun run dev`
- Core setup/test:
  - `cd core && python3 -m venv .venv && source .venv/bin/activate`
  - `cd core && pip install -e ".[arxiv,ml,dev]"`
  - `cd core && scholardevclaw --help`
  - `cd core && pytest`
- Local stack (optional): `docker compose -f docker/docker-compose.yml up -d`

## Project Conventions
- Do not edit `main` directly; use feature/integration branches (integration branch pattern: `integration/<paper-name>`).
- Keep autonomous behavior transparent: explicit phase logs, confidence reporting, and bounded retries.
- Prefer targeted edits over full-file rewrites; preserve public interfaces across agent/core boundaries.
- Match documented CLI in `core/src/scholardevclaw/cli.py` (current subcommands include `analyze`, `search`, `suggest`, `integrate`, `specs`, `demo`).

## Integration Points
- Agent→Core subprocess bridge: `agent/src/bridges/python-subprocess.ts`.
- Agent→Core HTTP bridge and endpoints: `agent/src/bridges/python-http.ts` and `core/src/scholardevclaw/api/server.py`.
- Agent↔Convex state sync: `agent/src/api/convex.ts`, `convex/integrations.ts`, `convex/integrations-mutations.ts`.
- Optional GitHub PR integration: `agent/src/api/github.ts`.

## Security
- Secrets are env-based (`GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, `CONVEX_URL`, OpenClaw token). Never hardcode credentials.
- Treat repo path inputs and subprocess execution as high-risk boundaries; validate before invoking autonomous runs.
- Keep approval gates before PR creation and avoid unbounded retries.
- Avoid privileged Docker mounts (for example Docker socket) unless explicitly required by the task.
