# ScholarDevClaw — Product Updates & Roadmap

## 0) Last Updated + Changelog

**Last updated:** 2026-02-17

### 2026-02-17
- Completed Planner mode for multi-spec migration strategies:
  - New `scholardevclaw planner` CLI command
  - Analyzes repo and suggests multiple compatible specs
  - Dependency ordering (normalization -> activation -> attention)
  - Combined impact estimation (speedup, memory, benefits)
  - Category filtering support
  - Added to pipeline with `run_planner()` and `run_multi_integrate()`
- Completed Critic mode for patch verification:
  - New `scholardevclaw critic` CLI command
  - Syntax validation for generated code
  - Import validation (detect missing imports)
  - Anti-pattern detection (range(len), bare except, global, etc.)
  - Security checks (eval/exec detection)
  - Transform safety (balanced brackets)
  - Severity classification (error/warning)
- Completed end-to-end regression suite for key workflows:
  - Created `tests/e2e/` directory with 46 e2e tests covering analyze, map, generate, validate, integrate, preflight, search, specs, and suggest workflows.
  - Tests run against real nanoGPT test repository and use pipeline functions directly.
  - All e2e tests passing (46 passed).
- Completed major agent orchestration hardening:
  - resumable run checkpoints + heartbeat recovery,
  - deterministic retry budgeting with backoff,
  - patch branch safety guardrails,
  - confidence/risk policy gates with persisted guardrail reasons.
- Added core payload-contract reliability upgrades:
  - schema metadata + compatibility checks,
  - deterministic compatibility policy matrix.
- Completed TUI/operator MVP+ quality improvements:
  - run details,
  - artifact viewer,
  - validation scorecard highlights.
- Added explicit approval decision ingestion:
  - Convex mutation to create approval records,
  - Convex query to list approvals for integration,
  - Agent Convex client updated with createApproval and listApprovals methods,
  - Local run store updated to persist approval records with new ApprovalRecord interface,
  - Orchestrator records approval decisions when received (approved/rejected with notes).

### 2026-02-16
- Strengthened preflight and integration safety guidance.
- Added normalized validation scorecards in shared pipeline outputs.

### 2026-02-15
- Baseline roadmap and product framing consolidated in `UPDATES.md`.

## 1) Overall Idea

ScholarDevClaw is a **research-driven programming AI assistant** designed to help both:
- **Researchers (especially AI/ML):** convert research ideas/papers into implementation-ready specs.
- **Developers/teams:** map those specs to real codebases, generate patch artifacts, validate outcomes, and iterate safely.

In short, the product bridges this gap:

**Research insight → Engineering action → Validated impact**

---

## 2) What the Product Does

At a high level, ScholarDevClaw:
1. Analyzes a target repository (multi-language awareness).
2. Extracts or searches research specifications.
3. Maps research changes to likely code targets.
4. Generates patch artifacts.
5. Runs validation and reports outcomes.

This enables a repeatable workflow for taking high-value research ideas into practical code changes.

---

## 3) Overall Structure (Current)

The project is organized as a monorepo with clear responsibilities:

- `core/` (Python): execution engine
  - CLI entrypoint (`scholardevclaw`)
  - Optional Textual TUI
  - FastAPI endpoints
  - Repo analysis, research extraction/search, mapping, patch generation, validation
  - Shared `application/pipeline.py` seam used by interfaces

- `agent/` (TypeScript): control-plane orchestration
  - Phase orchestration and lifecycle handling
  - Bridges to Python core (subprocess/HTTP)
  - Operational integration with state systems

- `convex/`: state and lifecycle persistence integration
- `docker/`: local multi-service orchestration
- `test_repos/`: sandbox/demo targets

Conceptual lifecycle:
1. Repo Intelligence
2. Research Intelligence
3. Mapping
4. Patch Generation
5. Validation
6. Report

---

## 4) Implemented Till Now (Snapshot)

### Core interfaces
- CLI with workflows including:
  - `analyze`, `search`, `suggest`, `map`, `generate`, `validate`, `integrate`, `specs`, `demo`, `tui`
- FastAPI service with core endpoints for analyze/extract/map/generate/validate.
- Optional TUI support installed via `.[tui]`.
- Payload schema metadata and compatibility checks added for integration/validation payloads.

### TUI (MVP+) implemented
- Wizard-style execution for key workflows.
- Action-aware input enable/disable behavior.
- Live log streaming from pipeline callbacks during runs.
- Run status feedback (`Running`, `Done`, `Failed`).
- Run history pane with:
  - run id,
  - action,
  - status,
  - duration,
  - quick rerun by run id.
- Run details view for history entries, including:
  - inputs (repo/spec/query/options),
  - key outputs (selected spec, mapping targets, validation state),
  - artifact hints (written/generated files),
  - error + timing summary.
- Artifact viewer in TUI run history for inspecting generated file content and transformation summaries.
- Validation scorecard highlights visible in run details.
- Payload compatibility warnings surfaced in run logs/details.
- Agent process launcher controls in TUI (start/stop + log stream).

### Shared execution seam
- `core/src/scholardevclaw/application/pipeline.py` now centralizes workflow execution.
- CLI/TUI reuse this seam to keep behavior consistent and reduce duplication.
- Validation scorecard normalization added in shared pipeline output.
- Preflight checks strengthened with actionable recommendations and strict `require_clean` enforcement.

### Agent orchestration engine (major progress)
- New `agent/src/index.ts` entrypoint supports `run`, `resume`, and heartbeat (`processPendingWork`) modes.
- Resumable orchestration implemented in `agent/src/orchestrator.ts` with persisted phase checkpoints.
- Local run persistence added via `agent/src/utils/run-store.ts` (with tests) for durable state recovery.
- Convex-backed resume path now reconstructs phase context/results and continues from next phase.
- Deterministic retry budgeting + exponential backoff added for phase failures (notably validation).
- Patch safety guardrails enforced in phase 4:
  - blocks protected branches (`main`, `master`, `develop`, etc.),
  - requires `integration/` prefix,
  - rejects empty/no-op patch outputs.
- Confidence/risk guardrail policy layer added (`agent/src/utils/guardrails.ts`):
  - mapping-confidence gates,
  - validation speedup/loss-drift gates,
  - mandatory approval holds when policies trigger.
- Guardrail reason persistence added across local snapshots and Convex integration status metadata.
- Explicit approval decision ingestion implemented:
  - Convex `createApproval` mutation to record approve/reject decisions with phase, notes, timestamp.
  - Convex `listApprovals` query to retrieve approval history for an integration.
  - Agent `ConvexClientWrapper` updated with `createApproval()` and `listApprovals()` methods.
  - Local `RunStore` enhanced with `ApprovalRecord` interface and `addApproval()`/`getApprovals()` methods.
  - Orchestrator now records approval decisions (approved/rejected) locally and in Convex when waiting for approval.

### Documentation updates completed
- Root `AGENTS.md` expanded into an operational handbook for agents/contributors.
- `README.md` updated with a detailed Core + TUI getting-started path and practical workflows.
- `README.md` now documents scorecards, schema compatibility, resumable orchestration, and guardrails.

---

## 5) Future Advancements (Toward World-Class Researcher + Dev AI)

Below is a prioritized roadmap to make ScholarDevClaw category-defining.

### A) Research-to-Code reliability
1. Strong schema/versioning for research specs and inter-phase payloads.
2. Better normalization of extracted specs (ambiguity handling + provenance fields).
3. Confidence calibration and explicit uncertainty reporting at each phase.
4. Cross-paper composition (combine multiple compatible techniques in one plan).

### B) Engineering safety and trust
1. Preflight safety checks before patch generation/integration:
   - dirty repo detection,
   - environment/dependency checks,
   - language/framework compatibility checks.
2. Dry-run mode for full integrate flow with no write/apply.
3. Revert/rollback support and deterministic recovery flow.
4. Policy guardrails (approval gates for risky transformations).

### C) TUI v2 (operator-grade experience)
1. Run details view (inputs, outputs, artifacts, errors, timing).
2. Rich artifact explorer for generated files and transformation summaries.
3. Cancellable jobs + progress indicators for long operations.
4. Better keyboard-first navigation and command palette.
5. Session persistence and resume behavior across restarts.

### D) Validation and benchmarking
1. Standard benchmark suite across representative repos.
2. Before/after scorecards:
   - correctness,
   - performance,
   - maintainability.
3. Automated regression checks for mapping and patch fidelity.
4. Golden test packs for paper-spec to patch consistency.

### E) Platform and ecosystem
1. Unified run records across CLI/TUI/API for observability.
2. Optional cloud execution mode with secure remote runners.
3. Team collaboration features (review queues, run sharing, approvals).
   - Shared run history across team members
   - Patch review workflows
   - Multi-user approval gates
4. Reusable plugin system for custom analyzers/spec providers/validators.
   - Custom analyzers for new languages/frameworks
   - External spec providers (beyond built-in papers)
   - Domain-specific validators

### F) AI-native enhancements
1. Planner mode: propose multi-step migration strategy before writing changes.
   - Propose coherent multi-paper migration strategies (e.g., RMSNorm + SwiGLU + RoPE together)
   - Show dependency ordering between changes
   - Estimate combined impact of multiple techniques
2. Critic mode: independent verifier to challenge generated patches.
   - Check for common bugs (typos, missing imports, syntax errors)
   - Validate against known anti-patterns
   - Pre-execution safety review
3. Long-horizon memory for project context and prior integration decisions.
   - Project context retention across sessions
   - Prior integration decisions history
   - Learned preferences (preferred specs, validation thresholds)
4. Experiment loop mode: generate hypotheses, run validation, rank outcomes.
   - Generate variants of patches with different parameters
   - Run A/B validation comparisons
   - Rank outcomes by metrics

---

## 6) North-Star Product Outcomes

To become world-class, ScholarDevClaw should optimize for:
- **Reliability:** high-confidence, reproducible outputs.
- **Transparency:** clear reasoning, logs, and artifacts per decision.
- **Safety:** controlled automation with reviewability.
- **Velocity:** faster research-to-production cycle for real teams.
- **Generality:** useful across AI/ML and broader software engineering domains.

---

## 7) Suggested Near-Term Execution Plan (Practical)

### Phase 1 (immediate)
- ~~Add run detail view in TUI.~~ ✅ Implemented.
- ~~Add integrate dry-run mode.~~ ✅ Implemented.
- ~~Add preflight checks and clear failure guidance.~~ ✅ Implemented.

### Phase 2
- ~~Add artifact browser + richer validation scorecards.~~ ✅ Implemented.
- ~~Add payload schema versioning + compatibility checks.~~ ✅ Implemented.
- ~~Add end-to-end regression suite for key workflows.~~ ✅ Implemented.
- Add planner mode for multi-spec migration strategies. 🚧 In Progress
- Add critic mode for patch verification.

### Phase 3
- Add long-horizon memory for project context.
- Add experiment loop mode for hypothesis testing.
- Add plugin system and team collaboration review layer.
- Add cloud-ready execution profiles.

### Agent/Orchestration (current active track)
- ~~Run persistence + resumable orchestration checkpoints.~~ ✅ Implemented.
- ~~Deterministic retry backoff + branch safety guardrails.~~ ✅ Implemented.
- ~~Policy-based approval gates with persisted guardrail reasons.~~ ✅ Implemented.
- ~~Explicit approval decision ingestion (approve/reject records).~~ ✅ Implemented.
- ~~End-to-end orchestration regression suite.~~ ✅ Implemented.
- ~~Planner mode:~~ ✅ Implemented.
- ~~Critic mode:~~ ✅ Implemented.
  - Syntax validation
  - Import validation
  - Anti-pattern detection
  - Security checks
- Next: long-horizon memory for project context.

---

## 8) Product Positioning Statement

ScholarDevClaw is becoming a **research-to-engineering operating system**:
- It helps researchers operationalize ideas,
- helps developers ship safe improvements,
- and helps teams trust AI-assisted code evolution with measurable outcomes.
