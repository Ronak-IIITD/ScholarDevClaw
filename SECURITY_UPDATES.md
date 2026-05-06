# ScholarDevClaw Security and Reliability Audit

Date: 2026-05-06
Scope: full repository source audit across `agent/`, `core/`, `web/`, `convex/`, `docker/`, `scripts/`, `landing/`, `docs/`, `vendor/`, and `install/`

This document records the repository audit findings captured on 2026-05-06. The focus is real breakage, unsafe behavior, contract drift, and claims that do not match the implementation.

## 🔴 CRITICAL

1. Patch application can corrupt files by overwriting them with truncated 500-character fragments.
   - `core/src/scholardevclaw/patch_generation/generator.py:1358-1362` truncates `original` and `modified` to `[:500]`.
   - `core/src/scholardevclaw/application/pipeline.py:423-456` applies `modified` as the full replacement text inside `_apply_patch_to_copy()`.
   - `core/src/scholardevclaw/api/server.py:959-969` truncates patch response payloads again at the API layer.
   - Result: generated transformations can destroy files instead of editing them.

2. Validation is not trustworthy and can report success after failing tests.
   - `core/src/scholardevclaw/validation/runner.py:321-326` only aborts when `not test_result.passed and test_result.error`.
   - `core/src/scholardevclaw/validation/runner.py:452-467` normal pytest failures return `passed=False` with no `error`, so validation continues.
   - `core/tests/unit/test_validation_runner.py:367-385` explicitly codifies this behavior.
   - `core/src/scholardevclaw/validation/runner.py:491-598` runs a synthetic benchmark detached from the actual repo/patch.
   - `core/src/scholardevclaw/validation/runner.py:445-450` skips pytest entirely in Docker sandbox mode and marks tests as passed.

3. The Paper to Code web flow is placeholder code, not a working product path.
   - `web/src/pages/PaperToCodePage.tsx:259` labels the pipeline as simulated.
   - `web/src/pages/PaperToCodePage.tsx:276-280` posts to `${API_BASE}/api/from-paper`.
   - `web/src/pages/PaperToCodePage.tsx:261` and `web/src/pages/PaperToCodePage.tsx:332-345` only keep and send `file.name`, not the file.
   - No `/api/from-paper` route exists in the FastAPI server.

4. The documented one-line install path is broken.
   - `README.md:18-21` recommends `curl .../install.sh | bash`.
   - `landing/install.sh:139-147` runs `pip install .` and then installs the repo root from GitHub.
   - The only Python package metadata is `core/pyproject.toml`.
   - Verified behavior:
     - `python -m pip install --dry-run .` at repo root fails because the root is not a Python project.
     - `python -m pip install --dry-run git+<repo-root>` fails for the same reason.

## 🟠 HIGH

1. The TypeScript subprocess bridge breaks phase contracts.
   - `agent/src/bridges/python-subprocess.ts:288-299` collapses the mapping output into a spec string and drops the real mapping payload during patch generation.
   - `agent/src/bridges/python-subprocess.ts:306-309` ignores the patch payload during validation.
   - `agent/src/phases/phase4-patch.ts:13` and `agent/src/phases/phase5-validation.ts:14` assume those payloads survive between phases.

2. Subprocess patch generation can infer the spec name as `"[object Object]"`.
   - `agent/src/bridges/python-subprocess.ts:291` forwards `mappingRecord?.research_spec`.
   - `agent/src/bridges/python-subprocess.ts:331-345` converts arbitrary values with `String(...)`.

3. Approval gates are not actually mandatory when Convex is absent.
   - `agent/src/orchestrator.ts:649-653` auto-approves after one second.
   - `README.md:77` claims mandatory approval gates.

4. The production web dashboard cannot talk to a secured core API over HTTP.
   - `core/src/scholardevclaw/api/server.py:148-166` enforces auth on non-exempt routes.
   - `docker/docker-compose.prod.yml:36-39` requires `SCHOLARDEVCLAW_API_AUTH_KEY`.
   - `web/src/lib/api.ts:31-35` does not attach `Authorization` on HTTP requests.
   - WebSocket token handling exists, but the normal HTTP client path is unauthenticated.

5. The `generate` command documentation does not match the implemented CLI.
   - `README.md:106` and `demo.md:128` show `generate . rmsnorm`.
   - `core/src/scholardevclaw/cli.py:3746-3776` only supports repo/spec generation behind `--use-specs`.
   - `core/src/scholardevclaw/cli.py:835-910` defaults to the plan/understanding artifact path instead.

6. OpenClaw integration is mostly narrative.
   - `agent/src/utils/config.ts:16-20` only reads env vars.
   - `vendor/openclaw/` is effectively empty.
   - `git submodule status` fails with `no submodule mapping found in .gitmodules for path 'vendor/openclaw'`.

7. Dashboard pipeline state is single-process memory and only supports one active run.
   - `core/src/scholardevclaw/api/routes/dashboard.py:137-143` stores state in module globals.
   - `core/src/scholardevclaw/api/routes/dashboard.py:223-226` rejects concurrent runs with HTTP 409.

## 🟡 MEDIUM

1. The README claim of GitHub + Papers with Code + Stack Overflow research is false on the main execution path.
   - `core/src/scholardevclaw/research_intelligence/web_research.py:209-230` never calls Stack Overflow in `search_all_sources()`.
   - `core/src/scholardevclaw/application/pipeline.py:1012-1042` only serializes GitHub and Papers with Code results.

2. Multi-language support is overstated.
   - `core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py:1-8` claims 10+ languages.
   - `core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py:16-77` configures six languages.
   - `core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py:140-142` leaves `_setup_parsers()` empty.
   - `LanguageStats` counters exist but are not meaningfully populated before being returned.

3. External web-research integrations have timeouts but no retry, backoff, or rate-limit handling.
   - `core/src/scholardevclaw/research_intelligence/web_research.py:107-123` performs a one-shot GET.
   - `core/src/scholardevclaw/research_intelligence/web_research.py:258-263`, `:336-339`, and `:377-380` collapse non-200 responses to empty results.

4. Agent health checks are largely synthetic.
   - `agent/src/utils/health.ts:116-123` hardcodes event-loop lag to zero.
   - `agent/src/utils/health.ts:134-140` always reports the Python bridge as healthy.

5. Convex phase-result schemas are too loose to catch contract drift.
   - `convex/schema.ts:23-28` uses `v.any()` for all phase result payloads.

6. The tests miss several high-risk paths.
   - `core/tests/unit/test_pipeline.py:317-346` swaps out the real analyzer/extractor/generator/validator stack with fakes.
   - `core/tests/e2e/test_validate.py:24-58` mostly checks response shape or skip conditions.
   - `core/tests/unit/test_patch_generator.py:488-504` does not verify full-file integrity after transformations.
   - There is no focused `python-subprocess` bridge test file covering the broken handoff path.

## 🔵 LOW

1. Deployment claims overreach the actual repository contents.
   - `README.md:225-235` references systemd, PM2, Fly.io, and Railway deployment options.
   - Matching configuration files such as `ecosystem.config.*`, `*.service`, `fly.toml`, and `railway.json` are not present in the repository.

2. `install/install.sh` has the same root-directory packaging bug as the landing installer.
   - `install/install.sh:67-73` runs `pip3 install -e ".[tui,security]"` from the repo root.

## 📊 CLAIMS AUDIT

| Claimed Feature | Reality | Evidence |
| --- | --- | --- |
| Multi-language support (Python, JS/TS, Go, Rust, Java) | Partial | `core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py:16-77`, `:140-142` |
| Real-time arXiv API integration | Partial | Real arXiv search helpers exist in `core/src/scholardevclaw/research_intelligence/extractor.py:728-768`; broader extraction still depends on LLM-assisted paths later in the module |
| GitHub + Papers with Code + Stack Overflow web research | Partial | `core/src/scholardevclaw/research_intelligence/web_research.py:209-230` does not invoke Stack Overflow in the main aggregate search path |
| Smart matching of research to code patterns | Real | `core/src/scholardevclaw/mapping/engine.py` implements exact, fuzzy, import-based, and LLM-assisted mapping tiers |
| TUI with run history and artifact inspection | Real | `core/src/scholardevclaw/tui/app.py:545-627`, `:995-1026` |
| Resumable checkpoints and heartbeat recovery | Partial | `agent/src/utils/run-store.ts:56-139`, `agent/src/orchestrator.ts:458-567`; dashboard state is still in-memory only |
| Mandatory approval gates | Partial | `agent/src/orchestrator.ts:649-653` auto-approves when Convex is unavailable |
| Docker deployment | Real | `docker/docker-compose.yml`, `docker/docker-compose.prod.yml`, `scripts/runbook.sh` |
| Systemd and PM2 deployment | Fiction | Claimed in `README.md:225-235`; supporting config files not found |
| Fly.io and Railway deployment | Fiction | Claimed in `README.md:225-235`; supporting deploy files not found |

## 🎯 TOP 5 HIGHEST-IMPACT FIXES

1. Remove transformation truncation and stop using preview snippets as full-file patch payloads.
2. Rewrite validation so it executes the actual patched repository copy, fails on any test failure, and runs pytest inside Docker mode.
3. Replace the Paper to Code placeholder flow with a real backend route, real file upload, and payloads that match the API contract.
4. Fix or remove subprocess bridge mode until mapping and patch payloads survive phase handoff correctly.
5. Repair the install and documentation contract: one-line installer, `generate` examples, and unsupported deployment claims.

## Notes

- This was a source audit plus targeted command verification.
- The full test suite was not run as part of this audit record.
