# ScholarDevClaw — Product Updates & Roadmap

## 0) Last Updated + Changelog

**Last updated:** 2026-02-27

### 2026-02-27
- **TUI Clipboard & Image Support** (`core/src/scholardevclaw/tui/clipboard.py` — NEW, 315 lines):
  - `ClipboardManager`: Cross-platform clipboard support (macOS/Linux/Windows)
  - Read/write text from system clipboard
  - Read images from clipboard (PNG, JPEG)
  - Save pasted images to `~/.scholardevclaw/attached_images/`
  - `ImageInputHandler` for handling image attachments
  - Drag-and-drop file support with format validation
  - Image path context for AI agent integration
  - Copy AI output to clipboard
  - Auto-cleanup of old attachments

- **TUI Integration**:
  - Lazy loading of clipboard module to avoid textual dependency issues
  - Exports available via `scholardevclaw.tui`

- **New Tests** (21 new tests):
  - `test_tui_clipboard.py` — 21 tests: clipboard operations, image handling, cross-platform

- **Total: 803 tests passing** (was 782)

### 2026-02-26
- **Encryption at Rest** (`core/src/scholardevclaw/auth/encryption.py` — NEW, 191 lines):
  - `EncryptionManager`: Fernet-based encryption with PBKDF2-HMAC-SHA256 key derivation (600k iterations)
  - Salt persistence, enable/disable/unlock, change_password with re-encryption
  - `FallbackEncryptionManager`: No-op fallback when `cryptography` is not installed
  - `get_encryption_manager()` factory for graceful degradation
  - Store integration: `enable_encryption()`, `unlock_encryption()`, `disable_encryption()`, `is_encryption_enabled()`

- **Rate Limiting** (`core/src/scholardevclaw/auth/rate_limit.py` — NEW, 224 lines):
  - `RateLimiter`: Per-key sliding window rate limiter
  - `RateLimitConfig`: Configurable per-minute/hour/day limits with burst support
  - `KeyUsageStats` and `UsageRecord` for tracking
  - Persistent usage data with auto-recovery from corrupted files
  - Store integration: `set_rate_limit()`, `get_key_usage()`, `get_api_key_with_rate_check()`

- **Import/Export** (`core/src/scholardevclaw/auth/import_export.py` — NEW, 291 lines):
  - `AuthExporter`: Export to JSON (full/redacted) and .env format, file output support
  - `AuthImporter`: Import from JSON, .env, and 1Password CSV formats
  - Auto-detect provider from key format (Anthropic/OpenAI/GitHub/Google/Custom)
  - `ImportResult` for tracking imported/skipped/error counts
  - Store integration: `export_json()`, `export_env()`, `import_keys_from_env()`, `import_keys_from_json()`, `import_keys_from_1password()`

- **Multi-Profile / Workspace Support** (in `store.py`):
  - `save_profile_as()`, `load_profile()`, `list_profiles()`, `delete_profile()`
  - Profile isolation with file-per-workspace storage under `~/.scholardevclaw/profiles/`
  - File permission hardening (chmod 600) on profile files

- **Key Expiration Alerts** (in `store.py`):
  - `get_expiring_keys(within_days)`: Find keys expiring within a time window
  - `deactivate_expired_keys()`: Auto-deactivate past-expiry keys
  - `set_key_expiry(key_id, expires_at)`: Set/update expiry with ISO 8601 validation

- **File Permission Hardening** (in `store.py`):
  - `_harden_file()` sets chmod 600 on auth.json, .env, and profile files

- **Rewritten `store.py`** (818 lines): Complete rewrite integrating encryption, rate limiting, multi-profile, expiration, import/export, file hardening. Fixed duplicate `__init__` bug.

- **Rewritten `cli.py`** (705 lines): Added CLI commands: `rotate`, `audit`, `export`, `import`, `encrypt`, `profiles`, `usage`, `expiry`

- **Updated `__init__.py`**: Exports for RateLimiter, RateLimitConfig, KeyUsageStats, AuthExporter, AuthImporter, ImportResult

- **Updated `pyproject.toml`**: Added `crypto = ["cryptography>=42.0.0"]` optional dependency

- **Bug fix: `disable_encryption()` password verification** — `disable_encryption("wrongpass")` previously succeeded because `unlock()` only derives a key without verification. Fixed to attempt actual decryption of auth data before disabling.

- **New Tests** (220 tests, all passing):
  - `test_auth_encryption.py` — 43 tests: key derivation, EncryptionManager, FallbackEncryptionManager, store integration
  - `test_auth_rate_limit.py` — 32 tests: config, records, stats, limiter, store integration
  - `test_auth_import_export.py` — 46 tests: export JSON/env, import JSON/env/1Password CSV, provider detection, store roundtrip
  - `test_auth_profiles.py` — 28 tests: multi-profile CRUD, key expiration, file permissions
  - `test_auth_cli_extended.py` — 71 tests: rotate, audit, export, import, encrypt, profiles, usage, expiry CLI commands

- **Total: 688 tests passing** (was 145 auth tests + existing suite)

### 2026-02-26 (advanced features)
- **OAuth 2.0 Flows** (`core/src/scholardevclaw/auth/oauth.py` — NEW, 410 lines):
  - `OAuthProvider` base class with authorization code + PKCE flow
  - `GoogleOAuthProvider` and `GitHubOAuthProvider` implementations
  - `OAuthToken` with auto-refresh capability
  - `OAuthTokenStore` for secure token persistence
  - `OAuthManager` for high-level flow management
  - Token exchange, refresh, user info retrieval

- **Hardware Key Support** (`core/src/scholardevclaw/auth/hardware_keys.py` — NEW, 271 lines):
  - `HardwareKeyManager` for YubiKey/PKCS#11
  - YubiKey PIV slot detection and key generation
  - Sign/encrypt operations with hardware keys
  - PKCS#11 module support for external HSMs
  - Hardware key reference storage in config

- **Team / Multi-User Support** (`core/src/scholardevclaw/auth/team.py` — NEW, 389 lines):
  - `Team`, `TeamMember`, `TeamInvite` classes
  - Role-based access: ADMIN, DEVELOPER, VIEWER
  - Granular permissions: READ_KEYS, WRITE_KEYS, ROTATE_KEYS, DELETE_KEYS, MANAGE_USERS, etc.
  - `TeamStore` for team CRUD operations
  - `TeamAccessControl` for permission checking
  - Invite system with expiry

- **API Usage Analytics** (`core/src/scholardevclaw/auth/analytics.py` — NEW, 415 lines):
  - `UsageTracker` for recording API calls with cost estimation
  - Provider pricing (Anthropic, OpenAI, Google, GitHub)
  - `UsageAnalytics` with daily/provider/endpoint breakdown
  - `UsageDashboard` for summary views
  - Cost alerts and budget warnings

- **Secret Rotation Automation** (`core/src/scholardevclaw/auth/rotation.py` — NEW, 397 lines):
  - `RotationPolicy` for automated rotation scheduling
  - `AnthropicRotationProvider` and `OpenAIRotationProvider` implementations
  - `RotationScheduler` with policy management
  - Auto-rotation for due keys
  - Rotation history logging

- **Key Request/Approval Workflow** (`core/src/scholardevclaw/auth/approval.py` — NEW, 371 lines):
  - `KeyRequest` with types: NEW_KEY, KEY_ROTATION, KEY_RENEWAL
  - `ApprovalWorkflow` for request lifecycle
  - Approve/reject with notifications
  - Request validation and rate limiting
  - `RequestValidator` for policy enforcement

- **New Tests** (94 new tests):
  - `test_auth_oauth.py` — 18 tests: OAuth flow, token store, provider implementations
  - `test_auth_hardware_keys.py` — 9 tests: YubiKey detection, PKCS#11, key management
  - `test_auth_team.py` — 18 tests: team CRUD, roles, permissions, invites
  - `test_auth_analytics.py` — 17 tests: usage tracking, cost estimation, dashboards
  - `test_auth_rotation.py` — 16 tests: rotation policies, scheduler, providers
  - `test_auth_approval.py` — 16 tests: workflow, approvals, validation

- **Total: 782 tests passing** (was 688)

### 2026-02-25 (continued - session 2)
- Added **Key Rotation Feature** (`core/src/scholardevclaw/auth/types.py`, `store.py`):
  - `KeyRotationEntry` class to track rotation history
  - `rotate_api_key()` method with reason tracking
  - `get_rotation_history()` to retrieve rotation events
  - `get_keys_needing_rotation()` to find old keys
  - `mark_key_for_rotation()` to flag keys for rotation

- Added **Key Scopes/Permissions** (`core/src/scholardevclaw/auth/types.py`):
  - `KeyScope` enum: READ_ONLY, READ_WRITE, ADMIN, CUSTOM
  - `set_key_scope()` to change key permissions
  - Scope persisted with API key

- Added **Audit Logging** (`core/src/scholardevclaw/auth/audit.py`):
  - `AuditLogger` class for tracking all auth events
  - Events: KEY_ACCESSED, KEY_ADDED, KEY_REMOVED, KEY_ROTATED, LOGIN_SUCCESS, LOGIN_FAILED, etc.
  - Fingerprint tracking (key not stored in logs)
  - `get_events()` with filtering by key_id and event_type
  - `get_failed_logins()` for security monitoring
  - `clear_old_events()` for retention management
  - Integrated with AuthStore for automatic logging

- Added **11 Audit Tests** (`core/tests/unit/test_auth_audit.py`):
  - Logger initialization and event logging
  - Event filtering and retrieval
  - Failed login tracking
  - Store integration tests

- **Total: 145 tests passing** (was 123)

### 2026-02-25 (continued)
- Added **Enhanced Auth Module**:
  - Key format validation for Anthropic, OpenAI, GitHub, Google
  - Email validation for profiles
  - Key name validation
  - Provider-specific key format hints
  - API key fingerprinting (SHA256 hash for identification)
  - Optional validation on add_api_key() and create_profile()

- Added **More Edge Case Tests** (`core/tests/unit/test_auth.py`): 17 new tests
  - Key format validation tests (Anthropic, OpenAI, GitHub)
  - Email validation tests
  - Key name validation tests
  - Key fingerprint tests
  - Invalid expiry format handling

- Added **Agent Integration Tests** (`core/tests/e2e/test_auth_agent.py`): 13 tests
  - Auth + analyze workflow
  - Environment variable precedence
  - Multiple providers
  - Key activation/deactivation
  - Profile with subscription tiers
  - Auth backup/restore
  - Key metadata persistence
  - Provider switching
  - File permissions
  - Large number of keys

- Added **Security Tests** (`core/tests/unit/test_auth_security.py`): 12 tests
  - API key masking in output
  - Fingerprint doesn't reveal key
  - Logout removes all sensitive data
  - Env override key not stored
  - Random key/profile IDs
  - Invalid JSON handling
  - Path traversal prevention
  - Large key handling

- Added **Extended CLI Tests** (`core/tests/unit/test_auth_cli.py`): 10 new tests
  - Multiple keys in JSON output
  - Status with profile display
  - Remove last key behavior
  - Default after remove
  - Empty key handling
  - Special characters in key names
  - Subscription tier display
  - Empty list output
  - Setup with all providers

- **Total: 123 tests passing** (was 72)

### 2026-02-25
- Added **Auth Module Tests** (robust test coverage):
  - **Unit Tests** (`core/tests/unit/test_auth.py`): 35 tests
    - APIKey tests: generation, masking, validation, expiry, serialization
    - UserProfile tests: creation, update, serialization
    - AuthConfig tests: active key selection, provider fallback
    - AuthStore tests: CRUD operations, env override, profile management, logout
    - Edge cases: empty keys, unicode names, key uniqueness, invalid JSON recovery
  - **CLI Tests** (`core/tests/unit/test_auth_cli.py`): 27 tests
    - Status, list, add, remove, default commands
    - Login, logout (with confirm + force)
    - Setup wizard (env key detection, manual key entry)
    - JSON output parsing
    - Error handling: invalid provider, missing keys
  - **Integration Tests** (`core/tests/e2e/test_auth.py`): 10 tests
    - Full workflow: add key → check status → list → remove
    - Profile workflow: create, update, retrieve
    - Environment variable override
    - Multiple providers
    - Key rotation
    - Auth persistence across store instances
    - Auth file format validation
    - CLI + store integration
  - **Total: 72 tests passing**

### 2026-02-23
- Added **Rollback Support** (P1 feature):
  - **Rollback Module** (`core/src/scholardevclaw/rollback/`):
    - `RollbackStore`: Persistent storage for rollback snapshots (~/.scholardevclaw/rollbacks/)
    - `RollbackManager`: Manages snapshot creation, file change recording, and rollback execution
    - `RollbackSnapshot`: Captures git state, file states, and change history
    - Support for file modifications, creations, deletions, branch creations, and commit tracking
  - **CLI Commands**:
    - `scholardevclaw rollback list <repo>`: List all rollback snapshots
    - `scholardevclaw rollback show <repo> --snapshot-id <id>`: View snapshot details
    - `scholardevclaw rollback run <repo> [--snapshot-id <id>] [--force]`: Execute rollback
    - `scholardevclaw rollback status <repo>`: Check current rollback state
    - `scholardevclaw rollback delete <repo> --snapshot-id <id>`: Remove snapshot
  - **Integration with Integrate Workflow**:
    - Automatic snapshot creation before integration changes
    - Snapshot marked as 'applied' after successful integration
    - Rollback snapshot ID included in integration result payload
  - **Unit Tests** (24 tests): All passing

- Added **GitHub App for PR Reviews** (P1 feature):
  - **GitHub App Module** (`core/src/scholardevclaw/github_app/`):
    - `GitHubAppClient`: JWT-based authentication, installation tokens, full GitHub API support
    - `WebhookHandler`: Processes PR events, runs integrations, posts results
    - `create_app()` / `create_router()`: FastAPI integration
    - `manifest.json`: GitHub App manifest for easy setup
  - **Features**:
    - Automatic analysis on PR open/reopen/update
    - Check Run status updates (queued → in_progress → completed)
    - PR comments with validation results, speedup metrics, change summaries
    - Support for custom integration handlers
    - Configurable auto-apply and approval requirements
  - **CLI Commands**:
    - `scholardevclaw github-app setup`: Show setup instructions
    - `scholardevclaw github-app manifest [--server-url URL]`: Generate app manifest
    - `scholardevclaw github-app server [--port PORT]`: Start webhook server
    - `scholardevclaw github-app status`: Check configuration status
    - `scholardevclaw github-app test-webhook`: Test webhook signature
  - **Environment Variables**:
    - `GITHUB_APP_ID`: App ID from GitHub
    - `GITHUB_APP_PRIVATE_KEY`: Path to private key PEM file
    - `GITHUB_APP_WEBHOOK_SECRET`: Webhook secret
    - `GITHUB_APP_REPOS`: Comma-separated allowed repos
    - `GITHUB_APP_AUTO_APPLY`: Auto-apply safe patches
    - `GITHUB_APP_REQUIRE_APPROVAL`: Require approval before apply
  - **Unit Tests** (16 tests): All passing

- Added **Security Scanning** (P1 feature):
  - **Security Module** (`core/src/scholardevclaw/security/`):
    - `BanditScanner`: Python-specific security scanning
    - `SemgrepScanner`: Multi-language security scanning
    - `SecurityScanner`: Unified scanner combining both tools
  - **Features**:
    - Bandit: Detects Python-specific security issues (eval, assert, hardcoded credentials, etc.)
    - Semgrep: Multi-language scanning (JavaScript, Java, Go, Rust, etc.)
    - CWE mapping for findings
    - Severity levels: HIGH, MEDIUM, LOW, INFO
    - Configurable exclude patterns
    - JSON output with detailed findings
  - **CLI Commands**:
    - `scholardevclaw security <repo>`: Run full security scan
    - `scholardevclaw security <repo> check`: Check tool availability
    - `scholardevclaw security <repo> --tools bandit,semgrep`: Run specific tools
    - `scholardevclaw security <repo> -v`: Verbose output with findings
  - **Optional Dependencies**:
    - Install with: `pip install -e ".[security]"`
    - Installs: bandit>=1.7.0, semgrep>=1.0.0
  - **Unit Tests** (11 tests): All passing

- Added **Interactive Agent Mode**:
  - **Agent Module** (`core/src/scholardevclaw/agent/`):
    - `AgentEngine`: Core processing with natural language understanding
    - `AgentREPL`: Interactive terminal with Rich UI
    - Session management with context/history
  - **Features**:
    - Natural language command parsing (e.g., "analyze my-project", "search normalization")
    - Smart context awareness - remembers current repository
    - Rich terminal UI with colors, tables, panels
    - Streaming output for long-running operations
    - Command suggestions and next steps
    - Error recovery with helpful messages
  - **CLI Commands**:
    - `scholardevclaw agent` - Start interactive mode
    - `scholardevclaw agent "analyze ./repo"` - Run single command
    - `scholardevclaw agent "search attention"` - Search papers
  - **Quick Start**:
    ```
    $ scholardevclaw agent
    ScholarDevClaw > analyze ./my-project
    ScholarDevClaw > integrate rmsnorm
    ScholarDevClaw > suggest improvements
    ```
  - **Installer Script** (`install/install.sh`):
    - One-command install: `curl -L scholardevclaw.com/install | bash`
    - Auto-detects OS and Python
    - Creates launcher script
    - Adds to PATH

- Added **Real-time Streaming Agent**:
  - `StreamingAgentEngine`: Event-based streaming for real-time UI updates
  - `StreamEvent`: Progress, output, error, suggestion events
  - All commands now stream in real-time:
    - `[progress] 📊 Analyzing...`
    - `[output] ✅ Found 5 languages`
    - `[suggestion] 💡 Try: integrate rmsnorm`
  - Async generator-based streaming for non-blocking output
  - Better UX with immediate feedback

- Added **Local Auth (API Key) Support** (MVP):
  - **Auth Module** (`core/src/scholardevclaw/auth/`):
    - `AuthStore`: Local auth storage (`~/.scholardevclaw/auth.json`)
    - `AuthConfig`, `APIKey`, `UserProfile`, `AuthStatus`
  - **Features**:
    - Bring-your-own API keys (Anthropic, OpenAI, GitHub, Custom)
    - Local profile storage and default key selection
    - Environment override: `SCHOLARDEVCLAW_API_KEY`
  - **CLI Commands**:
    - `scholardevclaw auth setup` - Interactive setup wizard
    - `scholardevclaw auth login` - Add API key
    - `scholardevclaw auth status` - Show auth status
    - `scholardevclaw auth list` - List keys
    - `scholardevclaw auth remove --key-id <id>` - Remove key
    - `scholardevclaw auth default --key-id <id>` - Set default key

### 2026-02-20
- Added production-ready features:
  - **Prometheus Metrics Collection** (41 tests):
    - Counter, Gauge, Histogram metric types
    - MetricsRegistry for managing all metrics
    - Pre-defined metrics for requests, integrations, patches, validations, errors, workflows
    - FastAPI middleware for automatic HTTP request tracking
    - /metrics endpoint for Prometheus scraping
    - Path normalization for high-cardinality routes
  - **Structured Logging** (16 tests):
    - JSON output mode for production
    - Trace ID, Request ID, User ID context
    - LogContext for timing operations
    - Standard field injection
  - **API Rate Limiting** (19 tests):
    - Token bucket algorithm for burst handling
    - Sliding window algorithm for precise limiting
    - FastAPI middleware with IP-based rate limiting
    - Per-minute, per-hour, burst limits
  - **OpenAPI/Swagger Documentation**:
    - Custom OpenAPI schema with examples
    - Tagged endpoints with descriptions
    - /docs/json and /docs/version routes
  - **Connection Pooling** (24 tests):
    - Generic ConnectionPool with acquire/release pattern
    - Connection validation and idle timeout
    - HTTPConnectionPool singleton with httpx
    - AsyncHTTPConnectionPool for async clients
    - Connection stats tracking
  - **CLI Progress Bars** (18 tests):
    - ProgressBar with ETA, rate, and percentage
    - ProgressConfig for customization
    - progress_iter for iterable progress
    - Spinner for indeterminate operations
    - MultiProgress for parallel bars
  - **Standardized Error Codes** (27 tests):
    - ErrorCategory enum (VAL, REP, RES, MAP, PAT, VRU, INT, NET, CFG, SYS, PRM, RSC, TIM, RTL, PLG, WRK)
    - ErrorSeverity enum (info, warning, error, critical)
    - ErrorCode dataclass with code, message, http_status, remediation
    - ErrorCodes class with 40+ predefined error codes
    - AppException for structured error handling
  - **Distributed Request Tracing** (35 tests):
    - Span class for timing operations
    - TraceContext for managing spans and baggage
    - @span decorator for function tracing
    - Tracer class with export callbacks
    - Header extraction/injection for distributed traces
    - TraceMiddleware for FastAPI/Starlette
  - **Production Docker Setup**:
    - Multi-stage Dockerfiles for core and agent
    - Non-root user, health checks
    - Production docker-compose with resource limits
    - Nginx reverse proxy with SSL, rate limiting
    - Prometheus + Grafana for monitoring

### 2026-02-17
- Added reliability and stability layer:
  - Health check system (memory, disk, environment, filesystem)
  - Liveness and readiness probes
  - Circuit breaker pattern for external calls
  - Input validation layer (path, spec, string, enum, range validators)
  - Graceful shutdown with cleanup handlers
  - Resource manager for cleanup
  - Health API endpoints (/health, /health/live, /health/ready)
- Completed Workflow DAG Engine for agent orchestration:
  - DAG-based execution with parallel node support
  - Node dependencies and topological sorting
  - Conditional branching (on_success, on_failure, always)
  - Event streaming for real-time progress
  - Retry with exponential backoff
  - Cycle detection
  - Phase nodes factory functions (analyze, research, mapping, patch, validation, report)
  - Integration with Planner, Critic, Experiment nodes
  - Workflow persistence store for resumability
  - Workflow templates library (6 templates)
  - Metrics collector for performance monitoring
  - Webhook notifier for external integrations
  - Dynamic workflow builder with quick workflow creation
- Completed performance optimizations and agent tools:
  - Added parallel processing utilities (parallel_map, LazyFileScanner, ParallelGit)
  - Added improved error handling (ErrorContext, RetryConfig, OperationTimer)
  - Colored console logging with timestamps
  - Added AgentTools class for TypeScript agent with better tool wrappers
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
  - Transform safety (balanced brackets check)
  - Severity classification (error/warning)
- Completed Context Engine with Long-horizon Memory:
  - New `scholardevclaw context` CLI with subcommands:
    - init: Initialize project context from repo analysis
    - history: Show integration history
    - summary: Show project context summary
    - recommend: Get AI recommendation based on past runs
    - set: Set user preferences
    - clear: Clear project memory
    - list: List tracked projects
  - Persistent memory store (~/.scholardevclaw/context/)
  - Integration history tracking
  - Agent Brain for context-aware recommendations
  - Auto-approval based on confidence and past success
  - User preferences learning
- Completed Experiment Loop Mode:
  - New `scholardevclaw experiment` CLI command
  - Variant generation with different parameters
  - A/B comparison with validation
  - Results ranking by score
  - Metrics tracking (speedup, loss change)
  - Best variant recommendation
- Completed Plugin System:
  - New `scholardevclaw plugin` CLI with subcommands:
    - list: List discovered and loaded plugins
    - load: Load a plugin
    - unload: Unload a plugin
    - analyze: Run analyzer plugin
    - validate: Run validator plugin
    - scaffold: Create plugin scaffold
    - info: Show plugin info
  - PluginManager for plugin discovery and loading
  - Plugin interfaces (AnalyzerPlugin, SpecProviderPlugin, ValidatorPlugin)
  - Built-in plugins: javalang, jsts, rustlang, security
  - Plugin scaffold generator
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
- ~~Long-horizon memory with Context Engine:~~ ✅ Implemented
  - Project context storage
  - Integration history tracking
  - Agent Brain for recommendations
  - User preferences learning
- ~~Experiment loop mode:~~ ✅ Implemented
  - Variant generation with different parameters
  - A/B comparison with validation
  - Results ranking by score
  - Best variant recommendation
- Next: experiment loop mode for hypothesis testing.

---

## 8) Product Positioning Statement

ScholarDevClaw is becoming a **research-to-engineering operating system**:
- It helps researchers operationalize ideas,
- helps developers ship safe improvements,
- and helps teams trust AI-assisted code evolution with measurable outcomes.

---

## 9) Future Implementations

### A) Enhanced Research Intelligence
- **Multi-source paper extraction**: Add support for PubMed, IEEE Xplore, ACL, Google Scholar
- **Research similarity search**: Find related papers that could combine well together
- **Better spec extraction**: Fine-tuned LLM for code-oriented research extraction
- **Citation graph analysis**: Understand paper dependencies, successors, and related work
- **arXiv API integration**: Direct paper fetching and metadata extraction

### B) Advanced Mapping & Code Understanding
- **Code embeddings**: Use semantic similarity for better mapping targets
- **Complex refactoring support**: Handle larger-scale architectural changes
- **Dependency graph analysis**: Understand import graphs for safer changes
- **Call graph generation**: Map function/method call relationships
- **Cross-file refactoring**: Coordinated changes across multiple files

### C) Enhanced Validation
- **Property-based testing**: Integrate with Hypothesis for generative testing
- **Fuzzing integration**: AFL/libFuzzer integration for robustness testing
- **Security scanning**: Integrate with Bandit, Semgrep, CodeQL
- **Performance profiling**: CPU, memory, GPU profiling integration
- **Mutation testing**: Verify test quality with mutation testing
- **Benchmark suite**: Standardized benchmark repo collection

### D) User Experience
- **Web UI**: Browser-based dashboard (beyond terminal TUI)
- **VSCode extension**: Inline suggestions, diff viewing, quick actions
- **GitHub App**: PR comments, automated reviews, status checks
- **GitHub Actions integration**: CI/CD integration for automatic validation
- **JetBrains plugin**: IntelliJ/PyCharm integration

### E) Notifications & Integrations
- **Slack webhooks**: Real-time status notifications
- **Discord webhooks**: Community/team notifications
- **Jira integration**: Create tickets for manual review items
- **Email notifications**: For approval requests, completed runs
- **PagerDuty integration**: Alert on critical failures

### F) Enterprise Features
- **Audit logging**: Track all operations for compliance
- **SSO/SAML**: Enterprise authentication (Okta, Auth0, etc.)
- **Custom spec repositories**: Private spec stores for organizations
- **Team dashboards**: Aggregate metrics across team members
- **Role-based access control**: Granular permissions
- **Data residency options**: Control where data is stored

### G) Advanced Automation
- **Scheduled runs**: Cron-like integration scheduling
- **Webhook triggers**: Trigger runs on git push, PR creation
- **Auto-apply safe patches**: Low-risk changes auto-merged with config
- **Batch processing**: Process multiple repos/specs in parallel
- ~~**Rollback support**: One-click revert of applied changes~~ ✅ Implemented 2026-02-23

### H) AI/ML Enhancements
- **Fine-tuned models**: Domain-specific models for ML research
- **Retrieval-augmented generation**: Better context for code generation
- **Multi-model support**: Allow switching between different LLMs
- **Confidence calibration**: Better uncertainty quantification
- **Active learning**: Learn from user corrections

### I) Quick Wins (Lower Effort, High Impact)

| Feature | Effort | Impact | Priority | Status |
|---------|--------|--------|----------|--------|
| GitHub App for PR reviews | Medium | High | P1 | ✅ Done |
| Bandit/Semgrep security integration | Low | High | P1 | ✅ Done |
| Slack/Discord notifications | Low | Medium | P2 | |
| Rollback support | Medium | High | P1 | ✅ Done |
| Benchmark suite | Medium | High | P2 | |
| VSCode extension | Medium | High | P2 | |
| Web UI dashboard | High | High | P3 | |
| SSO/SAML auth | High | Medium | P3 | |
