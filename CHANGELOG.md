# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- End-to-end `demo` command: auto-clones nanoGPT, runs full pipeline across multiple specs, writes patch artifacts, summary report
- PyPI distribution: renamed to `scholardevclaw`, SPDX license, PEP 561 typing, clean sdist/wheel
- CI/CD pipeline with GitHub Actions (lint, type-check, test, coverage, Docker build)
- Auto-release workflow (PyPI publish, GitHub Release, Docker image push on tag)
- End-to-end `demo` command for full pipeline demonstration
- CHANGELOG.md following Keep a Changelog format
- Web dashboard (React + TypeScript + Vite) with real-time pipeline visualization
- Plugin ecosystem with hook system, registry, and community extensions
- Multi-repo support for cross-repository analysis and knowledge transfer

## [0.1.0] - 2026-03-06

### Added
- **Phase 1:** Auth restructure with 18-provider support + unified LLM HTTP client
- **Phase 2:** Fixed runtime bugs in pipeline (p.summary vs p.abstract, p.published.isoformat)
- **Phase 3:** Real tree-sitter AST extraction for 6 languages (Python, JS, TS, Go, Rust, Java)
- **Phase 4:** Wired LLM module into pipeline for real extraction, analysis, and web research
- **Phase 5:** Real subprocess-based validation benchmarks (replaced fake metrics)
- **Phase 6:** Dynamic research knowledge base with 16 specs, arXiv HTTP fallback, LLM discovery
- **Phase 7:** Expanded patch generator to 15 templates, 10+ CST transformers, LLM synthesis fallback
- **Phase 8:** Rewritten mapping engine with 6-tier matching (exact, fuzzy, import, text-scan, legacy, LLM)
- Smart Agent Engine with query classification, budget management, and execution routing
- Advanced shell capabilities with OS detection, terminal mode, slash commands
- Comprehensive security audit fixing 77+ vulnerabilities
- TUI wizard with workflow launcher, live logs, run history, agent launcher
- TypeScript orchestrator with 6-phase pipeline execution
- FastAPI server with 6 REST endpoints, API key auth, CORS, rate limiting
- Docker infrastructure with dev/prod compose, nginx, prometheus, grafana
- 851 tests passing across unit and integration suites

### Security
- Comprehensive security audit: timing-safe auth, path traversal prevention, SSRF protection
- Input validation, rate limiting, secret management, subprocess sandboxing
- Nginx hardening: HSTS, CSP, server_tokens off, body limits, IP restrictions

[Unreleased]: https://github.com/Ronak-IIITD/ScholarDevClaw/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ronak-IIITD/ScholarDevClaw/releases/tag/v0.1.0
