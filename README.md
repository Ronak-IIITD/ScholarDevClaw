# ScholarDevClaw / nanogpt-opt

Detect, apply, and revert **10 proven transformer optimizations** as validated, reversible patches — with an honest validator that never reports a pass for work it did not do.

![nanogpt-opt demo](demo-nanogpt.gif)

## The four commands

`nanogpt-opt` is a lean CLI over a frozen scope of 10 research-backed specs:

`rmsnorm` · `preln_transformer` · `qknorm` · `swiglu` · `flashattention2` · `grouped_query_attention` · `rope` · `alibi` · `lion` · `cosine_warmup`

| Command | What it does |
|---------|--------------|
| `analyze <repo>` | Offline scan (tree-sitter, no LLM): Python files, model classes, applicable specs. |
| `suggest <repo> [--spec]` | Maps specs to concrete `file:line` targets with confidence + strategy. |
| `apply <repo> [--spec] [--write\|--revert]` | **Dry-run by default** (unified diff, zero writes). `--write` applies atomically per spec and records `.nanogpt-opt/manifest.json`; `--revert` restores **byte-identical** originals and removes created files. Re-running `--write` on a patched repo is a clean no-op. |
| `validate <repo>` | Honest scorecard: `compile` / `tests` / `benchmark`, each `pass`, `fail`, or `skipped`. A stage that never ran is `skipped` — never a pass. Exit 1 on any fail. |

Every command accepts `--json`.

## Quick start

```bash
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git && cd ScholarDevClaw
pip install -e core

nanogpt-opt analyze /path/to/repo
nanogpt-opt suggest /path/to/repo --spec rmsnorm
nanogpt-opt apply   /path/to/repo --spec rmsnorm            # dry-run: unified diff
nanogpt-opt apply   /path/to/repo --spec rmsnorm --write    # apply + manifest
nanogpt-opt validate /path/to/repo                          # honest pass/fail/skipped
nanogpt-opt apply   /path/to/repo --revert                  # byte-identical restore
```

No GPU required; `analyze` needs no network and no LLM.

## Demo (54s)

[`demo-nanogpt.tape`](demo-nanogpt.tape) renders [`demo-nanogpt.mp4`](demo-nanogpt.mp4) / `demo-nanogpt.gif` — the full loop against a live clone of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT):

1. `analyze` → 10/10 applicable specs
2. `suggest --spec rmsnorm` → target at `model.py:18`, confidence 75
3. `apply` dry-run → unified diff, zero writes
4. `apply --write` → `model.py` transformed + `rmsnorm.py` created + manifest
5. second `--write` → `nothing to apply` (idempotent)
6. `validate` → compile pass, tests skipped, benchmark fail → **`exit=1`** (honest)
7. `git status` → dirty; `--revert` → restored; `git status` → **clean**

Render it yourself: `vhs demo-nanogpt.tape` (requires [vhs](https://github.com/charmbracelet/vhs), `ttyd`, `ffmpeg`).

## Proof on external repositories

Every roundtrip below was verified by SHA-256 of all `.py` files before/after: **byte-identical revert**, manifest fully cleaned, idempotent second `--write`.

| External repo | Files | Specs applied | Result |
|---------------|-------|---------------|--------|
| `karpathy/nanoGPT` | 5 | all 10 | 10 files changed + 8 created; revert byte-identical (12.6s) |
| `lucidrains/vit-pytorch` | 86 | `rmsnorm` (15 targets) | 20 changed + 5 created; revert byte-identical (177s) |
| `lucidrains/x-transformers` | 43 | `rmsnorm` (12 targets) | 12 changed + 5 created; revert byte-identical (203s) |

`validate` on each reported honestly: compilation passed; tests/benchmark reported `fail` or `skipped` for real environmental reasons (missing optional deps, `bench.py` runtime failure) — never a fabricated pass.

## Honesty guarantees

- **Skipped ≠ pass.** Tests without test files (or without pytest) and benchmarks without scripts report `skipped`; the summary counts them separately; exit code reflects only real failures.
- **No fake benchmark numbers.** A benchmark case scores `1.0` only when it actually imported and executed (`import_ok`); AST-identical but unexecuted candidates score `0.75` (`ast_matched`). The committed no-torch baseline (`core/benchmarks/benchmark_report.json`, aggregate `0.725`) gates CI at `current ≥ baseline − 0.1`.
- **Dry-run default.** `apply` writes nothing until `--write`; created files are create-only — an existing path is never clobbered and never recorded for revert.
- **Byte-identical revert.** Transformed files are restored from manifest originals; `--revert` walks specs backwards with a no-clobber guard.

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│ nanogpt-opt (nano_cli.py) — analyze/suggest/apply/validate│
├──────────────────────────────────────────────────────────┤
│ Python core pipeline (shared by scholardevclaw CLI/TUI/API)│
│  • repo intelligence (tree-sitter analyzer)               │
│  • mapping (alias matcher, confidence ≥ 70)               │
│  • patch generation (libcst transformers, reversible)     │
│  • validation + benchmark harness (honest scoring)        │
└──────────────────────────────────────────────────────────┘
```

The legacy `scholardevclaw` CLI, Textual TUI, FastAPI server, and TypeScript orchestrator remain in the repo and are covered by CI.

## Docs

- [Quick Start Guide (legacy surfaces)](demo.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Notes](ARCHITECTURE.md)
- [Agent Handbook](AGENTS.md)

## Contributing and community

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Use [GitHub Discussions](https://github.com/Ronak-IIITD/ScholarDevClaw/discussions) for questions and show-and-tell, and [GitHub Issues](https://github.com/Ronak-IIITD/ScholarDevClaw/issues) for bugs or feature requests.
