# ScholarDevClaw v2 🚀

**Autonomous Research-to-Code AI Agent for All Developers**

ScholarDevClaw is an open-source research-to-code AI agent that analyzes your codebase, discovers relevant ML papers and implementations, and generates validated code improvements automatically.
If you search for **ScholarDevClaw**, this repository and landing page are the canonical project sources.

## 🌟 What's New in v2

- **Multi-Language Support**: Python, JavaScript/TypeScript, Go, Rust, Java, and more
- **Real-Time Research**: arXiv API integration for live paper search
- **Web Research**: GitHub, Papers with Code, Stack Overflow search
- **Smart Matching**: Automatically matches research to your code patterns
- **Flexible Deployment**: Self-hosted or cloud

## 🚀 Quick Start

```bash
# One-line install (recommended)
curl -fsSL https://ronak-iiitd.github.io/ScholarDevClaw/install.sh | bash

# Or via pip
pip install scholardevclaw

# Or from source
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw/core
pip install -e ".[arxiv,ml,tui]"

# Test installation
scholardevclaw --help
scholardevclaw demo
```

## 🧭 Core + TUI Getting Started (Step-by-Step)

This section is the fastest way to run ScholarDevClaw on your own project using either CLI commands or the interactive TUI.

### 1) One-time setup

```bash
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw/core

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml,tui,dev]"
```

### 2) Verify install

```bash
scholardevclaw --help
scholardevclaw specs --list
```

### 3) Launch the TUI

From `core/` with the virtual environment active:

```bash
scholardevclaw tui
```

What you get in TUI right now:
- Workflow wizard for `analyze`, `suggest`, `search`, `specs`, `map`, `generate`, `validate`, `integrate`
- Live execution logs while a workflow runs
- Run status (`Running`, `Done`, `Failed`)
- Run history with run ID, action, duration, and quick rerun
- Run details inspector for any history run (inputs, outputs, artifacts, errors, timing)
- Artifact viewer to inspect generated files and transformation summaries from history runs
- Validation scorecard highlights (pass/fail summary, speedup/loss deltas) in run details
- Payload schema metadata and compatibility warnings across integrate/validate flows
- Deterministic schema policy (major mismatch blocks compatibility; minor/patch drift emits migration notes)
- Agent orchestration run persistence with resumable phase checkpoints and heartbeat recovery
- Agent phase guardrails: protected-branch blocking in patch phase + deterministic retry/backoff budgeting in validation phase
- Mandatory approval gates on low-confidence mapping or risky validation deltas with persisted guardrail reasons

### 4) Example TUI workflow (real project)

1. Open TUI with `scholardevclaw tui`.
2. Set **Repository path** to your target repo (for example `/home/user/my-model-repo`).
3. Select **Analyze repository** and run.
4. Switch to **Suggest improvements** and run.
5. Pick a spec (for example `rmsnorm`) and run **Map spec to repository**.
6. Run **Generate patch artifacts** and optionally set an output directory.
7. Run **Validate repository**.
8. Use **Run History** pane to rerun any previous workflow quickly.

### 5) Example CLI workflow (same result, command-first)

```bash
# go to your repository
cd /path/to/your/repo

# analyze project structure
scholardevclaw analyze .

# get improvement suggestions
scholardevclaw suggest .

# map a paper spec to your code locations
scholardevclaw map . rmsnorm

# generate patch artifacts
scholardevclaw generate . rmsnorm --output-dir ./integration-patch

# validate
scholardevclaw validate .
```

### 6) Search workflow examples

```bash
# local + arXiv + web search
scholardevclaw search "layer normalization" --arxiv --web

# category exploration
scholardevclaw specs --categories
```

### 7) Common issues

- If `scholardevclaw tui` fails, ensure you installed with `.[tui]` and activated `.venv`.
- If arXiv/web search is slow, start with local flows (`analyze`, `suggest`, `specs`) first.
- Use `--output-json` on CLI commands when you need machine-readable output.

## 📖 Usage

All commands work through the unified `scholardevclaw` CLI:

```bash
# See all commands
scholardevclaw --help

# Analyze any codebase (auto-detects language)
scholardevclaw analyze /path/to/your/repo

# Search for research papers
scholardevclaw search "layer normalization" --arxiv --web

# Get AI-powered improvement suggestions
scholardevclaw suggest /path/to/your/repo

# Full integration workflow
scholardevclaw integrate /path/to/your/repo rmsnorm

# List available paper specs
scholardevclaw specs --list

# Run demo
scholardevclaw demo

# Optional: install and launch TUI
pip install -e ".[tui]"
scholardevclaw tui
```

## 🛠️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ScholarDevClaw v2                        │
├─────────────────────────────────────────────────────────────┤
│  Unified CLI Interface                                      │
│  • One command for all operations                          │
│  • Multi-language support                                   │
│  • Web + arXiv research                                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Code        │   │ Research     │   │  Code        │
│  Analyzer    │   │ Engine       │   │  Generator   │
│ (tree-sitter)│   │(arXiv + Web)│   │ (multi-lang) │
└───────────────┘   └───────────────┘   └───────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Validation + Patch Artifacts                │
│  • Mapping confidence • Generated files • Outcome checks    │
└─────────────────────────────────────────────────────────────┘
```

## 🌍 Supported Languages

| Language | Status | Frameworks |
|----------|--------|------------|
| Python | ✅ Full | PyTorch, TensorFlow, Django, Flask, FastAPI |
| JavaScript | ✅ Full | Express, React, Vue, Angular |
| TypeScript | ✅ Full | Next.js, NestJS |
| Go | ✅ Basic | Gin, Echo |
| Rust | ✅ Basic | Actix, Rocket |
| Java | ✅ Basic | Spring, Maven |
| C/C++ | 🚧 Planned | - |
| Ruby | 🚧 Planned | Rails |

## 📚 Research Sources

- **arXiv**: 2.4M+ papers in CS, ML, Physics
- **GitHub**: Search for implementations
- **Papers with Code**: ML implementations
- **Stack Overflow**: Technical discussions
- **Technical Blogs**: Coming soon

## 🏗️ Deployment Options

### Self-Hosted (Local)

```bash
# Development stack (core-api + agent + convex)
cp docker/.env.example docker/.env
bash scripts/runbook.sh dev up
bash scripts/runbook.sh dev health

# Production stack (nginx + web-ui + core-api + agent + monitoring)
cp docker/.env.example docker/.env
# set required vars in docker/.env:
#   SCHOLARDEVCLAW_API_AUTH_KEY
#   SCHOLARDEVCLAW_ALLOWED_REPO_DIRS
bash scripts/runbook.sh prod preflight
bash scripts/runbook.sh prod up
bash scripts/runbook.sh prod health

# Systemd service
sudo systemctl start scholardevclaw

# PM2
pm2 start ecosystem.config.js
```

### Cloud

- **Fly.io**: `fly deploy`
- **Railway**: `railway up`
- **AWS EC2**: See docs/DEPLOYMENT.md

## 📖 Documentation

- [Quick Start Guide](demo.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Overview](AGENTS.md)
- [API Reference](docs/API.md)

## 🎯 Example Workflows

### Improve a Python ML Project

```bash
# 1. Analyze
cd my-ml-project
scholardevclaw analyze .

# 2. Get suggestions
scholardevclaw suggest .

# 3. Integrate RMSNorm
scholardevclaw integrate . rmsnorm
```

### Improve a JavaScript Backend

```bash
# Analyze Express.js app
scholardevclaw analyze ./my-api

# Search for caching papers
scholardevclaw search "api caching" --web

# Get suggestions
scholardevclaw suggest ./my-api
```

## 🔧 Configuration

```bash
# Environment variables
cat > .env << 'EOF'
CORE_API_URL=http://localhost:8000
GITHUB_TOKEN=your_github_token
ANTHROPIC_API_KEY=your_anthropic_key

# Production hardening for core API
SCHOLARDEVCLAW_API_AUTH_KEY=replace-with-strong-random-key
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/absolute/path/to/allowed/repos
SCHOLARDEVCLAW_ENABLE_HSTS=true
EOF
```

## 🧪 Testing

```bash
# Run tests
cd core
pytest

# Run demo
scholardevclaw demo

# Test with your repo
scholardevclaw analyze /path/to/repo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a PR

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- tree-sitter for multi-language parsing
- arXiv for paper access
- Papers with Code for implementations

## 📞 Support

- GitHub Issues: https://github.com/Ronak-IIITD/ScholarDevClaw/issues
- Discussions: https://github.com/Ronak-IIITD/ScholarDevClaw/discussions

---

**Built with ❤️ for researchers and developers**

Transform your codebase with cutting-edge research automatically. 🚀
