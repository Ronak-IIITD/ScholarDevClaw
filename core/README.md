# ScholarDevClaw

**Autonomous ML Research Integration Engine**

Analyzes repositories, finds research improvements from arXiv papers, maps them onto your codebase, generates validated patches, and reports outcomes.

## Installation

```bash
pip install scholardevclaw
```

With optional extras:

```bash
# With arXiv search
pip install scholardevclaw[arxiv]

# With ML/PyTorch support
pip install scholardevclaw[ml]

# With TUI interface
pip install scholardevclaw[tui]

# Everything
pip install scholardevclaw[all]
```

## Quick Start

```bash
# Analyze a repository
scholardevclaw analyze /path/to/your/repo

# Search for relevant research papers
scholardevclaw search "attention mechanism optimization"

# Get research-based improvement suggestions
scholardevclaw suggest /path/to/your/repo

# Run the full pipeline demo on nanoGPT
scholardevclaw demo
```

## Features

- **6-Language AST Analysis**: Python, JavaScript, TypeScript, Go, Rust, Java via tree-sitter
- **16 Built-in Research Specs**: RMSNorm, FlashAttention, SwiGLU, RoPE, GQA, and more
- **15 Code Templates + 10+ CST Transformers**: Production-quality patch generation via libcst
- **Real Subprocess Benchmarks**: Actual timing and memory measurements (no fake metrics)
- **6-Tier Mapping Engine**: Exact, fuzzy, import, text-scan, legacy, and LLM semantic matching
- **18 LLM Providers**: Anthropic, OpenAI, Groq, Mistral, DeepSeek, Cohere, and more
- **arXiv Integration**: HTTP fallback when `arxiv` package not installed
- **FastAPI Server**: 6 REST endpoints with API key auth, CORS, rate limiting
- **Interactive TUI**: Textual-based wizard with live logs, run history, agent launcher

## Architecture

```
scholardevclaw/
├── repo_intelligence/     # Tree-sitter AST extraction
├── research_intelligence/ # Paper search, extraction, web research
├── mapping/               # Code-to-research matching
├── patch_generation/      # Template + CST transformer patches
├── validation/            # Subprocess benchmark runner
├── application/           # Pipeline orchestration
├── llm/                   # Multi-provider LLM client
├── api/                   # FastAPI server
├── tui/                   # Textual TUI
├── agent/                 # Smart agent engine
└── cli.py                 # CLI entry point
```

## Development

```bash
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw/core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,arxiv,ml,tui]"
pytest tests/ -x -q  # 851 tests
```

## License

MIT
