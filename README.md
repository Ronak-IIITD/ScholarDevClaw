# ScholarDevClaw

Autonomous Research-to-Production ML Integration Engine

## Overview

ScholarDevClaw is an autonomous AI agent that integrates cutting-edge ML research papers into existing PyTorch codebases. It analyzes repository architecture, extracts paper specifications, maps changes, generates patches, and validates with benchmarks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 OpenClaw Gateway (TypeScript)                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ScholarDevClaw Agent                                │   │
│  │  • 6-phase orchestration                            │   │
│  │  • Convex state management                          │   │
│  │  • GitHub PR creation                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python ML Core                               │
│  ┌──────────────┬──────────────┬───────────────────────┐  │
│  │ Repo Intel   │ Research     │ Mapping Engine         │  │
│  │ (libcst)     │ (specs)      │ (compatibility)        │  │
│  └──────────────┴──────────────┴───────────────────────┘  │
│  ┌──────────────┬──────────────┬───────────────────────┐  │
│  │ Patch Gen    │ Validation   │ CLI                    │  │
│  │ (libcst)     │ (pytest)     │ (Typer)                │  │
│  └──────────────┴──────────────┴───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/scholardevclaw/scholardevclaw.git
cd scholardevclaw

# Install Python core
cd core
pip install -e .

# Install Node.js agent
cd ../agent
npm install
```

## Quick Start

```bash
# Clone test repository (nanoGPT)
git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt

# Run demo
scholardevclaw demo
```

## CLI Commands

```bash
# Analyze repository
scholardevclaw analyze test_repos/nanogpt

# List available paper specs
scholardevclaw specs --list

# Extract research spec
scholardevclaw extract rmsnorm

# Map changes to repository
scholardevclaw map test_repos/nanogpt rmsnorm

# Generate patch
scholardevclaw generate test_repos/nanogpt rmsnorm

# Validate changes
scholardevclaw validate test_repos/nanogpt
```

## Available Paper Specifications

- **RMSNorm** (1910.07467) - Root Mean Square Layer Normalization
- **SwiGLU** - Swish-Gated Linear Unit
- **FlashAttention** (2205.14135) - IO-Aware Exact Attention
- **RoPE** (2104.09864) - Rotary Position Embedding
- **GQA** (2305.13245) - Grouped Query Attention

## Project Structure

```
scholardevclaw/
├── agent/                    # TypeScript/OpenClaw agent
│   ├── src/
│   │   ├── orchestrator.ts   # Main workflow controller
│   │   ├── phases/           # 6 phase executors
│   │   ├── bridges/          # Python subprocess + HTTP bridges
│   │   └── api/              # Convex & GitHub clients
│   └── workspace/            # OpenClaw workspace files
│
├── core/                     # Python ML engines
│   └── src/scholardewclaw/
│       ├── repo_intelligence/   # AST parsing (libcst)
│       ├── research_intelligence/  # Paper specs
│       ├── mapping/             # Architecture mapping
│       ├── patch_generation/    # Code generation
│       ├── validation/          # Test & benchmark runner
│       └── cli.py              # Command-line interface
│
├── convex/                   # Database schema & functions
├── docker/                   # Dockerfiles & compose
└── test_repos/               # Test repositories
```

## Docker Deployment

```bash
cd docker
docker-compose up
```

## Safety Features

- **Never writes to main branch** - Always creates feature branches
- **User approval required** - Stops before PR creation
- **Max 2 retries** - Prevents infinite loops
- **Sandboxed execution** - Runs in Docker containers

## License

MIT

## Contributing

See CONTRIBUTING.md for guidelines.
