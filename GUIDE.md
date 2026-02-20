# ScholarDevClaw - Installation & Usage Guide

A step-by-step guide to install and use ScholarDevClaw, your autonomous research-to-code agent.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verify Installation](#3-verify-installation)
4. [Available Commands](#4-available-commands)
5. [Common Workflows](#5-common-workflows)
6. [Configuration](#6-configuration)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

Before installing ScholarDevClaw, ensure you have:

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| Python | 3.10+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| Git | Any | `git --version` |

**Optional:**
- Bun (for TypeScript agent): `bun --version`
- Docker (for local stack): `docker --version`

---

## 2. Installation

### Step 1: Fork the Repository

1. Go to https://github.com/Ronak-IIITD/ScholarDevClaw
2. Click the **Fork** button in the top right
3. Select your GitHub account

### Step 2: Clone Your Fork

```bash
# Replace YOUR_USERNAME with your GitHub username
git clone https://github.com/YOUR_USERNAME/ScholarDevClaw.git
cd ScholarDevClaw
```

### Step 3: Set Up Python Core

```bash
# Navigate to core directory
cd core

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,tui]"

# Return to root directory
cd ..
```

### Step 4: Set Up TypeScript Agent (Optional)

```bash
# Navigate to agent directory
cd agent

# Install dependencies
npm install

# Build the agent
npm run build

# Return to root directory
cd ..
```

---

## 3. Verify Installation

### Check CLI is Working

```bash
cd core
source .venv/bin/activate
scholardevclaw --help
```

You should see output like:
```
ScholarDevClaw - Autonomous Research-to-Code Agent v2.0
...
```

### Run Demo

```bash
scholardevclaw demo
```

This runs a quick demo with the nanoGPT test repository.

---

## 4. Available Commands

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analyze` | Analyze repository structure | `scholardevclaw analyze ./my-project` |
| `search` | Search for papers and implementations | `scholardevclaw search "layer normalization"` |
| `suggest` | Get AI-powered improvement suggestions | `scholardevclaw suggest ./my-project` |
| `specs` | List available paper specifications | `scholardevclaw specs --list` |
| `map` | Map spec to repository locations | `scholardevclaw map ./my-project rmsnorm` |
| `generate` | Generate patch artifacts | `scholardevclaw generate ./my-project rmsnorm` |
| `validate` | Run validation on repository | `scholardevclaw validate ./my-project` |
| `integrate` | Full integration workflow | `scholardevclaw integrate ./my-project rmsnorm` |

### Advanced Commands

| Command | Description | Example |
|---------|-------------|---------|
| `planner` | Plan multi-spec migration strategy | `scholardevclaw planner ./my-project` |
| `critic` | Verify generated patches | `scholardevclaw critic ./my-project rmsnorm` |
| `context` | Manage project context and memory | `scholardevclaw context ./my-project summary` |
| `experiment` | Run hypothesis experiments | `scholardevclaw experiment ./my-project rmsnorm` |
| `plugin` | Manage plugins | `scholardevclaw plugin list` |
| `tui` | Launch interactive terminal UI | `scholardevclaw tui` |
| `demo` | Run demo with nanoGPT | `scholardevclaw demo` |

---

## 5. Common Workflows

### Workflow 1: Analyze a Repository

```bash
# Basic analysis
scholardevclaw analyze ./my-project

# With JSON output
scholardevclaw analyze ./my-project --output-json
```

**Output:** Shows detected languages, frameworks, patterns, and structure.

---

### Workflow 2: Get Improvement Suggestions

```bash
scholardevclaw suggest ./my-project
```

**Output:** Lists applicable research improvements (e.g., RMSNorm, FlashAttention).

---

### Workflow 3: Full Integration

```bash
# Dry run first (recommended)
scholardevclaw integrate ./my-project rmsnorm --dry-run

# Full integration with output
scholardevclaw integrate ./my-project rmsnorm --output-dir ./patches
```

**What it does:**
1. Analyzes your repository
2. Extracts spec from paper
3. Maps spec to your code
4. Generates patches
5. Validates changes

---

### Workflow 4: Plan Multi-Spec Migration

```bash
# See recommended specs and execution order
scholardevclaw planner ./my-project

# Focus on specific categories
scholardevclaw planner ./my-project --categories normalization,attention
```

**Output:** Shows recommended specs, dependency order, and combined impact estimate.

---

### Workflow 5: Verify Patches Before Applying

```bash
scholardevclaw critic ./my-project rmsnorm
```

**Checks:**
- Syntax validation
- Import validation
- Anti-pattern detection
- Security checks

---

### Workflow 6: Run Experiments with Variants

```bash
scholardevclaw experiment ./my-project rmsnorm --variants 3
```

**Output:** Compares multiple parameter variants and ranks by score.

---

### Workflow 7: Use Interactive TUI

```bash
scholardevclaw tui
```

Launches an interactive terminal interface for:
- Running workflows
- Viewing logs
- Managing history

---

### Workflow 8: Manage Project Context

```bash
# Initialize context
scholardevclaw context ./my-project init

# View summary
scholardevclaw context ./my-project summary

# See history
scholardevclaw context ./my-project history

# Get AI recommendations
scholardevclaw context ./my-project recommend
```

---

## 6. Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: GitHub API access
GITHUB_TOKEN=your_github_token

# Optional: Anthropic API for AI features
ANTHROPIC_API_KEY=your_api_key

# Optional: Convex for persistence
CONVEX_URL=your_convex_url

# Python core API URL (default: http://localhost:8000)
CORE_API_URL=http://localhost:8000
```

### Configuration File

Settings are in `agent/src/utils/config.ts`:

```typescript
export const config = {
  execution: {
    defaultMode: 'step_approval',  // or 'autonomous'
    maxRetries: 2,
    benchmarkTimeout: 300,
    guardrails: {
      mappingMinConfidence: 75,
      validationMinSpeedup: 1.01,
    },
  },
};
```

---

## 7. Troubleshooting

### "Repository not found"

```bash
# Use absolute or relative path
scholardevclaw analyze /home/user/my-project
scholardevclaw analyze ../my-project
```

### "nanoGPT not found for demo"

```bash
# Clone test repository
cd test_repos
git clone https://github.com/karpathy/nanoGPT.git nanogpt
cd ..
scholardevclaw demo
```

### "Python module not found"

```bash
# Ensure virtual environment is activated
cd core
source .venv/bin/activate
pip install -e ".[dev,tui]"
```

### "TUI not working"

```bash
# Install TUI dependencies
pip install -e ".[tui]"
```

### "Permission denied"

```bash
# Make sure you have write permissions
chmod -R u+w ./my-project
```

---

## Quick Reference Card

```bash
# Setup
cd core && source .venv/bin/activate

# Most common commands
scholardevclaw analyze ./repo              # Analyze repo
scholardevclaw suggest ./repo              # Get suggestions
scholardevclaw specs --list                # List available specs
scholardevclaw integrate ./repo rmsnorm    # Apply spec
scholardevclaw planner ./repo              # Plan multi-spec
scholardevclaw critic ./repo rmsnorm       # Verify patches
scholardevclaw tui                         # Interactive UI

# Help
scholardevclaw --help
scholardevclaw <command> --help
```

---

## 8. Production Deployment

### Docker Deployment

For production deployment, use Docker:

```bash
# Development
docker compose -f docker/docker-compose.yml up -d

# Production (with monitoring)
docker compose -f docker/docker-compose.prod.yml up -d
```

### Production Features

ScholarDevClaw includes production-ready features:

| Feature | Description |
|---------|-------------|
| **Prometheus Metrics** | `/metrics` endpoint for monitoring |
| **Health Checks** | `/health`, `/health/live`, `/health/ready` |
| **Rate Limiting** | Built-in API rate limiting |
| **Request Tracing** | Distributed tracing with trace IDs |
| **Structured Logging** | JSON logging for production |
| **Error Codes** | Standardized error codes with remediation |

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /health/ready` | Readiness probe |
| `GET /health/live` | Liveness probe |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Swagger UI |
| `POST /repo/analyze` | Analyze repository |
| `POST /research/extract` | Extract research spec |
| `POST /mapping/map` | Map spec to code |
| `POST /patch/generate` | Generate patches |
| `POST /validation/run` | Run validation |

### Monitoring Stack

Production setup includes:
- **Prometheus** (port 9090) - Metrics collection
- **Grafana** (port 3000) - Dashboards and visualization
- **Nginx** (port 80/443) - Reverse proxy with SSL

```bash
# Access Grafana
open http://localhost:3000

# Access Prometheus
open http://localhost:9090
```

---

## Next Steps

1. Run `scholardevclaw demo` to see it in action
2. Try `scholardevclaw suggest ./your-project` on your codebase
3. Use `scholardevclaw tui` for interactive workflow
4. Explore `scholardevclaw specs --list` to see available improvements

For more details, visit: https://github.com/Ronak-IIITD/ScholarDevClaw
