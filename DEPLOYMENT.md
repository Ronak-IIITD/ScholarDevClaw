# ScholarDevClaw Deployment Guide

This guide covers deployment scenarios for ScholarDevClaw, focusing on the landing page and core services.

## Table of Contents

- [Quick Start](#quick-start)
- [Production Readiness Instructions (Week 1 + Week 2)](#production-readiness-instructions-week-1--week-2)
- [Deployment Options](#deployment-options)
  - [Option 1: Landing Page (GitHub Pages)](#option-1-landing-page-github-pages)
  - [Option 2: Production API Stack (Docker Compose)](#option-2-production-api-stack-docker-compose)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Production Readiness Instructions (Week 1 + Week 2)

This section consolidates all operational instructions introduced in recent hardening passes (quality gates, env hardening, observability, and CI checks).

### 1) Core security + repo confinement (must-have)

Set these in `docker/.env`:

```bash
SCHOLARDEVCLAW_API_AUTH_KEY=$(openssl rand -base64 32)
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_CORS_ORIGINS=https://scholardevclaw.ai,https://www.scholardevclaw.ai
SCHOLARDEVCLAW_ENABLE_HSTS=true
```

Why:
- Core API rejects unauthenticated requests when not in dev mode.
- Repo paths are confined to allowed roots.
- Explicit CORS is required for approved API clients.

### 2) Agent ↔ Core connection (must-have)

Set these in `docker/.env`:

```bash
CORE_BRIDGE_MODE=http
CORE_API_URL=http://core-api:8000
SCHOLARDEVCLAW_API_AUTH_KEY=<same value as core-api>
OPENCLAW_TOKEN=...
OPENCLAW_API_URL=...
CONVEX_URL=...
```

Why:
- In HTTP bridge mode, the agent must send `Authorization: Bearer <SCHOLARDEVCLAW_API_AUTH_KEY>`.
- Missing token propagation causes 401 failures from core-api.

### 3) Provider connection setup (required based on provider)

Always set provider/model explicitly in production:

```bash
SCHOLARDEVCLAW_API_PROVIDER=anthropic
SCHOLARDEVCLAW_API_MODEL=claude-sonnet-4-20250514
```

Then set the matching key(s):

```bash
ANTHROPIC_API_KEY=...
# OPENAI_API_KEY=...
# OPENROUTER_API_KEY=...
# GROQ_API_KEY=...
# DEEPSEEK_API_KEY=...
# MISTRAL_API_KEY=...
# COHERE_API_KEY=...
# TOGETHER_API_KEY=...
# FIREWORKS_API_KEY=...
# OLLAMA_HOST=http://localhost:11434
```

### 4) Quality gates and integration safety

Integration workflow now includes quality-gate checks (mapping/validation thresholds) and attaches gate summaries in integration payloads.

Operator action:
- Review quality gate policy in `docs/quality-gates.md`.
- Keep risky changes behind approval gates in production mode.

### 5) Observability + SLO baseline

Prometheus rules are loaded from `docker/alerts.yml` and include:
- high 5xx ratio,
- high p95 latency,
- core target down.

Operator action:
- Confirm Prometheus can load `/etc/prometheus/alerts.yml`.
- Verify Grafana credentials are changed from defaults.
- Review `docs/sre/slo.md`.

### 6) CI enforcement and docs consistency

`docs-lint` runs in CI and validates canonical install URL + command coverage consistency across docs.

Operator action:
- Run locally before push:
```bash
python scripts/docs_lint.py
```

### 7) Minimal release verification commands

Run before deploying:

```bash
scholardevclaw deploy-check --env-file docker/.env --output-json
scholardevclaw doctor production -v
python scripts/docs_lint.py
cd core && python -m pytest tests/unit/test_api_server.py tests/unit/test_health.py tests/unit/test_pipeline.py -q
cd core && python -m pytest tests/e2e/test_benchmark_scenarios.py -q
cd agent && bun run build
```

---

## Quick Start

### For Users (Just Want to Try It)
```bash
# One-line install
curl -fsSL https://ronak-iiitd.github.io/ScholarDevClaw/install.sh | bash

# Or via pip
pip install scholardevclaw

# Test installation
scholardevclaw --help
scholardevclaw demo
```

### For Developers (Want to Deploy)
```bash
# Clone repository
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw

# Option 1: Deploy landing page to GitHub Pages (automatic on push)
# Just push changes to landing/ directory

# Option 2: Run locally for development
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml,tui,dev]"
scholardevclaw tui
```

---

## Deployment Options

### Option 1: Landing Page (GitHub Pages)

The static landing page in `landing/` is deployed by
`.github/workflows/pages.yml` whenever `landing/**` changes on `main`.

One-time repository setup:

1. Open **Settings > Pages**.
2. Select **Deploy from a branch**.
3. Choose `gh-pages` and `/ (root)`.

The public URL is `https://ronak-iiitd.github.io/ScholarDevClaw/`.

### Option 2: Production API Stack (Docker Compose)

The production compose stack runs the FastAPI core, TypeScript agent, nginx,
Prometheus, and Grafana. The retired React dashboard is not part of this stack.

```bash
cp docker/.env.example docker/.env
# Set the required secrets and allowed repository roots in docker/.env.
bash scripts/runbook.sh prod preflight
bash scripts/runbook.sh prod up
bash scripts/runbook.sh prod health
```

Nginx exposes the API and operational endpoints over HTTPS. Grafana and
Prometheus remain internal Docker services by default.

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. CI Workflow (`.github/workflows/ci.yml`)
**Triggers**: Push to `main`, Pull requests, Manual dispatch
**Jobs**:
- Python lint (Ruff)
- Python type check (Mypy)
- Python tests (3.10, 3.11, 3.12)
- Agent build + test (Bun)
- Docker build smoke test
- Quality gate

**Path filters**: Triggers on changes to:
- `core/**`
- `agent/**`
- `docker/**`
- `landing/**`
- `.github/workflows/**`

#### 2. Pages Workflow (`.github/workflows/pages.yml`)
**Triggers**: Push to `main` (changes to `landing/**`), Manual dispatch
**Action**: Deploys landing page to `gh-pages` branch
**URL**: `https://ronak-iiitd.github.io/ScholarDevClaw/`

#### 3. Release Workflow (`.github/workflows/release.yml`)
**Triggers**: Push tags (`v*`), Manual dispatch
**Jobs**:
- Validate version matches pyproject.toml
- Run full test suite
- Publish to PyPI
- Create GitHub Release
- Build and push Docker images

### Running CI Locally

```bash
# Python lint
cd core
ruff check src/ tests/
ruff format --check src/ tests/

# Python type check
cd core
pip install -e ".[dev]"
mypy src/scholardevclaw/cli.py src/scholardevclaw/api/server.py --ignore-missing-imports --follow-imports=skip --disable-error-code no-any-return

# Python tests
cd core
python -m pytest tests/ -x -q --tb=short

# Agent build
cd agent
bun install --frozen-lockfile
bun run build
bun tsc --noEmit
bun run test

# Docker build
docker build -f docker/Dockerfile.core -t scholardevclaw-core:ci .
docker build -f docker/Dockerfile.agent -t scholardevclaw-agent:ci .
```

---

## Troubleshooting

### CI Workflow Not Triggering

**Problem**: CI doesn't run on every push
**Solution**: Check if changes are in filtered paths
```bash
# CI triggers on changes to:
# - core/**
# - agent/**
# - docker/**
# - landing/**
# - .github/workflows/**

# If you changed files outside these paths, CI won't trigger
# To force CI, use workflow_dispatch in GitHub Actions UI
```

### GitHub Pages Not Showing

**Problem**: Landing page not visible at GitHub Pages URL
**Solution**:
1. Check if GitHub Pages is enabled in repository settings
2. Verify source is set to `gh-pages` branch
3. Check if `gh-pages` branch exists
4. Wait a few minutes for deployment to complete

### Docker Build Fails

**Problem**: Docker build fails with errors
**Solution**:
```bash
# Clean up Docker cache
docker system prune -a

# Rebuild without cache
docker compose -f docker/docker-compose.prod.yml build --no-cache

# Check logs
docker compose -f docker/docker-compose.prod.yml logs
```

### SSL Certificate Errors

**Problem**: Browser shows SSL errors
**Solution**:
```bash
# For self-signed certificates (development)
# Click "Advanced" → "Proceed to localhost (unsafe)" in browser

# For production, use Let's Encrypt
cd docker
./generate-ssl.sh --letsencrypt yourdomain.com

# Verify certificate
openssl x509 -in docker/ssl/cert.pem -text -noout
```

### Agent Not Connecting

**Problem**: Agent can't connect to core API
**Solution**:
```bash
# Check if core-api is healthy
docker compose -f docker/docker-compose.prod.yml ps core-api

# Check agent logs
docker compose -f docker/docker-compose.prod.yml logs agent

# Verify network connectivity
docker compose -f docker/docker-compose.prod.yml exec agent ping core-api

# Check environment variables
docker compose -f docker/docker-compose.prod.yml exec agent env | grep CORE_API_URL
```

### Permission Denied Errors

**Problem**: Permission denied when running scripts
**Solution**:
```bash
# Make scripts executable
chmod +x docker/generate-ssl.sh
chmod +x landing/install.sh

# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Port Already in Use

**Problem**: Port 80 or 443 already in use
**Solution**:
```bash
# Find process using port
sudo lsof -i :80
sudo lsof -i :443

# Stop conflicting service
sudo systemctl stop nginx
sudo systemctl stop apache2

# Or change ports in docker-compose.prod.yml
# ports:
#   - "8080:80"
#   - "8443:443"
```

---

## Additional Resources

- **Documentation**: See `README.md`, `ARCHITECTURE.md`, `GUIDE.md`
- **API Reference**: `https://localhost/docs` (when running)
- **GitHub Issues**: https://github.com/Ronak-IIITD/ScholarDevClaw/issues
- **Changelog**: See `CHANGELOG.md`

---

## Support

If you encounter issues not covered here:
1. Check the logs: `docker compose -f docker/docker-compose.prod.yml logs`
2. Search existing GitHub issues
3. Create a new issue with:
   - Error messages
   - Steps to reproduce
   - Environment details (OS, Docker version, etc.)
