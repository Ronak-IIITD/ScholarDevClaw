# ScholarDevClaw Deployment Guide

This guide covers all deployment scenarios for ScholarDevClaw, from simple landing page deployment to full production stack.

## Table of Contents

- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
  - [Option 1: Landing Page (GitHub Pages)](#option-1-landing-page-github-pages)
  - [Option 2: Web Dashboard (Docker)](#option-2-web-dashboard-docker)
  - [Option 3: Full Production Stack (Docker Compose)](#option-3-full-production-stack-docker-compose)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### For Users (Just Want to Try It)
```bash
# One-line install
curl -fsSL https://ronak-iitd.github.io/ScholarDevClaw/install.sh | bash

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

# Option 2: Deploy web dashboard with Docker
cd docker
./generate-ssl.sh
docker compose -f docker-compose.prod.yml up -d

# Option 3: Run locally for development
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml,tui,dev]"
scholardevclaw tui
```

---

## Deployment Options

### Option 1: Landing Page (GitHub Pages)

**What it is**: Static HTML page with install instructions and feature showcase
**Where it's deployed**: GitHub Pages (`gh-pages` branch)
**URL**: `https://ronak-iitd.github.io/ScholarDevClaw/`
**Requirements**: None (just push to GitHub)

#### How It Works
1. The landing page is in the `landing/` directory
2. GitHub Actions workflow (`.github/workflows/pages.yml`) deploys it automatically
3. Every push to `main` that changes `landing/**` triggers deployment
4. The workflow deploys to the `gh-pages` branch using `peaceiris/actions-gh-pages@v4`

#### Setup (One-Time)
1. Go to your GitHub repository → Settings → Pages
2. Set **Source** to "Deploy from a branch"
3. Set **Branch** to `gh-pages` and folder to `/ (root)`
4. Click Save
5. Your landing page will be available at: `https://ronak-iitd.github.io/ScholarDevClaw/`

#### Deploy Changes
```bash
# Make changes to landing/index.html
git add landing/
git commit -m "Update landing page"
git push origin main
# GitHub Actions will automatically deploy to gh-pages branch
```

#### Files
- `landing/index.html` - Main landing page (893 lines)
- `landing/install.sh` - One-line install script (261 lines)
- `landing/favicon.svg` - Custom favicon
- `landing/404.html` - Custom 404 page
- `landing/robots.txt` - SEO configuration
- `landing/sitemap.xml` - SEO sitemap

---

### Option 2: Web Dashboard (Docker)

**What it is**: React dashboard with real-time pipeline visualization
**Where it's deployed**: Docker container (nginx)
**URL**: `https://localhost` (or your domain)
**Requirements**: Docker, SSL certificates, API server running

#### Architecture
```
┌─────────────────┐
│   nginx (443)   │ ← SSL termination
├─────────────────┤
│  web-ui (80)    │ ← React dashboard
├─────────────────┤
│  core-api (8000)│ ← FastAPI backend
├─────────────────┤
│  agent          │ ← TypeScript orchestrator
└─────────────────┘
```

#### Prerequisites
1. Docker and Docker Compose installed
2. SSL certificates generated
3. Environment variables configured

#### Step-by-Step Deployment

##### 1. Generate SSL Certificates
```bash
cd docker

# For local development (self-signed)
./generate-ssl.sh

# For production domain
./generate-ssl.sh example.com

# For Let's Encrypt (requires certbot)
./generate-ssl.sh --letsencrypt example.com
```

##### 2. Configure Environment Variables
```bash
# Copy example environment file
cp docker/.env.example docker/.env

# Edit environment variables
nano docker/.env
```

Required environment variables:
```bash
# API Keys (optional but recommended)
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# Grafana (required for monitoring)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure_password_here

# Logging
LOG_LEVEL=INFO
```

##### 3. Build and Deploy
```bash
cd docker

# Build images
docker compose -f docker-compose.prod.yml build

# Start services
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

##### 4. Access the Dashboard
- **Web Dashboard**: `https://localhost`
- **API Documentation**: `https://localhost/docs` (internal networks only)
- **Health Check**: `https://localhost/health`
- **Prometheus**: `http://localhost:9090` (internal)
- **Grafana**: `http://localhost:3000` (internal)

#### Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80, 443 | Reverse proxy with SSL |
| web-ui | 80 | React dashboard |
| core-api | 8000 | FastAPI backend |
| agent | - | TypeScript orchestrator |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboard |

#### Useful Commands
```bash
# Stop all services
docker compose -f docker-compose.prod.yml down

# View logs for specific service
docker compose -f docker-compose.prod.yml logs -f core-api

# Restart a service
docker compose -f docker-compose.prod.yml restart core-api

# Update and restart
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d

# Clean up
docker compose -f docker-compose.prod.yml down -v
docker system prune -a
```

---

### Option 3: Full Production Stack (Docker Compose)

**What it is**: Complete stack with API, web dashboard, monitoring, and security
**Where it's deployed**: Docker containers
**URL**: `https://your-domain.com`
**Requirements**: Domain, SSL certificates, server with Docker

#### Production Checklist

##### 1. Domain Setup
```bash
# Point your domain to your server's IP
# Example DNS records:
# A     scholardevclaw.ai     → your_server_ip
# AAAA  scholardevclaw.ai     → your_server_ipv6
```

##### 2. SSL Certificates
```bash
# Option A: Let's Encrypt (recommended)
cd docker
./generate-ssl.sh --letsencrypt scholardevclaw.ai

# Option B: Your own certificates
cp /path/to/your/cert.pem docker/ssl/cert.pem
cp /path/to/your/key.pem docker/ssl/key.pem
```

##### 3. Security Configuration
```bash
# Update nginx.conf with your domain
sed -i 's/server_name _;/server_name scholardevclaw.ai;/' docker/nginx.conf

# Update security headers
# Review and update CSP, HSTS, and other headers in docker/nginx.conf
```

##### 4. Environment Variables
```bash
# Create production environment file
cat > docker/.env << EOF
# API Keys
ANTHROPIC_API_KEY=your_production_key
GITHUB_TOKEN=your_production_token
OPENCLAW_TOKEN=your_openclaw_token
CORE_API_URL=http://core-api:8000

# Core API hardening (required)
SCHOLARDEVCLAW_API_AUTH_KEY=$(openssl rand -base64 32)
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_ENABLE_HSTS=true

# Grafana (use strong passwords!)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)

# Logging
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Monitoring
PROMETHEUS_RETENTION=30d
EOF
```

##### 5. Deploy
```bash
cd docker

# Build production images
docker compose -f docker-compose.prod.yml build

# Start with production profile
docker compose -f docker-compose.prod.yml up -d

# Verify all services are healthy
docker compose -f docker-compose.prod.yml ps

# Preferred (from repo root): operator runbook
bash scripts/runbook.sh prod preflight
bash scripts/runbook.sh prod up
bash scripts/runbook.sh prod health
```

##### 6. Monitoring Setup
```bash
# Access Grafana
open http://localhost:3000

# Default credentials (change immediately!)
# Username: admin
# Password: (from GRAFANA_ADMIN_PASSWORD in .env)

# Import dashboards
# - Docker metrics
# - nginx metrics
# - Application metrics
```

##### 7. Backup Strategy
```bash
# Backup volumes
docker run --rm -v scholardevclaw_prometheus-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /data
docker run --rm -v scholardevclaw_grafana-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/grafana-$(date +%Y%m%d).tar.gz /data

# Backup configuration
tar czf backups/config-$(date +%Y%m%d).tar.gz docker/
```

##### 8. Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
cd docker
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Verify
docker compose -f docker-compose.prod.yml ps
```

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
- `web/**`
- `.github/workflows/**`

#### 2. Pages Workflow (`.github/workflows/pages.yml`)
**Triggers**: Push to `main` (changes to `landing/**`), Manual dispatch
**Action**: Deploys landing page to `gh-pages` branch
**URL**: `https://ronak-iitd.github.io/ScholarDevClaw/`

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
# - web/**
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
docker compose -f docker-compose.prod.yml build --no-cache

# Check logs
docker compose -f docker-compose.prod.yml logs
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

### Web Dashboard Not Loading

**Problem**: Dashboard shows blank page or errors
**Solution**:
```bash
# Check if API server is running
docker compose -f docker-compose.prod.yml ps core-api

# Check API logs
docker compose -f docker-compose.prod.yml logs core-api

# Verify API is accessible
curl http://localhost:8000/health

# Check browser console for errors
# Open Developer Tools → Console
```

### Agent Not Connecting

**Problem**: Agent can't connect to core API
**Solution**:
```bash
# Check if core-api is healthy
docker compose -f docker-compose.prod.yml ps core-api

# Check agent logs
docker compose -f docker-compose.prod.yml logs agent

# Verify network connectivity
docker compose -f docker-compose.prod.yml exec agent ping core-api

# Check environment variables
docker compose -f docker-compose.prod.yml exec agent env | grep CORE_API_URL
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
1. Check the logs: `docker compose -f docker-compose.prod.yml logs`
2. Search existing GitHub issues
3. Create a new issue with:
   - Error messages
   - Steps to reproduce
   - Environment details (OS, Docker version, etc.)
