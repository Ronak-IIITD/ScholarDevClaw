# APPLY.md — Manual Requirements for Deployment

This document lists all manual configurations, API keys, and external service setups that **cannot be automated by the AI agent** and require human intervention before ScholarDevClaw can run.

---

## 1. Required API Keys & Secrets

### A. Core AI Services

| Key | Purpose | How to Get | Required For |
|-----|---------|------------|--------------|
| `ANTHROPIC_API_KEY` | Claude API for LLM reasoning | [ Anthropic Console](https://console.anthropic.com/) | Agent orchestration, patch generation, research analysis |
| `OPENCLAW_TOKEN` | OpenClaw runner authentication | Contact OpenClaw team | Agent heartbeat, phase execution |

### B. External APIs

| Key | Purpose | How to Get | Required For |
|-----|---------|------------|--------------|
| `GITHUB_TOKEN` | GitHub API access | [GitHub Settings → Tokens](https://github.com/settings/tokens) | Repo cloning, PR creation, API rate limits |
| `ARXIV_API_KEY` | arXiv enhanced access | [arXiv API](https://arxiv.org/help/api) - no key needed for basic | Faster paper search |
| `IEEE_API_KEY` | IEEE Xplore access | [IEEE Xplore](https://developer.ieee.org/) | Academic paper retrieval |
| `PUBMED_API_KEY` | PubMed access | [NCBI Account](https://www.ncbi.nlm.nih.gov/account/) | Biomedical research extraction |

### C. Infrastructure

| Key | Purpose | How to Get | Required For |
|-----|---------|------------|--------------|
| `CONVEX_URL` | Convex deployment | [Convex Dashboard](https://dashboard.convex.dev/) | State persistence, run history |
| `CONVEX_DEPLOY_KEY` | Convex admin | Generated in Convex dashboard | Deployment management |

---

## 2. External Service Accounts

### Must Be Created Manually

| Service | Account Type | Signup URL | Purpose |
|---------|--------------|------------|---------|
| **Convex** | Cloud deployment | https://convex.dev | State/lifecycle tracking |
| **GitHub** | Personal/Org | https://github.com | Repo access, OAuth |
| **Anthropic** | API account | https://anthropic.com | LLM inference |

### Optional but Recommended

| Service | Signup URL | Purpose |
|---------|------------|---------|
| Slack | https://slack.com | Notifications |
| Stripe | https://stripe.com | Future payment processing |
| Grafana Cloud | https://grafana.com | Observability dashboards |

---

## 3. Manual Infrastructure Setup

### A. Convex Deployment

```bash
# 1. Install Convex CLI
npm install -g convex

# 2. Login to Convex
npx convex dev

# 3. This will:
#    - Create a Convex project
#    - Generate CONVEX_URL
#    - Create deployment key
#    - Update convex/schema.ts

# 4. Add to .env:
CONVEX_URL=https://your-deployment.convex.cloud
CONVEX_DEPLOY_KEY=your-deploy-key
```

**AI Cannot Do:** Convex requires OAuth login and project creation through browser.

---

### B. GitHub OAuth App (For Web UI)

If building the web frontend, you need a GitHub OAuth app:

```bash
# 1. Go to: https://github.com/settings/developers

# 2. Click "New OAuth App"
#    - Application name: ScholarDevClaw
#    - Homepage URL: http://localhost:3000
#    - Authorization callback: http://localhost:3000/api/auth/callback/github

# 3. Copy Client ID
# 4. Generate Client Secret

# 5. Add to .env:
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret
```

**AI Cannot Do:** Requires GitHub account login and OAuth app creation.

---

### C. Domain & SSL (Production)

```bash
# 1. Register domain (manual)
#    - GoDaddy, Namecheap, Cloudflare, etc.

# 2. For SSL with Certbot:
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com

# OR use Cloudflare (recommended):
#    - Create Cloudflare account
#    - Update nameservers
#    - Enable "Full" SSL mode
```

**AI Cannot Do:** Domain registration requires payment and legal agreement.

---

## 4. Local Development Setup

### A. Clone Repository

```bash
# Manual - requires git
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw
```

**AI Cannot Do:** Requires git installation and network access.

---

### B. Python Environment

```bash
# 1. Create virtual environment
cd core
python3 -m venv .venv

# 2. Activate
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install with dependencies
pip install -e ".[arxiv,ml,dev,tui]"
```

**AI Cannot Do:** Requires Python 3.10+ installation on host machine.

---

### C. Node.js/Bun Setup

```bash
# Install Bun (if not present)
curl -fsSL https://bun.sh/install | bash

# Install dependencies
cd ../agent
bun install
bun run build
```

**AI Cannot Do:** Requires runtime installation on host machine.

---

### D. Environment Variables

Create `.env` in project root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENCLAW_TOKEN=your-token
GITHUB_TOKEN=ghp_...
CONVEX_URL=https://your-convex.convex.cloud

# Optional (for full features)
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret
STRIPE_SECRET_KEY=sk_test_...
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

**AI Cannot Do:** Requires copying `.env.example` and filling in real keys.

---

## 5. Cloud Deployment Prerequisites

### A. Docker & Container Registry

```bash
# Install Docker
# https://docs.docker.com/get-docker/

# For cloud deployment:
# - Docker Hub account OR
# - AWS ECR OR
# - GCP Container Registry
```

**AI Cannot Do:** Requires Docker installation and cloud account creation.

---

### B. Cloud Platform Account

Choose one:

| Platform | Signup | Setup |
|----------|--------|-------|
| **Fly.io** | https://fly.io | `flyctl install` |
| **Railway** | https://railway.app | `railway login` |
| **AWS** | https://aws.amazon.com | Create IAM user |
| **GCP** | https://console.cloud.google.com | Create project |
| **Azure** | https://azure.microsoft.com | Create subscription |

**AI Cannot Do:** Requires credit card and account creation.

---

### C. Database/State Service

If not using Convex, you need to provision:

| Option | Manual Setup |
|--------|---------------|
| PostgreSQL | `CREATE DATABASE scholardevclaw;` |
| Redis | `brew install redis` or cloud provider |
| S3 | Create bucket, get credentials |

---

## 6. Payment & Billing Setup

### Future Requirements (Not Yet Implemented)

| Item | Manual Action |
|------|---------------|
| **Stripe** | Create Stripe account, get API keys |
| **Payment webhooks** | Configure webhook URL in Stripe dashboard |
| **Email service** | Sign up for SendGrid/Postmark |
| **Domain purchase** | Buy domain from registrar |

---

## 7. CI/CD Pipeline Setup

### GitHub Actions (Optional)

To enable automated testing:

```bash
# 1. Create .github/workflows/ci.yml
# 2. Add GitHub Secrets:
#    - PYPI_TOKEN (for package publishing)
#    - DOCKER_USERNAME
#    - DOCKER_PASSWORD
```

**AI Cannot Do:** Requires GitHub repository settings access.

---

## 8. Checklist: What AI Can vs Cannot Do

### ✅ AI Can Do Automatically

- Install Python/Bun dependencies via pip/bun
- Run database migrations
- Generate code from templates
- Execute tests locally
- Build Docker images
- Deploy via `docker-compose up`

### ❌ AI Cannot Do (Requires Human)

- Create accounts on external services
- Obtain API keys (requires payment/login)
- Configure OAuth apps
- Purchase domains
- Set up cloud infrastructure
- Access secrets management systems
- Sign up for paid services
- Configure SSL certificates
- Grant admin permissions
- Review and approve security policies

---

## 9. Quick Start (Manual Steps First)

Before running ScholarDevClaw, complete these manually:

```bash
# 1. Get API keys (see Section 1)
#    - ANTHROPIC_API_KEY (required)
#    - GITHUB_TOKEN (required)
#    - CONVEX_URL (optional for local)

# 2. Create .env file
cp .env.example .env
# Edit .env with your keys

# 3. Install dependencies
cd core && pip install -e ".[arxiv,ml,dev]"
cd ../agent && bun install

# 4. Run locally
cd core && uvicorn scholardevclaw.api.server:app --reload
```

---

## 10. Support

If stuck on any manual step:
- **Convex setup:** https://docs.convex.dev/quick-start
- **Anthropic keys:** https://console.anthropic.com/
- **GitHub tokens:** https://github.com/settings/tokens
- **Docker:** https://docs.docker.com/get-docker/

---

## Summary

| Category | Manual Action Required |
|----------|----------------------|
| **API Keys** | 3 minimum (Anthropic, GitHub, OpenClaw) |
| **Accounts** | 1 minimum (Convex recommended) |
| **Runtime** | Python 3.10+, Node.js/Bun |
| **Cloud** | Optional but recommended for production |
| **Domain** | Only for production HTTPS |

**Total manual setup time:** ~30 minutes for local, ~2 hours for production cloud.
