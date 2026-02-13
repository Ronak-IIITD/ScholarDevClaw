# ScholarDevClaw Deployment Guide

This guide covers both self-hosted (local) and cloud deployment options for ScholarDevClaw.

## Quick Start - Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+ (for OpenClaw agent)
- Git
- 4GB RAM minimum

### 1. Clone and Setup

```bash
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw

# Setup Python core
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml]"

# Setup TypeScript agent
cd ../agent
npm install
# or: bun install
```

### 2. Configure Environment

```bash
# Create .env file in project root
cat > .env << 'EOF'
# Python Core
SCHOLARDEVCLAW_CORE_PATH=./core/src
SCHOLARDEVCLAW_WORKSPACE=~/.scholardevclaw/workspace
SCHOLARDEVCLAW_LOG_PATH=~/.scholardevclaw/logs

# Optional: GitHub API token for better rate limits
GITHUB_TOKEN=your_github_token_here

# Optional: Anthropic API for LLM features
ANTHROPIC_API_KEY=your_anthropic_key_here
EOF
```

### 3. Test Installation

```bash
# Test Python core
cd core
source .venv/bin/activate
scholardevclaw --help
scholardevclaw demo

# Test agent (in another terminal)
cd agent
bun run src/index.ts --help
```

---

## Self-Hosted Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Create deployment directory
mkdir scholardevclaw-deploy
cd scholardevclaw-deploy

# Copy docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  scholardevclaw-core:
    build:
      context: ../ScholarDevClaw/core
      dockerfile: Dockerfile
    volumes:
      - ./workspace:/workspace
      - ./logs:/logs
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - SCHOLARDEVCLAW_WORKSPACE=/workspace
      - SCHOLARDEVCLAW_LOG_PATH=/logs
    ports:
      - "8000:8000"
    command: uvicorn scholardevclaw.api.server:app --host 0.0.0.0 --port 8000

  scholardevclaw-agent:
    build:
      context: ../ScholarDevClaw/agent
      dockerfile: Dockerfile
    depends_on:
      - scholardevclaw-core
    volumes:
      - ./workspace:/workspace
      - ./logs:/logs
    environment:
      - SCHOLARDEVCLAW_CORE_URL=http://scholardevclaw-core:8000
      - SCHOLARDEVCLAW_WORKSPACE=/workspace
      - SCHOLARDEVCLAW_LOG_PATH=/logs
    command: bun run src/index.ts heartbeat

  # Optional: Convex for state management
  convex:
    image: ghcr.io/getconvex/convex-backend:latest
    ports:
      - "3210:3210"
    volumes:
      - convex-data:/convex/data

volumes:
  convex-data:
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f scholardevclaw-core
docker-compose logs -f scholardevclaw-agent
```

### Option 2: Systemd Service (Linux)

```bash
# Create service file
sudo tee /etc/systemd/system/scholardevclaw.service << 'EOF'
[Unit]
Description=ScholarDevClaw Research Agent
After=network.target

[Service]
Type=simple
User=scholardevclaw
Group=scholardevclaw
WorkingDirectory=/opt/scholardevclaw
ExecStart=/opt/scholardevclaw/run.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create run script
sudo tee /opt/scholardevclaw/run.sh << 'EOF'
#!/bin/bash
cd /opt/scholardevclaw/agent
exec bun run src/index.ts heartbeat
EOF

sudo chmod +x /opt/scholardevclaw/run.sh
sudo systemctl daemon-reload
sudo systemctl enable scholardevclaw
sudo systemctl start scholardevclaw
```

### Option 3: PM2 (Node.js Process Manager)

```bash
# Install PM2 globally
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'scholardevclaw-core',
      script: 'uvicorn',
      args: 'scholardevclaw.api.server:app --host 0.0.0.0 --port 8000',
      cwd: './core',
      interpreter: './core/.venv/bin/python',
      env: {
        SCHOLARDEVCLAW_WORKSPACE: './workspace',
        SCHOLARDEVCLAW_LOG_PATH: './logs',
      },
      log_file: './logs/core.log',
      error_file: './logs/core-error.log',
      out_file: './logs/core-out.log',
      merge_logs: true,
    },
    {
      name: 'scholardevclaw-agent',
      script: './agent/src/index.ts',
      interpreter: 'bun',
      args: 'heartbeat',
      env: {
        SCHOLARDEVCLAW_CORE_URL: 'http://localhost:8000',
        SCHOLARDEVCLAW_WORKSPACE: './workspace',
        SCHOLARDEVCLAW_LOG_PATH: './logs',
      },
      cron_restart: '*/5 * * * *',  # Run heartbeat every 5 minutes
      log_file: './logs/agent.log',
    },
  ],
};
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Generate startup script
```

---

## Cloud Deployment

### Option 1: Fly.io (Recommended)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Create fly.toml
cat > fly.toml << 'EOF'
app = "scholardevclaw"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  SCHOLARDEVCLAW_WORKSPACE = "/workspace"
  SCHOLARDEVCLAW_LOG_PATH = "/logs"

[mounts]
  source = "scholardevclaw_data"
  destination = "/workspace"

[[services]]
  internal_port = 8000
  protocol = "tcp"
  
  [[services.ports]]
    handlers = ["http"]
    port = 80
    
  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
EOF

# Deploy
fly launch
fly deploy
```

### Option 2: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Option 3: AWS EC2

```bash
# Launch Ubuntu 22.04 instance
# SSH into instance

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv nodejs npm git docker.io

# Install Bun
curl -fsSL https://bun.sh/install | bash

# Clone and setup
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw
./scripts/setup.sh

# Run with Docker Compose
docker-compose up -d
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLARDEVCLAW_CORE_PATH` | Python core source path | `./core/src` |
| `SCHOLARDEVCLAW_WORKSPACE` | Workspace directory | `~/.scholardevclaw/workspace` |
| `SCHOLARDEVCLAW_LOG_PATH` | Log directory | `~/.scholardevclaw/logs` |
| `SCHOLARDEVCLAW_CORE_URL` | Core API URL | `http://localhost:8000` |
| `GITHUB_TOKEN` | GitHub API token | None |
| `ANTHROPIC_API_KEY` | Anthropic/Claude API key | None |
| `CONVEX_URL` | Convex deployment URL | None |

### Proxy Configuration

If behind corporate proxy:

```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
```

---

## Usage After Deployment

### CLI

```bash
# Analyze repository
scholardevclaw analyze https://github.com/user/repo

# Search for improvements
scholardevclaw suggest https://github.com/user/repo

# Full integration
scholardevclaw integrate https://github.com/user/repo --mode autonomous
```

### API

```bash
# Start integration
POST /api/v1/integrations
{
  "repo_url": "https://github.com/user/repo",
  "mode": "step_approval",
  "focus": "performance"
}

# Check status
GET /api/v1/integrations/{id}

# Approve phase
POST /api/v1/integrations/{id}/approve
{
  "phase": 3
}
```

### Web Dashboard

```bash
# Access dashboard
open http://localhost:3000

# Features:
# - View all integrations
# - Review diffs
# - Approve phases
# - View benchmarks
# - Manage settings
```

---

## Monitoring

### Health Checks

```bash
# Core health
curl http://localhost:8000/health

# Agent health
curl http://localhost:3000/api/health
```

### Logs

```bash
# View logs
tail -f ~/.scholardevclaw/logs/scholardevclaw-*.log

# Or with Docker
docker-compose logs -f
```

### Metrics

```bash
# Prometheus metrics (if enabled)
curl http://localhost:8000/metrics
```

---

## Backup and Recovery

```bash
# Backup workspace
tar -czf backup-$(date +%Y%m%d).tar.gz ~/.scholardevclaw/workspace

# Restore
tar -xzf backup-20240101.tar.gz -C ~/
```

---

## Troubleshooting

### Common Issues

1. **Python not found**
   ```bash
   # Ensure Python 3.10+ is installed
   python3 --version
   # Use pyenv if needed
   pyenv install 3.11
   ```

2. **Tree-sitter parsers not available**
   ```bash
   # Install language-specific parsers
   pip install tree-sitter-python tree-sitter-javascript
   ```

3. **Permission denied**
   ```bash
   # Fix permissions
   sudo chown -R $(whoami) ~/.scholardevclaw
   ```

4. **Out of memory**
   ```bash
   # Increase swap
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Security Considerations

- Never commit API keys to git
- Use .env files or secret management
- Enable HTTPS in production
- Use non-root user in containers
- Regularly update dependencies
- Scan for vulnerabilities: `pip-audit`, `npm audit`

---

## Support

- GitHub Issues: https://github.com/Ronak-IIITD/ScholarDevClaw/issues
- Documentation: https://github.com/Ronak-IIITD/ScholarDevClaw/tree/main/docs
