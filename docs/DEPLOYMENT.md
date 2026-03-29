# ScholarDevClaw Deployment Guide (Canonical)

This is the canonical deployment reference and is aligned with the current repo runtime files:

- `docker/docker-compose.yml` (development stack)
- `docker/docker-compose.prod.yml` (production-like stack)
- `docker/.env.example` (required env variables)

## 1) Prerequisites

- Docker + Docker Compose plugin (`docker compose`)
- Git

## 2) Development deployment (recommended first)

From repository root:

```bash
cp docker/.env.example docker/.env
bash scripts/runbook.sh dev up
bash scripts/runbook.sh dev health
```

Services:

- `core-api` (FastAPI) on `http://localhost:8000`
- `agent` (TypeScript orchestrator, internal worker)
- `convex` (optional state backend)

## 3) Production-like deployment (nginx + web + monitoring)

From repository root:

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` and set these **required** values:

- `SCHOLARDEVCLAW_API_AUTH_KEY`
- `SCHOLARDEVCLAW_ALLOWED_REPO_DIRS`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`

Then deploy:

```bash
bash scripts/runbook.sh prod preflight
bash scripts/runbook.sh prod up
bash scripts/runbook.sh prod health
```

You can also use direct compose commands if needed:

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.prod.yml up -d --build
```

## 4) Health and readiness checks

Core API endpoints:

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /metrics`

Examples:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics
```

## 5) Operational commands

Development stack:

```bash
bash scripts/runbook.sh dev logs
bash scripts/runbook.sh dev down
```

Production-like stack:

```bash
bash scripts/runbook.sh prod logs
bash scripts/runbook.sh prod logs core-api
bash scripts/runbook.sh prod down
```

## 6) Notes on API/agent env names

- Use `CORE_API_URL` for agent-to-core API connectivity.
- Do **not** use legacy `SCHOLARDEVCLAW_CORE_URL` (stale docs/old examples).

## 7) Security baseline

- Keep `SCHOLARDEVCLAW_API_AUTH_KEY` set in production-like deployments.
- Keep `SCHOLARDEVCLAW_ALLOWED_REPO_DIRS` restricted to intended repo roots.
- Keep `SCHOLARDEVCLAW_ENABLE_HSTS=true` when serving over HTTPS.
