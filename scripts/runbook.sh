#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/docker/.env}"
DEV_COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yml"
PROD_COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.prod.yml"

log() {
  printf '[runbook] %s\n' "$*"
}

fail() {
  printf '[runbook] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
ScholarDevClaw operator runbook

Usage:
  bash scripts/runbook.sh dev  <preflight|setup|up|down|ps|logs|health> [service]
  bash scripts/runbook.sh prod <preflight|up|down|ps|logs|health> [service]
  bash scripts/runbook.sh help

Examples:
  bash scripts/runbook.sh dev setup
  bash scripts/runbook.sh dev up
  bash scripts/runbook.sh dev health
  bash scripts/runbook.sh prod preflight
  bash scripts/runbook.sh prod up
  bash scripts/runbook.sh prod logs core-api

Environment:
  ENV_FILE   Override env file path (default: docker/.env)
EOF
}

require_docker() {
  command -v docker >/dev/null 2>&1 || fail "docker CLI not found. Install Docker first."
  docker compose version >/dev/null 2>&1 || fail "docker compose plugin not available."
}

ensure_env_file() {
  [[ -f "$ENV_FILE" ]] || fail "Missing env file: $ENV_FILE (copy docker/.env.example -> docker/.env)"
}

compose_cmd() {
  local mode="$1"
  shift

  local compose_file="$DEV_COMPOSE_FILE"
  if [[ "$mode" == "prod" ]]; then
    compose_file="$PROD_COMPOSE_FILE"
  fi

  docker compose --env-file "$ENV_FILE" -f "$compose_file" "$@"
}

env_get() {
  local key="$1"
  awk -F= -v key="$key" '
    /^[[:space:]]*#/ { next }
    $1 == key {
      sub(/^[^=]*=/, "", $0)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      gsub(/^"|"$/, "", $0)
      print $0
      exit
    }
  ' "$ENV_FILE"
}

check_not_placeholder() {
  local key="$1"
  local value="$2"

  case "$value" in
    "" | *change_me* | *your_* | *replace* | *example*)
      fail "${key} in $ENV_FILE looks unset/placeholder"
      ;;
  esac
}

prod_preflight() {
  require_docker
  ensure_env_file

  log "Running prod preflight checks..."

  local required_keys=(
    "SCHOLARDEVCLAW_API_AUTH_KEY"
    "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS"
    "GRAFANA_ADMIN_USER"
    "GRAFANA_ADMIN_PASSWORD"
  )

  for key in "${required_keys[@]}"; do
    local value
    value="$(env_get "$key" || true)"
    check_not_placeholder "$key" "$value"
  done

  local allowed_dirs
  allowed_dirs="$(env_get "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS" || true)"
  IFS=':' read -r -a dirs <<<"$allowed_dirs"
  for dir in "${dirs[@]}"; do
    [[ "$dir" == /* ]] || fail "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS must contain absolute paths: '$dir'"
  done

  [[ -f "$ROOT_DIR/docker/ssl/cert.pem" ]] || fail "Missing docker/ssl/cert.pem (run docker/generate-ssl.sh)"
  [[ -f "$ROOT_DIR/docker/ssl/key.pem" ]] || fail "Missing docker/ssl/key.pem (run docker/generate-ssl.sh)"

  compose_cmd prod config >/dev/null
  log "Prod preflight passed."
}

dev_preflight() {
  require_docker
  ensure_env_file
  compose_cmd dev config >/dev/null
  log "Dev preflight passed."
}

dev_setup() {
  require_docker
  ensure_env_file
  dev_preflight
  log "Building sandbox image..."
  docker build -f docker/sandbox.Dockerfile -t sdc-sandbox:latest .
  log "Dev setup complete."
}

probe_host_http() {
  local url="$1"
  python - "$url" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        status = int(response.getcode() or 0)
    ok = 200 <= status < 400
    print(f"{url} -> {status}")
    raise SystemExit(0 if ok else 1)
except Exception as exc:
    print(f"{url} -> ERROR: {exc}")
    raise SystemExit(1)
PY
}

dev_health() {
  require_docker
  log "Checking dev stack status..."
  compose_cmd dev ps
  probe_host_http "http://localhost:8000/health"
  probe_host_http "http://localhost:8000/health/live"
  probe_host_http "http://localhost:8000/health/ready"
  log "Dev health checks passed."
}

prod_health() {
  require_docker
  log "Checking prod stack status..."
  compose_cmd prod ps
  compose_cmd prod exec -T core-api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"
  compose_cmd prod exec -T core-api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live').read()"
  compose_cmd prod exec -T nginx wget -q --spider http://localhost/health
  log "Prod health checks passed."
}

MODE="${1:-help}"
ACTION="${2:-help}"
SERVICE="${3:-}"

case "$MODE" in
  help|-h|--help)
    usage
    exit 0
    ;;
  dev|prod)
    ;;
  *)
    usage
    fail "Unknown mode: $MODE"
    ;;
esac

case "$MODE:$ACTION" in
  dev:preflight)
    dev_preflight
    ;;
  dev:setup)
    dev_setup
    ;;
  dev:up)
    dev_preflight
    compose_cmd dev up -d --build
    ;;
  dev:down)
    require_docker
    compose_cmd dev down
    ;;
  dev:ps)
    require_docker
    compose_cmd dev ps
    ;;
  dev:logs)
    require_docker
    if [[ -n "$SERVICE" ]]; then
      compose_cmd dev logs -f "$SERVICE"
    else
      compose_cmd dev logs -f
    fi
    ;;
  dev:health)
    dev_health
    ;;

  prod:preflight)
    prod_preflight
    ;;
  prod:up)
    prod_preflight
    compose_cmd prod up -d --build
    ;;
  prod:down)
    require_docker
    compose_cmd prod down
    ;;
  prod:ps)
    require_docker
    compose_cmd prod ps
    ;;
  prod:logs)
    require_docker
    if [[ -n "$SERVICE" ]]; then
      compose_cmd prod logs -f "$SERVICE"
    else
      compose_cmd prod logs -f
    fi
    ;;
  prod:health)
    prod_health
    ;;
  *)
    usage
    fail "Unknown action: $ACTION"
    ;;
esac
