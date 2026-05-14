# ScholarDevClaw API Reference

This document covers the HTTP and WebSocket surfaces exposed by [core/src/scholardevclaw/api/server.py](../core/src/scholardevclaw/api/server.py) and [core/src/scholardevclaw/api/routes/dashboard.py](../core/src/scholardevclaw/api/routes/dashboard.py).

## Base URL

- Local default: `http://localhost:8000`
- Production behind nginx: `/api/*` is proxied to the FastAPI core service

## Authentication

ScholarDevClaw uses a single bearer token for API access when `SCHOLARDEVCLAW_API_AUTH_KEY` is set on the server.

### Set the server key

```bash
export SCHOLARDEVCLAW_API_AUTH_KEY="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
```

Start the API with:

```bash
export SCHOLARDEVCLAW_ALLOWED_REPO_DIRS="/absolute/path/to/repos"
uvicorn scholardevclaw.api.server:app --reload
```

### Send the key from clients

```bash
export SDC_API="http://localhost:8000"
export SDC_TOKEN="paste-the-same-token-here"
```

Use it on authenticated routes:

```bash
curl -H "Authorization: Bearer $SDC_TOKEN" "$SDC_API/repo/analyze"
```

### Auth-exempt routes

These remain available without a bearer token:

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`
- `GET /metrics`

## Health

### `GET /health`

Returns a lightweight launch-safe health payload.

```bash
curl "$SDC_API/health"
```

Example response:

```json
{
  "status": "ok",
  "version": "2.0",
  "spec_count": 19
}
```

### `GET /health/live`

Container liveness probe.

```bash
curl "$SDC_API/health/live"
```

### `GET /health/ready`

Container readiness probe.

```bash
curl "$SDC_API/health/ready"
```

## Core Pipeline Endpoints

### `POST /repo/analyze`

Analyze a repository with tree-sitter and return models, training loops, dependencies, and test files.

```bash
curl -X POST "$SDC_API/repo/analyze" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repoPath": "/absolute/path/to/repo"
  }'
```

### `POST /research/extract`

Extract a research spec from a built-in spec name, arXiv ID, or PDF source.

```bash
curl -X POST "$SDC_API/research/extract" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "arxiv:2205.14135",
    "sourceType": "arxiv"
  }'
```

### `POST /mapping/map`

Map a normalized research spec onto a repository analysis payload.

```bash
curl -X POST "$SDC_API/mapping/map" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repoAnalysis": {
      "elements": [],
      "imports": []
    },
    "researchSpec": {
      "paper": {
        "title": "Root Mean Square Layer Normalization",
        "authors": ["Biao Zhang", "Rico Sennrich"],
        "arxiv": "1910.07467",
        "year": 2019
      },
      "algorithm": {
        "name": "RMSNorm",
        "replaces": "LayerNorm",
        "description": "Simplified layer normalization without mean-centering",
        "formula": "x / sqrt(mean(x^2) + eps) * gamma",
        "category": "normalization"
      },
      "implementation": {
        "moduleName": "RMSNorm",
        "parentClass": "nn.Module",
        "parameters": ["ndim", "eps"]
      },
      "changes": {
        "type": "replace",
        "targetPattern": "LayerNorm",
        "insertionPoints": ["Block class"],
        "replacement": "RMSNorm",
        "expectedBenefits": ["speedup"]
      }
    }
  }'
```

### `POST /patch/generate`

Generate patch artifacts from a mapping result for a confined repository path.

```bash
curl -X POST "$SDC_API/patch/generate" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repoPath": "/absolute/path/to/repo",
    "mapping": {
      "targets": [],
      "strategy": "replace",
      "confidence": 80,
      "researchSpec": {}
    }
  }'
```

### `POST /validation/run`

Validate patch artifacts. The runner performs syntax checks, repo tests, benchmark timing, and when patch artifacts are present it now adds numerical correctness, regression snapshot, and diff readability checks.

```bash
curl -X POST "$SDC_API/validation/run" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repoPath": "/absolute/path/to/repo",
    "patch": {
      "algorithm_name": "RMSNorm",
      "paper_reference": "arXiv:1910.07467",
      "new_files": [
        {
          "path": "rmsnorm.py",
          "content": "class RMSNorm:\n    pass\n"
        }
      ],
      "transformations": []
    }
  }'
```

### `POST /from-paper`

Run the full paper-to-code flow in one request: extraction, repo analysis, mapping, patch generation, and validation.

```bash
curl -X POST "$SDC_API/from-paper" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repoPath": "/absolute/path/to/repo",
    "paperSource": "arxiv:2106.09685",
    "sourceType": "arxiv"
  }'
```

## Dashboard API

These routes are mounted under `/api`.

### `GET /api/specs`

List the built-in paper specs shown in the dashboard spec browser.

```bash
curl -H "Authorization: Bearer $SDC_TOKEN" "$SDC_API/api/specs"
```

### `GET /api/specs/{name}`

Get one dashboard-ready spec record.

```bash
curl -H "Authorization: Bearer $SDC_TOKEN" "$SDC_API/api/specs/rmsnorm"
```

### `GET /api/pipeline/status`

Return the in-memory status of the active dashboard pipeline run.

```bash
curl -H "Authorization: Bearer $SDC_TOKEN" "$SDC_API/api/pipeline/status"
```

### `POST /api/pipeline/run`

Launch a dashboard pipeline run.

```bash
curl -X POST "$SDC_API/api/pipeline/run" \
  -H "Authorization: Bearer $SDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/absolute/path/to/repo",
    "spec_names": ["rmsnorm", "swiglu"],
    "skip_validate": false,
    "output_dir": null
  }'
```

### `POST /api/demo`

Launch the one-click demo pipeline against the bundled nanoGPT repo.

```bash
curl -X POST "$SDC_API/api/demo" \
  -H "Authorization: Bearer $SDC_TOKEN"
```

### `GET /api/pipeline/stream/{run_id}`

Server-Sent Events stream for the active run.

```bash
curl -N -H "Authorization: Bearer $SDC_TOKEN" "$SDC_API/api/pipeline/stream/<run_id>"
```

### `WS /api/ws/pipeline`

WebSocket feed for live dashboard updates.

When API auth is enabled, send the auth frame first:

```json
{"type":"auth","token":"<same bearer token>"}
```

Then keep the socket alive with:

```json
{"type":"ping"}
```

Typical server messages:

```json
{"type":"auth_ok"}
{"type":"pipeline_snapshot","run":{"run_id":"abc123","status":"running","steps":[]}}
{"type":"pipeline_step","run_id":"abc123","step":"analyze","status":"completed"}
{"type":"pipeline_complete","run_id":"abc123","status":"completed","total_seconds":12.4}
{"type":"pipeline_error","run_id":"abc123","error":"..."}
```

## Notes

- Repository paths are confined by `SCHOLARDEVCLAW_ALLOWED_REPO_DIRS`.
- `SCHOLARDEVCLAW_DEV_MODE=true` disables auth and confinement fail-closed behavior for local development only.
- The dashboard routes use in-memory run state. For multi-instance deployments, move that state to Redis or Convex.
- OpenAPI remains available at `GET /openapi.json`.
