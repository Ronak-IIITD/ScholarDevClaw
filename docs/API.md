# ScholarDevClaw API Reference

This document describes the HTTP endpoints exposed by the core service in [core/src/scholardevclaw/api/server.py](../core/src/scholardevclaw/api/server.py).

## Base URL

- Local default: `http://localhost:8000`

## Health

### `GET /health`

Returns service health.

Response:

```json
{
  "status": "ok"
}
```

## Repository Analysis

### `POST /repo/analyze`

Request body:

```json
{
  "repoPath": "/absolute/path/to/repo"
}
```

Response includes parsed repository summary used by the agent phases.

## Research Extraction

### `POST /research/extract`

Request body:

```json
{
  "source": "rmsnorm",
  "sourceType": "pdf"
}
```

`sourceType` supports `pdf` and `arxiv`.

## Mapping

### `POST /mapping/map`

Request body:

```json
{
  "repoAnalysis": {},
  "researchSpec": {}
}
```

Response includes `targets`, `strategy`, and `confidence`.

## Patch Generation

### `POST /patch/generate`

Request body:

```json
{
  "mapping": {}
}
```

Response includes `newFiles`, `transformations`, and `branchName`.

## Validation

### `POST /validation/run`

Request body:

```json
{
  "patch": {},
  "repoPath": "/absolute/path/to/repo"
}
```

Response includes pass/fail status, stage, metrics, and logs.

## Notes

- Paths must exist on the machine running the core service.
- API currently reflects the v1/v2 transition and may evolve with stricter schemas.
