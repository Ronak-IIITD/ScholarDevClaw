# ScholarDevClaw Web Dashboard

Next.js 16 (App Router) dashboard for the ScholarDevClaw research-to-code integration engine.

## Features

- **Dashboard** — Server health, pipeline status, quick actions (demo run, custom run), live WebSocket feed
- **Specs Browser** — Search/filter all 22 paper specifications with categories
- **Spec Detail** — Full spec metadata, description, run instructions
- **Live Run View** — Real-time pipeline step timeline via WebSocket, live log feed

## Design System

Mirrors the landing page editorial theme:
- Colors: `--paper:#f5f2eb`, `--ink:#0c0c0c`, `--accent:#c8410a` (terracotta), `--accent2:#1a3a2a` (deep green)
- Fonts: DM Serif Display (headings), DM Mono (mono), DM Sans (body)
- Grain overlay, paper/ink aesthetic

## Getting Started

```bash
cd web
bun install
bun dev
```

Runs on `http://localhost:3000` by default.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend FastAPI base URL |
| `NEXT_PUBLIC_API_KEY` | *(empty)* | API key for authenticated requests (optional in dev mode) |

## API Integration

The dashboard consumes the existing FastAPI endpoints:

- `GET /health` — Server health + spec count
- `GET /api/specs` — List all specs
- `GET /api/specs/{name}` — Single spec
- `POST /api/pipeline/run` — Start pipeline
- `POST /api/demo` — Start demo run
- `GET /api/pipeline/status` — Current run status
- `WS /api/ws/pipeline` — Real-time progress (already implemented in backend)
- `GET /api/pipeline/stream/{run_id}` — SSE fallback

CORS is pre-configured in the backend to allow `http://localhost:3000`.

## Project Structure

```
web/
├── app/
│   ├── layout.tsx          # Root layout with DM fonts + Nav
│   ├── globals.css         # Design system (paper/ink/terracotta)
│   ├── page.tsx            # Dashboard home
│   ├── specs/
│   │   ├── page.tsx        # Specs browser
│   │   └── [name]/page.tsx # Spec detail
│   └── run/page.tsx        # Live pipeline run view
├── lib/
│   ├── api.ts              # Typed API client
│   ├── types.ts            # Shared TypeScript types
│   └── usePipelineSocket.ts # WebSocket hook
└── components/
    ├── Nav.tsx
    ├── StatusBadge.tsx
    ├── StepTimeline.tsx
    ├── SpecCard.tsx
    ├── MetricCard.tsx
    └── LogFeed.tsx
```

## Scripts

```bash
bun dev      # Development server
bun build    # Production build
bun start    # Production server
bun lint     # ESLint
```

## Backend Requirements

The FastAPI server must be running:

```bash
cd ../core
source .venv/bin/activate
uvicorn scholardevclaw.api.server:app --reload --port 8000
```

Or via Docker Compose:

```bash
docker compose -f ../docker/docker-compose.yml up -d
```