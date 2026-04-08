# ScholarDevClaw SLO Baseline (Week 2)

## Service scope

- `core-api` FastAPI service (`/health`, `/metrics`, pipeline endpoints)
- Scraped by Prometheus job: `scholardevclaw-core`

## Initial SLO targets

1. **Availability SLO**: 99.5% monthly for core API successful responses.
2. **Latency SLO**: p95 request latency under 2.0s (5-minute windows).
3. **Reliability SLO**: 5xx error ratio below 5% over rolling 10 minutes.

## Alerting policy

Prometheus alert rules are defined in:

- `docker/alerts.yml`

Enabled via:

- `docker/prometheus.yml` `rule_files`
- `docker/docker-compose.prod.yml` mount at `/etc/prometheus/alerts.yml`

## Implemented alerts

- `ScholarDevClawHighErrorRate` (page)
- `ScholarDevClawHighLatencyP95` (warning)
- `ScholarDevClawCoreDown` (page)

## Observability improvements in week 2

- Request correlation via `X-Request-ID` middleware in API responses.
- Request ID propagation for unauthorized responses to ease incident tracing.

## Next steps

- Add burn-rate multi-window alerts (e.g., 1h/6h pairs).
- Add dashboard panels tied to SLO objectives and error budget.
- Add per-endpoint latency/error segmentation for workflow endpoints.
