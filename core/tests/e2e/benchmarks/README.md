# E2E Benchmark Skeleton

This directory contains the initial benchmark skeleton for measuring research-to-code quality trends.

## Initial scenarios

See `scenarios.json` for a machine-readable list of benchmark cases.

Each scenario should define:

- `id`: stable benchmark identifier
- `spec`: integration spec (e.g., `rmsnorm`)
- `repo`: fixture repository key
- `assertions`: required high-level outcome checks

## Goal

Keep this benchmark set lightweight in CI while enabling expansion into a full scorecard pipeline.
