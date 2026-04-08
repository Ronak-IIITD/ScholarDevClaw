# ScholarDevClaw Quality Gates

This document defines the minimum quality bar for production integration workflows.

## Integration quality gate policy (v1.0)

The integration pipeline enforces these thresholds:

- `mapping_target_count_min`: **1**
- `mapping_confidence_min`: **70.0**
- `validation_speedup_min`: **0.95x**
- `validation_abs_loss_change_pct_max`: **5.0%**

## Gate semantics

- **Fail**: hard stop for the integration workflow.
- **Warn**: workflow can continue, but warnings are surfaced in payload + logs.
- **Pass**: threshold satisfied.

## Current enforcement location

- `core/src/scholardevclaw/application/pipeline.py`
  - Mapping quality gates are evaluated immediately after map resolution.
  - Final quality gates are evaluated after validation.
  - Gate summaries are included in integration payloads under `quality_gates`.

## Future expansion

Planned additions:

- language-tier-specific thresholds,
- repo-size runtime budgets,
- change-risk-aware approval gates,
- benchmark-profile-specific gate overrides.
