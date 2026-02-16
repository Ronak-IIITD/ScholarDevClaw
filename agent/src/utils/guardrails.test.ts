import { describe, expect, it } from 'vitest';

import {
  evaluateMappingGuardrails,
  evaluateValidationGuardrails,
} from './guardrails.js';

describe('guardrails', () => {
  it('triggers mapping guardrail on low confidence', () => {
    const decision = evaluateMappingGuardrails(
      {
        confidence: 62,
        strategy: 'replace',
        targets: [{ file: 'model.py', line: 10, currentCode: 'x', replacementRequired: true }],
      },
      75,
    );

    expect(decision.triggered).toBe(true);
    expect(decision.reasons.length).toBeGreaterThan(0);
  });

  it('triggers validation guardrail on poor speedup and high loss drift', () => {
    const decision = evaluateValidationGuardrails(
      {
        passed: true,
        stage: 'benchmark',
        comparison: {
          speedup: 0.97,
          lossChange: 7.3,
          passed: false,
        },
        logs: '',
      },
      1.01,
      5,
    );

    expect(decision.triggered).toBe(true);
    expect(decision.reasons.some((item) => item.includes('speedup'))).toBe(true);
    expect(decision.reasons.some((item) => item.includes('loss change'))).toBe(true);
  });
});
