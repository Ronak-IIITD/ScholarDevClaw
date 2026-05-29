import { describe, expect, it } from 'vitest';

import {
  evaluateMappingGuardrails,
  evaluateValidationGuardrails,
} from './guardrails.js';

describe('guardrails', () => {
  // ── Mapping guardrails ───────────────────────────────────────────────
  describe('evaluateMappingGuardrails', () => {
    it('triggers on low confidence', () => {
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
      expect(decision.reasons.some((r) => r.includes('confidence'))).toBe(true);
    });

    it('triggers on zero targets', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 90,
          strategy: 'replace',
          targets: [],
        },
        75,
      );

      expect(decision.triggered).toBe(true);
      expect(decision.reasons.some((r) => r.includes('no targets'))).toBe(true);
    });

    it('triggers on both low confidence and zero targets', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 50,
          strategy: 'replace',
          targets: [],
        },
        75,
      );

      expect(decision.triggered).toBe(true);
      expect(decision.reasons.length).toBe(2);
    });

    it('does not trigger when all criteria pass', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 95,
          strategy: 'replace',
          targets: [{ file: 'model.py', line: 10, currentCode: 'x', replacementRequired: true }],
        },
        75,
      );

      expect(decision.triggered).toBe(false);
      expect(decision.reasons.length).toBe(0);
    });

    it('does not trigger at exact threshold', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 75,
          strategy: 'replace',
          targets: [{ file: 'model.py', line: 10, currentCode: 'x', replacementRequired: true }],
        },
        75,
      );

      expect(decision.triggered).toBe(false);
    });

    it('handles undefined confidence gracefully', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: undefined as any,
          strategy: 'replace',
          targets: [{ file: 'model.py', line: 10, currentCode: 'x', replacementRequired: true }],
        },
        75,
      );

      // undefined ?? 0 === 0, which is < 75
      expect(decision.triggered).toBe(true);
    });

    it('handles undefined targets gracefully', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 90,
          strategy: 'replace',
          targets: undefined as any,
        },
        75,
      );

      // targets?.length is undefined, || 0 === 0
      expect(decision.triggered).toBe(true);
      expect(decision.reasons.some((r) => r.includes('no targets'))).toBe(true);
    });

    it('minConfidence of 0 accepts any confidence', () => {
      const decision = evaluateMappingGuardrails(
        {
          confidence: 1,
          strategy: 'replace',
          targets: [{ file: 'model.py', line: 10, currentCode: 'x', replacementRequired: true }],
        },
        0,
      );

      expect(decision.triggered).toBe(false);
    });
  });

  // ── Validation guardrails ────────────────────────────────────────────
  describe('evaluateValidationGuardrails', () => {
    it('triggers on failed validation', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: false,
          stage: 'import_check',
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(true);
      expect(decision.reasons.some((r) => r.includes('did not pass'))).toBe(true);
    });

    it('triggers on poor speedup and high loss drift', () => {
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

    it('does not trigger when all criteria pass', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          comparison: {
            speedup: 1.5,
            lossChange: 2.0,
            passed: true,
          },
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(false);
      expect(decision.reasons.length).toBe(0);
    });

    it('does not trigger at exact speedup threshold', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          comparison: {
            speedup: 1.01,
            lossChange: 0,
            passed: true,
          },
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(false);
    });

    it('does not trigger at exact loss change threshold', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          comparison: {
            speedup: 1.5,
            lossChange: 5.0,
            passed: true,
          },
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(false);
    });

    it('triggers on negative loss change exceeding threshold', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          comparison: {
            speedup: 1.5,
            lossChange: -6.0,
            passed: true,
          },
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(true);
      expect(decision.reasons.some((r) => r.includes('loss change'))).toBe(true);
    });

    it('handles missing comparison gracefully', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          logs: '',
        },
        1.01,
        5,
      );

      // passed=true, no comparison => speedup/lossChange are undefined, skip those checks
      expect(decision.triggered).toBe(false);
    });

    it('handles null comparison fields', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: true,
          stage: 'benchmark',
          comparison: {
            speedup: null as any,
            lossChange: null as any,
            passed: true,
          },
          logs: '',
        },
        1.01,
        5,
      );

      // typeof null is 'object', not 'number', so checks are skipped
      expect(decision.triggered).toBe(false);
    });

    it('triggers all three reasons at once', () => {
      const decision = evaluateValidationGuardrails(
        {
          passed: false,
          stage: 'benchmark',
          comparison: {
            speedup: 0.5,
            lossChange: 100,
            passed: false,
          },
          logs: '',
        },
        1.01,
        5,
      );

      expect(decision.triggered).toBe(true);
      expect(decision.reasons.length).toBe(3);
    });
  });
});
