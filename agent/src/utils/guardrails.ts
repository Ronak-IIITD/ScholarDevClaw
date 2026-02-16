import type { MappingResult, ValidationResult } from '../bridges/python-subprocess.js';

export interface GuardrailDecision {
  triggered: boolean;
  reasons: string[];
}

export function evaluateMappingGuardrails(
  mapping: MappingResult,
  minConfidence: number,
): GuardrailDecision {
  const reasons: string[] = [];

  if ((mapping.confidence ?? 0) < minConfidence) {
    reasons.push(
      `Mapping confidence ${mapping.confidence} is below threshold ${minConfidence}`,
    );
  }

  if ((mapping.targets?.length || 0) === 0) {
    reasons.push('Mapping produced no targets for patch generation');
  }

  return {
    triggered: reasons.length > 0,
    reasons,
  };
}

export function evaluateValidationGuardrails(
  validation: ValidationResult,
  minSpeedup: number,
  maxLossChangePct: number,
): GuardrailDecision {
  const reasons: string[] = [];

  if (!validation.passed) {
    reasons.push(`Validation did not pass at stage '${validation.stage}'`);
  }

  const speedup = validation.comparison?.speedup;
  if (typeof speedup === 'number' && speedup < minSpeedup) {
    reasons.push(`Validation speedup ${speedup.toFixed(3)}x is below threshold ${minSpeedup.toFixed(3)}x`);
  }

  const lossChange = validation.comparison?.lossChange;
  if (typeof lossChange === 'number' && Math.abs(lossChange) > maxLossChangePct) {
    reasons.push(
      `Validation loss change ${lossChange.toFixed(3)}% exceeds threshold ${maxLossChangePct.toFixed(3)}%`,
    );
  }

  return {
    triggered: reasons.length > 0,
    reasons,
  };
}
