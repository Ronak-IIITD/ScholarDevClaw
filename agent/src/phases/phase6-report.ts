import { logger } from '../utils/logger.js';
import type { Phase6Context, PhaseResult } from './types.js';

interface Report {
  metadata: {
    integrationId: string;
    repoUrl: string;
    paper: string;
    algorithm: string;
    createdAt: string;
  };
  summary: {
    status: string;
    confidence: number;
    changesMade: number;
    filesModified: string[];
    newFiles: string[];
  };
  whatChanged: string;
  why: string;
  observedImpact: {
    metricsComparison: unknown;
    meetsExpectations: boolean;
  };
  riskNotes: string[];
  diffPreview: string;
  testResults: {
    unitTestsPassed: boolean;
    benchmarkResults: unknown;
  };
  recommendation: {
    action: 'approve' | 'review' | 'reject';
    confidence: number;
    notes: string;
  };
}

export async function executePhase6(
  context: Phase6Context
): Promise<PhaseResult<Report>> {
  logger.info('=== Phase 6: Report Generation ===');

  try {
    const report = generateReport(context);
    
    logger.info('Phase 6 completed', {
      recommendation: report.recommendation.action,
      confidence: report.recommendation.confidence,
    });

    return {
      success: true,
      data: report,
      confidence: 100,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 6 error', { error: message });
    return { success: false, error: message };
  }
}

function generateReport(context: Phase6Context): Report {
  const { repoAnalysis, researchSpec, mapping, patch, validation } = context;
  
  const algorithmName = researchSpec?.algorithm?.name || 'Unknown';
  const paperTitle = researchSpec?.paper?.title || 'Unknown';

  return {
    metadata: {
      integrationId: context.repoPath,
      repoUrl: context.repoPath,
      paper: paperTitle,
      algorithm: algorithmName,
      createdAt: new Date().toISOString(),
    },
    summary: {
      status: validation?.passed ? 'completed' : 'needs_review',
      confidence: validation?.passed ? 95 : 60,
      changesMade: patch?.transformations?.length || 0,
      filesModified: patch?.transformations?.map(t => t.file) || [],
      newFiles: patch?.newFiles?.map(f => f.path) || [],
    },
    whatChanged: generateWhatChanged(mapping, researchSpec),
    why: generateWhy(researchSpec),
    observedImpact: {
      metricsComparison: validation?.comparison || {},
      meetsExpectations: validation?.passed || false,
    },
    riskNotes: generateRiskNotes(validation, mapping),
    diffPreview: generateDiffPreview(patch),
    testResults: {
      unitTestsPassed: validation?.passed || false,
      benchmarkResults: validation?.comparison || {},
    },
    recommendation: {
      action: generateRecommendationAction(validation),
      confidence: validation?.passed ? 95 : 50,
      notes: generateRecommendationNotes(validation, researchSpec),
    },
  };
}

function generateWhatChanged(mapping: any, spec: any): string {
  const algorithmName = spec?.algorithm?.name || 'the research';
  const targetPattern = spec?.changes?.targetPattern || 'components';
  
  return `This integration applies ${algorithmName} by replacing ${targetPattern} in the codebase.`;
}

function generateWhy(spec: any): string {
  const algorithmName = spec?.algorithm?.name || 'The technique';
  const benefits = spec?.changes?.expectedBenefits?.join(', ') || 'improved performance';
  
  return `${algorithmName} offers ${benefits}.`;
}

function generateRiskNotes(validation: any, mapping: any): string[] {
  const notes: string[] = [];
  
  if (validation?.comparison?.speedup < 1.0) {
    notes.push('Performance regression detected - review required');
  }
  
  if (validation?.comparison?.lossChange > 5) {
    notes.push('Significant loss change observed - verify model quality');
  }
  
  if (mapping?.confidence < 70) {
    notes.push('Low mapping confidence - manual review recommended');
  }
  
  return notes;
}

function generateDiffPreview(patch: any): string {
  if (!patch) return 'No patch generated';
  
  const lines: string[] = [];
  
  if (patch.newFiles?.length) {
    lines.push(`New files: ${patch.newFiles.map((f: any) => f.path).join(', ')}`);
  }
  
  if (patch.transformations?.length) {
    lines.push(`Modified: ${patch.transformations.map((t: any) => t.file).join(', ')}`);
  }
  
  return lines.join('\n') || 'No changes';
}

function generateRecommendationAction(validation: any): 'approve' | 'review' | 'reject' {
  if (!validation) return 'review';
  if (!validation.passed) return 'reject';
  if (validation.comparison?.speedup < 1.05) return 'review';
  return 'approve';
}

function generateRecommendationNotes(validation: any, spec: any): string {
  const algorithmName = spec?.algorithm?.name || 'The integration';
  
  if (!validation) {
    return `${algorithmName} validation could not be completed. Manual review required.`;
  }
  
  if (validation.passed) {
    return `${algorithmName} passed validation with ${validation.comparison?.speedup?.toFixed(2)}x speedup. Ready for integration.`;
  }
  
  return `${algorithmName} validation failed at ${validation.stage}. Review required before proceeding.`;
}

export function validatePhase6Input(context: Phase6Context): string | null {
  if (!context.validation) {
    return 'Validation result is required (complete Phase 5)';
  }

  return null;
}
