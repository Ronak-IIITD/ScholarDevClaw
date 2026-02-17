import { FunctionNode, WorkflowNode, ParallelNode, ConditionalNode } from './node.js';
import type { WorkflowState, NodeResult } from './types.js';
import type { PythonSubprocessBridge, PythonHttpBridge } from '../bridges/python-bridge.js';
import { logger } from '../utils/logger.js';

export interface PhaseConfig {
  id: string;
  name: string;
  timeout?: number;
  retryCount?: number;
}

export function createAnalyzeNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase1-analyze',
      name: config?.name || 'Repository Analysis',
      timeout: config?.timeout || 300000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 1: Repository Analysis');
      const repoPath = context.repoPath as string;
      
      const result = await bridge.analyzeRepo(repoPath);
      
      if (!result.success || !result.data) {
        throw new Error(`Phase 1 failed: ${result.error || 'No data'}`);
      }
      
      state.context.repoAnalysis = result.data;
      return result.data;
    }
  );
}

export function createResearchNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase2-research',
      name: config?.name || 'Research Extraction',
      dependencies: ['phase1-analyze'],
      timeout: config?.timeout || 300000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 2: Research Extraction');
      const paperSource = context.paperSource as string;
      const sourceType = context.sourceType as 'pdf' | 'arxiv';
      
      const result = await bridge.extractResearch(paperSource, sourceType);
      
      if (!result.success || !result.data) {
        throw new Error(`Phase 2 failed: ${result.error || 'No data'}`);
      }
      
      state.context.researchSpec = result.data;
      return result.data;
    }
  );
}

export function createMappingNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase3-mapping',
      name: config?.name || 'Architecture Mapping',
      dependencies: ['phase1-analyze', 'phase2-research'],
      timeout: config?.timeout || 300000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 3: Architecture Mapping');
      const repoAnalysis = state.context.repoAnalysis;
      const researchSpec = state.context.researchSpec;
      
      if (!repoAnalysis || !researchSpec) {
        throw new Error('Missing repo analysis or research spec');
      }
      
      const result = await bridge.mapArchitecture(repoAnalysis, researchSpec);
      
      if (!result.success || !result.data) {
        throw new Error(`Phase 3 failed: ${result.error || 'No data'}`);
      }
      
      state.context.mapping = result.data;
      return result.data;
    }
  );
}

export function createPatchNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase4-patch',
      name: config?.name || 'Patch Generation',
      dependencies: ['phase3-mapping'],
      timeout: config?.timeout || 600000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 4: Patch Generation');
      const mapping = state.context.mapping;
      
      if (!mapping) {
        throw new Error('Missing mapping');
      }
      
      const result = await bridge.generatePatch(mapping);
      
      if (!result.success || !result.data) {
        throw new Error(`Phase 4 failed: ${result.error || 'No data'}`);
      }
      
      state.context.patch = result.data;
      return result.data;
    }
  );
}

export function createValidationNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase5-validation',
      name: config?.name || 'Validation',
      dependencies: ['phase4-patch'],
      timeout: config?.timeout || 600000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 5: Validation');
      const patch = state.context.patch;
      const repoPath = context.repoPath as string;
      
      if (!patch) {
        throw new Error('Missing patch');
      }
      
      const result = await bridge.validate(patch, repoPath);
      
      if (!result.success || !result.data) {
        throw new Error(`Phase 5 failed: ${result.error || 'No data'}`);
      }
      
      state.context.validation = result.data;
      return result.data;
    }
  );
}

export function createReportNode(
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'phase6-report',
      name: config?.name || 'Report Generation',
      dependencies: ['phase5-validation'],
      timeout: config?.timeout || 60000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Phase 6: Report Generation');
      
      const report = {
        repoAnalysis: state.context.repoAnalysis,
        researchSpec: state.context.researchSpec,
        mapping: state.context.mapping,
        patch: state.context.patch,
        validation: state.context.validation,
        completedAt: new Date().toISOString(),
      };
      
      state.context.report = report;
      return report;
    }
  );
}

export function createPlannerNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'planner',
      name: config?.name || 'Multi-Spec Planning',
      timeout: config?.timeout || 120000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Planner');
      const repoPath = context.repoPath as string;
      const maxSpecs = (context.maxSpecs as number) || 5;

      const response = await fetch(`${(bridge as PythonHttpBridge)['baseUrl'] || 'http://localhost:8000'}/planner/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoPath, maxSpecs }),
      });

      if (!response.ok) {
        throw new Error(`Planner failed: ${response.statusText}`);
      }

      const result = await response.json();
      state.context.plannerResult = result;
      return result;
    }
  );
}

export function createCriticNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'critic',
      name: config?.name || 'Code Critic',
      dependencies: ['phase4-patch'],
      timeout: config?.timeout || 120000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Critic');
      const patch = state.context.patch;
      const repoPath = context.repoPath as string;
      const spec = context.spec as string;

      const response = await fetch(`${(bridge as PythonHttpBridge)['baseUrl'] || 'http://localhost:8000'}/critic/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoPath, spec, patchResult: patch }),
      });

      if (!response.ok) {
        throw new Error(`Critic failed: ${response.statusText}`);
      }

      const result = await response.json();
      state.context.criticResult = result;

      if (result.payload?.issue_count > 0) {
        state.context.criticIssues = result.payload.issues;
      }

      return result;
    }
  );
}

export function createExperimentNode(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  config?: Partial<PhaseConfig>
): WorkflowNode {
  return new FunctionNode(
    {
      id: config?.id || 'experiment',
      name: config?.name || 'Experiment Loop',
      timeout: config?.timeout || 600000,
      retryCount: config?.retryCount || 0,
    },
    async (context: Record<string, unknown>, state: WorkflowState) => {
      logger.info('Executing Experiment Loop');
      const repoPath = context.repoPath as string;
      const spec = context.spec as string;
      const variants = (context.variants as number) || 3;

      const response = await fetch(`${(bridge as PythonHttpBridge)['baseUrl'] || 'http://localhost:8000'}/experiment/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoPath, spec, variants }),
      });

      if (!response.ok) {
        throw new Error(`Experiment failed: ${response.statusText}`);
      }

      const result = await response.json();
      state.context.experimentResult = result;
      return result;
    }
  );
}

export function createIntegrationWorkflow(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  options?: {
    includeCritic?: boolean;
    includePlanner?: boolean;
  }
): WorkflowNode[] {
  const nodes: WorkflowNode[] = [
    createAnalyzeNode(bridge),
    createResearchNode(bridge),
    createMappingNode(bridge),
    createPatchNode(bridge),
  ];

  if (options?.includeCritic) {
    nodes.push(createCriticNode(bridge));
  }

  nodes.push(createValidationNode(bridge));
  nodes.push(createReportNode());

  return nodes;
}

export function createExperimentWorkflow(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  options?: {
    includeCritic?: boolean;
  }
): WorkflowNode[] {
  const nodes: WorkflowNode[] = [
    createExperimentNode(bridge),
  ];

  if (options?.includeCritic) {
    const criticNode = createCriticNode(bridge);
    criticNode.dependencies = ['experiment'];
    nodes.push(criticNode);
  }

  return nodes;
}