import { PythonSubprocessBridge, PythonHttpBridge } from '../bridges/python-bridge.js';
import { FunctionNode, WorkflowNode, ConditionalNode } from './node.js';
import type { WorkflowState } from './types.js';
import { logger } from '../utils/logger.js';

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: 'integration' | 'experiment' | 'analysis' | 'custom';
  createNodes: (bridge: PythonSubprocessBridge | PythonHttpBridge) => WorkflowNode[];
}

export const templates: WorkflowTemplate[] = [
  {
    id: 'full-integration',
    name: 'Full Integration',
    description: 'Complete 6-phase integration workflow',
    category: 'integration',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createResearchNode(bridge),
      createMappingNode(bridge),
      createPatchNode(bridge),
      createValidationNode(bridge),
      createReportNode(),
    ],
  },
  {
    id: 'quick-analyze',
    name: 'Quick Analysis',
    description: 'Just analyze and suggest improvements',
    category: 'analysis',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createSuggestNode(bridge),
    ],
  },
  {
    id: 'planner-workflow',
    name: 'Planner Workflow',
    description: 'Plan and execute multi-spec migration',
    category: 'integration',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createPlannerNode(bridge),
      createMultiExecuteNode(bridge),
    ],
  },
  {
    id: 'critic-workflow',
    name: 'Critic Workflow',
    description: 'Generate patch and verify with critic',
    category: 'integration',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createResearchNode(bridge),
      createMappingNode(bridge),
      createPatchNode(bridge),
      createCriticNode(bridge),
      createValidationNode(bridge),
    ],
  },
  {
    id: 'experiment-workflow',
    name: 'Experiment Workflow',
    description: 'Run hypothesis experiments with variants',
    category: 'experiment',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createExperimentNode(bridge),
    ],
  },
  {
    id: 'safe-integration',
    name: 'Safe Integration',
    description: 'Integration with critic and validation gates',
    category: 'integration',
    createNodes: (bridge) => [
      createAnalyzeNode(bridge),
      createResearchNode(bridge),
      createMappingNode(bridge),
      createPatchNode(bridge),
      createCriticNode(bridge),
      createValidationGateNode(bridge),
      createReportNode(),
    ],
  },
];

export function getTemplate(id: string): WorkflowTemplate | undefined {
  return templates.find(t => t.id === id);
}

export function getTemplatesByCategory(category: string): WorkflowTemplate[] {
  return templates.filter(t => t.category === category);
}

export function listTemplates(): { id: string; name: string; description: string; category: string }[] {
  return templates.map(({ id, name, description, category }) => ({
    id,
    name,
    description,
    category,
  }));
}

function createAnalyzeNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'analyze', name: 'Repository Analysis', timeout: 300000 },
    async (context, state) => {
      const result = await bridge.analyzeRepo(context.repoPath as string);
      if (!result.success) throw new Error(result.error);
      state.context.analysis = result.data;
      return result.data;
    }
  );
}

function createResearchNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'research', name: 'Research Extraction', dependencies: ['analyze'], timeout: 300000 },
    async (context, state) => {
      const result = await bridge.extractResearch(
        context.paperSource as string,
        context.sourceType as 'pdf' | 'arxiv'
      );
      if (!result.success) throw new Error(result.error);
      state.context.research = result.data;
      return result.data;
    }
  );
}

function createMappingNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'mapping', name: 'Architecture Mapping', dependencies: ['analyze', 'research'], timeout: 300000 },
    async (context, state) => {
      const result = await bridge.mapArchitecture(
        state.context.analysis,
        state.context.research
      );
      if (!result.success) throw new Error(result.error);
      state.context.mapping = result.data;
      return result.data;
    }
  );
}

function createPatchNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'patch', name: 'Patch Generation', dependencies: ['mapping'], timeout: 600000 },
    async (context, state) => {
      const result = await bridge.generatePatch(state.context.mapping);
      if (!result.success) throw new Error(result.error);
      state.context.patch = result.data;
      return result.data;
    }
  );
}

function createValidationNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'validation', name: 'Validation', dependencies: ['patch'], timeout: 600000 },
    async (context, state) => {
      const result = await bridge.validate(state.context.patch, context.repoPath as string);
      if (!result.success) throw new Error(result.error);
      state.context.validation = result.data;
      return result.data;
    }
  );
}

function createReportNode(): WorkflowNode {
  return new FunctionNode(
    { id: 'report', name: 'Report Generation', dependencies: ['validation'], timeout: 60000 },
    async (context, state) => {
      return {
        analysis: state.context.analysis,
        research: state.context.research,
        mapping: state.context.mapping,
        patch: state.context.patch,
        validation: state.context.validation,
      };
    }
  );
}

function createCriticNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'critic', name: 'Code Critic', dependencies: ['patch'], timeout: 120000 },
    async (context, state) => {
      const baseUrl = (bridge as PythonHttpBridge).baseUrl || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/critic/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repoPath: context.repoPath,
          patchResult: state.context.patch,
        }),
      });
      if (!response.ok) throw new Error(`Critic failed: ${response.statusText}`);
      const result = await response.json();
      state.context.critic = result;
      return result;
    }
  );
}

function createValidationGateNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'validation-gate', name: 'Validation Gate', dependencies: ['critic', 'validation'], timeout: 60000 },
    async (context, state) => {
      const critic = state.context.critic as any;
      const validation = state.context.validation as any;
      
      const gatePassed = 
        (critic?.payload?.issue_count || 0) === 0 &&
        (validation?.passed || false);
      
      state.context.gatePassed = gatePassed;
      
      if (!gatePassed) {
        logger.warn('Validation gate not passed');
      }
      
      return { passed: gatePassed, critic, validation };
    }
  );
}

function createSuggestNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'suggest', name: 'Get Suggestions', dependencies: ['analyze'], timeout: 120000 },
    async (context, state) => {
      const baseUrl = (bridge as PythonHttpBridge).baseUrl || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoPath: context.repoPath }),
      });
      if (!response.ok) throw new Error(`Suggest failed: ${response.statusText}`);
      const result = await response.json();
      state.context.suggestions = result;
      return result;
    }
  );
}

function createPlannerNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'planner', name: 'Multi-Spec Planner', dependencies: ['analyze'], timeout: 120000 },
    async (context, state) => {
      const baseUrl = (bridge as PythonHttpBridge).baseUrl || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/planner/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repoPath: context.repoPath,
          maxSpecs: (context.maxSpecs as number) || 5,
        }),
      });
      if (!response.ok) throw new Error(`Planner failed: ${response.statusText}`);
      const result = await response.json();
      state.context.plan = result;
      return result;
    }
  );
}

function createMultiExecuteNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'multi-execute', name: 'Execute Planned Specs', dependencies: ['planner'], timeout: 600000 },
    async (context, state) => {
      const plan = state.context.plan as any;
      const specs = plan?.payload?.selected_specs || [];
      
      const results = [];
      for (const spec of specs) {
        const mapping = await bridge.mapArchitecture(
          state.context.analysis,
          { spec_name: spec.name }
        );
        const patch = await bridge.generatePatch(mapping.data);
        const validation = await bridge.validate(patch.data, context.repoPath as string);
        results.push({ spec, mapping: mapping.data, patch: patch.data, validation: validation.data });
      }
      
      state.context.multiResults = results;
      return results;
    }
  );
}

function createExperimentNode(bridge: PythonSubprocessBridge | PythonHttpBridge): WorkflowNode {
  return new FunctionNode(
    { id: 'experiment', name: 'Experiment Loop', dependencies: ['analyze'], timeout: 600000 },
    async (context, state) => {
      const baseUrl = (bridge as PythonHttpBridge).baseUrl || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/experiment/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repoPath: context.repoPath,
          spec: context.spec,
          variants: (context.variants as number) || 3,
        }),
      });
      if (!response.ok) throw new Error(`Experiment failed: ${response.statusText}`);
      const result = await response.json();
      state.context.experiment = result;
      return result;
    }
  );
}
