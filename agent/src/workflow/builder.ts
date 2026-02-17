import { PythonSubprocessBridge, PythonHttpBridge } from '../bridges/python-bridge.js';
import { WorkflowNode, FunctionNode, ConditionalNode } from './node.js';
import type { NodeConfig } from './node.js';

export interface NodeDefinition {
  id: string;
  type: 'function' | 'conditional' | 'parallel';
  config: NodeConfig;
  handler?: string;
  condition?: string;
  trueNode?: string;
  falseNode?: string;
  children?: string[];
}

export interface WorkflowDefinition {
  id: string;
  name: string;
  description?: string;
  nodes: NodeDefinition[];
  edges: { from: string; to: string }[];
}

export interface DynamicWorkflowBuilderOptions {
  bridge: PythonSubprocessBridge | PythonHttpBridge;
  context?: Record<string, unknown>;
}

export class DynamicWorkflowBuilder {
  private bridge: PythonSubprocessBridge | PythonHttpBridge;
  private nodes: Map<string, WorkflowNode> = new Map();
  private nodeDefs: Map<string, NodeDefinition> = new Map();
  private context: Record<string, unknown>;

  constructor(options: DynamicWorkflowBuilderOptions) {
    this.bridge = options.bridge;
    this.context = options.context || {};
  }

  addNodeDefinition(definition: NodeDefinition): DynamicWorkflowBuilder {
    this.nodeDefs.set(definition.id, definition);
    return this;
  }

  addNodeDefinitions(definitions: NodeDefinition[]): DynamicWorkflowBuilder {
    for (const def of definitions) {
      this.addNodeDefinition(def);
    }
    return this;
  }

  addEdge(from: string, to: string): DynamicWorkflowBuilder {
    const fromDef = this.nodeDefs.get(from);
    if (fromDef) {
      fromDef.config.dependencies = [...(fromDef.config.dependencies || []), to];
    }
    return this;
  }

  addEdges(edges: { from: string; to: string }[]): DynamicWorkflowBuilder {
    for (const edge of edges) {
      this.addEdge(edge.from, edge.to);
    }
    return this;
  }

  build(): WorkflowNode[] {
    for (const [id, def] of this.nodeDefs.entries()) {
      const node = this.createNode(def);
      if (node) {
        this.nodes.set(id, node);
      }
    }

    return Array.from(this.nodes.values());
  }

  private createNode(def: NodeDefinition): WorkflowNode | null {
    switch (def.type) {
      case 'function':
        return this.createFunctionNode(def);
      case 'conditional':
        return this.createConditionalNode(def);
      case 'parallel':
        return this.createParallelNode(def);
      default:
        return null;
    }
  }

  private createFunctionNode(def: NodeDefinition): WorkflowNode {
    const handler = this.getHandler(def.handler || def.id);
    
    return new FunctionNode(def.config, async (context, state) => {
      return handler(this.bridge, context, state);
    });
  }

  private createConditionalNode(def: NodeDefinition): WorkflowNode {
    const trueNode = def.trueNode ? this.nodeDefs.get(def.trueNode) : undefined;
    const falseNode = def.falseNode ? this.nodeDefs.get(def.falseNode) : undefined;

    const conditionFn = this.getCondition(def.condition || 'default');

    const trueWorkflowNode = trueNode ? this.createFunctionNode(trueNode) : undefined;
    const falseWorkflowNode = falseNode ? this.createFunctionNode(falseNode) : undefined;

    return new ConditionalNode(
      def.config,
      conditionFn,
      trueWorkflowNode || new FunctionNode({ id: 'noop', name: 'Noop' }, async () => null),
      falseWorkflowNode
    );
  }

  private createParallelNode(def: NodeDefinition): WorkflowNode {
    const children = def.children || [];
    const childNodes = children
      .map(id => this.nodeDefs.get(id))
      .filter((c): c is NodeDefinition => c !== undefined)
      .map(c => this.createFunctionNode(c));

    const wrapperConfig: NodeConfig = {
      id: def.id + '-wrapper',
      name: def.config.name + ' (parallel)',
      dependencies: def.config.dependencies,
      parallel: true,
    };

    return new (class extends FunctionNode {
      constructor() {
        super(wrapperConfig, async (context, state) => {
          return Promise.all(childNodes.map(node => node.execute(context, state)));
        });
      }
    })();
  }

  private getHandler(name: string): (bridge: any, context: any, state: any) => Promise<unknown> {
    const handlers: Record<string, (bridge: any, context: any, state: any) => Promise<unknown>> = {
      analyze: async (bridge, context) => {
        const result = await bridge.analyzeRepo(context.repoPath as string);
        if (!result.success) throw new Error(result.error);
        return result.data;
      },
      research: async (bridge, context, state) => {
        const result = await bridge.extractResearch(
          context.paperSource as string,
          context.sourceType as 'pdf' | 'arxiv'
        );
        if (!result.success) throw new Error(result.error);
        return result.data;
      },
      mapping: async (_bridge, context, state) => {
        const { bridge } = this;
        const result = await (bridge as any).mapArchitecture(
          state.context.analysis,
          state.context.research
        );
        if (!result.success) throw new Error(result.error);
        return result.data;
      },
      patch: async (_bridge, context, state) => {
        const { bridge } = this;
        const result = await (bridge as any).generatePatch(state.context.mapping);
        if (!result.success) throw new Error(result.error);
        return result.data;
      },
      validation: async (_bridge, context, state) => {
        const { bridge } = this;
        const result = await (bridge as any).validate(state.context.patch, context.repoPath as string);
        if (!result.success) throw new Error(result.error);
        return result.data;
      },
      report: async (_context, state) => {
        return {
          analysis: state.context.analysis,
          research: state.context.research,
          mapping: state.context.mapping,
          patch: state.context.patch,
          validation: state.context.validation,
        };
      },
    };

    return handlers[name] || (async () => ({ id: name, executed: true }));
  }

  private getCondition(name: string): (context: any, state: any) => boolean {
    const conditions: Record<string, (context: any, state: any) => boolean> = {
      default: () => true,
      hasAnalysis: (_context, state) => !!state.context.analysis,
      hasResearch: (_context, state) => !!state.context.research,
      hasMapping: (_context, state) => !!state.context.mapping,
      hasPatch: (_context, state) => !!state.context.patch,
      hasValidation: (_context, state) => !!state.context.validation,
      validationPassed: (_context, state) => {
        const validation = state.context.validation as any;
        return validation?.passed === true;
      },
      criticPassed: (_context, state) => {
        const critic = state.context.critic as any;
        return (critic?.payload?.issue_count || 0) === 0;
      },
    };

    return conditions[name] || conditions.default;
  }

  setContext(key: string, value: unknown): DynamicWorkflowBuilder {
    this.context[key] = value;
    return this;
  }

  getContext(): Record<string, unknown> {
    return { ...this.context };
  }

  static fromDefinition(
    definition: WorkflowDefinition,
    options: DynamicWorkflowBuilderOptions
  ): WorkflowNode[] {
    const builder = new DynamicWorkflowBuilder(options);
    builder.addNodeDefinitions(definition.nodes);
    builder.addEdges(definition.edges);
    return builder.build();
  }
}

export function createQuickWorkflow(
  bridge: PythonSubprocessBridge | PythonHttpBridge,
  steps: string[]
): WorkflowNode[] {
  const builder = new DynamicWorkflowBuilder({ bridge });

  const stepHandlers: Record<string, (bridge: any) => WorkflowNode> = {
    analyze: (b) => new FunctionNode(
      { id: 'analyze', name: 'Analyze', timeout: 300000 },
      async (context) => {
        const result = await b.analyzeRepo(context.repoPath as string);
        if (!result.success) throw new Error(result.error);
        return result.data;
      }
    ),
    research: (b) => new FunctionNode(
      { id: 'research', name: 'Research', timeout: 300000 },
      async (context) => {
        const result = await b.extractResearch(
          context.paperSource as string,
          context.sourceType as 'pdf' | 'arxiv'
        );
        if (!result.success) throw new Error(result.error);
        return result.data;
      }
    ),
    map: (b) => new FunctionNode(
      { id: 'map', name: 'Map', timeout: 300000 },
      async (_context, state) => {
        const result = await b.mapArchitecture(
          state.context.analysis,
          state.context.research
        );
        if (!result.success) throw new Error(result.error);
        return result.data;
      }
    ),
    generate: (b) => new FunctionNode(
      { id: 'generate', name: 'Generate', timeout: 600000 },
      async (_context, state) => {
        const result = await b.generatePatch(state.context.mapping);
        if (!result.success) throw new Error(result.error);
        return result.data;
      }
    ),
    validate: (b) => new FunctionNode(
      { id: 'validate', name: 'Validate', timeout: 600000 },
      async (context, state) => {
        const result = await b.validate(state.context.patch, context.repoPath as string);
        if (!result.success) throw new Error(result.error);
        return result.data;
      }
    ),
  };

  const nodes: WorkflowNode[] = [];
  let previousId: string | null = null;

  for (const step of steps) {
    const handler = stepHandlers[step];
    if (!handler) continue;

    const node = handler(bridge);
    
    if (previousId) {
      const prevNode = nodes.find(n => n.id === previousId);
      if (prevNode) {
        prevNode.dependencies.push(node.id);
      }
    }

    nodes.push(node);
    previousId = node.id;
  }

  return nodes;
}
