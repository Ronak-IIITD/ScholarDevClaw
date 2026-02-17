import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { dirname } from 'path';
import { WorkflowState, NodeResult } from './types.js';

export interface WorkflowSnapshot {
  workflowId: string;
  status: string;
  config: object;
  state: {
    workflowId: string;
    status: string;
    startedAt?: string;
    completedAt?: string;
    context: Record<string, unknown>;
    error?: string;
  };
  nodeResults: Record<string, NodeResult>;
  savedAt: string;
}

export class WorkflowStore {
  private storeDir: string;

  constructor(storeDir: string = './workflow-runs') {
    this.storeDir = storeDir;
  }

  private async ensureDir(): Promise<void> {
    if (!existsSync(this.storeDir)) {
      await mkdir(this.storeDir, { recursive: true });
    }
  }

  private getFilePath(workflowId: string): string {
    return `${this.storeDir}/${workflowId}.json`;
  }

  async save(workflowId: string, config: object, state: WorkflowState): Promise<void> {
    await this.ensureDir();

    const snapshot: WorkflowSnapshot = {
      workflowId,
      status: state.status,
      config,
      state: {
        workflowId: state.workflowId,
        status: state.status,
        startedAt: state.startedAt,
        completedAt: state.completedAt,
        context: state.context,
        error: state.error,
      },
      nodeResults: Object.fromEntries(state.nodeResults),
      savedAt: new Date().toISOString(),
    };

    await writeFile(
      this.getFilePath(workflowId),
      JSON.stringify(snapshot, null, 2)
    );
  }

  async load(workflowId: string): Promise<WorkflowSnapshot | null> {
    const filePath = this.getFilePath(workflowId);

    if (!existsSync(filePath)) {
      return null;
    }

    try {
      const content = await readFile(filePath, 'utf-8');
      return JSON.parse(content) as WorkflowSnapshot;
    } catch {
      return null;
    }
  }

  async list(): Promise<WorkflowSnapshot[]> {
    await this.ensureDir();

    try {
      const { readdir } = await import('fs/promises');
      const files = await readdir(this.storeDir);
      
      const snapshots: WorkflowSnapshot[] = [];
      
      for (const file of files) {
        if (!file.endsWith('.json')) continue;
        
        try {
          const content = await readFile(`${this.storeDir}/${file}`, 'utf-8');
          snapshots.push(JSON.parse(content));
        } catch {
          continue;
        }
      }
      
      return snapshots.sort((a, b) => 
        new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime()
      );
    } catch {
      return [];
    }
  }

  async delete(workflowId: string): Promise<void> {
    const { unlink } = await import('fs/promises');
    const filePath = this.getFilePath(workflowId);
    
    if (existsSync(filePath)) {
      await unlink(filePath);
    }
  }

  async getByStatus(status: string): Promise<WorkflowSnapshot[]> {
    const all = await this.list();
    return all.filter(s => s.status === status);
  }

  async getRecent(limit: number = 10): Promise<WorkflowSnapshot[]> {
    const all = await this.list();
    return all.slice(0, limit);
  }
}

export class ResumableWorkflow {
  private engine: DAGEngine;
  private store: WorkflowStore;
  private workflowId: string;

  constructor(
    engine: DAGEngine,
    store: WorkflowStore,
    workflowId: string
  ) {
    this.engine = engine;
    this.store = store;
    this.workflowId = workflowId;
  }

  async save(): Promise<void> {
    const state = this.engine.getState();
    await this.store.save(this.workflowId, {}, state);
  }

  static async resume(
    engine: DAGEngine,
    store: WorkflowStore,
    workflowId: string,
    nodes: WorkflowNode[]
  ): Promise<{ state: WorkflowState; nodeResults: Map<string, NodeResult> } | null> {
    const snapshot = await store.load(workflowId);
    
    if (!snapshot) {
      return null;
    }

    for (const node of nodes) {
      const result = snapshot.nodeResults[node.id];
      if (result) {
        engine.getState().nodeResults.set(node.id, result);
      }
    }

    return {
      state: {
        workflowId: snapshot.state.workflowId,
        status: snapshot.state.status as any,
        startedAt: snapshot.state.startedAt,
        completedAt: snapshot.state.completedAt,
        nodeResults: new Map(Object.entries(snapshot.nodeResults)),
        context: snapshot.state.context,
        error: snapshot.state.error,
      },
      nodeResults: new Map(Object.entries(snapshot.nodeResults)),
    };
  }
}

import { DAGEngine } from './engine.js';
import { WorkflowNode } from './node.js';
