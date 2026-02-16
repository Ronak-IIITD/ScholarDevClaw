import { mkdir, readFile, readdir, writeFile } from 'fs/promises';
import { join } from 'path';

export type RunSnapshotStatus =
  | 'pending'
  | 'running'
  | 'awaiting_approval'
  | 'completed'
  | 'failed';

export interface ApprovalRecord {
  phase: number;
  action: 'approved' | 'rejected';
  notes?: string;
  createdAt: string;
}

export interface RunSnapshot {
  runId: string;
  integrationId?: string;
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  mode: 'step_approval' | 'autonomous';
  status: RunSnapshotStatus;
  currentPhase: number;
  phaseResults: Record<number, unknown>;
  context: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  retryCount: number;
  lastErrorPhase?: number;
  awaitingReason?: string;
  guardrailReasons?: string[];
  errorMessage?: string;
  approvals: ApprovalRecord[];
}

export class RunStore {
  private readonly runsDir: string;

  constructor(baseDir: string = join(process.cwd(), 'workspace', 'runs')) {
    this.runsDir = baseDir;
  }

  async initialize(): Promise<void> {
    await mkdir(this.runsDir, { recursive: true });
  }

  private runPath(runId: string): string {
    return join(this.runsDir, `${runId}.json`);
  }

  async save(snapshot: RunSnapshot): Promise<void> {
    await this.initialize();
    await writeFile(this.runPath(snapshot.runId), `${JSON.stringify(snapshot, null, 2)}\n`, 'utf8');
  }

  async get(runId: string): Promise<RunSnapshot | null> {
    try {
      const content = await readFile(this.runPath(runId), 'utf8');
      return JSON.parse(content) as RunSnapshot;
    } catch {
      return null;
    }
  }

  async list(): Promise<RunSnapshot[]> {
    await this.initialize();
    const files = await readdir(this.runsDir);
    const snapshots: RunSnapshot[] = [];

    for (const file of files) {
      if (!file.endsWith('.json')) {
        continue;
      }
      try {
        const content = await readFile(join(this.runsDir, file), 'utf8');
        snapshots.push(JSON.parse(content) as RunSnapshot);
      } catch {
      }
    }

    return snapshots.sort((a, b) => (a.updatedAt > b.updatedAt ? -1 : 1));
  }

  async listByStatus(statuses: RunSnapshotStatus[]): Promise<RunSnapshot[]> {
    const all = await this.list();
    return all.filter((snapshot) => statuses.includes(snapshot.status));
  }

  async addApproval(
    runId: string,
    phase: number,
    action: 'approved' | 'rejected',
    notes?: string,
  ): Promise<void> {
    const snapshot = await this.get(runId);
    if (!snapshot) {
      throw new Error(`Run snapshot not found: ${runId}`);
    }

    const approval: ApprovalRecord = {
      phase,
      action,
      notes,
      createdAt: new Date().toISOString(),
    };

    snapshot.approvals = snapshot.approvals || [];
    snapshot.approvals.push(approval);
    snapshot.updatedAt = new Date().toISOString();

    await this.save(snapshot);
  }

  async getApprovals(runId: string): Promise<ApprovalRecord[]> {
    const snapshot = await this.get(runId);
    return snapshot?.approvals || [];
  }
}
