import { Mutation } from "../_generated/dataModel";
import { v } from "convex/server";

export const create: Mutation = async ({ db }, args: {
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  mode?: "step_approval" | "autonomous";
}) => {
  const id = await db.insert("integrations", {
    repoUrl: args.repoUrl,
    paperUrl: args.paperUrl,
    paperPdfPath: args.paperPdfPath,
    status: "pending",
    mode: args.mode || "step_approval",
    currentPhase: 0,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    retryCount: 0,
  });
  return id;
};

// SECURITY: Allowed status values to prevent arbitrary state injection
const ALLOWED_STATUSES = new Set([
  "pending", "running", "awaiting_approval", "completed", "failed", "cancelled",
]);

export const updateStatus: Mutation = async ({ db }, args: {
  id: string;
  status: string;
  currentPhase?: number;
  awaitingReason?: string;
  guardrailReasons?: string[];
  updatedAt: number;
}) => {
  // SECURITY: Validate status against allowlist
  if (!ALLOWED_STATUSES.has(args.status)) {
    throw new Error(`Invalid status: ${args.status}`);
  }
  await db.patch(args.id as any, {
    status: args.status,
    currentPhase: args.currentPhase,
    awaitingReason: args.awaitingReason,
    guardrailReasons: args.guardrailReasons,
    updatedAt: args.updatedAt,
  });
};

// SECURITY: Whitelist allowed fields to prevent arbitrary field injection
const ALLOWED_PHASE_FIELDS = new Set([
  "repoAnalysis", "researchSpec", "mapping", "patch", "validation", "report",
  "confidence", "phaseLogs",
]);

export const savePhaseResult: Mutation = async ({ db }, args: {
  id: string;
  field: string;
  result: any;
  updatedAt: number;
}) => {
  // SECURITY: Reject fields not in the allowlist (prevents prototype pollution / field injection)
  if (!ALLOWED_PHASE_FIELDS.has(args.field)) {
    throw new Error(`Disallowed field: ${args.field}`);
  }

  const update: Record<string, any> = {
    updatedAt: args.updatedAt,
  };
  update[args.field] = args.result;
  
  await db.patch(args.id as any, update);
};

export const setConfidence: Mutation = async ({ db }, args: {
  id: string;
  confidence: number;
  updatedAt: number;
}) => {
  await db.patch(args.id as any, {
    confidence: args.confidence,
    updatedAt: args.updatedAt,
  });
};

export const setError: Mutation = async ({ db }, args: {
  id: string;
  errorMessage: string;
  status: string;
  updatedAt: number;
}) => {
  await db.patch(args.id as any, {
    errorMessage: args.errorMessage,
    status: args.status,
    updatedAt: args.updatedAt,
  });
};

export const incrementRetry: Mutation = async ({ db }, args: { id: string }) => {
  const doc = await db.get(args.id as any);
  if (!doc) return { retryCount: 0 };
  
  const retryCount = (doc.retryCount || 0) + 1;
  await db.patch(args.id as any, { retryCount });
  
  return { retryCount };
};

export const createApproval: Mutation = async ({ db }, args: {
  integrationId: string;
  phase: number;
  action: "approved" | "rejected";
  notes?: string;
}) => {
  await db.insert("approvals", {
    integrationId: args.integrationId as any,
    phase: args.phase,
    action: args.action,
    notes: args.notes,
    createdAt: Date.now(),
  });
};
