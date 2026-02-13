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

export const updateStatus: Mutation = async ({ db }, args: {
  id: string;
  status: string;
  currentPhase?: number;
  updatedAt: number;
}) => {
  await db.patch(args.id as any, {
    status: args.status,
    currentPhase: args.currentPhase,
    updatedAt: args.updatedAt,
  });
};

export const savePhaseResult: Mutation = async ({ db }, args: {
  id: string;
  field: string;
  result: any;
  updatedAt: number;
}) => {
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
