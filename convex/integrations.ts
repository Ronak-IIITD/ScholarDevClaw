import { Query } from "../_generated/dataModel";
import { v } from "convex/server";

export const list: Query = async ({ db }) => {
  return await db.query("integrations").order("desc").take(100);
};

export const listByStatus = async ({ db }, { status }: { status: string }) => {
  return await db
    .query("integrations")
    .withIndex("by_status", (q) => q.eq("status", status))
    .take(100);
};

export const get: Query = async ({ db }, { id }: { id: string }) => {
  return await db.get(id as any);
};

export const listApprovals: Query = async ({ db }, { integrationId }: { integrationId: string }) => {
  return await db
    .query("approvals")
    .withIndex("by_integration", (q) => q.eq("integrationId", integrationId as any))
    .collect();
};
