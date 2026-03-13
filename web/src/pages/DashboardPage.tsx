import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getHealth, getSpecs, getPipelineStatus } from "@/lib/api";
import type { SpecSummary, PipelineRunStatus } from "@/types/api";
import StatusBadge from "@/components/StatusBadge";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  Server,
  Wifi,
  WifiOff,
} from "lucide-react";

export default function DashboardPage() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [pipeline, setPipeline] = useState<PipelineRunStatus | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    Promise.all([
      getHealth()
        .then(() => setHealthy(true))
        .catch(() => setHealthy(false)),
      getSpecs()
        .then(setSpecs)
        .catch(() => {}),
      getPipelineStatus()
        .then(setPipeline)
        .catch(() => {}),
    ]).catch((e) => setError(String(e)));
  }, []);

  const categories = [...new Set(specs.map((s) => s.category))];

  return (
    <div>
      <h2 className="text-2xl font-bold text-white">Dashboard</h2>
      <p className="mt-1 text-sm text-gray-400">
        Research-to-code integration overview
      </p>

      {error && (
        <div className="mt-4 rounded-lg bg-red-900/20 border border-red-800 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Stats grid */}
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* API Health */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-gray-400">
              <Server size={16} />
              <span className="text-xs font-medium uppercase tracking-wide">
                API Server
              </span>
            </div>
            {healthy === null ? (
              <span className="h-2 w-2 rounded-full bg-gray-600" />
            ) : healthy ? (
              <Wifi size={16} className="text-green-400" />
            ) : (
              <WifiOff size={16} className="text-red-400" />
            )}
          </div>
          <p className="mt-3 text-2xl font-bold text-white">
            {healthy === null ? "..." : healthy ? "Online" : "Offline"}
          </p>
          <p className="text-xs text-gray-500">
            {healthy
              ? "Backend connected at :8000"
              : "Start with: uvicorn scholardevclaw.api.server:app"}
          </p>
        </div>

        {/* Paper Specs */}
        <Link
          to="/specs"
          className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 hover:border-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2 text-gray-400">
            <BookOpen size={16} />
            <span className="text-xs font-medium uppercase tracking-wide">
              Paper Specs
            </span>
          </div>
          <p className="mt-3 text-2xl font-bold text-white">{specs.length}</p>
          <p className="text-xs text-gray-500">
            {categories.length} categories
          </p>
        </Link>

        {/* Pipeline Status */}
        <Link
          to="/pipeline"
          className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 hover:border-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2 text-gray-400">
            <Activity size={16} />
            <span className="text-xs font-medium uppercase tracking-wide">
              Pipeline
            </span>
          </div>
          <div className="mt-3">
            <StatusBadge
              status={pipeline?.status ?? "idle"}
              size="md"
            />
          </div>
          <p className="mt-1 text-xs text-gray-500">
            {pipeline?.steps.length ?? 0} steps executed
          </p>
        </Link>

        {/* Completed Steps */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
          <div className="flex items-center gap-2 text-gray-400">
            <CheckCircle2 size={16} />
            <span className="text-xs font-medium uppercase tracking-wide">
              Steps Done
            </span>
          </div>
          <p className="mt-3 text-2xl font-bold text-white">
            {pipeline?.steps.filter((s) => s.status === "completed").length ?? 0}
          </p>
          <p className="text-xs text-gray-500">
            {pipeline?.total_seconds
              ? `Total: ${pipeline.total_seconds.toFixed(1)}s`
              : "No runs yet"}
          </p>
        </div>
      </div>

      {/* Category breakdown */}
      {specs.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-300">
            Spec Categories
          </h3>
          <div className="mt-3 flex flex-wrap gap-2">
            {categories.sort().map((cat) => {
              const count = specs.filter((s) => s.category === cat).length;
              return (
                <span
                  key={cat}
                  className="rounded-lg bg-gray-800 px-3 py-1.5 text-xs text-gray-300"
                >
                  {cat}{" "}
                  <span className="ml-1 text-gray-500">{count}</span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Quick start */}
      <div className="mt-8 rounded-xl border border-gray-800 bg-gray-900/40 p-6">
        <h3 className="text-sm font-semibold text-gray-300">Quick Start</h3>
        <div className="mt-3 space-y-2 font-mono text-xs text-gray-400">
          <p>
            <span className="text-gray-500"># Start the API server</span>
          </p>
          <p className="text-brand-400">
            $ uvicorn scholardevclaw.api.server:app --reload
          </p>
          <p className="mt-2">
            <span className="text-gray-500"># Start the dashboard</span>
          </p>
          <p className="text-brand-400">$ cd web && npm run dev</p>
          <p className="mt-2">
            <span className="text-gray-500"># Run a demo pipeline via CLI</span>
          </p>
          <p className="text-brand-400">
            $ scholardevclaw demo --skip-validate
          </p>
        </div>
      </div>
    </div>
  );
}
