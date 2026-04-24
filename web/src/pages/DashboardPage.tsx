import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getHealth, getSpecs, getPipelineStatus } from "@/lib/api";
import type { SpecSummary, PipelineRunStatus } from "@/types/api";
import StatusBadge from "@/components/StatusBadge";
import {
  Activity,
  BookOpen,
  Server,
  Sparkles,
  ArrowRight,
  Code2,
  Terminal,
} from "lucide-react";

export default function DashboardPage() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [pipeline, setPipeline] = useState<PipelineRunStatus | null>(null);

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
    ]).catch(() => {});
  }, []);

  const categories = [...new Set(specs.map((s) => s.category))];

  return (
    <div className="max-w-6xl mx-auto space-y-12 pb-12 animate-fade-in">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-[2.5rem] border border-white/5 bg-gradient-to-b from-gray-900 to-black p-10 md:p-14 shadow-2xl">
        {/* Abstract glow effects */}
        <div className="absolute top-0 right-0 -mt-32 -mr-32 h-96 w-96 rounded-full bg-brand-600/20 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-0 left-0 -mb-32 -ml-32 h-96 w-96 rounded-full bg-violet-600/20 blur-[100px] pointer-events-none" />
        
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-sm text-gray-300 backdrop-blur-md mb-8">
            <Sparkles size={14} className="text-brand-400" />
            <span>ScholarDevClaw v2.0</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-white mb-6 leading-[1.1]">
            Research papers to{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 via-violet-400 to-brand-400 animate-shimmer" style={{ backgroundSize: "200% 100%" }}>
              production code.
            </span>
          </h1>
          <p className="text-xl text-gray-400 mb-10 max-w-2xl leading-relaxed font-light">
            Instantly transform arXiv papers into production-ready codebases. 
            Drop a PDF, and the autonomous engine builds the architecture, logic, and tests.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <Link
              to="/paper-to-code"
              className="group flex items-center gap-2 rounded-2xl bg-white text-gray-950 px-8 py-4 font-semibold hover:bg-gray-100 transition-all shadow-[0_0_40px_rgba(255,255,255,0.15)] hover:shadow-[0_0_60px_rgba(255,255,255,0.25)] hover:scale-[1.02]"
            >
              <Sparkles size={18} className="text-brand-600" />
              Try Paper → Code
              <ArrowRight size={18} className="text-gray-400 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/pipeline"
              className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-8 py-4 font-medium text-white hover:bg-white/10 transition-all backdrop-blur-md hover:scale-[1.02]"
            >
              <Activity size={18} className="text-gray-400" />
              View Active Pipeline
            </Link>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Server Status */}
        <div className="group relative overflow-hidden rounded-3xl border border-white/5 bg-gray-900/40 p-7 hover:bg-gray-900/60 transition-all backdrop-blur-xl">
          <div className="absolute top-0 right-0 p-5 opacity-5 group-hover:opacity-10 transition-opacity">
            <Server size={80} />
          </div>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3 text-gray-400">
              <div className="p-2 rounded-xl bg-gray-800/80 border border-white/5">
                <Server size={18} className="text-brand-400" />
              </div>
              <span className="text-sm font-medium tracking-wide">System Status</span>
            </div>
          </div>
          <div className="flex items-end justify-between">
            <div>
              <p className="text-3xl font-bold text-white tracking-tight mb-1">
                {healthy === null ? "..." : healthy ? "Online" : "Offline"}
              </p>
              <p className="text-sm text-gray-500 font-medium">
                {healthy ? "Backend connected" : "No response"}
              </p>
            </div>
            {healthy === null ? (
              <span className="h-3 w-3 rounded-full bg-gray-600 animate-pulse mb-2" />
            ) : healthy ? (
              <span className="h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.8)] mb-2 animate-pulse" />
            ) : (
              <span className="h-3 w-3 rounded-full bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8)] mb-2" />
            )}
          </div>
        </div>

        {/* Knowledge Base */}
        <Link to="/specs" className="group relative overflow-hidden rounded-3xl border border-white/5 bg-gray-900/40 p-7 hover:bg-gray-900/60 transition-all hover:-translate-y-1 backdrop-blur-xl">
          <div className="absolute top-0 right-0 p-5 opacity-5 group-hover:opacity-10 transition-opacity">
            <BookOpen size={80} />
          </div>
          <div className="flex items-center gap-3 text-gray-400 mb-6">
            <div className="p-2 rounded-xl bg-gray-800/80 border border-white/5">
              <BookOpen size={18} className="text-violet-400" />
            </div>
            <span className="text-sm font-medium tracking-wide">Knowledge Base</span>
          </div>
          <p className="text-3xl font-bold text-white tracking-tight mb-1">{specs.length || "0"}</p>
          <p className="text-sm text-gray-500 font-medium">
            Papers in <span className="text-gray-300">{categories.length}</span> categories
          </p>
        </Link>

        {/* Pipeline Runs */}
        <Link to="/pipeline" className="group relative overflow-hidden rounded-3xl border border-white/5 bg-gray-900/40 p-7 hover:bg-gray-900/60 transition-all hover:-translate-y-1 backdrop-blur-xl">
          <div className="absolute top-0 right-0 p-5 opacity-5 group-hover:opacity-10 transition-opacity">
            <Activity size={80} />
          </div>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3 text-gray-400">
              <div className="p-2 rounded-xl bg-gray-800/80 border border-white/5">
                <Activity size={18} className="text-blue-400" />
              </div>
              <span className="text-sm font-medium tracking-wide">Active Pipeline</span>
            </div>
          </div>
          <div className="mb-3">
            <StatusBadge status={pipeline?.status ?? "idle"} size="md" />
          </div>
          <p className="text-sm text-gray-500 font-medium">
            {pipeline?.steps.length || 0} steps executed
          </p>
        </Link>

        {/* Modules Generated */}
        <div className="group relative overflow-hidden rounded-3xl border border-white/5 bg-gray-900/40 p-7 hover:bg-gray-900/60 transition-all backdrop-blur-xl">
          <div className="absolute top-0 right-0 p-5 opacity-5 group-hover:opacity-10 transition-opacity">
            <Code2 size={80} />
          </div>
          <div className="flex items-center gap-3 text-gray-400 mb-6">
            <div className="p-2 rounded-xl bg-gray-800/80 border border-white/5">
              <Code2 size={18} className="text-emerald-400" />
            </div>
            <span className="text-sm font-medium tracking-wide">Code Generated</span>
          </div>
          <p className="text-3xl font-bold text-white tracking-tight mb-1">
            {pipeline?.steps.filter((s) => s.step.startsWith("generate") && s.status === "completed").length ?? 0}
          </p>
          <p className="text-sm text-gray-500 font-medium">
            Modules successfully built
          </p>
        </div>
      </div>

      {/* Developer Tools */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded-3xl border border-white/5 bg-gray-900/20 overflow-hidden backdrop-blur-xl flex flex-col">
          <div className="border-b border-white/5 bg-gray-900/40 px-6 py-5 flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-gray-800 border border-white/5">
              <Terminal size={16} className="text-gray-400" />
            </div>
            <h3 className="text-base font-semibold text-white">CLI Quick Start</h3>
          </div>
          <div className="p-8 flex-1 flex flex-col justify-center">
            <div className="rounded-2xl bg-black p-6 font-mono text-[13px] leading-relaxed text-gray-400 border border-white/5 shadow-2xl">
              <div className="flex items-center gap-4 hover:bg-white/5 p-2 rounded-xl transition-colors group">
                <span className="text-gray-600 select-none">$</span>
                <span className="text-brand-400 font-medium">scholardevclaw</span>
                <span className="text-white">from-paper</span>
                <span className="text-emerald-400">"arxiv:1706.03762"</span>
              </div>
              <div className="flex items-center gap-4 hover:bg-white/5 p-2 rounded-xl transition-colors group">
                <span className="text-gray-600 select-none">$</span>
                <span className="text-brand-400 font-medium">scholardevclaw</span>
                <span className="text-white">demo</span>
                <span className="text-gray-500">--live</span>
              </div>
              <div className="flex items-center gap-4 hover:bg-white/5 p-2 rounded-xl transition-colors group mt-4">
                <span className="text-gray-600 select-none">#</span>
                <span className="text-gray-500 italic">Start the background API server</span>
              </div>
              <div className="flex items-center gap-4 hover:bg-white/5 p-2 rounded-xl transition-colors group">
                <span className="text-gray-600 select-none">$</span>
                <span className="text-brand-400 font-medium">uvicorn</span>
                <span className="text-white">scholardevclaw.api.server:app</span>
                <span className="text-gray-500">--reload</span>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-white/5 bg-gray-900/20 p-8 flex flex-col backdrop-blur-xl">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-1.5 rounded-lg bg-gray-800 border border-white/5">
              <BookOpen size={16} className="text-gray-400" />
            </div>
            <h3 className="text-base font-semibold text-white">Categories</h3>
          </div>
          {categories.length > 0 ? (
            <div className="flex flex-wrap gap-3">
              {categories.sort().map((cat) => {
                const count = specs.filter((s) => s.category === cat).length;
                return (
                  <div
                    key={cat}
                    className="flex items-center justify-between w-full rounded-2xl bg-white/5 px-5 py-4 text-sm hover:bg-white/10 transition-colors border border-white/5"
                  >
                    <span className="text-gray-300 font-medium capitalize tracking-wide">{cat}</span>
                    <span className="flex h-7 min-w-7 items-center justify-center rounded-full bg-black/50 border border-white/5 text-xs font-medium text-gray-400 shadow-inner">
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-white/5 rounded-2xl">
              <BookOpen size={24} className="text-gray-600 mb-3" />
              <p className="text-sm text-gray-500 font-medium">No specs loaded yet.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
