import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  Link2,
  FileText,
  Brain,
  Map,
  Code2,
  FlaskConical,
  Package,
  Sparkles,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowRight,
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

type PhaseId =
  | "ingest"
  | "understand"
  | "plan"
  | "generate"
  | "execute"
  | "scaffold";

type PhaseStatus = "pending" | "running" | "completed" | "failed";

interface Phase {
  id: PhaseId;
  label: string;
  icon: React.ElementType;
  description: string;
  status: PhaseStatus;
  detail?: string;
  duration?: number;
}

interface GeneratedModule {
  id: string;
  path: string;
  lines: number;
  description: string;
  code?: string;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const INITIAL_PHASES: Phase[] = [
  {
    id: "ingest",
    label: "Ingest",
    icon: FileText,
    description: "Parse PDF, extract equations & algorithms",
    status: "pending",
  },
  {
    id: "understand",
    label: "Understand",
    icon: Brain,
    description: "Deep comprehension of paper concepts",
    status: "pending",
  },
  {
    id: "plan",
    label: "Plan",
    icon: Map,
    description: "Decompose into implementation modules",
    status: "pending",
  },
  {
    id: "generate",
    label: "Generate",
    icon: Code2,
    description: "Produce code for each module",
    status: "pending",
  },
  {
    id: "execute",
    label: "Validate",
    icon: FlaskConical,
    description: "Run tests & score reproducibility",
    status: "pending",
  },
  {
    id: "scaffold",
    label: "Product",
    icon: Package,
    description: "API, docs, CI & deployment scaffold",
    status: "pending",
  },
];

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function cls(...parts: (string | false | undefined | null)[]) {
  return parts.filter(Boolean).join(" ");
}

function formatDuration(s: number) {
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

function PhaseNode({ phase, index }: { phase: Phase; index: number }) {
  const Icon = phase.icon;
  const statusColors: Record<PhaseStatus, string> = {
    pending: "border-gray-700 bg-gray-900/40 text-gray-500",
    running:
      "border-blue-500/60 bg-blue-950/40 text-blue-400 shadow-[0_0_20px_rgba(59,130,246,0.15)]",
    completed: "border-emerald-500/50 bg-emerald-950/30 text-emerald-400",
    failed: "border-red-500/50 bg-red-950/30 text-red-400",
  };
  const dotColors: Record<PhaseStatus, string> = {
    pending: "bg-gray-600",
    running: "bg-blue-400 animate-pulse",
    completed: "bg-emerald-400",
    failed: "bg-red-400",
  };

  return (
    <div
      className={cls(
        "relative flex items-center gap-4 rounded-xl border px-5 py-4 transition-all duration-500",
        statusColors[phase.status]
      )}
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* Status dot */}
      <div className={cls("h-2.5 w-2.5 rounded-full shrink-0", dotColors[phase.status])} />

      {/* Icon */}
      <div
        className={cls(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
          phase.status === "running"
            ? "bg-blue-500/20"
            : phase.status === "completed"
            ? "bg-emerald-500/20"
            : "bg-gray-800/60"
        )}
      >
        {phase.status === "running" ? (
          <Loader2 size={20} className="animate-spin" />
        ) : phase.status === "completed" ? (
          <CheckCircle2 size={20} />
        ) : phase.status === "failed" ? (
          <XCircle size={20} />
        ) : (
          <Icon size={20} />
        )}
      </div>

      {/* Text */}
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold">{phase.label}</p>
        <p className="mt-0.5 text-xs text-gray-400 truncate">
          {phase.detail || phase.description}
        </p>
      </div>

      {/* Duration */}
      {phase.duration != null && (
        <span className="shrink-0 rounded-md bg-gray-800/60 px-2 py-1 text-xs font-mono text-gray-300">
          {formatDuration(phase.duration)}
        </span>
      )}
    </div>
  );
}

function ModuleCard({
  mod,
  expanded,
  onToggle,
}: {
  mod: GeneratedModule;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    if (mod.code) {
      navigator.clipboard.writeText(mod.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  }, [mod.code]);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/50 overflow-hidden transition-all">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/40 transition-colors"
      >
        {expanded ? (
          <ChevronDown size={14} className="text-gray-500 shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-gray-500 shrink-0" />
        )}
        <Code2 size={14} className="text-brand-400 shrink-0" />
        <span className="flex-1 truncate font-mono text-sm text-gray-200">
          {mod.path}
        </span>
        <span className="shrink-0 text-xs text-gray-500">{mod.lines} lines</span>
      </button>

      {expanded && mod.code && (
        <div className="relative border-t border-gray-800">
          <button
            onClick={handleCopy}
            className="absolute right-2 top-2 rounded-md bg-gray-800 p-1.5 text-gray-400 hover:text-white transition-colors z-10"
            title="Copy code"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <pre className="max-h-80 overflow-auto p-4 text-xs leading-relaxed text-gray-300 font-mono">
            <code>{mod.code}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main Page                                                           */
/* ------------------------------------------------------------------ */

export default function PaperToCodePage() {
  const [mode, setMode] = useState<"url" | "upload">("url");
  const [paperUrl, setPaperUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [phases, setPhases] = useState<Phase[]>(INITIAL_PHASES);
  const [modules, setModules] = useState<GeneratedModule[]>([]);
  const [expandedModule, setExpandedModule] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [paperTitle, setPaperTitle] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  /* Simulated pipeline — replace with real WebSocket / API calls */
  const runPipeline = useCallback(async () => {
    const source = mode === "url" ? paperUrl.trim() : fileName;
    if (!source) {
      setError(mode === "url" ? "Please enter a paper URL or arXiv ID" : "Please upload a PDF");
      return;
    }

    setError("");
    setIsRunning(true);
    setModules([]);
    setPaperTitle("");
    setPhases(INITIAL_PHASES.map((p) => ({ ...p, status: "pending", detail: undefined, duration: undefined })));

    const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

    try {
      // Start the from-paper pipeline via POST
      const res = await fetch(`${API_BASE}/api/from-paper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, provider: "openrouter" }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Pipeline failed: ${body}`);
      }

      const data = await res.json();

      // Update phases based on response
      setPhases((prev) =>
        prev.map((p) => ({
          ...p,
          status: "completed" as PhaseStatus,
          duration: data.phase_durations?.[p.id] ?? undefined,
        }))
      );

      setPaperTitle(data.paper_title || source);

      if (data.modules) {
        setModules(
          data.modules.map((m: Record<string, unknown>, i: number) => ({
            id: String(m.id || `mod-${i}`),
            path: String(m.path || m.id || `module_${i}.py`),
            lines: Number(m.lines || m.estimated_lines || 0),
            description: String(m.description || ""),
            code: m.code ? String(m.code) : undefined,
          }))
        );
      }
    } catch (err) {
      setError(String(err));
      setPhases((prev) => {
        const first_pending = prev.findIndex((p) => p.status === "pending" || p.status === "running");
        if (first_pending >= 0) {
          return prev.map((p, i) =>
            i === first_pending ? { ...p, status: "failed" as PhaseStatus, detail: String(err) } : p
          );
        }
        return prev;
      });
    } finally {
      setIsRunning(false);
    }
  }, [mode, paperUrl, fileName]);

  const handleFileDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type === "application/pdf") {
        setFileName(file.name);
        setMode("upload");
      }
    },
    []
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setFileName(file.name);
        setMode("upload");
      }
    },
    []
  );

  const completedCount = phases.filter((p) => p.status === "completed").length;
  const totalDuration = phases.reduce((sum, p) => sum + (p.duration || 0), 0);
  const hasResults = completedCount > 0 || modules.length > 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 shadow-lg shadow-blue-500/20">
            <Sparkles size={20} className="text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Paper → Code</h2>
            <p className="text-sm text-gray-400">
              Drop a research paper and watch it transform into a working codebase
            </p>
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className="rounded-2xl border border-gray-800 bg-gradient-to-b from-gray-900/80 to-gray-950/80 p-6 backdrop-blur-sm">
        {/* Mode tabs */}
        <div className="flex gap-1 rounded-lg bg-gray-800/50 p-1 w-fit">
          <button
            onClick={() => setMode("url")}
            className={cls(
              "flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all",
              mode === "url"
                ? "bg-brand-600 text-white shadow-sm"
                : "text-gray-400 hover:text-gray-200"
            )}
          >
            <Link2 size={15} /> URL / arXiv ID
          </button>
          <button
            onClick={() => setMode("upload")}
            className={cls(
              "flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all",
              mode === "upload"
                ? "bg-brand-600 text-white shadow-sm"
                : "text-gray-400 hover:text-gray-200"
            )}
          >
            <Upload size={15} /> Upload PDF
          </button>
        </div>

        {/* URL input */}
        {mode === "url" && (
          <div className="mt-5">
            <input
              type="text"
              value={paperUrl}
              onChange={(e) => setPaperUrl(e.target.value)}
              placeholder="arxiv:1706.03762  or  https://arxiv.org/abs/2305.13245  or  DOI"
              disabled={isRunning}
              className="w-full rounded-xl border border-gray-700 bg-gray-950/80 px-5 py-3.5 font-mono text-sm text-gray-200 placeholder-gray-600 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500/50 disabled:opacity-50 transition-colors"
              onKeyDown={(e) => e.key === "Enter" && runPipeline()}
            />
          </div>
        )}

        {/* Upload area */}
        {mode === "upload" && (
          <div
            onDrop={handleFileDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
            className={cls(
              "mt-5 flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed p-10 transition-all",
              dragOver
                ? "border-brand-400 bg-brand-950/20"
                : "border-gray-700 bg-gray-950/40 hover:border-gray-600"
            )}
          >
            <Upload
              size={32}
              className={dragOver ? "text-brand-400" : "text-gray-500"}
            />
            <p className="text-sm text-gray-400">
              {fileName ? (
                <span className="text-brand-300 font-medium">{fileName}</span>
              ) : (
                <>
                  Drop your PDF here or{" "}
                  <span className="text-brand-400 underline">browse</span>
                </>
              )}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>
        )}

        {/* Error */}
        {error && (
          <p className="mt-3 rounded-lg bg-red-950/30 border border-red-900/50 px-4 py-2 text-sm text-red-400">
            {error}
          </p>
        )}

        {/* Launch button */}
        <div className="mt-5">
          <button
            onClick={runPipeline}
            disabled={isRunning}
            className={cls(
              "group flex items-center gap-2.5 rounded-xl px-6 py-3 text-sm font-semibold transition-all",
              isRunning
                ? "bg-gray-800 text-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-600/25 hover:shadow-blue-600/40 hover:brightness-110"
            )}
          >
            {isRunning ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Running pipeline…
              </>
            ) : (
              <>
                <Sparkles size={16} />
                Transform Paper to Code
                <ArrowRight
                  size={16}
                  className="transition-transform group-hover:translate-x-0.5"
                />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Pipeline Phases */}
      {(isRunning || hasResults) && (
        <div className="space-y-6">
          {/* Summary bar */}
          <div className="flex flex-wrap items-center gap-4 rounded-xl border border-gray-800 bg-gray-900/50 px-5 py-3">
            {paperTitle && (
              <span className="text-sm font-medium text-gray-200 truncate max-w-md">
                📄 {paperTitle}
              </span>
            )}
            <span className="text-xs text-gray-500">
              {completedCount}/{phases.length} phases
            </span>
            {totalDuration > 0 && (
              <span className="text-xs text-gray-500 font-mono">
                ⏱ {formatDuration(totalDuration)}
              </span>
            )}
            {completedCount === phases.length && (
              <span className="ml-auto rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-400">
                ✓ Complete
              </span>
            )}
          </div>

          {/* Phase list */}
          <div className="grid gap-3">
            {phases.map((phase, i) => (
              <PhaseNode key={phase.id} phase={phase} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Generated Modules */}
      {modules.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Code2 size={18} className="text-brand-400" />
            <h3 className="text-lg font-semibold text-white">
              Generated Modules
            </h3>
            <span className="rounded-full bg-brand-500/15 px-2.5 py-0.5 text-xs font-medium text-brand-300">
              {modules.length}
            </span>
          </div>
          <div className="grid gap-2">
            {modules.map((mod) => (
              <ModuleCard
                key={mod.id}
                mod={mod}
                expanded={expandedModule === mod.id}
                onToggle={() =>
                  setExpandedModule(expandedModule === mod.id ? null : mod.id)
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
