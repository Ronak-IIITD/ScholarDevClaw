import { useEffect, useState, useCallback } from "react";
import { getSpecs, getPipelineStatus, startPipelineRun } from "@/lib/api";
import { usePipelineWs } from "@/hooks/usePipelineWs";
import type { SpecSummary, PipelineRunStatus } from "@/types/api";
import SpecCard from "@/components/SpecCard";
import StatusBadge from "@/components/StatusBadge";
import PipelineTimelinePanel from "@/components/PipelineTimelinePanel";
import ValidationScorecardCard from "@/components/ValidationScorecardCard";
import TrustMetadataCard from "@/components/TrustMetadataCard";
import ActionableErrorPanel from "@/components/ActionableErrorPanel";
import {
  Play,
  FolderOpen,
  Wifi,
  WifiOff,
  RotateCcw,
  Clock,
} from "lucide-react";

export default function PipelinePage() {
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [selectedSpecs, setSelectedSpecs] = useState<Set<string>>(new Set());
  const [repoPath, setRepoPath] = useState("");
  const [skipValidate, setSkipValidate] = useState(false);
  const [serverStatus, setServerStatus] = useState<PipelineRunStatus | null>(
    null
  );
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState("");

  const {
    connected,
    liveSteps,
    pipelineStatus,
    totalSeconds,
    timelineEvents,
    reset,
  } =
    usePipelineWs();

  // Load specs and current status on mount
  useEffect(() => {
    getSpecs()
      .then(setSpecs)
      .catch(() => {});
    getPipelineStatus()
      .then(setServerStatus)
      .catch(() => {});
  }, []);

  const toggleSpec = useCallback((name: string) => {
    setSelectedSpecs((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedSpecs(new Set(specs.map((s) => s.name)));
  }, [specs]);

  const clearSelection = useCallback(() => {
    setSelectedSpecs(new Set());
  }, []);

  const handleRun = useCallback(async () => {
    if (!repoPath.trim()) {
      setLaunchError("Repository path is required");
      return;
    }
    setLaunching(true);
    setLaunchError("");
    reset();
    try {
      const result = await startPipelineRun({
        repo_path: repoPath.trim(),
        spec_names: [...selectedSpecs],
        skip_validate: skipValidate,
        output_dir: null,
      });
      setServerStatus(result);
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setLaunching(false);
    }
  }, [repoPath, selectedSpecs, skipValidate, reset]);

  const hasLiveSignal =
    liveSteps.length > 0 || timelineEvents.length > 0 || pipelineStatus !== "idle";
  const isRunning = pipelineStatus === "running" || serverStatus?.status === "running";
  const displaySteps = hasLiveSignal ? liveSteps : serverStatus?.steps ?? [];
  const orderedDisplaySteps = [...displaySteps].sort(
    (a, b) => (a.sequence ?? 0) - (b.sequence ?? 0)
  );
  const displayStatus = hasLiveSignal ? pipelineStatus : serverStatus?.status ?? "idle";
  const elapsedSeconds = totalSeconds > 0 ? totalSeconds : serverStatus?.total_seconds ?? 0;
  const completedCount = orderedDisplaySteps.filter(
    (step) => step.status === "completed"
  ).length;

  const latestValidateStep = [...orderedDisplaySteps]
    .reverse()
    .find((step) => step.step.startsWith("validate:") && step.status === "completed");
  const latestGenerateStep = [...orderedDisplaySteps]
    .reverse()
    .find((step) => step.step.startsWith("generate:") && step.status === "completed");
  const latestError =
    launchError ||
    [...timelineEvents].reverse().find((event) => event.eventType === "error")?.error ||
    [...orderedDisplaySteps].reverse().find((step) => Boolean(step.error))?.error ||
    null;

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Pipeline</h2>
          <p className="mt-1 text-sm text-gray-400">
            Run the full research-to-code pipeline
          </p>
        </div>
        <div className="flex items-center gap-2">
          {connected ? (
            <span className="flex items-center gap-1.5 text-xs text-green-400">
              <Wifi size={14} />
              Live
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-red-400">
              <WifiOff size={14} />
              Disconnected
            </span>
          )}
        </div>
      </div>

      {/* Configuration panel */}
      <div className="mt-6 rounded-xl border border-gray-800 bg-gray-900/60 p-6">
        <h3 className="text-sm font-semibold text-gray-200">Configuration</h3>

        {/* Repo path */}
        <div className="mt-4">
          <label className="flex items-center gap-2 text-xs font-medium text-gray-400">
            <FolderOpen size={14} />
            Repository Path
          </label>
          <input
            type="text"
            value={repoPath}
            onChange={(e) => setRepoPath(e.target.value)}
            placeholder="/path/to/your/repo"
            disabled={isRunning}
            className="mt-1.5 w-full rounded-lg border border-gray-800 bg-gray-950 px-4 py-2.5 font-mono text-sm text-gray-200 placeholder-gray-600 focus:border-brand-600 focus:outline-none focus:ring-1 focus:ring-brand-600 disabled:opacity-50"
          />
        </div>

        {/* Options row */}
        <div className="mt-4 flex items-center gap-6">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={skipValidate}
              onChange={(e) => setSkipValidate(e.target.checked)}
              disabled={isRunning}
              className="rounded border-gray-700 bg-gray-800 text-brand-600 focus:ring-brand-600"
            />
            Skip validation
          </label>
        </div>

        {/* Spec selection */}
        <div className="mt-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-400">
              Select Specs ({selectedSpecs.size} of {specs.length})
            </span>
            <div className="flex gap-2">
              <button
                onClick={selectAll}
                disabled={isRunning}
                className="text-xs text-brand-400 hover:text-brand-300 disabled:opacity-50"
              >
                Select All
              </button>
              <span className="text-gray-700">|</span>
              <button
                onClick={clearSelection}
                disabled={isRunning}
                className="text-xs text-gray-400 hover:text-gray-300 disabled:opacity-50"
              >
                Clear
              </button>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 max-h-80 overflow-y-auto pr-1">
            {specs.map((spec) => (
              <SpecCard
                key={spec.name}
                spec={spec}
                selected={selectedSpecs.has(spec.name)}
                onToggle={isRunning ? undefined : toggleSpec}
              />
            ))}
          </div>
        </div>

        {launchError && (
          <p className="mt-4 text-sm text-red-400">{launchError}</p>
        )}

        {/* Launch button */}
        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={isRunning || launching}
            className="flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isRunning || launching ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Running...
              </>
            ) : (
              <>
                <Play size={16} />
                Run Pipeline
              </>
            )}
          </button>
          {displayStatus !== "idle" && !isRunning && (
            <button
              onClick={reset}
              className="flex items-center gap-2 rounded-lg border border-gray-700 px-4 py-2.5 text-sm text-gray-300 hover:bg-gray-800 transition-colors"
            >
              <RotateCcw size={14} />
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Run workspace */}
      {(displaySteps.length > 0 || displayStatus !== "idle") && (
        <div className="mt-8 space-y-4">
          <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
            <div className="flex flex-wrap items-center gap-4">
              <StatusBadge status={displayStatus} size="md" />
              <span className="text-sm text-gray-300">
                Completed steps: <span className="font-medium">{completedCount}</span>
              </span>
              <span className="flex items-center gap-1 text-sm text-gray-400">
                <Clock size={14} />
                {elapsedSeconds.toFixed(1)}s elapsed
              </span>
            </div>
          </div>

          <ActionableErrorPanel error={latestError} />

          <div className="grid gap-4 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <PipelineTimelinePanel
                steps={orderedDisplaySteps}
                timelineEvents={timelineEvents}
                status={displayStatus}
              />
            </div>
            <div className="space-y-4">
              <ValidationScorecardCard
                latestValidateStepData={
                  (latestValidateStep?.data as Record<string, unknown> | undefined) ?? null
                }
              />
              <TrustMetadataCard
                latestGenerateStepData={
                  (latestGenerateStep?.data as Record<string, unknown> | undefined) ?? null
                }
                latestValidateStepData={
                  (latestValidateStep?.data as Record<string, unknown> | undefined) ?? null
                }
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
