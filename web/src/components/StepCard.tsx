import type { PipelineStepResult } from "@/types/api";
import StatusBadge from "./StatusBadge";
import {
  Search,
  Lightbulb,
  MapPin,
  Hammer,
  ShieldCheck,
  AlertTriangle,
  Scan,
} from "lucide-react";

function stepIcon(step: string) {
  if (step === "analyze") return Scan;
  if (step === "suggest") return Lightbulb;
  if (step.startsWith("map:")) return MapPin;
  if (step.startsWith("generate:")) return Hammer;
  if (step.startsWith("validate:")) return ShieldCheck;
  if (step === "error") return AlertTriangle;
  return Search;
}

function stepLabel(step: string): string {
  if (step === "analyze") return "Repository Analysis";
  if (step === "suggest") return "Research Suggestions";
  if (step === "specs_resolved") return "Specs Resolved";
  if (step.startsWith("map:")) return `Mapping: ${step.slice(4)}`;
  if (step.startsWith("generate:")) return `Patch Generation: ${step.slice(9)}`;
  if (step.startsWith("validate:")) return `Validation: ${step.slice(9)}`;
  if (step === "error") return "Error";
  return step;
}

interface StepCardProps {
  result: PipelineStepResult;
}

export default function StepCard({ result }: StepCardProps) {
  const Icon = stepIcon(result.step);
  const data = result.data as Record<string, unknown>;

  return (
    <div className="animate-fade-in rounded-xl border border-gray-800 bg-gray-900/80 p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={`flex h-9 w-9 items-center justify-center rounded-lg ${
              result.status === "completed"
                ? "bg-green-900/40 text-green-400"
                : result.status === "running"
                ? "bg-yellow-900/40 text-yellow-400"
                : result.status === "failed"
                ? "bg-red-900/40 text-red-400"
                : "bg-gray-800 text-gray-400"
            }`}
          >
            <Icon size={18} />
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-200">
              {stepLabel(result.step)}
            </h3>
            {result.duration_seconds > 0 && (
              <p className="text-xs text-gray-500">
                {result.duration_seconds.toFixed(2)}s
              </p>
            )}
          </div>
        </div>
        <StatusBadge status={result.status} />
      </div>

      {/* Data summary */}
      {Object.keys(data).length > 0 && result.status === "completed" && (
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {Object.entries(data).map(([key, value]) => {
            // Skip complex nested objects for the summary
            if (typeof value === "object" && value !== null && !Array.isArray(value))
              return null;
            const display = Array.isArray(value)
              ? `${value.length} items`
              : String(value);
            return (
              <div
                key={key}
                className="rounded-lg bg-gray-800/60 px-3 py-2"
              >
                <p className="text-xs text-gray-500 truncate">{key}</p>
                <p className="text-sm font-medium text-gray-300 truncate">
                  {display}
                </p>
              </div>
            );
          })}
        </div>
      )}

      {result.error && (
        <p className="mt-2 rounded-lg bg-red-900/20 px-3 py-2 text-xs text-red-400">
          {result.error}
        </p>
      )}
    </div>
  );
}
