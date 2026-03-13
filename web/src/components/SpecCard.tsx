import type { SpecSummary } from "@/types/api";
import { FileText, ArrowRightLeft } from "lucide-react";

const CATEGORY_COLORS: Record<string, string> = {
  normalization: "bg-purple-900/40 text-purple-300 border-purple-800",
  activation: "bg-orange-900/40 text-orange-300 border-orange-800",
  attention: "bg-blue-900/40 text-blue-300 border-blue-800",
  optimization: "bg-green-900/40 text-green-300 border-green-800",
  architecture: "bg-cyan-900/40 text-cyan-300 border-cyan-800",
  regularization: "bg-pink-900/40 text-pink-300 border-pink-800",
  embedding: "bg-yellow-900/40 text-yellow-300 border-yellow-800",
  training: "bg-indigo-900/40 text-indigo-300 border-indigo-800",
};

interface SpecCardProps {
  spec: SpecSummary;
  selected?: boolean;
  onToggle?: (name: string) => void;
}

export default function SpecCard({ spec, selected, onToggle }: SpecCardProps) {
  const catColor =
    CATEGORY_COLORS[spec.category] ??
    "bg-gray-800/40 text-gray-300 border-gray-700";

  return (
    <div
      onClick={() => onToggle?.(spec.name)}
      className={`group cursor-pointer rounded-xl border p-4 transition-all ${
        selected
          ? "border-brand-500 bg-brand-950/30 ring-1 ring-brand-500/30"
          : "border-gray-800 bg-gray-900/60 hover:border-gray-700"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-100 truncate">
              {spec.algorithm || spec.name}
            </h3>
            <span
              className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-medium ${catColor}`}
            >
              {spec.category}
            </span>
          </div>
          <p className="mt-1 text-xs text-gray-400 truncate">{spec.title}</p>
        </div>
        {selected && (
          <div className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-600 text-white text-xs shrink-0">
            &#10003;
          </div>
        )}
      </div>

      {spec.description && (
        <p className="mt-2 text-xs text-gray-500 line-clamp-2">
          {spec.description}
        </p>
      )}

      <div className="mt-3 flex items-center gap-3 text-xs text-gray-500">
        {spec.replaces && (
          <span className="flex items-center gap-1">
            <ArrowRightLeft size={12} />
            Replaces {spec.replaces}
          </span>
        )}
        {spec.arxiv_id && (
          <a
            href={`https://arxiv.org/abs/${spec.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="flex items-center gap-1 hover:text-brand-400 transition-colors"
          >
            <FileText size={12} />
            arXiv
          </a>
        )}
      </div>
    </div>
  );
}
