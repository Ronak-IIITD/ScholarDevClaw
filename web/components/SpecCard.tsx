"use client";

import Link from "next/link";
import type { SpecSummary } from "@/lib/types";

interface SpecCardProps {
  spec: SpecSummary;
}

export function SpecCard({ spec }: SpecCardProps) {
  return (
    <Link href={`/specs/${spec.name}`} className="spec-card" aria-label={`View ${spec.algorithm} spec`}>
      <div className="algo">{spec.algorithm}</div>
      <div className="paper-title">{spec.title}</div>
      <div className="meta">
        <span className="cat">{spec.category}</span>
        <span>{spec.replaces ? `replaces ${spec.replaces}` : "new technique"}</span>
      </div>
      {spec.arxiv_id && (
        <a
          href={`https://arxiv.org/abs/${spec.arxiv_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mono small"
          style={{ color: "var(--accent)" }}
          onClick={(e) => e.stopPropagation()}
        >
          arXiv:{spec.arxiv_id}
        </a>
      )}
    </Link>
  );
}