"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { api } from "@/lib/api";
import type { SpecSummary } from "@/lib/types";

interface SpecDetailPageProps {
  params: Promise<{ name: string }>;
}

export default function SpecDetailPage({ params }: SpecDetailPageProps) {
  const [spec, setSpec] = useState<SpecSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSpec = async () => {
    const { name } = await params;
    try {
      const s = await api.specs.get(name);
      setSpec(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Spec not found");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSpec();
  }, [params]);

  if (loading) {
    return (
      <>
        <Nav />
        <main className="main"><div className="empty">Loading spec…</div></main>
        <footer className="footer"><span>ScholarDevClaw Dashboard</span></footer>
      </>
    );
  }

  if (error || !spec) {
    return (
      <>
        <Nav />
        <main className="main">
          <header className="page-head">
            <div className="page-eyebrow">Specifications</div>
            <h1 className="page-title">Not Found</h1>
          </header>
          <div className="card">
            <p className="mono" style={{ color: "var(--err)" }}>Spec not found: {error}</p>
          </div>
        </main>
        <footer className="footer"><span>ScholarDevClaw Dashboard</span></footer>
      </>
    );
  }

  return (
    <>
      <Nav />
      <main className="main" role="main">
        <header className="page-head">
          <div className="page-eyebrow">Specifications / {spec.name}</div>
          <h1 className="page-title">{spec.algorithm}</h1>
          <p className="page-sub">{spec.title}</p>
        </header>

        <div className="grid grid-2" style={{ marginBottom: "1.5rem" }}>
          <div className="card">
            <div className="card-title">Metadata</div>
            <dl style={{ display: "grid", gap: "0.6rem", fontSize: "0.85rem" }}>
              <div className="spread"><dt className="muted mono small">Category</dt><dd>{spec.category}</dd></div>
              <div className="spread"><dt className="muted mono small">Replaces</dt><dd>{spec.replaces || "—"}</dd></div>
              <div className="spread"><dt className="muted mono small">arXiv</dt><dd>
                {spec.arxiv_id ? (
                  <a href={`https://arxiv.org/abs/${spec.arxiv_id}`} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)" }}>
                    {spec.arxiv_id}
                  </a>
                ) : (
                  <span className="muted">—</span>
                )}
              </dd></div>
              <div className="spread"><dt className="muted mono small">Spec ID</dt><dd className="mono">{spec.name}</dd></div>
            </dl>
          </div>

          <div className="card">
            <div className="card-title">Description</div>
            <p style={{ lineHeight: 1.7 }}>{spec.description || "No description available."}</p>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Run This Spec</div>
          <p className="small muted" style={{ marginBottom: "1rem" }}>
            Use the <a href="/run" style={{ color: "var(--accent)" }}>Run page</a> or API to execute this spec against a repository.
          </p>
          <div className="code-block mono">{JSON.stringify(
            {
              repo_path: "/path/to/your/repo",
              spec_names: [spec.name],
              skip_validate: false,
            },
            null,
            2
          )}</div>
        </div>
      </main>
      <footer className="footer">
        <span>ScholarDevClaw Dashboard</span>
        <span>Spec: {spec.name}</span>
      </footer>
    </>
  );
}