"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { SpecCard } from "@/components/SpecCard";
import { api } from "@/lib/api";
import type { SpecSummary } from "@/lib/types";

export default function SpecsPage() {
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");

  const loadSpecs = async () => {
    try {
      const s = await api.specs.list();
      setSpecs(s);
    } catch (e) {
      console.error("Failed to load specs:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSpecs();
  }, []);

  const categories = ["all", ...new Set(specs.map((s) => s.category).filter(Boolean))];

  const filtered = specs.filter((s) => {
    const matchesSearch =
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.title.toLowerCase().includes(search.toLowerCase()) ||
      s.algorithm.toLowerCase().includes(search.toLowerCase()) ||
      s.description.toLowerCase().includes(search.toLowerCase());
    const matchesCat = category === "all" || s.category === category;
    return matchesSearch && matchesCat;
  });

  return (
    <>
      <Nav />
      <main className="main" role="main">
        <header className="page-head">
          <div className="page-eyebrow">Specifications</div>
          <h1 className="page-title">Paper Specifications</h1>
          <p className="page-sub">
            Browse {specs.length} research paper specifications with CST transformers for code integration
          </p>
        </header>

        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div className="row" style={{ gap: "1rem", flexWrap: "wrap" }}>
            <div className="search" style={{ flex: 1, minWidth: "240px" }}>
              <input
                type="search"
                className="input"
                placeholder="Search specs…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search specifications"
              />
            </div>
            <select
              className="input"
              style={{ width: "auto", minWidth: "180px" }}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              aria-label="Filter by category"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c === "all" ? "All Categories" : c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {loading ? (
          <div className="empty">Loading specs…</div>
        ) : filtered.length === 0 ? (
          <div className="empty">No specs match your filters</div>
        ) : (
          <div className="grid grid-3">
            {filtered.map((spec) => (
              <SpecCard key={spec.name} spec={spec} />
            ))}
          </div>
        )}
      </main>
      <footer className="footer">
        <span>ScholarDevClaw Dashboard</span>
        <span>{filtered.length} of {specs.length} specs</span>
      </footer>
    </>
  );
}