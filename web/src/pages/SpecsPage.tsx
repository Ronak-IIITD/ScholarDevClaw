import { useEffect, useState } from "react";
import { getSpecs } from "@/lib/api";
import type { SpecSummary } from "@/types/api";
import SpecCard from "@/components/SpecCard";
import { Search, Filter } from "lucide-react";

export default function SpecsPage() {
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("");

  useEffect(() => {
    getSpecs()
      .then(setSpecs)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const categories = [...new Set(specs.map((s) => s.category))].sort();

  const filtered = specs.filter((s) => {
    const matchSearch =
      !search ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.algorithm.toLowerCase().includes(search.toLowerCase()) ||
      s.title.toLowerCase().includes(search.toLowerCase());
    const matchCat = !categoryFilter || s.category === categoryFilter;
    return matchSearch && matchCat;
  });

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Paper Specs</h2>
          <p className="mt-1 text-sm text-gray-400">
            {specs.length} research specifications across {categories.length}{" "}
            categories
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg bg-red-900/20 border border-red-800 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Search and filter bar */}
      <div className="mt-6 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search specs..."
            className="w-full rounded-lg border border-gray-800 bg-gray-900 py-2.5 pl-10 pr-4 text-sm text-gray-200 placeholder-gray-600 focus:border-brand-600 focus:outline-none focus:ring-1 focus:ring-brand-600"
          />
        </div>
        <div className="relative">
          <Filter
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="appearance-none rounded-lg border border-gray-800 bg-gray-900 py-2.5 pl-10 pr-8 text-sm text-gray-200 focus:border-brand-600 focus:outline-none focus:ring-1 focus:ring-brand-600"
          >
            <option value="">All categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="mt-8 flex items-center justify-center py-12">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-700 border-t-brand-500" />
        </div>
      ) : (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((spec) => (
            <SpecCard key={spec.name} spec={spec} />
          ))}
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <p className="mt-8 text-center text-sm text-gray-500">
          No specs match your search.
        </p>
      )}
    </div>
  );
}
