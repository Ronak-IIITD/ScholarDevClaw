"""
Cross-repository analysis — compare patterns, frameworks, and languages.

Given a set of :class:`RepoProfile` snapshots produced by
:class:`MultiRepoManager`, the :class:`CrossRepoAnalyzer` computes pairwise
similarity scores and aggregated overlap metrics across three dimensions:

- **Patterns**: which code patterns (e.g. attention, normalization) appear in
  which repos, and how much the pattern sets overlap.
- **Frameworks**: which frameworks (e.g. pytorch, tensorflow) each repo uses.
- **Languages**: primary and secondary language overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .manager import RepoProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PatternOverlap:
    """Which code patterns appear across the analysed repos.

    Attributes:
        pattern: The code pattern name (e.g. ``"normalization"``).
        repos: Repo IDs that contain this pattern.
        details: Mapping from repo_id -> list of specific occurrences.
    """

    pattern: str
    repos: list[str] = field(default_factory=list)
    details: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "repos": self.repos,
            "repo_count": len(self.repos),
            "details": self.details,
        }


@dataclass
class FrameworkComparison:
    """Framework usage comparison across repos.

    Attributes:
        framework: Framework name (e.g. ``"pytorch"``).
        repos: Repo IDs that use this framework.
    """

    framework: str
    repos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "repos": self.repos,
            "repo_count": len(self.repos),
        }


@dataclass
class LanguageOverlap:
    """Language usage overlap across repos.

    Attributes:
        language: The programming language name.
        repos: Repo IDs that use this language.
        total_lines: Aggregate line count across all repos for this language.
        total_files: Aggregate file count across all repos for this language.
    """

    language: str
    repos: list[str] = field(default_factory=list)
    total_lines: int = 0
    total_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "repos": self.repos,
            "repo_count": len(self.repos),
            "total_lines": self.total_lines,
            "total_files": self.total_files,
        }


@dataclass
class ComparisonResult:
    """Full cross-repo comparison output.

    Attributes:
        repo_ids: The IDs of all repos included in the comparison.
        repo_names: Human-readable names keyed by repo_id.
        pattern_overlaps: Per-pattern overlap data.
        framework_comparisons: Per-framework comparison data.
        language_overlaps: Per-language overlap data.
        pairwise_similarity: Mapping ``"repoA:repoB"`` -> similarity float.
        shared_patterns: Patterns that appear in ALL repos.
        unique_patterns: Mapping repo_id -> patterns exclusive to that repo.
        summary: Human-readable summary string.
    """

    repo_ids: list[str] = field(default_factory=list)
    repo_names: dict[str, str] = field(default_factory=dict)
    pattern_overlaps: list[PatternOverlap] = field(default_factory=list)
    framework_comparisons: list[FrameworkComparison] = field(default_factory=list)
    language_overlaps: list[LanguageOverlap] = field(default_factory=list)
    pairwise_similarity: dict[str, float] = field(default_factory=dict)
    shared_patterns: list[str] = field(default_factory=list)
    unique_patterns: dict[str, list[str]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_ids": self.repo_ids,
            "repo_names": self.repo_names,
            "repo_count": len(self.repo_ids),
            "pattern_overlaps": [po.to_dict() for po in self.pattern_overlaps],
            "framework_comparisons": [fc.to_dict() for fc in self.framework_comparisons],
            "language_overlaps": [lo.to_dict() for lo in self.language_overlaps],
            "pairwise_similarity": self.pairwise_similarity,
            "shared_patterns": self.shared_patterns,
            "unique_patterns": self.unique_patterns,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# CrossRepoAnalyzer
# ---------------------------------------------------------------------------


class CrossRepoAnalyzer:
    """Compare patterns, frameworks, and languages across multiple repos.

    Usage::

        profiles = manager.get_ready_profiles()
        analyzer = CrossRepoAnalyzer(profiles)
        result = analyzer.compare()
    """

    def __init__(self, profiles: list[RepoProfile]) -> None:
        self._profiles = profiles
        self._by_id: dict[str, RepoProfile] = {p.repo_id: p for p in profiles}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self) -> ComparisonResult:
        """Run full cross-repo comparison and return a :class:`ComparisonResult`."""
        if len(self._profiles) < 2:
            return ComparisonResult(
                repo_ids=[p.repo_id for p in self._profiles],
                repo_names={p.repo_id: p.name for p in self._profiles},
                summary="Need at least 2 repos for comparison.",
            )

        pattern_overlaps = self._analyze_patterns()
        framework_comparisons = self._analyze_frameworks()
        language_overlaps = self._analyze_languages()
        pairwise = self._compute_pairwise_similarity()
        shared, unique = self._compute_shared_and_unique_patterns()

        summary = self._build_summary(
            pattern_overlaps, framework_comparisons, language_overlaps, pairwise
        )

        return ComparisonResult(
            repo_ids=[p.repo_id for p in self._profiles],
            repo_names={p.repo_id: p.name for p in self._profiles},
            pattern_overlaps=pattern_overlaps,
            framework_comparisons=framework_comparisons,
            language_overlaps=language_overlaps,
            pairwise_similarity=pairwise,
            shared_patterns=shared,
            unique_patterns=unique,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Pattern analysis
    # ------------------------------------------------------------------

    def _analyze_patterns(self) -> list[PatternOverlap]:
        """Gather all patterns across repos and record which repos have each."""
        all_patterns: dict[str, PatternOverlap] = {}

        for profile in self._profiles:
            for pattern_name, occurrences in profile.patterns.items():
                if pattern_name not in all_patterns:
                    all_patterns[pattern_name] = PatternOverlap(pattern=pattern_name)
                po = all_patterns[pattern_name]
                if profile.repo_id not in po.repos:
                    po.repos.append(profile.repo_id)
                po.details[profile.repo_id] = occurrences

        # Sort by number of repos descending, then alphabetically
        result = sorted(all_patterns.values(), key=lambda p: (-len(p.repos), p.pattern))
        return result

    # ------------------------------------------------------------------
    # Framework analysis
    # ------------------------------------------------------------------

    def _analyze_frameworks(self) -> list[FrameworkComparison]:
        """Gather all frameworks across repos."""
        framework_map: dict[str, list[str]] = {}

        for profile in self._profiles:
            for fw in profile.frameworks:
                fw_lower = fw.lower()
                if fw_lower not in framework_map:
                    framework_map[fw_lower] = []
                if profile.repo_id not in framework_map[fw_lower]:
                    framework_map[fw_lower].append(profile.repo_id)

        result = [
            FrameworkComparison(framework=fw, repos=repos)
            for fw, repos in sorted(framework_map.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
        return result

    # ------------------------------------------------------------------
    # Language analysis
    # ------------------------------------------------------------------

    def _analyze_languages(self) -> list[LanguageOverlap]:
        """Aggregate language stats across all repos."""
        lang_data: dict[str, dict[str, Any]] = {}

        for profile in self._profiles:
            # From languages list (always present)
            for lang in profile.languages:
                lang_lower = lang.lower()
                if lang_lower not in lang_data:
                    lang_data[lang_lower] = {"repos": [], "total_lines": 0, "total_files": 0}
                if profile.repo_id not in lang_data[lang_lower]["repos"]:
                    lang_data[lang_lower]["repos"].append(profile.repo_id)

            # Enrich with language_stats if available
            for ls in profile.language_stats:
                lang_lower = ls.get("language", "").lower()
                if not lang_lower:
                    continue
                if lang_lower not in lang_data:
                    lang_data[lang_lower] = {"repos": [], "total_lines": 0, "total_files": 0}
                if profile.repo_id not in lang_data[lang_lower]["repos"]:
                    lang_data[lang_lower]["repos"].append(profile.repo_id)
                lang_data[lang_lower]["total_lines"] += ls.get("line_count", 0)
                lang_data[lang_lower]["total_files"] += ls.get("file_count", 0)

        result = [
            LanguageOverlap(
                language=lang,
                repos=data["repos"],
                total_lines=data["total_lines"],
                total_files=data["total_files"],
            )
            for lang, data in sorted(
                lang_data.items(), key=lambda kv: (-len(kv[1]["repos"]), kv[0])
            )
        ]
        return result

    # ------------------------------------------------------------------
    # Pairwise similarity
    # ------------------------------------------------------------------

    def _compute_pairwise_similarity(self) -> dict[str, float]:
        """Compute Jaccard-like similarity between every repo pair.

        Similarity is the weighted average of pattern, framework, and language
        Jaccard indices (patterns 0.5, frameworks 0.3, languages 0.2).
        """
        ids = [p.repo_id for p in self._profiles]
        result: dict[str, float] = {}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self._by_id[ids[i]], self._by_id[ids[j]]
                pattern_sim = self._jaccard(set(a.patterns.keys()), set(b.patterns.keys()))
                framework_sim = self._jaccard(
                    {f.lower() for f in a.frameworks},
                    {f.lower() for f in b.frameworks},
                )
                language_sim = self._jaccard(
                    {l.lower() for l in a.languages},
                    {l.lower() for l in b.languages},
                )
                # Weighted average
                similarity = (pattern_sim * 0.5) + (framework_sim * 0.3) + (language_sim * 0.2)
                key = f"{ids[i]}:{ids[j]}"
                result[key] = round(similarity, 4)

        return result

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Jaccard index of two sets.  Returns 1.0 when both are empty."""
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Shared / unique patterns
    # ------------------------------------------------------------------

    def _compute_shared_and_unique_patterns(
        self,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Identify patterns shared by all repos and patterns unique to each."""
        all_ids = {p.repo_id for p in self._profiles}
        pattern_repos: dict[str, set[str]] = {}

        for profile in self._profiles:
            for pname in profile.patterns:
                if pname not in pattern_repos:
                    pattern_repos[pname] = set()
                pattern_repos[pname].add(profile.repo_id)

        shared = sorted(p for p, repos in pattern_repos.items() if repos == all_ids)
        unique: dict[str, list[str]] = {pid: [] for pid in all_ids}
        for pname, repos in pattern_repos.items():
            if len(repos) == 1:
                unique[next(iter(repos))].append(pname)
        # Sort unique lists
        for pid in unique:
            unique[pid].sort()

        return shared, unique

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        pattern_overlaps: list[PatternOverlap],
        framework_comparisons: list[FrameworkComparison],
        language_overlaps: list[LanguageOverlap],
        pairwise: dict[str, float],
    ) -> str:
        """Build a human-readable summary."""
        n = len(self._profiles)
        names = [p.name for p in self._profiles]
        names_str = ", ".join(names)

        lines = [
            f"Cross-repo comparison of {n} repositories: {names_str}",
            "",
        ]

        # Patterns
        shared_pats = [po for po in pattern_overlaps if len(po.repos) == n]
        lines.append(f"Patterns: {len(pattern_overlaps)} total, {len(shared_pats)} shared by all")

        # Frameworks
        shared_fws = [fc for fc in framework_comparisons if len(fc.repos) == n]
        lines.append(
            f"Frameworks: {len(framework_comparisons)} total, {len(shared_fws)} shared by all"
        )

        # Languages
        shared_langs = [lo for lo in language_overlaps if len(lo.repos) == n]
        lines.append(
            f"Languages: {len(language_overlaps)} total, {len(shared_langs)} shared by all"
        )

        # Pairwise
        if pairwise:
            avg = sum(pairwise.values()) / len(pairwise)
            best_pair = max(pairwise, key=pairwise.get)  # type: ignore[arg-type]
            best_val = pairwise[best_pair]
            a_id, b_id = best_pair.split(":")
            a_name = self._by_id[a_id].name
            b_name = self._by_id[b_id].name
            lines.append("")
            lines.append(f"Average pairwise similarity: {avg:.1%}")
            lines.append(f"Most similar pair: {a_name} <-> {b_name} ({best_val:.1%})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Spec relevance matrix
    # ------------------------------------------------------------------

    def spec_relevance_matrix(self) -> dict[str, dict[str, bool]]:
        """Build a matrix of which specs are relevant to which repos.

        Returns ``{spec_name: {repo_id: True/False, ...}, ...}`` based
        on each repo's suggestion list.
        """
        all_specs: set[str] = set()
        repo_specs: dict[str, set[str]] = {}

        for profile in self._profiles:
            specs_for_repo: set[str] = set()
            for suggestion in profile.suggestions:
                spec_name = suggestion.get("spec") or suggestion.get("name", "")
                if spec_name:
                    specs_for_repo.add(spec_name)
                    all_specs.add(spec_name)
            repo_specs[profile.repo_id] = specs_for_repo

        matrix: dict[str, dict[str, bool]] = {}
        for spec in sorted(all_specs):
            matrix[spec] = {
                profile.repo_id: spec in repo_specs.get(profile.repo_id, set())
                for profile in self._profiles
            }
        return matrix
