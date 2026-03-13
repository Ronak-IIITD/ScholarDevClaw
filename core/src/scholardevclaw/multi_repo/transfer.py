"""
Knowledge transfer engine — discover and plan transferable improvements.

Given a set of analysed :class:`RepoProfile` snapshots, the
:class:`KnowledgeTransferEngine` discovers improvements (specs/suggestions)
that were suggested for one repository and could benefit another, then
assembles ranked :class:`TransferPlan` objects.

Transfer scoring considers:

- Shared patterns between source and target repos.
- Framework compatibility.
- Language compatibility.
- Suggestion category relevance.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

from .manager import RepoProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TransferDirection(str, enum.Enum):
    """Direction of a knowledge transfer opportunity."""

    SOURCE_TO_TARGET = "source_to_target"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class TransferOpportunity:
    """A single transferable improvement between two repos.

    Attributes:
        spec_name: The spec or suggestion name.
        source_repo_id: Repo where this spec was originally suggested.
        target_repo_id: Repo that could benefit from the same spec.
        direction: Whether the transfer is one-way or bidirectional.
        confidence: Estimated transferability confidence (0-100).
        rationale: Human-readable explanation of why this is transferable.
        shared_patterns: Patterns the two repos have in common.
        shared_frameworks: Frameworks the two repos share.
        category: The suggestion/spec category (e.g. ``"normalization"``).
    """

    spec_name: str
    source_repo_id: str
    target_repo_id: str
    direction: TransferDirection = TransferDirection.SOURCE_TO_TARGET
    confidence: int = 0
    rationale: str = ""
    shared_patterns: list[str] = field(default_factory=list)
    shared_frameworks: list[str] = field(default_factory=list)
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "source_repo_id": self.source_repo_id,
            "target_repo_id": self.target_repo_id,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "shared_patterns": self.shared_patterns,
            "shared_frameworks": self.shared_frameworks,
            "category": self.category,
        }


@dataclass
class TransferPlan:
    """Ranked list of transfer opportunities between two repos.

    Attributes:
        source_repo_id: The source repository.
        source_name: Human-readable name of the source repo.
        target_repo_id: The target repository.
        target_name: Human-readable name of the target repo.
        opportunities: Ordered (highest confidence first) transfer items.
        overall_score: Aggregate transferability score (0-100).
        summary: Human-readable plan description.
    """

    source_repo_id: str
    source_name: str
    target_repo_id: str
    target_name: str
    opportunities: list[TransferOpportunity] = field(default_factory=list)
    overall_score: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_repo_id": self.source_repo_id,
            "source_name": self.source_name,
            "target_repo_id": self.target_repo_id,
            "target_name": self.target_name,
            "opportunities": [opp.to_dict() for opp in self.opportunities],
            "opportunity_count": len(self.opportunities),
            "overall_score": self.overall_score,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# KnowledgeTransferEngine
# ---------------------------------------------------------------------------


class KnowledgeTransferEngine:
    """Discover transferable improvements between repositories.

    Usage::

        profiles = manager.get_ready_profiles()
        engine = KnowledgeTransferEngine(profiles)
        plans = engine.discover()
    """

    def __init__(self, profiles: list[RepoProfile]) -> None:
        self._profiles = profiles
        self._by_id: dict[str, RepoProfile] = {p.repo_id: p for p in profiles}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[TransferPlan]:
        """Discover all transfer plans between repo pairs.

        Returns a list of :class:`TransferPlan` objects, one per directed
        pair (source -> target), sorted by ``overall_score`` descending.
        Plans with zero opportunities are excluded.
        """
        if len(self._profiles) < 2:
            return []

        plans: list[TransferPlan] = []
        ids = [p.repo_id for p in self._profiles]

        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                source = self._by_id[ids[i]]
                target = self._by_id[ids[j]]
                plan = self._build_plan(source, target)
                if plan.opportunities:
                    plans.append(plan)

        plans.sort(key=lambda p: -p.overall_score)
        return plans

    def discover_for_pair(self, source_id: str, target_id: str) -> TransferPlan | None:
        """Discover transfer opportunities for a specific source -> target pair.

        Returns ``None`` if either repo ID is not found.
        """
        source = self._by_id.get(source_id)
        target = self._by_id.get(target_id)
        if source is None or target is None:
            return None
        return self._build_plan(source, target)

    # ------------------------------------------------------------------
    # Plan building
    # ------------------------------------------------------------------

    def _build_plan(self, source: RepoProfile, target: RepoProfile) -> TransferPlan:
        """Build a transfer plan from *source* to *target*."""
        shared_patterns = self._shared_patterns(source, target)
        shared_frameworks = self._shared_frameworks(source, target)
        shared_languages = self._shared_languages(source, target)

        # Extract specs suggested for source that are NOT already suggested for target
        source_specs = self._extract_spec_names(source)
        target_specs = self._extract_spec_names(target)

        candidates = source_specs - target_specs
        opportunities: list[TransferOpportunity] = []

        for spec_name in sorted(candidates):
            confidence, rationale = self._score_transfer(
                spec_name=spec_name,
                source=source,
                target=target,
                shared_patterns=shared_patterns,
                shared_frameworks=shared_frameworks,
                shared_languages=shared_languages,
            )

            # Only include non-trivial transfers
            if confidence < 10:
                continue

            # Determine category from source suggestions
            category = self._get_category(source, spec_name)

            # Check if target also has the spec suggested (bidirectional)
            direction = TransferDirection.SOURCE_TO_TARGET
            if spec_name in target_specs:
                direction = TransferDirection.BIDIRECTIONAL

            opportunities.append(
                TransferOpportunity(
                    spec_name=spec_name,
                    source_repo_id=source.repo_id,
                    target_repo_id=target.repo_id,
                    direction=direction,
                    confidence=confidence,
                    rationale=rationale,
                    shared_patterns=shared_patterns,
                    shared_frameworks=shared_frameworks,
                    category=category,
                )
            )

        # Sort by confidence descending
        opportunities.sort(key=lambda o: -o.confidence)

        overall = (
            round(sum(o.confidence for o in opportunities) / len(opportunities))
            if opportunities
            else 0
        )

        summary = self._build_summary(source, target, opportunities, overall)

        return TransferPlan(
            source_repo_id=source.repo_id,
            source_name=source.name,
            target_repo_id=target.repo_id,
            target_name=target.name,
            opportunities=opportunities,
            overall_score=overall,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_transfer(
        self,
        *,
        spec_name: str,
        source: RepoProfile,
        target: RepoProfile,
        shared_patterns: list[str],
        shared_frameworks: list[str],
        shared_languages: list[str],
    ) -> tuple[int, str]:
        """Score the transferability of *spec_name* from source to target.

        Returns (confidence 0-100, rationale string).
        """
        score = 0
        reasons: list[str] = []

        # Base: shared language means the spec is more likely applicable
        if shared_languages:
            lang_bonus = min(30, len(shared_languages) * 15)
            score += lang_bonus
            reasons.append(f"shared languages: {', '.join(shared_languages)}")

        # Framework compatibility — strong signal
        if shared_frameworks:
            fw_bonus = min(30, len(shared_frameworks) * 15)
            score += fw_bonus
            reasons.append(f"shared frameworks: {', '.join(shared_frameworks)}")

        # Pattern overlap — if source and target share patterns related to
        # the spec, transfer is more likely to succeed
        category = self._get_category(source, spec_name)
        if category and category in [p.lower() for p in shared_patterns]:
            score += 20
            reasons.append(f"target has related '{category}' patterns")
        elif shared_patterns:
            score += 10
            reasons.append(f"{len(shared_patterns)} shared code patterns")

        # Size heuristic: if target has similar complexity
        if source.element_count > 0 and target.element_count > 0:
            ratio = min(source.element_count, target.element_count) / max(
                source.element_count, target.element_count
            )
            if ratio > 0.3:
                score += 10
                reasons.append("similar codebase size")

        # Cap at 100
        score = min(100, score)

        rationale = "; ".join(reasons) if reasons else "minimal overlap"
        return score, rationale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_spec_names(profile: RepoProfile) -> set[str]:
        """Extract spec names from a profile's suggestions."""
        names: set[str] = set()
        for suggestion in profile.suggestions:
            name = suggestion.get("spec") or suggestion.get("name", "")
            if name:
                names.add(name)
        return names

    @staticmethod
    def _get_category(profile: RepoProfile, spec_name: str) -> str:
        """Get the category for a spec from the profile's suggestions."""
        for suggestion in profile.suggestions:
            name = suggestion.get("spec") or suggestion.get("name", "")
            if name == spec_name:
                return suggestion.get("category", "")
        return ""

    @staticmethod
    def _shared_patterns(a: RepoProfile, b: RepoProfile) -> list[str]:
        return sorted(set(a.patterns.keys()) & set(b.patterns.keys()))

    @staticmethod
    def _shared_frameworks(a: RepoProfile, b: RepoProfile) -> list[str]:
        return sorted({f.lower() for f in a.frameworks} & {f.lower() for f in b.frameworks})

    @staticmethod
    def _shared_languages(a: RepoProfile, b: RepoProfile) -> list[str]:
        return sorted({l.lower() for l in a.languages} & {l.lower() for l in b.languages})  # noqa: E741

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        source: RepoProfile,
        target: RepoProfile,
        opportunities: list[TransferOpportunity],
        overall_score: int,
    ) -> str:
        """Build a human-readable transfer plan summary."""
        if not opportunities:
            return f"No transferable improvements found from {source.name} to {target.name}."

        n = len(opportunities)
        top_specs = [o.spec_name for o in opportunities[:3]]
        top_str = ", ".join(top_specs)
        if n > 3:
            top_str += f" (+{n - 3} more)"

        lines = [
            f"Transfer plan: {source.name} -> {target.name}",
            f"  {n} transferable improvement(s), overall confidence: {overall_score}%",
            f"  Top candidates: {top_str}",
        ]
        return "\n".join(lines)
