"""
Multi-repository support for ScholarDevClaw.

Provides cross-repo analysis, comparison, and knowledge transfer:

  - **RepoProfile**: snapshot of a single repo's analysis + suggestions
  - **MultiRepoManager**: coordinate analysis across N repos
  - **CrossRepoAnalyzer**: compare patterns, frameworks, languages across repos
  - **KnowledgeTransferEngine**: discover transferable improvements between repos
"""

from __future__ import annotations

from .manager import (
    MultiRepoManager,
    RepoProfile,
    RepoProfileStatus,
)
from .analysis import (
    CrossRepoAnalyzer,
    ComparisonResult,
    PatternOverlap,
    FrameworkComparison,
    LanguageOverlap,
)
from .transfer import (
    KnowledgeTransferEngine,
    TransferOpportunity,
    TransferPlan,
    TransferDirection,
)

__all__ = [
    # Manager
    "MultiRepoManager",
    "RepoProfile",
    "RepoProfileStatus",
    # Analysis
    "CrossRepoAnalyzer",
    "ComparisonResult",
    "PatternOverlap",
    "FrameworkComparison",
    "LanguageOverlap",
    # Transfer
    "KnowledgeTransferEngine",
    "TransferOpportunity",
    "TransferPlan",
    "TransferDirection",
]
