# Research Intelligence module
from .citation_graph import (
    CitationAnalyzer,
    CitationGraph,
    CitationNode,
    CitationPath,
)
from .embeddings import EmbeddingIndex
from .extractor import ResearchExtractor
from .paper_sources import (
    ArxivSource,
    IEEESource,
    Paper,
    PaperSourceAggregator,
    PubmedSource,
    SearchResult,
    deduplicate_papers,
    get_paper_source,
)
from .similarity import (
    ResearchRecommendationEngine,
    ResearchSimilaritySearch,
    SimilarPaper,
)

__all__ = [
    "ResearchExtractor",
    "EmbeddingIndex",
    "Paper",
    "SearchResult",
    "ArxivSource",
    "PubmedSource",
    "IEEESource",
    "PaperSourceAggregator",
    "deduplicate_papers",
    "get_paper_source",
    "CitationNode",
    "CitationPath",
    "CitationGraph",
    "CitationAnalyzer",
    "SimilarPaper",
    "ResearchSimilaritySearch",
    "ResearchRecommendationEngine",
]
