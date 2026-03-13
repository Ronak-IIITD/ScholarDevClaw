# Research Intelligence module
from .citation_graph import (
    CitationAnalyzer,
    CitationGraph,
    CitationNode,
    CitationPath,
)
from .enhanced_extractor import (
    EnhancedSpecExtractor,
    ExtractedAlgorithm,
    PaperSpec,
    extract_spec_from_arxiv,
)
from .extractor import ResearchExtractor
from .paper_sources import (
    ArxivSource,
    IEEESource,
    Paper,
    PaperSourceAggregator,
    PubmedSource,
    SearchResult,
    get_paper_source,
)
from .similarity import (
    ResearchRecommendationEngine,
    ResearchSimilaritySearch,
    SimilarPaper,
)

__all__ = [
    "ResearchExtractor",
    "Paper",
    "SearchResult",
    "ArxivSource",
    "PubmedSource",
    "IEEESource",
    "PaperSourceAggregator",
    "get_paper_source",
    "CitationNode",
    "CitationPath",
    "CitationGraph",
    "CitationAnalyzer",
    "SimilarPaper",
    "ResearchSimilaritySearch",
    "ResearchRecommendationEngine",
    "ExtractedAlgorithm",
    "PaperSpec",
    "EnhancedSpecExtractor",
    "extract_spec_from_arxiv",
]
