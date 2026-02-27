# Research Intelligence module
from .extractor import ResearchExtractor
from .paper_sources import (
    Paper,
    SearchResult,
    ArxivSource,
    PubmedSource,
    IEEESource,
    PaperSourceAggregator,
    get_paper_source,
)
from .citation_graph import (
    CitationNode,
    CitationPath,
    CitationGraph,
    CitationAnalyzer,
)
from .similarity import (
    SimilarPaper,
    ResearchSimilaritySearch,
    ResearchRecommendationEngine,
)
from .enhanced_extractor import (
    ExtractedAlgorithm,
    PaperSpec,
    EnhancedSpecExtractor,
    extract_spec_from_arxiv,
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
