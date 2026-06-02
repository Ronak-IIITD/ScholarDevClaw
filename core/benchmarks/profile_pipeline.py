#!/usr/bin/env python3
"""Profile ScholarDevClaw pipeline hot paths against test_repos/nanogpt.

Runs each major module in isolation with cProfile, then writes a summary.
"""

import cProfile
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_PATH = Path(__file__).resolve().parent.parent.parent / "test_repos" / "nanogpt"
OUTPUT_DIR = Path(__file__).resolve().parent / "profile_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def _profile(label: str, func, *args, **kwargs):
    """Run *func* under cProfile and save stats + print summary."""
    print(f"\n{'=' * 60}")
    print(f"  Profiling: {label}")
    print(f"  Repo: {REPO_PATH}")
    print(f"{'=' * 60}")

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - t0

    stats_file = OUTPUT_DIR / f"{label.replace(' ', '_').replace('/', '_')}.prof"
    profiler.dump_stats(str(stats_file))

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print(f"\n  Wall time: {elapsed:.3f}s")
    print(f"\n  Top 30 by cumulative time:")
    stats.print_stats(30)

    print(f"\n  Top 20 by tottime (self-time):")
    stats.sort_stats("tottime")
    stats.print_stats(20)

    print(f"\n  Stats saved to: {stats_file}")
    return result


# ---------------------------------------------------------------------------
# Individual profiling targets
# ---------------------------------------------------------------------------


def profile_tree_sitter_analyze():
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer(REPO_PATH)
    return analyzer.analyze()


def profile_tree_sitter_detect():
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer(REPO_PATH)
    return analyzer.detect_languages()


def profile_similarity_find():
    """Profile similarity search with synthetic papers (no network needed)."""
    from scholardevclaw.research_intelligence.similarity import SimilarityFinder

    finder = SimilarityFinder()
    # Generate synthetic papers to stress-test the scorer
    papers = []
    for i in range(200):
        papers.append(
            {
                "paper_id": f"arxiv:{i:04d}",
                "title": f"Efficient Transformer Architecture {i} for Deep Learning",
                "abstract": (
                    "We propose a novel approach to transformer optimization that "
                    f"achieves state-of-the-art results on benchmark {i}. Our method "
                    "reduces computational cost while maintaining accuracy."
                ),
                "year": 2020 + (i % 6),
                "source": "arxiv",
            }
        )
    return finder.find_similar(
        "efficient transformer deep learning optimization",
        papers,
        max_results=10,
    )


def profile_mapping_engine():
    """Profile mapping engine with a synthetic spec against nanogpt analysis."""
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.mapping.engine import MappingEngine

    analyzer = TreeSitterAnalyzer(REPO_PATH)
    analysis = analyzer.analyze()

    # Synthetic spec that targets nanogpt patterns
    spec = {
        "algorithm": {"name": "FlashAttention"},
        "changes": {
            "target_patterns": [
                "GPT",
                "Block",
                "CausalSelfAttention",
                "MLP",
                "model.py",
                "train.py",
                "self.ln_1",
                "self.ln_2",
                "nn.Embedding",
                "nn.Linear",
                "nn.LayerNorm",
            ],
            "replacement": "flash_attention_optimized_version",
        },
    }

    engine = MappingEngine(analysis.__dict__, spec, llm_assistant=None)
    return engine.map()


def profile_mapping_text_scan():
    """Profile the text scan tier specifically."""
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.mapping.engine import MappingEngine

    analyzer = TreeSitterAnalyzer(REPO_PATH)
    analysis = analyzer.analyze()

    spec = {
        "algorithm": {"name": "FlashAttention"},
        "changes": {
            "target_patterns": [
                "self.ln_1",
                "self.ln_2",
                "nn.Embedding",
                "nn.Linear",
                "nn.LayerNorm",
                "nn.GELU",
                "self.wpe",
                "self.wte",
                "F.scaled_dot_product_attention",
                "torch.nn.functional",
            ],
            "replacement": "flash_attention_optimized_version",
        },
    }

    engine = MappingEngine(analysis.__dict__, spec, llm_assistant=None)
    # Call the text scan directly
    seen: set[tuple[str, int]] = set()
    return engine._text_scan_for_patterns(
        spec["changes"]["target_patterns"],
        spec["changes"]["replacement"],
        seen,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not REPO_PATH.is_dir():
        print(f"ERROR: test repo not found at {REPO_PATH}")
        sys.exit(1)

    print(f"ScholarDevClaw Performance Profiler")
    print(f"Test repo: {REPO_PATH}")
    print(f"Results dir: {OUTPUT_DIR}")

    # 1. Tree-sitter analysis
    _profile("tree_sitter_analyze", profile_tree_sitter_analyze)

    # 2. Language detection (subset of analyze)
    _profile("tree_sitter_detect_languages", profile_tree_sitter_detect)

    # 3. Similarity search
    _profile("similarity_find_similar_200papers", profile_similarity_find)

    # 4. Full mapping engine
    _profile("mapping_engine_full", profile_mapping_engine)

    # 5. Mapping text scan tier
    _profile("mapping_text_scan", profile_mapping_text_scan)

    print(f"\n{'=' * 60}")
    print(f"  All profiles saved to: {OUTPUT_DIR}")
    print(f"  Use `python -m pstats {OUTPUT_DIR}/<file>.prof` to inspect")
    print(f"{'=' * 60}")
