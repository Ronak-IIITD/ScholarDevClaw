"""ScholarDevClaw - Autonomous ML Research Integration Engine Core"""

__version__ = "0.1.0"
__author__ = "ScholarDevClaw Team"

from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser
from scholardevclaw.research_intelligence.extractor import ResearchExtractor
from scholardevclaw.mapping.engine import MappingEngine
from scholardevclaw.patch_generation.generator import PatchGenerator
from scholardevclaw.validation.runner import ValidationRunner

__all__ = [
    "PyTorchRepoParser",
    "ResearchExtractor",
    "MappingEngine",
    "PatchGenerator",
    "ValidationRunner",
]
