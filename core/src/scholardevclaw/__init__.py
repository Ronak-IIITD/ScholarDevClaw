"""ScholarDevClaw - Autonomous ML Research Integration Engine Core"""

from importlib.metadata import PackageNotFoundError, version

from scholardevclaw.exceptions import ScholarDevClawError
from scholardevclaw.mapping.engine import MappingEngine
from scholardevclaw.patch_generation.generator import PatchGenerator
from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser
from scholardevclaw.research_intelligence.extractor import ResearchExtractor
from scholardevclaw.validation.runner import ValidationRunner

try:
    __version__ = version("scholardevclaw")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "ScholarDevClaw Team"

__all__ = [
    "ScholarDevClawError",
    "PyTorchRepoParser",
    "ResearchExtractor",
    "MappingEngine",
    "PatchGenerator",
    "ValidationRunner",
]
