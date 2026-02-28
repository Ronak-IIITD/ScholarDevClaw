"""
Agent reflection system for self-evaluation and improvement.

Provides:
- Output quality assessment
- Error analysis
- Success/failure pattern recognition
- Improvement suggestions
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReflectionType(Enum):
    """Types of reflection"""

    OUTPUT_QUALITY = "output_quality"
    ERROR_ANALYSIS = "error_analysis"
    SUCCESS_PATTERN = "success_pattern"
    IMPROVEMENT = "improvement"
    FEEDBACK_INTEGRATION = "feedback_integration"


class QualityRating(Enum):
    """Quality rating levels"""

    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class Reflection:
    """A reflection entry"""

    id: str
    reflection_type: ReflectionType
    content: str
    quality_rating: QualityRating | None = None
    related_output: str = ""
    outcome: str = ""
    lessons_learned: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionReport:
    """Summary of reflections"""

    total_reflections: int
    avg_quality: float
    success_patterns: list[str]
    areas_for_improvement: list[str]
    recommendations: list[str]


class AgentReflector:
    """Agent self-reflection system"""

    def __init__(self):
        self.reflections: list[Reflection] = []

    def reflect_on_output(
        self,
        output: str,
        expected: str | None = None,
    ) -> Reflection:
        """Reflect on generated output"""
        quality = self._assess_quality(output, expected)

        reflection = Reflection(
            id=str(uuid.uuid4()),
            reflection_type=ReflectionType.OUTPUT_QUALITY,
            content=output[:500],
            quality_rating=quality,
            related_output=expected or "",
        )

        reflection.lessons_learned = self._extract_lessons(output)
        reflection.suggestions = self._generate_suggestions(quality)

        self.reflections.append(reflection)
        return reflection

    def analyze_error(
        self,
        error: str,
        context: str | None = None,
    ) -> Reflection:
        """Analyze an error and extract learnings"""
        reflection = Reflection(
            id=str(uuid.uuid4()),
            reflection_type=ReflectionType.ERROR_ANALYSIS,
            content=error,
            related_output=context or "",
            outcome="failed",
        )

        reflection.lessons_learned = self._analyze_error_patterns(error)
        reflection.suggestions = self._suggest_error_fixes(error)

        self.reflections.append(reflection)
        return reflection

    def reflect_on_success(
        self,
        output: str,
        context: str | None = None,
    ) -> Reflection:
        """Reflect on successful outcome"""
        reflection = Reflection(
            id=str(uuid.uuid4()),
            reflection_type=ReflectionType.SUCCESS_PATTERN,
            content=output[:500],
            quality_rating=QualityRating.EXCELLENT,
            outcome="success",
        )

        reflection.lessons_learned = self._extract_success_patterns(output)
        reflection.suggestions = []

        self.reflections.append(reflection)
        return reflection

    def integrate_feedback(
        self,
        feedback: str,
        original_output: str,
    ) -> Reflection:
        """Integrate user feedback"""
        reflection = Reflection(
            id=str(uuid.uuid4()),
            reflection_type=ReflectionType.FEEDBACK_INTEGRATION,
            content=feedback,
            related_output=original_output,
        )

        reflection.lessons_learned = self._extract_feedback_lessons(feedback)
        reflection.suggestions = self._translate_feedback_to_improvements(feedback)

        self.reflections.append(reflection)
        return reflection

    def _assess_quality(self, output: str, expected: str | None) -> QualityRating:
        """Assess output quality"""
        if not output:
            return QualityRating.VERY_POOR

        length_score = min(1.0, len(output) / 1000)
        has_structure = any(marker in output for marker in ["```", "#", "- ", "* "])
        has_code = "```" in output or "def " in output or "class " in output

        score = length_score * 0.3 + (0.3 if has_structure else 0) + (0.4 if has_code else 0)

        if expected:
            overlap = len(set(output.lower().split()) & set(expected.lower().split()))
            overlap_score = min(1.0, overlap / len(set(expected.lower().split())))
            score = (score + overlap_score) / 2

        if score >= 0.8:
            return QualityRating.EXCELLENT
        elif score >= 0.6:
            return QualityRating.GOOD
        elif score >= 0.4:
            return QualityRating.FAIR
        elif score >= 0.2:
            return QualityRating.POOR
        else:
            return QualityRating.VERY_POOR

    def _extract_lessons(self, output: str) -> list[str]:
        """Extract lessons from output"""
        lessons = []

        if len(output) < 50:
            lessons.append("Output was too brief - may need more detail")

        if "error" in output.lower():
            lessons.append("Output contained errors that need addressing")

        if "undefined" in output.lower() or "not found" in output.lower():
            lessons.append("Missing information detected")

        return lessons

    def _generate_suggestions(self, rating: QualityRating) -> list[str]:
        """Generate improvement suggestions"""
        if rating == QualityRating.EXCELLENT:
            return ["Continue similar approach", "Document what worked well"]
        elif rating == QualityRating.GOOD:
            return ["Minor refinements possible", "Consider adding more context"]
        elif rating == QualityRating.FAIR:
            return ["Need more detail", "Consider restructuring output"]
        elif rating == QualityRating.POOR:
            return ["Significant improvement needed", "Review requirements again"]
        else:
            return ["Restart with clearer requirements"]

    def _analyze_error_patterns(self, error: str) -> list[str]:
        """Analyze error patterns"""
        lessons = []

        error_lower = error.lower()

        if "syntax" in error_lower:
            lessons.append("Syntax error - check code carefully")
        if "import" in error_lower:
            lessons.append("Import error - check dependencies")
        if "timeout" in error_lower:
            lessons.append("Operation took too long - optimize or chunk")
        if "permission" in error_lower:
            lessons.append("Permission issue - check access rights")
        if "not found" in error_lower:
            lessons.append("Missing resource - verify paths")

        return lessons

    def _suggest_error_fixes(self, error: str) -> list[str]:
        """Suggest fixes for errors"""
        fixes = []

        error_lower = error.lower()

        if "syntax" in error_lower:
            fixes.append("Review code syntax carefully")
        if "import" in error_lower:
            fixes.append("Verify all imports are available")
        if "null" in error_lower or "none" in error_lower:
            fixes.append("Add null checks")
        if "index" in error_lower:
            fixes.append("Check array/list bounds")

        return fixes

    def _extract_success_patterns(self, output: str) -> list[str]:
        """Extract patterns from successful outputs"""
        patterns = []

        if len(output) > 100:
            patterns.append("Good level of detail")

        if "```" in output:
            patterns.append("Code examples helpful")

        if "# " in output:
            patterns.append("Clear structure with headers")

        return patterns

    def _extract_feedback_lessons(self, feedback: str) -> list[str]:
        """Extract lessons from feedback"""
        lessons = []

        feedback_lower = feedback.lower()

        if "good" in feedback_lower or "great" in feedback_lower:
            lessons.append("What worked: continue approach")

        if "wrong" in feedback_lower or "incorrect" in feedback_lower:
            lessons.append("Error in output - need correction")

        if "more" in feedback_lower:
            lessons.append("Need more detail")

        if "less" in feedback_lower:
            lessons.append("Too verbose - need conciseness")

        return lessons

    def _translate_feedback_to_improvements(self, feedback: str) -> list[str]:
        """Convert feedback to actionable improvements"""
        improvements = []

        feedback_lower = feedback.lower()

        if "wrong" in feedback_lower:
            improvements.append("Verify accuracy before outputting")
        if "unclear" in feedback_lower:
            improvements.append("Add more explanation")
        if "missing" in feedback_lower:
            improvements.append("Include missing information")
        if "confusing" in feedback_lower:
            improvements.append("Restructure for clarity")

        return improvements

    def generate_report(self) -> ReflectionReport:
        """Generate reflection summary report"""
        if not self.reflections:
            return ReflectionReport(
                total_reflections=0,
                avg_quality=0.0,
                success_patterns=[],
                areas_for_improvement=[],
                recommendations=[],
            )

        ratings = [r.quality_rating for r in self.reflections if r.quality_rating]
        avg_quality = sum(r.value for r in ratings) / len(ratings) if ratings else 0

        success_patterns = []
        improvements = []
        recommendations = []

        for r in self.reflections:
            if r.reflection_type == ReflectionType.SUCCESS_PATTERN:
                success_patterns.extend(r.lessons_learned)
            elif r.reflection_type == ReflectionType.ERROR_ANALYSIS:
                improvements.extend(r.suggestions)
            elif r.reflection_type == ReflectionType.IMPROVEMENT:
                recommendations.extend(r.suggestions)

        return ReflectionReport(
            total_reflections=len(self.reflections),
            avg_quality=avg_quality,
            success_patterns=list(set(success_patterns))[:5],
            areas_for_improvement=list(set(improvements))[:5],
            recommendations=list(set(recommendations))[:5],
        )
