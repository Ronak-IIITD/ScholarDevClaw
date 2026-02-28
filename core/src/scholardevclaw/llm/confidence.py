"""
Confidence calibration for better uncertainty quantification.

Provides:
- Confidence scoring from model responses
- Calibration tracking
- Prediction accuracy history
- Calibration curves
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ConfidenceLevel(Enum):
    """Confidence levels"""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Prediction:
    """A single prediction with confidence"""

    id: str
    prediction: str
    confidence: float  # 0-1
    confidence_level: ConfidenceLevel
    category: str  # "mapping", "validation", "generation"
    actual: str = ""
    correct: bool | None = None
    created_at: str = ""
    resolved_at: str = ""


@dataclass
class CalibrationMetrics:
    """Calibration metrics"""

    accuracy: float  # Overall accuracy
    confidence_error: float  # |confidence - accuracy|
    calibration_score: float  # Lower is better
    n_samples: int


class ConfidenceCalibrator:
    """Calibrate model confidence scores"""

    def __init__(self):
        self.predictions: list[Prediction] = []
        self._calibration_cache: CalibrationMetrics | None = None

    def add_prediction(
        self,
        prediction: str,
        confidence: float,
        category: str,
    ) -> Prediction:
        """Add a new prediction"""
        level = self._calculate_level(confidence)

        pred = Prediction(
            id=f"pred_{len(self.predictions)}",
            prediction=prediction,
            confidence=confidence,
            confidence_level=level,
            category=category,
            created_at=datetime.now().isoformat(),
        )

        self.predictions.append(pred)
        self._calibration_cache = None

        return pred

    def resolve_prediction(
        self,
        prediction_id: str,
        actual: str,
    ) -> bool:
        """Mark prediction as resolved with actual outcome"""
        for pred in self.predictions:
            if pred.id == prediction_id:
                pred.actual = actual
                pred.correct = pred.prediction == actual
                pred.resolved_at = datetime.now().isoformat()
                self._calibration_cache = None
                return True
        return False

    def get_metrics(self) -> CalibrationMetrics:
        """Calculate calibration metrics"""
        resolved = [p for p in self.predictions if p.correct is not None]

        if not resolved:
            return CalibrationMetrics(
                accuracy=0.0,
                confidence_error=0.0,
                calibration_score=0.0,
                n_samples=0,
            )

        accuracy = sum(1 for p in resolved if p.correct) / len(resolved)

        confidence_error = sum(abs(p.confidence - (1 if p.correct else 0)) for p in resolved) / len(
            resolved
        )

        calibration_score = self._calculate_ece(resolved)

        return CalibrationMetrics(
            accuracy=accuracy,
            confidence_error=confidence_error,
            calibration_score=calibration_score,
            n_samples=len(resolved),
        )

    def _calculate_ece(self, predictions: list[Prediction], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bins = [[] for _ in range(n_bins)]

        for pred in predictions:
            bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(pred)

        ece = 0.0
        total = len(predictions)

        for bin_preds in bins:
            if not bin_preds:
                continue

            bin_acc = sum(1 for p in bin_preds if p.correct) / len(bin_preds)
            bin_conf = sum(p.confidence for p in bin_preds) / len(bin_preds)

            ece += (len(bin_preds) / total) * abs(bin_acc - bin_conf)

        return ece

    def _calculate_level(self, confidence: float) -> ConfidenceLevel:
        """Calculate confidence level from score"""
        if confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.4:
            return ConfidenceLevel.LOW
        elif confidence < 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def get_by_category(self, category: str) -> list[Prediction]:
        """Get predictions by category"""
        return [p for p in self.predictions if p.category == category]

    def get_calibration_curve(self) -> list[dict[str, float]]:
        """Get data for calibration curve"""
        resolved = [p for p in self.predictions if p.correct is not None]
        n_bins = 10

        curve = []
        for i in range(n_bins):
            bin_min = i / n_bins
            bin_max = (i + 1) / n_bins

            bin_preds = [p for p in resolved if bin_min <= p.confidence < bin_max]

            if bin_preds:
                accuracy = sum(1 for p in bin_preds if p.correct) / len(bin_preds)
                avg_confidence = sum(p.confidence for p in bin_preds) / len(bin_preds)

                curve.append(
                    {
                        "bin": i,
                        "confidence": avg_confidence,
                        "accuracy": accuracy,
                        "count": len(bin_preds),
                    }
                )

        return curve


class AdaptiveConfidence:
    """Adaptive confidence based on multiple signals"""

    @staticmethod
    def calculate(
        validation_score: float = 0.0,
        test_coverage: float = 0.0,
        patch_size: float = 0.0,
        complexity: float = 0.0,
        historical_accuracy: float | None = None,
    ) -> float:
        """Calculate calibrated confidence from multiple signals"""

        weights = {
            "validation": 0.35,
            "coverage": 0.25,
            "size": 0.15,
            "complexity": 0.15,
            "historical": 0.10,
        }

        size_score = AdaptiveConfidence._size_score(patch_size)
        complexity_score = 1.0 - min(complexity, 1.0)

        score = (
            weights["validation"] * validation_score
            + weights["coverage"] * test_coverage
            + weights["size"] * size_score
            + weights["complexity"] * complexity_score
        )

        if historical_accuracy is not None:
            score = score * 0.9 + weights["historical"] * historical_accuracy

        return max(0.0, min(1.0, score))

    @staticmethod
    def _size_score(size: float) -> float:
        """Calculate score from patch size"""
        if size < 10:
            return 1.0
        elif size < 50:
            return 0.9
        elif size < 100:
            return 0.7
        elif size < 200:
            return 0.5
        else:
            return 0.3


class UncertaintyEstimator:
    """Estimate uncertainty in predictions"""

    @staticmethod
    def estimate_from_variance(responses: list[str]) -> float:
        """Estimate uncertainty from multiple responses"""
        if len(responses) < 2:
            return 0.0

        unique_responses = set(responses)
        variance = 1.0 - (len(unique_responses) / len(responses))

        return variance

    @staticmethod
    def estimate_from_confidence(confidences: list[float]) -> float:
        """Estimate uncertainty from confidence scores"""
        if not confidences:
            return 1.0

        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        return min(1.0, variance * 2)

    @staticmethod
    def get_uncertainty_level(uncertainty: float) -> str:
        """Get uncertainty level description"""
        if uncertainty < 0.2:
            return "very confident"
        elif uncertainty < 0.4:
            return "somewhat confident"
        elif uncertainty < 0.6:
            return "neutral"
        elif uncertainty < 0.8:
            return "uncertain"
        else:
            return "very uncertain"


def quick_confidence(
    validation_score: float,
    test_coverage: float,
) -> float:
    """Quick confidence calculation"""
    return AdaptiveConfidence.calculate(
        validation_score=validation_score,
        test_coverage=test_coverage,
    )
