"""
Auto-apply safe patches with configurable risk thresholds.

Provides:
- Risk assessment for patches
- Safe patch classification
- Auto-apply rules engine
- Approval workflow integration
- Audit logging
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class RiskLevel(Enum):
    """Patch risk levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


class AutoApplyDecision(Enum):
    """Auto-apply decisions"""

    AUTO_APPLY = "auto_apply"
    APPROVAL_REQUIRED = "approval_required"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


@dataclass
class PatchAssessment:
    """Assessment of a patch"""

    patch_id: str
    risk_level: RiskLevel
    confidence: float  # 0-1
    size_score: float  # 0-1 (smaller = safer)
    scope_score: float  # 0-1 (narrower = safer)
    test_coverage: float  # 0-1
    validation_score: float  # 0-1
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoApplyRule:
    """Rule for auto-applying patches"""

    id: str
    name: str
    description: str = ""
    risk_threshold: RiskLevel = RiskLevel.LOW
    max_file_changes: int = 5
    max_lines_added: int = 100
    min_confidence: float = 0.8
    require_tests: bool = True
    require_validation: bool = True
    enabled: bool = True


@dataclass
class ApplyRequest:
    """Request to apply a patch"""

    id: str
    patch_id: str
    repo_path: str
    branch: str
    risk_level: RiskLevel
    decision: AutoApplyDecision
    requested_at: str
    decided_at: str = ""
    decided_by: str = "system"
    reason: str = ""


class PatchAnalyzer:
    """Analyze patch for risk assessment"""

    @staticmethod
    def analyze(patch: dict[str, Any]) -> PatchAssessment:
        """Analyze a patch and return risk assessment"""
        patch_id = patch.get("id", str(uuid.uuid4()))

        size_score = PatchAnalyzer._assess_size(patch)
        scope_score = PatchAnalyzer._assess_scope(patch)
        test_coverage = PatchAnalyzer._assess_tests(patch)
        validation_score = PatchAnalyzer._assess_validation(patch)

        confidence = (size_score + scope_score + test_coverage + validation_score) / 4

        risk_level = PatchAnalyzer._calculate_risk(
            size_score, scope_score, test_coverage, validation_score
        )

        reasons = PatchAnalyzer._generate_reasons(
            size_score, scope_score, test_coverage, validation_score
        )

        return PatchAssessment(
            patch_id=patch_id,
            risk_level=risk_level,
            confidence=confidence,
            size_score=size_score,
            scope_score=scope_score,
            test_coverage=test_coverage,
            validation_score=validation_score,
            reasons=reasons,
            metadata=patch,
        )

    @staticmethod
    def _assess_size(patch: dict[str, Any]) -> float:
        """Assess patch size (smaller = safer)"""
        files = patch.get("files_changed", [])
        if not files:
            return 0.5

        total_lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)

        if total_lines < 10:
            return 1.0
        elif total_lines < 50:
            return 0.8
        elif total_lines < 200:
            return 0.6
        elif total_lines < 500:
            return 0.4
        else:
            return 0.2

    @staticmethod
    def _assess_scope(patch: dict[str, Any]) -> float:
        """Assess patch scope (narrower = safer)"""
        files = patch.get("files_changed", [])
        if not files:
            return 0.5

        critical_patterns = [
            "auth",
            "security",
            "password",
            "crypto",
            "key",
            "payment",
            "billing",
            "admin",
            "root",
        ]

        critical_count = 0
        for f in files:
            filename = f.get("filename", "").lower()
            if any(p in filename for p in critical_patterns):
                critical_count += 1

        if critical_count == 0:
            return 1.0
        elif critical_count <= 1:
            return 0.6
        else:
            return 0.2

    @staticmethod
    def _assess_tests(patch: dict[str, Any]) -> float:
        """Assess test coverage"""
        has_tests = patch.get("has_tests", False)
        test_files_changed = patch.get("test_files_changed", 0)

        if has_tests and test_files_changed > 0:
            return 1.0
        elif has_tests:
            return 0.7
        else:
            return 0.4

    @staticmethod
    def _assess_validation(patch: dict[str, Any]) -> float:
        """Assess validation results"""
        validation = patch.get("validation", {})

        passed_tests = validation.get("tests_passed", False)
        passed_security = validation.get("security_scan_passed", True)
        passed_syntax = validation.get("syntax_valid", True)

        score = 0
        if passed_syntax:
            score += 0.33
        if passed_tests:
            score += 0.33
        if passed_security:
            score += 0.34

        return score

    @staticmethod
    def _calculate_risk(size, scope, tests, validation) -> RiskLevel:
        """Calculate overall risk level"""
        scores = [size, scope, tests, validation]
        avg = sum(scores) / len(scores)

        if avg >= 0.8:
            return RiskLevel.SAFE
        elif avg >= 0.6:
            return RiskLevel.LOW
        elif avg >= 0.4:
            return RiskLevel.MEDIUM
        elif avg >= 0.2:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    @staticmethod
    def _generate_reasons(size, scope, tests, validation) -> list[str]:
        """Generate human-readable reasons"""
        reasons = []

        if size >= 0.8:
            reasons.append("Small patch (minimal risk)")
        elif size < 0.4:
            reasons.append("Large patch (higher risk)")

        if scope >= 0.8:
            reasons.append("Limited scope (few critical files)")
        elif scope < 0.4:
            reasons.append("Affects critical files (auth/security)")

        if tests >= 0.8:
            reasons.append("Has test coverage")
        elif tests < 0.4:
            reasons.append("No test coverage")

        if validation >= 0.8:
            reasons.append("All validations passed")
        elif validation < 0.4:
            reasons.append("Validations failed or incomplete")

        return reasons


class AutoApplyEngine:
    """Engine for auto-applying safe patches"""

    def __init__(self):
        self.rules: dict[str, AutoApplyRule] = {}
        self.requests: list[ApplyRequest] = []

    def add_rule(self, rule: AutoApplyRule):
        """Add an auto-apply rule"""
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def evaluate(
        self,
        assessment: PatchAssessment,
        context: dict | None = None,
    ) -> AutoApplyDecision:
        """Evaluate if patch should be auto-applied"""
        context = context or {}

        applicable_rules = [
            r
            for r in self.rules.values()
            if r.enabled and r.risk_threshold.value >= assessment.risk_level.value
        ]

        if not applicable_rules:
            return AutoApplyDecision.MANUAL_REVIEW

        for rule in applicable_rules:
            if assessment.risk_level.value > rule.risk_threshold.value:
                continue

            if assessment.confidence < rule.min_confidence:
                return AutoApplyDecision.APPROVAL_REQUIRED

            if rule.require_tests and assessment.test_coverage < 0.5:
                return AutoApplyDecision.APPROVAL_REQUIRED

            if rule.require_validation and assessment.validation_score < 0.7:
                return AutoApplyDecision.APPROVAL_REQUIRED

            if assessment.metadata.get("files_changed", []):
                file_count = len(assessment.metadata.get("files_changed", []))
                if file_count > rule.max_file_changes:
                    return AutoApplyDecision.APPROVAL_REQUIRED

        return AutoApplyDecision.AUTO_APPLY

    def create_request(
        self,
        assessment: PatchAssessment,
        repo_path: str,
        branch: str,
        context: dict | None = None,
    ) -> ApplyRequest:
        """Create an apply request"""
        decision = self.evaluate(assessment, context)

        request = ApplyRequest(
            id=str(uuid.uuid4()),
            patch_id=assessment.patch_id,
            repo_path=repo_path,
            branch=branch,
            risk_level=assessment.risk_level,
            decision=decision,
            requested_at=datetime.now().isoformat(),
        )

        self.requests.append(request)
        return request

    def approve(self, request_id: str, approver: str = "manual", reason: str = "") -> bool:
        """Approve a request"""
        for request in self.requests:
            if request.id == request_id:
                request.decision = AutoApplyDecision.AUTO_APPLY
                request.decided_at = datetime.now().isoformat()
                request.decided_by = approver
                request.reason = reason
                return True
        return False

    def reject(self, request_id: str, approver: str = "manual", reason: str = "") -> bool:
        """Reject a request"""
        for request in self.requests:
            if request.id == request_id:
                request.decision = AutoApplyDecision.REJECTED
                request.decided_at = datetime.now().isoformat()
                request.decided_by = approver
                request.reason = reason
                return True
        return False

    def get_pending_requests(self) -> list[ApplyRequest]:
        """Get pending apply requests"""
        return [r for r in self.requests if r.decision == AutoApplyDecision.APPROVAL_REQUIRED]


def create_default_rules() -> list[AutoApplyRule]:
    """Create default auto-apply rules"""
    return [
        AutoApplyRule(
            id="safe-patches",
            name="Safe Patches",
            description="Automatically apply safe, small, tested patches",
            risk_threshold=RiskLevel.SAFE,
            max_file_changes=3,
            max_lines_added=50,
            min_confidence=0.9,
            require_tests=True,
            require_validation=True,
        ),
        AutoApplyRule(
            id="low-risk",
            name="Low Risk",
            description="Auto-apply low risk patches with approval",
            risk_threshold=RiskLevel.LOW,
            max_file_changes=5,
            max_lines_added=100,
            min_confidence=0.8,
            require_tests=False,
            require_validation=True,
        ),
    ]
