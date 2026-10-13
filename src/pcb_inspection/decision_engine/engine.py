"""Rule-based decision layer, deliberately separate from any model.

Inspection stages report findings with a confidence. The rules here decide what
a finding means for the board. Rules live in configs/project.yaml, section
decision_engine.

Fail-safe behaviour (docs/requirements.md, FR-08):
  * PASS requires that every required stage ran and succeeded.
  * No required stages configured -> REVIEW.
  * A finding with no matching rule -> REVIEW.
  * A matching rule whose thresholds are not configured yet -> REVIEW.
  * Board verdict = most severe individual outcome (FAIL > REVIEW > PASS).

Confidence scores from different methods are not comparable; thresholds must
be tuned per defect type on validation data (docs/experiment_plan.md).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


_SEVERITY = {Verdict.PASS: 0, Verdict.REVIEW: 1, Verdict.FAIL: 2}
APPLIES_TO = ("any", "critical", "non_critical")


@dataclass(frozen=True)
class Finding:
    """One defect candidate reported by an inspection stage."""

    defect_type: str
    confidence: float
    source: str
    component_refdes: str | None = None
    component_critical: bool | None = None  # None: criticality unknown
    bbox_px: tuple[float, float, float, float] | None = None  # x, y, width, height in the capture

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class StageResult:
    """Outcome of one pipeline stage, e.g. image_quality or alignment."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Rule:
    defect_type: str
    fail_threshold: float | None
    review_threshold: float | None
    applies_to: str = "any"

    def __post_init__(self) -> None:
        if self.applies_to not in APPLIES_TO:
            raise ValueError(f"applies_to must be one of {APPLIES_TO}, got {self.applies_to!r}")
        for name in ("fail_threshold", "review_threshold"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.defect_type}: {name} must be in [0, 1] or null, got {value}")
        if self.configured and self.review_threshold > self.fail_threshold:
            raise ValueError(f"{self.defect_type}: review_threshold must not exceed fail_threshold")

    @property
    def configured(self) -> bool:
        return self.fail_threshold is not None and self.review_threshold is not None

    def matches(self, finding: Finding) -> bool:
        if finding.defect_type != self.defect_type:
            return False
        if self.applies_to == "critical":
            return finding.component_critical is True
        if self.applies_to == "non_critical":
            return finding.component_critical is False
        return True


@dataclass(frozen=True)
class DecisionConfig:
    rules: tuple[Rule, ...]
    required_stages: tuple[str, ...]

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> DecisionConfig:
        """Build from the decision_engine section of the project config."""
        rules = tuple(Rule(**rule) for rule in section.get("rules") or ())
        return cls(rules=rules, required_stages=tuple(section.get("required_stages") or ()))


@dataclass(frozen=True)
class Reason:
    verdict: Verdict
    message: str
    finding: Finding | None = None


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[Reason, ...]


def decide(findings: Iterable[Finding], stages: Iterable[StageResult], config: DecisionConfig) -> Decision:
    reasons: list[Reason] = []

    if not config.required_stages:
        reasons.append(Reason(Verdict.REVIEW, "no required inspection stages are configured"))
    results = {stage.name: stage for stage in stages}
    for name in config.required_stages:
        stage = results.get(name)
        if stage is None:
            reasons.append(Reason(Verdict.REVIEW, f"required stage '{name}' did not run"))
        elif not stage.ok:
            detail = f": {stage.detail}" if stage.detail else ""
            reasons.append(Reason(Verdict.REVIEW, f"required stage '{name}' failed{detail}"))

    for finding in findings:
        reasons.append(_evaluate(finding, config.rules))

    verdict = max((r.verdict for r in reasons), key=_SEVERITY.__getitem__, default=Verdict.PASS)
    if verdict is Verdict.PASS:
        reasons.append(Reason(Verdict.PASS, "all required stages succeeded; no finding reached a review threshold"))
    return Decision(verdict, tuple(reasons))


def _evaluate(finding: Finding, rules: Iterable[Rule]) -> Reason:
    where = f" at {finding.component_refdes}" if finding.component_refdes else ""
    label = f"{finding.defect_type}{where} (confidence {finding.confidence:.2f}, source {finding.source})"
    rule = next((r for r in rules if r.matches(finding)), None)
    if rule is None:
        return Reason(Verdict.REVIEW, f"{label}: no decision rule for this finding", finding)
    if not rule.configured:
        return Reason(Verdict.REVIEW, f"{label}: decision rule thresholds not configured", finding)
    if finding.confidence >= rule.fail_threshold:
        return Reason(Verdict.FAIL, f"{label}: at or above fail threshold {rule.fail_threshold}", finding)
    if finding.confidence >= rule.review_threshold:
        return Reason(Verdict.REVIEW, f"{label}: at or above review threshold {rule.review_threshold}", finding)
    return Reason(Verdict.PASS, f"{label}: below review threshold {rule.review_threshold}, ignored", finding)
