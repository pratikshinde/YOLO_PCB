"""Product-agnostic inspection results.

Decision states (docs/inspection_architecture.md):
    GOOD    every inspected region matched the product's GOOD reference bank, beyond the uncertainty band
    DEFECT  at least one enabled defect type found with sufficient evidence (location + evidence attached)
    REVIEW  the system cannot decide: unknown / ambiguous product, image quality, alignment, insufficient
            golden references, uncertain measurement, implausible result. Never a PCB defect by itself.
    ERROR   the inspection could not run (unreadable image, broken profile / reference, software error).
            The capture is still stored.

Confidence is never invented: when the engine has no calibrated confidence the field is None and the
measured scores are reported in the evidence instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GOOD, DEFECT, REVIEW, ERROR = "GOOD", "DEFECT", "REVIEW", "ERROR"
DECISIONS = (GOOD, DEFECT, REVIEW, ERROR)

# alignment_status values
ALIGN_OK, ALIGN_FAILED, ALIGN_NOT_RUN = "OK", "FAILED", "NOT_RUN"
# quality_status values
QUALITY_PASS, QUALITY_FLAGGED, QUALITY_FAIL, QUALITY_NOT_CHECKED = "PASS", "FLAGGED", "FAIL", "NOT_CHECKED"

# REVIEW / ERROR reason codes (machine-readable; the text explains)
UNKNOWN_PRODUCT = "UNKNOWN_PRODUCT"
AMBIGUOUS_PRODUCT = "AMBIGUOUS_PRODUCT"
IMAGE_QUALITY = "IMAGE_QUALITY"
NO_PCB = "NO_PCB"
ALIGNMENT = "ALIGNMENT"
NO_GOLDEN_REFERENCES = "NO_GOLDEN_REFERENCES"
INSUFFICIENT_GOLDEN_REFERENCES = "INSUFFICIENT_GOLDEN_REFERENCES"
UNCERTAIN = "UNCERTAIN"
NOT_CHECKABLE = "NOT_CHECKABLE"
IMPLAUSIBLE = "IMPLAUSIBLE"
NO_REGIONS = "NO_REGIONS"
PROFILE_INCOMPLETE = "PROFILE_INCOMPLETE"
SOFTWARE_ERROR = "SOFTWARE_ERROR"


@dataclass
class Defect:
    defect_id: str
    defect_type: str                       # project taxonomy, enabled in the product's inspection rules
    confidence: float | None               # None: no calibrated confidence (the measured score is in evidence)
    location: dict[str, Any]               # {"frame": "canonical_mm", "box_mm": [x0, y0, x1, y1], "center_mm": [x, y]}
    region: str | None                     # region of the product's region map (e.g. component refdes)
    evidence: dict[str, Any]               # measured values and evidence image paths
    severity: str | None                   # from the inspection rules; None = not defined
    status: str = "unconfirmed"            # a human review is stored separately, never written here


@dataclass
class Reason:
    code: str
    text: str


@dataclass
class ProductMatch:
    """Evidence that an image shows a given product: registration to that product's own golden reference."""
    product_id: str
    variant_id: str
    status: str                            # OK | FAILED | POSITION_INVALID | NO_PCB | INVALID_FRAME | ERROR
    inliers: int = 0
    inlier_ratio: float = 0.0
    fit_error_mm: float | None = None
    reason: str = ""
    registration: Any = field(default=None, repr=False)   # backend data, reused by the inspection (not serialised)

    def to_dict(self) -> dict[str, Any]:
        return {"product_id": self.product_id, "variant_id": self.variant_id, "status": self.status,
                "inliers": self.inliers, "inlier_ratio": self.inlier_ratio, "fit_error_mm": self.fit_error_mm,
                "reason": self.reason}


@dataclass
class InspectionOutcome:
    inspection_id: str
    product_id: str | None
    variant_id: str | None
    image_id: str
    board_id: str | None                   # physical board, if known (live inspection: usually None)
    alignment_status: str                  # OK | FAILED | NOT_RUN
    alignment_error: dict[str, Any] | None # measured alignment quality (inliers, fit error, rotation, ...)
    quality_status: str                    # PASS | FLAGGED | FAIL | NOT_CHECKED
    overall_decision: str                  # GOOD | DEFECT | REVIEW | ERROR
    confidence: float | None               # None unless the engine provides a calibrated value
    defects: list[Defect] = field(default_factory=list)
    reasons: list[Reason] = field(default_factory=list)
    engine: str | None = None              # e.g. REFERENCE_BASELINE
    engine_status: str | None = None       # e.g. EXPERIMENTAL
    identification: dict[str, Any] | None = None
    reference_bank: dict[str, Any] | None = None
    regions_checked: int | None = None
    uncertain_regions: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    evidence_files: dict[str, str] = field(default_factory=dict)
    seconds: float | None = None

    def __post_init__(self) -> None:
        if self.overall_decision not in DECISIONS:
            raise ValueError(f"overall_decision must be one of {DECISIONS}, got {self.overall_decision!r}")
        if self.overall_decision == DEFECT and not self.defects:
            raise ValueError("a DEFECT decision needs at least one defect with location and evidence")

    @property
    def reason_text(self) -> str:
        return "; ".join(r.text for r in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def review(code: str, text: str, **kw: Any) -> InspectionOutcome:
    """Shorthand used by engines: a REVIEW outcome with one reason (other fields via kw)."""
    values = dict(alignment_status=ALIGN_NOT_RUN, alignment_error=None, quality_status=QUALITY_NOT_CHECKED,
                  confidence=None)
    values.update(kw)
    reasons = values.pop("reasons", [])
    return InspectionOutcome(overall_decision=REVIEW, reasons=[Reason(code, text), *reasons], **values)
