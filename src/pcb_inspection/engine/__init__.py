"""PCB Inspection Engine Package.

Provides model-agnostic inspection architecture, multi-board statistical baselines,
canonical coordinate rectification, alignment quality gating, and dual raw/aligned storage.
"""

from .base import (
    BaseInspectionEngine,
    InspectionState,
    AlignmentResult,
    DefectCandidate,
    InspectionResult,
    InspectionRecord,
    RECORD_COLUMNS,
    PASS,
    FAIL,
    REVIEW,
)
from .baseline import GoodBaseline, build_good_baseline
from .classical import ClassicalInspectionEngine

__all__ = [
    "BaseInspectionEngine",
    "InspectionState",
    "AlignmentResult",
    "DefectCandidate",
    "InspectionResult",
    "InspectionRecord",
    "RECORD_COLUMNS",
    "PASS",
    "FAIL",
    "REVIEW",
    "GoodBaseline",
    "build_good_baseline",
    "ClassicalInspectionEngine",
]
