"""Core data models and abstract interfaces for the PCB inspection architecture.

Defines the model-agnostic inspection lifecycle:
    CAPTURED -> ALIGNING -> ALIGNMENT_OK / ALIGNMENT_FAILED -> INSPECTING -> GOOD / DEFECT / REVIEW

Supports:
- Physical PCB independence (multiple physical good boards)
- Canonical coordinate system (83 x 116 mm)
- Separation of alignment vs inspection
- Machine-readable inspection records
- Pluggable inspection backends (Classical CV now; YOLO / Anomaly AI later)
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class InspectionState(str, Enum):
    CAPTURED = "CAPTURED"
    ALIGNING = "ALIGNING"
    ALIGNMENT_OK = "ALIGNMENT_OK"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    INSPECTING = "INSPECTING"
    GOOD = "GOOD"
    DEFECT = "DEFECT"
    REVIEW = "REVIEW"


PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"


@dataclass
class AlignmentResult:
    """Outcome of detecting a PCB and rectifying it into the canonical frame."""
    success: bool
    status: str                         # OK | FAILED | POSITION_INVALID
    message: str = ""
    homography: np.ndarray | None = None  # raw px -> canonical px
    canonical_size: tuple[int, int] = (0, 0)  # (width_px, height_px)
    px_per_mm: float = 20.0
    inliers: int = 0
    inlier_ratio: float = 0.0
    fit_error_mm: float | None = None
    rotation_deg: float | None = None
    corners_canonical: list[list[float]] = field(default_factory=list)
    board_bbox_raw: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    seconds: float = 0.0

    @property
    def quality_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "fit_error_mm": round(self.fit_error_mm, 4) if self.fit_error_mm is not None else None,
            "rotation_deg": round(self.rotation_deg, 2) if self.rotation_deg is not None else None,
            "message": self.message,
        }


@dataclass
class DefectCandidate:
    """An abnormal region detected in canonical coordinates."""
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    area_mm2: float
    score: float                        # higher means more abnormal
    defect_type: str = "abnormal_region"
    refdes: str | None = None           # component reference if mapped (e.g. "R5")
    confidence: float = 0.0
    note: str = ""

    @property
    def box_mm(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) in canonical mm."""
        return (self.x_mm, self.y_mm, self.x_mm + self.width_mm, self.y_mm + self.height_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_mm": round(self.x_mm, 2),
            "y_mm": round(self.y_mm, 2),
            "width_mm": round(self.width_mm, 2),
            "height_mm": round(self.height_mm, 2),
            "area_mm2": round(self.area_mm2, 2),
            "score": round(self.score, 3),
            "defect_type": self.defect_type,
            "refdes": self.refdes,
            "confidence": round(self.confidence, 3),
            "note": self.note,
        }


@dataclass
class InspectionResult:
    """Final verdict and analysis for an inspected PCB image."""
    state: InspectionState
    result: str                         # PASS | FAIL | REVIEW
    anomaly_score: float = 0.0
    defects: list[DefectCandidate] = field(default_factory=list)
    canonical_image: np.ndarray | None = field(default=None, repr=False)
    aligned_path: Path | None = None
    evidence_path: Path | None = None
    report_text: str = ""
    message: str = ""
    stages: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def is_good(self) -> bool:
        return self.result == PASS

    @property
    def is_defect(self) -> bool:
        return self.result == FAIL

    @property
    def is_review(self) -> bool:
        return self.result == REVIEW


RECORD_COLUMNS = (
    "image_id", "session_id", "board_id", "dataset_type", "stage", "timestamp",
    "camera", "resolution", "raw_path", "aligned_path", "alignment_status",
    "alignment_quality", "inspection_status", "result", "anomaly_score",
    "defect_type", "defect_location", "quality_flags", "review_status", "sha256"
)


@dataclass
class InspectionRecord:
    """Complete machine-readable record for every captured PCB inspection."""
    image_id: str
    session_id: str
    board_id: str | None
    dataset_type: str                  # good | defect | test
    stage: str
    timestamp: str
    camera: str
    resolution: str
    raw_path: str
    aligned_path: str
    alignment_status: str
    alignment_quality: dict[str, Any]
    inspection_status: str             # GOOD | DEFECT | REVIEW
    result: str                        # PASS | FAIL | REVIEW
    anomaly_score: float
    defect_type: str | None
    defect_location: dict[str, Any] | None
    quality_flags: list[str]
    review_status: str                 # accepted | pending | quarantined
    sha256: str
    candidate_regions: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_row(self) -> dict[str, str]:
        loc_str = ""
        if self.defect_location:
            box = self.defect_location.get("box_mm", [])
            loc_str = f"x={box[0] if len(box)>0 else 0:.1f},y={box[1] if len(box)>1 else 0:.1f}"
            if self.defect_location.get("refdes"):
                loc_str = f"{self.defect_location['refdes']} ({loc_str})"
        return {
            "image_id": self.image_id,
            "session_id": self.session_id,
            "board_id": self.board_id or "-",
            "dataset_type": self.dataset_type,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "camera": self.camera,
            "resolution": self.resolution,
            "raw_path": self.raw_path,
            "aligned_path": self.aligned_path,
            "alignment_status": self.alignment_status,
            "alignment_quality": json.dumps(self.alignment_quality),
            "inspection_status": self.inspection_status,
            "result": self.result,
            "anomaly_score": f"{self.anomaly_score:.3f}",
            "defect_type": self.defect_type or "-",
            "defect_location": loc_str or "-",
            "quality_flags": ";".join(self.quality_flags),
            "review_status": self.review_status,
            "sha256": self.sha256,
        }


class BaseInspectionEngine(ABC):
    """Abstract interface for PCB inspection engines (Classical CV, YOLO, Anomaly, Hybrid)."""

    @abstractmethod
    def align(self, raw_image: np.ndarray) -> tuple[np.ndarray | None, AlignmentResult]:
        """Rectify raw camera frame into canonical PCB coordinates.

        Returns (canonical_image, AlignmentResult).
        """
        ...

    @abstractmethod
    def inspect(self, canonical_image: np.ndarray | None, alignment: AlignmentResult,
                board_id: str | None = None) -> InspectionResult:
        """Inspect rectified canonical image against the GOOD baseline.

        Returns InspectionResult with GOOD, DEFECT, or REVIEW state.
        """
        ...

    @abstractmethod
    def generate_evidence(self, raw_image: np.ndarray, canonical_image: np.ndarray | None,
                          result: InspectionResult, out_dir: Path) -> dict[str, Path]:
        """Generate human-reviewable evidence artifacts."""
        ...

    @abstractmethod
    def process_capture(self, raw_image: np.ndarray, capture_info: dict[str, Any],
                        session_dir: Path) -> InspectionRecord:
        """Full pipeline:
        1. Save raw image immediately (preserving raw evidence)
        2. Compute SHA-256
        3. Quality checks
        4. Alignment into canonical PCB frame
        5. Save canonical image
        6. Inspection against multi-board GOOD baseline
        7. Evidence generation
        8. Write machine-readable inspection record
        """
        ...
