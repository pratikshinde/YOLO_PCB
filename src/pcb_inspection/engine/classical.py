"""Classical Computer Vision implementation of the BaseInspectionEngine (EXPERIMENTAL, synthetic-tested only).

A pixel tolerance-map comparison ("difference above tolerance = candidate"). It is not trained AI and has not
been validated on real boards; the product station uses pcb_inspection.inspection (REFERENCE_BASELINE) instead.
Board dimensions come from the caller (product profile); nothing is assumed.

Implements:
1. PCB Detection & Quadrilateral Extraction
2. Fronto-Parallel Canonical Rectification (board size from EngineConfig)
3. Multi-Metric Alignment Quality Gate
4. Illumination & Color Normalization
5. Multi-Board GOOD Baseline Anomaly Detection
6. Component Verification (when component map provided)
7. Dual Storage Pipeline (Raw evidence preserved + Canonical image + Records)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..acquisition.camera import encode_png
from ..alignment.registration import RegistrationParams, RegistrationResult, register
from ..alignment.validation import AlignmentCheck, AlignmentValidationParams, validate_alignment
from ..comparison.board_inspection import InspectionReference
from ..comparison.components import ComponentMap, check_components
from ..image_quality.checks import ImageQuality, QualityParams, check_image
from ..preprocessing.board import (
    BoardSegmentationParams,
    CanonicalFrame,
    board_mask,
    board_quad,
    canonical_frame,
    warp_to_canonical,
)
from ..preprocessing.presence import NO_PCB, PresenceParams, detect_pcb
from .base import (
    FAIL,
    PASS,
    RECORD_COLUMNS,
    REVIEW,
    AlignmentResult,
    BaseInspectionEngine,
    DefectCandidate,
    InspectionRecord,
    InspectionResult,
    InspectionState,
)
from .baseline import GoodBaseline


def sha256_of(source: Path | str | bytes) -> str:
    """Compute SHA-256 hex digest of file or bytes."""
    h = hashlib.sha256()
    if isinstance(source, bytes):
        h.update(source)
    else:
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


@dataclass
class EngineConfig:
    board_dimensions_mm: tuple[float, float] | None = None    # (length, width) from the product profile; None = unknown
    px_per_mm: float = 20.0
    segmentation: BoardSegmentationParams = field(default_factory=BoardSegmentationParams)
    presence: PresenceParams = field(default_factory=PresenceParams)
    registration: RegistrationParams = field(default_factory=RegistrationParams)
    alignment: AlignmentValidationParams = field(default_factory=AlignmentValidationParams)
    quality: QualityParams = field(default_factory=QualityParams)
    min_defect_area_mm2: float = 0.5
    anomaly_threshold: float = 1.3
    illumination_mode: str = "global"


class ClassicalInspectionEngine(BaseInspectionEngine):
    """Experimental classical CV inspection engine with a multi-board GOOD baseline (not validated on real boards)."""

    def __init__(
        self,
        baseline: GoodBaseline,
        reference: InspectionReference | None = None,
        cmap: ComponentMap | None = None,
        config: EngineConfig = EngineConfig(),
    ):
        self.baseline = baseline
        self.reference = reference
        self.cmap = cmap
        self.config = config

    def align(self, raw_image: np.ndarray) -> tuple[np.ndarray | None, AlignmentResult]:
        """Detect board and warp into canonical coordinates."""
        start = time.perf_counter()
        if raw_image is None or raw_image.size == 0:
            return None, AlignmentResult(
                success=False, status="INVALID_FRAME", message="Empty or null image frame",
                seconds=time.perf_counter() - start
            )

        # 1. Detect PCB presence
        presence = detect_pcb(raw_image, self.config.presence, self.config.segmentation)
        if presence.status == NO_PCB:
            return None, AlignmentResult(
                success=False, status="NO_PCB", message=presence.reason,
                seconds=time.perf_counter() - start
            )

        # 2. Estimate registration to canonical baseline/reference
        ref_bgr = self.reference.image if self.reference is not None else self.baseline.median_canonical_bgr
        ref_mask = self.reference.mask if self.reference is not None else self.baseline.board_mask_canonical
        ref_quad = self.reference.quad if self.reference is not None else np.float32([
            [0, 0], [self.baseline.canonical_size[0] - 1, 0],
            [self.baseline.canonical_size[0] - 1, self.baseline.canonical_size[1] - 1],
            [0, self.baseline.canonical_size[1] - 1]
        ])
        ref_ppm = self.reference.frame.source_px_per_mm if self.reference is not None else self.config.px_per_mm

        reg = register(ref_bgr, ref_mask, raw_image, presence.mask, self.config.registration)
        check = validate_alignment(reg, ref_quad, ref_ppm, raw_image.shape, self.config.alignment)

        if not reg.success or check.status != "OK" or reg.homography is None:
            return None, AlignmentResult(
                success=False,
                status=check.status,
                message=f"Alignment quality gate rejected: {check.reason}",
                inliers=reg.inliers,
                inlier_ratio=reg.inlier_ratio,
                fit_error_mm=check.fit_error_mm,
                rotation_deg=check.rotation_deg,
                seconds=time.perf_counter() - start
            )

        # 3. Warp into canonical frame
        if self.reference is not None:
            to_canonical = self.reference.frame.homography @ reg.homography
            det = abs(np.linalg.det(reg.homography[:2, :2]))
            test_ppm = self.reference.frame.source_px_per_mm / float(np.sqrt(max(det, 1e-6)))
            canonical_size = self.reference.frame.size
        else:
            to_canonical = reg.homography
            test_ppm = self.config.px_per_mm
            canonical_size = self.baseline.canonical_size

        canonical_img, valid_mask = warp_to_canonical(
            raw_image, to_canonical, canonical_size, test_ppm, self.config.px_per_mm
        )

        corners_list = check.corners_px.tolist() if check.corners_px is not None else []
        alignment_res = AlignmentResult(
            success=True,
            status="OK",
            message=f"Aligned successfully ({reg.inliers} inliers, residual {check.fit_error_mm:.3f} mm)",
            homography=to_canonical,
            canonical_size=canonical_size,
            px_per_mm=self.config.px_per_mm,
            inliers=reg.inliers,
            inlier_ratio=reg.inlier_ratio,
            fit_error_mm=check.fit_error_mm,
            rotation_deg=check.rotation_deg,
            corners_canonical=corners_list,
            seconds=time.perf_counter() - start
        )
        return canonical_img, alignment_res

    def inspect(self, canonical_image: np.ndarray | None, alignment: AlignmentResult,
                board_id: str | None = None) -> InspectionResult:
        """Inspect rectified canonical image against multi-board GOOD baseline."""
        start = time.perf_counter()
        stages = {"alignment": f"{alignment.status}: {alignment.message}"}

        # 1. Alignment Quality Gate
        if not alignment.success or canonical_image is None:
            return InspectionResult(
                state=InspectionState.ALIGNMENT_FAILED,
                result=REVIEW,
                anomaly_score=0.0,
                message=f"Alignment gate rejected: {alignment.message}. (Not treated as a board defect)",
                stages=stages,
                seconds=time.perf_counter() - start
            )

        # 2. Multi-board Baseline Evaluation
        # If inspecting a board present in baseline, use leave-one-board-out
        try:
            active_baseline = self.baseline.without_board(board_id)
        except ValueError as e:
            return InspectionResult(
                state=InspectionState.REVIEW, result=REVIEW, anomaly_score=0.0,
                message=f"Not inspected: {e} (leave-one-board-out).", stages=stages,
                seconds=time.perf_counter() - start)
        norm_dev, diff_mask, anomaly_score = active_baseline.compute_anomaly(
            canonical_image, illumination_mode=self.config.illumination_mode
        )

        # Filter noise by morphological opening and area threshold
        px2 = self.config.px_per_mm ** 2
        k = max(1, round(0.2 * self.config.px_per_mm))
        clean_mask = cv2.morphologyEx(
            diff_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        candidates: list[DefectCandidate] = []

        for i in range(1, count):
            x, y, w, h, area = stats[i]
            area_mm2 = area / px2
            if area_mm2 < self.config.min_defect_area_mm2:
                continue

            member = labels == i
            region_score = float(norm_dev[member].mean())

            # Match against component map if present
            matched_refdes = None
            if self.cmap is not None:
                cx_mm, cy_mm = (x + w / 2) / self.config.px_per_mm, (y + h / 2) / self.config.px_per_mm
                for c in self.cmap.components:
                    x0, y0, x1, y1 = c.box_mm
                    if x0 <= cx_mm <= x1 and y0 <= cy_mm <= y1:
                        matched_refdes = c.refdes
                        break

            candidates.append(DefectCandidate(
                x_mm=x / self.config.px_per_mm,
                y_mm=y / self.config.px_per_mm,
                width_mm=w / self.config.px_per_mm,
                height_mm=h / self.config.px_per_mm,
                area_mm2=area_mm2,
                score=region_score,
                defect_type="missing_or_damaged_component" if matched_refdes else "surface_abnormality",
                refdes=matched_refdes,
                confidence=min(1.0, region_score / 2.0),
                note=f"Exceeded normal inter-board tolerance by {region_score:.1f}x",
            ))

        # Anomaly score is driven by candidate defect regions if present
        if candidates:
            effective_anomaly_score = max(c.score for c in candidates)
        else:
            effective_anomaly_score = 0.0

        # 3. Decision Logic
        # A board is normal (GOOD / PASS) if it has no significant abnormal regions
        # exceeding the multi-board tolerance threshold.
        if candidates:
            top = candidates[0]
            desc = f"{top.refdes} " if top.refdes else ""
            if top.score >= self.config.anomaly_threshold:
                state = InspectionState.DEFECT
                result = FAIL
                message = f"Defect detected: abnormal region {desc}(area {top.area_mm2:.1f} mm², score {top.score:.2f})."
            else:
                state = InspectionState.REVIEW
                result = REVIEW
                message = f"Borderline anomaly {desc}(area {top.area_mm2:.1f} mm², score {top.score:.2f}); recommended for review."
        else:
            state = InspectionState.GOOD
            result = PASS
            message = "Board normal: all features within verified GOOD baseline variation."

        stages["inspection"] = f"{result}: {message}"
        return InspectionResult(
            state=state,
            result=result,
            anomaly_score=effective_anomaly_score,
            defects=candidates,
            canonical_image=canonical_image,
            message=message,
            stages=stages,
            seconds=time.perf_counter() - start
        )

    def generate_evidence(
        self, raw_image: np.ndarray, canonical_image: np.ndarray | None,
        result: InspectionResult, out_dir: Path
    ) -> dict[str, Path]:
        """Generate human-reviewable evidence artifacts."""
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}

        if canonical_image is not None:
            # Annotated canonical image with bounding boxes
            annotated = canonical_image.copy()
            for d in result.defects:
                x0 = round(d.x_mm * self.config.px_per_mm)
                y0 = round(d.y_mm * self.config.px_per_mm)
                x1 = round((d.x_mm + d.width_mm) * self.config.px_per_mm)
                y1 = round((d.y_mm + d.height_mm) * self.config.px_per_mm)
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
                label = d.refdes or f"DEFECT ({d.score:.1f})"
                cv2.putText(annotated, label, (x0, max(15, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            ann_path = out_dir / "annotated.jpg"
            cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            artifacts["annotated"] = ann_path

            # Side-by-side evidence: Baseline Median vs Aligned Test Image
            if result.defects:
                d = result.defects[0]
                x0 = max(0, round((d.x_mm - 2) * self.config.px_per_mm))
                y0 = max(0, round((d.y_mm - 2) * self.config.px_per_mm))
                x1 = min(canonical_image.shape[1], round((d.x_mm + d.width_mm + 2) * self.config.px_per_mm))
                y1 = min(canonical_image.shape[0], round((d.y_mm + d.height_mm + 2) * self.config.px_per_mm))
                crop_test = canonical_image[y0:y1, x0:x1]
                crop_base = self.baseline.median_canonical_bgr[y0:y1, x0:x1]
                if crop_test.size > 0 and crop_base.size > 0:
                    evidence_pair = np.hstack([crop_base, crop_test])
                    ev_path = out_dir / "evidence.jpg"
                    cv2.imwrite(str(ev_path), evidence_pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    artifacts["evidence"] = ev_path

        return artifacts

    def process_capture(
        self, raw_image: np.ndarray, capture_info: dict[str, Any], session_dir: Path
    ) -> InspectionRecord:
        """Complete, traceable capture and inspection pipeline with dual storage."""
        image_id = capture_info.get("image_id", f"CAP_{datetime.now():%Y%m%d_%H%M%S}")
        session_id = capture_info.get("session_id", "LIVE_SESSION")
        board_id = capture_info.get("board_id")
        dataset_type = capture_info.get("dataset_type", "good")
        stage = capture_info.get("stage", "after_tht")
        timestamp = capture_info.get("timestamp", datetime.now().isoformat())
        camera_name = str(capture_info.get("camera", "camera"))
        res_str = f"{raw_image.shape[1]}x{raw_image.shape[0]}" if raw_image is not None else "0x0"

        # Step 1: Immediately save raw image (Original Evidence Preserved!)
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{image_id}.png"
        with raw_path.open("wb") as f:
            f.write(encode_png(raw_image))

        # Step 2: Compute SHA-256
        sha256 = sha256_of(raw_path)

        # Step 3: Quality check
        quality_check = check_image(raw_path, self.config.quality, self.config.board_dimensions_mm)
        quality_flags = list(quality_check.flags)

        # Step 4: Alignment to Canonical PCB coordinates
        canonical_img, alignment_res = self.align(raw_image)

        # Step 5: Save canonical image if alignment succeeded
        aligned_path_str = ""
        if canonical_img is not None and alignment_res.success:
            aligned_dir = session_dir / "aligned"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            aligned_file = aligned_dir / f"{image_id}_canonical.png"
            with aligned_file.open("wb") as f:
                f.write(encode_png(canonical_img))
            aligned_path_str = str(aligned_file)

        # Step 6: Inspection against multi-board GOOD baseline
        inspection_res = self.inspect(canonical_img, alignment_res, board_id=board_id)

        # Step 7: Generate human-reviewable evidence artifacts
        evidence_dir = session_dir / "evidence" / image_id
        artifacts = self.generate_evidence(raw_image, canonical_img, inspection_res, evidence_dir)

        # Defect location details
        defect_loc = None
        defect_type = None
        if inspection_res.defects:
            top = inspection_res.defects[0]
            defect_type = top.defect_type
            defect_loc = {
                "refdes": top.refdes,
                "box_mm": list(top.box_mm),
                "area_mm2": top.area_mm2,
                "confidence": top.confidence,
            }

        # Step 8: Build and save machine-readable inspection record
        record = InspectionRecord(
            image_id=image_id,
            session_id=session_id,
            board_id=board_id,
            dataset_type=dataset_type,
            stage=stage,
            timestamp=timestamp,
            camera=camera_name,
            resolution=res_str,
            raw_path=str(raw_path),
            aligned_path=aligned_path_str,
            alignment_status=alignment_res.status,
            alignment_quality=alignment_res.quality_summary,
            inspection_status=inspection_res.state.value,
            result=inspection_res.result,
            anomaly_score=inspection_res.anomaly_score,
            defect_type=defect_type,
            defect_location=defect_loc,
            quality_flags=quality_flags,
            review_status="accepted" if inspection_res.result == PASS else "pending",
            sha256=sha256,
            candidate_regions=[d.to_dict() for d in inspection_res.defects],
            message=inspection_res.message,
        )

        records_dir = session_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        record_json_path = records_dir / f"{image_id}.json"
        record_json_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

        # Step 9: Append to session summary.csv
        summary_path = session_dir / "summary.csv"
        new_summary = not summary_path.is_file()
        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RECORD_COLUMNS)
            if new_summary:
                writer.writeheader()
            writer.writerow(record.to_summary_row())

        return record
