"""Inspection station: one capture -> stored -> quality -> product -> inspection -> result -> records.

    1. save the original + image record (capture store)            FIRST; if this fails nothing else runs (ERROR)
    2. image quality (station camera thresholds)                    blocking flag -> REVIEW (image quality)
    3. product: AUTO identification or the operator's selection     unknown / ambiguous -> REVIEW
    4. product quality check (the product's thresholds, board size) blocking flag -> REVIEW (image quality)
    5. InspectionEngine.inspect(image, profile, reference_bank)     GOOD / DEFECT / REVIEW / ERROR
    6. result.json + evidence in the product's inspections folder; image record updated
Any exception after step 1 -> ERROR, recorded; the stored original stays. No step ever deletes or replaces data.

Product-agnostic: products, their reference banks and storage folders all come from configs/products/.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..comparison.pipeline import load_bgr
from ..config import load_config
from ..image_quality.checks import QualityParams, check_image
from ..inspection.bank import ReferenceBank
from ..inspection.engine import InspectionEngine, load_reference_bank
from ..inspection.identification import (
    AMBIGUOUS,
    AUTO,
    INVALID_FRAME,
    NO_PCB,
    Identification,
    IdentificationSettings,
    ProductIdentifier,
)
from ..inspection.types import (
    AMBIGUOUS_PRODUCT,
    ERROR,
    IMAGE_QUALITY,
    QUALITY_FAIL,
    QUALITY_FLAGGED,
    QUALITY_NOT_CHECKED,
    QUALITY_PASS,
    SOFTWARE_ERROR,
    UNKNOWN_PRODUCT,
    InspectionOutcome,
    Reason,
    review,
)
from ..inspection.types import NO_PCB as REASON_NO_PCB
from ..products.profile import ProductProfile, load_all_profiles, products_dir_of
from ..camera.base import CapturedFrame
from .capture_store import FAILED, INSPECTED, CaptureError, CaptureStore, StoredCapture
from .records import write_result

LIVE_INSPECTION, STORED_IMAGE = "live_inspection", "stored_image"


@dataclass
class StationResult:
    outcome: InspectionOutcome
    capture: StoredCapture
    result_path: Path | None
    identification: Identification | None
    product_name: str | None = None

    @property
    def decision(self) -> str:
        return self.outcome.overall_decision


def quality_status(flags: list[str], blocking: set[str]) -> str:
    if any(f in blocking for f in flags):
        return QUALITY_FAIL
    return QUALITY_FLAGGED if flags else QUALITY_PASS


class InspectionStation:
    def __init__(self, project_root: Path, settings_file: Path | None = None, products_dir: Path | None = None,
                 stage: str | None = None, station_id: str | None = None):
        self.project_root = project_root
        self.settings = load_config(settings_file or project_root / "configs" / "inspection.yaml")
        self.products_dir = products_dir or products_dir_of(project_root)
        self.stage, self.station_id = stage, station_id
        storage = self.settings["storage"]
        self.unresolved_root = project_root / storage["unresolved_product_root"]
        self.captures = CaptureStore(project_root / storage["capture_store"], project_root / "data", project_root)
        self.blocking = set((self.settings.get("quality") or {}).get("blocking_flags") or [])
        camera_cfg = (self.settings.get("station") or {}).get("camera_config")
        self.station_quality = self._quality_params(camera_cfg)
        self.engine = InspectionEngine()
        self.reload()

    # --- products ----------------------------------------------------------------------------
    def reload(self) -> None:
        """(Re)load every product profile and its golden reference bank (e.g. after new GOOD boards were added)."""
        self.profiles, self.problems = load_all_profiles(self.project_root, self.products_dir)
        self.banks: dict[str, ReferenceBank] = {p.product_id: load_reference_bank(p) for p in self.profiles}
        self.identifier = ProductIdentifier([(p, self.banks[p.product_id]) for p in self.profiles],
                                            IdentificationSettings.from_mapping(self.settings["identification"]))

    def warm_up(self) -> None:
        """Run each product's identification once on its own golden reference, so the first real capture does not
        pay the one-time start-up cost of feature extraction (measured ~9 s cold vs ~1 s warm). Stores nothing."""
        from ..inspection.engine import backend_for
        for p in self.profiles:
            bank = self.banks[p.product_id]
            if bank.can_identify and backend_for(p) is not None:
                try:
                    backend_for(p).match(bank.data.reference.image, p, bank)
                except Exception:  # noqa: BLE001 - warm-up is an optimisation only
                    pass

    def profile(self, product_id: str) -> ProductProfile | None:
        return next((p for p in self.profiles if p.product_id == product_id), None)

    def _quality_params(self, capture_config: str | None) -> QualityParams | None:
        if not capture_config:
            return None
        return QualityParams.from_mapping(load_config(self.project_root / capture_config)["quality_check"])

    # --- sessions ----------------------------------------------------------------------------
    def start_session(self, product_selection: str = AUTO, camera: dict[str, Any] | None = None,
                      mode: str = LIVE_INSPECTION) -> str:
        return self.captures.new_session(mode, product_selection, camera, self.stage, self.station_id)

    # --- one capture -------------------------------------------------------------------------
    def process(self, frame: CapturedFrame, session_id: str, product_selection: str = AUTO,
                board_id: str | None = None) -> StationResult:
        """Store the capture FIRST, then decide. Raises CaptureError only if the capture could not be stored."""
        stored = self.captures.save(session_id, frame, board_id=board_id, stage=self.stage)
        return self._decide(stored, frame.image, product_selection, board_id)

    def inspect_file(self, path: Path, product_selection: str = AUTO, board_id: str | None = None,
                     session_id: str | None = None) -> StationResult:
        """Inspect an image file. A file already under data/ is recorded in place (not copied); any other file is
        copied into the capture store first. The same file inspected again gets a new inspection_id.
        session_id: a stored-image session to record new files in (default: a new one for this file)."""
        path = path.resolve()
        stored = self.captures.find(path=path)
        if stored is None:
            data_dir = (self.project_root / "data").resolve()
            own_session = session_id is None
            if own_session:
                session_id = self.start_session(product_selection, {"kind": "stored image files"}, STORED_IMAGE)
            stored = self.captures.register_file(session_id, path, copy=not path.is_relative_to(data_dir),
                                                 board_id=board_id, stage=self.stage)
            if own_session:
                self.captures.end_session(session_id)
        image = load_bgr(self.project_root / stored.record["file_path"])
        return self._decide(stored, image, product_selection, board_id)

    def _decide(self, stored: StoredCapture, image: np.ndarray, selection: str, board_id: str | None) -> StationResult:
        start = time.perf_counter()
        inspection_id = self.captures.next_inspection_id(stored)
        ident: Identification | None = None
        profile: ProductProfile | None = None
        base = dict(inspection_id=inspection_id, image_id=stored.image_id, board_id=board_id)
        try:
            image_path = self.project_root / stored.record["file_path"]
            flags, q_status = [], QUALITY_NOT_CHECKED
            if self.station_quality is not None:
                flags = list(check_image(image_path, self.station_quality, None).flags)
                q_status = quality_status(flags, self.blocking)
            ident = (self.identifier.identify(image) if selection == AUTO
                     else self.identifier.select(selection, image))
            if ident.resolved:
                profile = self.profile(ident.product_id)
            ids = dict(base, product_id=ident.product_id, variant_id=ident.variant_id, quality_status=q_status,
                       identification=ident.to_dict())
            if q_status == QUALITY_FAIL:
                outcome = review(IMAGE_QUALITY, "image quality not sufficient: " + ", ".join(
                    f for f in flags if f in self.blocking), **ids)
            elif profile is None:
                outcome = self._unresolved(ident, ids)
            else:
                product_q = self._quality_params(profile.quality.get("capture_config"))
                if product_q is not None:
                    flags = list(check_image(image_path, product_q, profile.board_mm).flags)
                    ids["quality_status"] = q_status = quality_status(flags, self.blocking)
                if q_status == QUALITY_FAIL:
                    outcome = review(IMAGE_QUALITY, "image quality not sufficient: " + ", ".join(
                        f for f in flags if f in self.blocking), **ids)
                else:
                    evidence_dir = profile.inspections_dir / stored.session_id / inspection_id
                    outcome = self.engine.inspect(image, profile, self.banks[profile.product_id],
                                                  image_id=stored.image_id, inspection_id=inspection_id,
                                                  board_id=board_id, match=ident.match,
                                                  identification=ident.to_dict(), quality_status=q_status,
                                                  evidence_dir=evidence_dir)
            outcome.details.setdefault("quality_flags", flags)
        except Exception as e:  # noqa: BLE001 - the capture is stored; the failure is recorded as ERROR
            outcome = InspectionOutcome(**base, product_id=profile.product_id if profile else None,
                                        variant_id=profile.variant_id if profile else None,
                                        alignment_status="NOT_RUN", alignment_error=None,
                                        quality_status=QUALITY_NOT_CHECKED, overall_decision=ERROR, confidence=None,
                                        reasons=[Reason(SOFTWARE_ERROR, f"{type(e).__name__}: {e}")],
                                        identification=ident.to_dict() if ident else None,
                                        details={"traceback": traceback.format_exc(limit=8)})
        outcome.seconds = round(time.perf_counter() - start, 2)
        product_root = profile.storage_root if profile is not None else self.unresolved_root
        result_path = None
        try:
            result_path = write_result(product_root, stored.session_id, outcome, stored.record, self.captures.rel)
        finally:
            self.captures.update(
                stored, inspection_id=inspection_id, product_id=outcome.product_id, variant_id=outcome.variant_id,
                quality_status=outcome.quality_status, quality_flags=outcome.details.get("quality_flags", []),
                identification=(ident.to_dict() if ident else None),
                inspection_status=FAILED if outcome.overall_decision == ERROR else INSPECTED,
                decision=outcome.overall_decision, decision_confidence=outcome.confidence,
                decision_reasons=[r.code for r in outcome.reasons],
                result_path=self.captures.rel(result_path) if result_path else None)
        return StationResult(outcome, stored, result_path, ident, profile.display_name if profile else None)

    def _unresolved(self, ident: Identification, ids: dict[str, Any]) -> InspectionOutcome:
        if ident.status == NO_PCB:
            return review(REASON_NO_PCB, f"no PCB detected: {ident.reason}", **ids)
        if ident.status == INVALID_FRAME:
            return review(IMAGE_QUALITY, f"frame not usable: {ident.reason}", **ids)
        code = AMBIGUOUS_PRODUCT if ident.status == AMBIGUOUS else UNKNOWN_PRODUCT
        return review(code, f"product not determined ({ident.status}): {ident.reason}", **ids)


def now() -> datetime:
    return datetime.now()
