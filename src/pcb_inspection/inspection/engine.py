"""The generic inspection engine: InspectionEngine.inspect(image, product_profile, reference_bank).

Product-agnostic and model-agnostic. The profile names its backend (engine.type); backends are registered in
BACKENDS. Today there is one, REFERENCE_BASELINE (experimental, classical). A trained detector or anomaly model
is added by registering another backend with the same four methods; profiles then select it. Nothing here
knows any product.

Guarantees, whatever the backend says:
  * an image is only ever compared with the reference bank of the product it was identified / selected as;
  * no golden references, too few GOOD boards, alignment or image problems -> REVIEW, never DEFECT;
  * a broken profile / reference or a software error -> ERROR (the capture is stored anyway, by the station);
  * DEFECT always carries defect type, location and evidence; confidence is None unless calibrated.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..products.profile import ProductProfile
from .bank import BROKEN, INSUFFICIENT, NOT_CONFIGURED, ReferenceBank
from .reference_baseline import ReferenceBaselineBackend
from .types import (
    ERROR,
    INSUFFICIENT_GOLDEN_REFERENCES,
    NO_GOLDEN_REFERENCES,
    PROFILE_INCOMPLETE,
    QUALITY_NOT_CHECKED,
    SOFTWARE_ERROR,
    InspectionOutcome,
    ProductMatch,
    Reason,
    review,
)


class InspectionBackend(Protocol):
    name: str

    def load_bank(self, profile: ProductProfile) -> ReferenceBank: ...
    def match(self, image: np.ndarray, profile: ProductProfile, bank: ReferenceBank) -> ProductMatch: ...
    def inspect(self, image: np.ndarray, profile: ProductProfile, bank: ReferenceBank, board_id: str | None = None,
                match: ProductMatch | None = None) -> tuple[dict[str, Any], Any]: ...
    def write_evidence(self, run: Any, out_dir: Path) -> dict[str, str]: ...
    def limitations(self, bank: ReferenceBank) -> list[str]: ...


BACKENDS: dict[str, type] = {ReferenceBaselineBackend.name: ReferenceBaselineBackend}
_instances: dict[str, Any] = {}


def backend_for(profile: ProductProfile) -> InspectionBackend | None:
    kind = profile.engine_type
    if kind not in BACKENDS:
        return None
    if kind not in _instances:
        _instances[kind] = BACKENDS[kind]()
    return _instances[kind]


def load_reference_bank(profile: ProductProfile) -> ReferenceBank:
    """The golden reference bank of this product variant (never raises: problems become the bank status)."""
    backend = backend_for(profile)
    if profile.engine_type is None:
        return ReferenceBank(profile.product_id, profile.variant_id, None, NOT_CONFIGURED,
                             "no inspection engine configured (variant engine.type UNKNOWN)")
    if backend is None:
        return ReferenceBank(profile.product_id, profile.variant_id, profile.engine_type, BROKEN,
                             f"unknown inspection engine {profile.engine_type!r}; registered: {sorted(BACKENDS)}")
    try:
        return backend.load_bank(profile)
    except Exception as e:  # noqa: BLE001 - a broken bank must not stop other products
        return ReferenceBank(profile.product_id, profile.variant_id, profile.engine_type, BROKEN,
                             f"golden reference bank failed to load: {type(e).__name__}: {e}")


class InspectionEngine:
    """Runs the backend named in the product profile and turns its result into an InspectionOutcome."""

    def inspect(self, image: np.ndarray | None, product_profile: ProductProfile, reference_bank: ReferenceBank, *,
                image_id: str = "", inspection_id: str = "", board_id: str | None = None,
                match: ProductMatch | None = None, identification: dict[str, Any] | None = None,
                quality_status: str = QUALITY_NOT_CHECKED, evidence_dir: Path | None = None) -> InspectionOutcome:
        start = time.perf_counter()
        p, bank = product_profile, reference_bank
        ids = dict(inspection_id=inspection_id, product_id=p.product_id, variant_id=p.variant_id, image_id=image_id,
                   board_id=board_id, quality_status=quality_status, identification=identification,
                   engine=p.engine_type, engine_status=p.engine.get("status"), reference_bank=bank.summary())
        try:
            outcome = self._inspect(image, p, bank, ids, board_id, match, evidence_dir)
        except Exception as e:  # noqa: BLE001 - reported as ERROR, never as a board result
            outcome = InspectionOutcome(**ids, alignment_status="NOT_RUN", alignment_error=None,
                                        overall_decision=ERROR, confidence=None,
                                        reasons=[Reason(SOFTWARE_ERROR, f"inspection failed: {type(e).__name__}: {e}")],
                                        details={"traceback": traceback.format_exc(limit=6)})
        outcome.seconds = round(time.perf_counter() - start, 2)
        return outcome

    def _inspect(self, image, p: ProductProfile, bank: ReferenceBank, ids: dict[str, Any], board_id, match,
                 evidence_dir: Path | None) -> InspectionOutcome:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return InspectionOutcome(**ids, alignment_status="NOT_RUN", alignment_error=None, overall_decision=ERROR,
                                     confidence=None, reasons=[Reason(SOFTWARE_ERROR, "no image data")])
        if (bank.product_id, bank.variant_id) != (p.product_id, p.variant_id):
            return InspectionOutcome(**ids, alignment_status="NOT_RUN", alignment_error=None, overall_decision=ERROR,
                                     confidence=None, reasons=[Reason(SOFTWARE_ERROR, (
                                         f"reference bank of {bank.product_id}/{bank.variant_id} offered for "
                                         f"{p.product_id}/{p.variant_id}: refused (never compared)"))])
        if match is not None and (match.product_id, match.variant_id) != (p.product_id, p.variant_id):
            match = None                       # evidence against another product's reference is never reused
        single_reference = False
        if bank.status == BROKEN:
            return InspectionOutcome(**ids, alignment_status="NOT_RUN", alignment_error=None, overall_decision=ERROR,
                                     confidence=None, reasons=[Reason(SOFTWARE_ERROR, bank.reason)])
        if bank.status == NOT_CONFIGURED:
            return review(NO_GOLDEN_REFERENCES, f"{p.display_name}: no golden references ({bank.reason})", **ids)
        if bank.status == INSUFFICIENT:
            if len(bank.boards) == 1 and bank.data is not None:
                single_reference = True
            else:
                return review(INSUFFICIENT_GOLDEN_REFERENCES, f"{p.display_name}: {bank.reason}", **ids)
        elif len(bank.boards) == 1:
            single_reference = True
        if not p.enabled_defect_types:
            return review(PROFILE_INCOMPLETE, f"{p.display_name}: no defect type is enabled in the inspection rules",
                          **ids)
        backend = backend_for(p)
        fields, run = backend.inspect(image, p, bank, board_id=board_id, match=match)
        details = {**fields.pop("details", {}), **fields.pop("details_extra", {})}
        if single_reference:
            details["reference_mode"] = "EXPERIMENTAL / SINGLE_REFERENCE"
            if ids.get("reference_bank"):
                ids["reference_bank"]["reference_mode"] = "EXPERIMENTAL / SINGLE_REFERENCE"
        decision = fields.pop("decision")
        limitations = backend.limitations(bank) + fields.pop("limitations", [])
        if single_reference:
            limitations.insert(0, "EXPERIMENTAL / SINGLE_REFERENCE: single GOOD board reference used; "
                                  "not production-valid (inter-board variance not captured)")
        outcome = InspectionOutcome(**ids, overall_decision=decision, confidence=None, details=details,
                                    limitations=limitations, **fields)
        if evidence_dir is not None:
            outcome.evidence_files = backend.write_evidence(run, evidence_dir)
        return outcome
