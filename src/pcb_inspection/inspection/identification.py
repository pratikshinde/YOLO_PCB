"""Product identification: which product (and variant) is under the camera?

AUTO: the image is registered against the geometric golden reference of every product whose profile allows
automatic identification and whose golden reference exists. Evidence per product = that registration (alignment
check, inliers, inlier ratio). A product is IDENTIFIED only if

    it registers (alignment check OK, or only the placement is out of limits), inliers >= min_inliers,
    inlier ratio >= min_inlier_ratio, and
    no other product passes within min_ratio_margin of it.

Otherwise the result is UNKNOWN or AMBIGUOUS and the station decides REVIEW: the image is never compared with
the references of a product it was not identified as. A product without golden references cannot be identified
automatically; it can only be selected by the operator (SELECT PRODUCT), and its inspection is then REVIEW.

The score reported is the measured inlier ratio, not a probability; confidence stays None (not calibrated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..products.profile import ProductProfile
from .bank import ReferenceBank
from .engine import backend_for
from .types import ProductMatch

IDENTIFIED, SELECTED, UNKNOWN, AMBIGUOUS = "IDENTIFIED", "SELECTED", "UNKNOWN", "AMBIGUOUS"
NO_PCB, INVALID_FRAME = "NO_PCB", "INVALID_FRAME"
AUTO = "AUTO"
RECOGNISED = ("OK", "POSITION_INVALID")      # alignment-check states that count as evidence for the product


@dataclass(frozen=True)
class IdentificationSettings:
    min_inlier_ratio: float
    min_inliers: int
    min_ratio_margin: float

    @classmethod
    def from_mapping(cls, m: dict[str, Any]) -> IdentificationSettings:
        return cls(float(m["min_inlier_ratio"]), int(m["min_inliers"]), float(m["min_ratio_margin"]))


@dataclass
class Identification:
    status: str                                   # IDENTIFIED | SELECTED | UNKNOWN | AMBIGUOUS | NO_PCB | INVALID_FRAME
    product_id: str | None
    variant_id: str | None
    method: str                                   # auto_reference_registration | operator_selected
    reason: str
    score: float | None = None                    # inlier ratio of the chosen product (a measurement, not a probability)
    confidence: float | None = None               # not calibrated -> None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    not_identifiable: list[dict[str, str]] = field(default_factory=list)
    match: ProductMatch | None = field(default=None, repr=False)

    @property
    def resolved(self) -> bool:
        return self.status in (IDENTIFIED, SELECTED) and self.product_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "product_id": self.product_id, "variant_id": self.variant_id,
                "method": self.method, "reason": self.reason, "score": self.score,
                "score_meaning": "SIFT inlier ratio against the product's own golden reference",
                "confidence": self.confidence, "candidates": self.candidates,
                "not_identifiable": self.not_identifiable}


class ProductIdentifier:
    def __init__(self, candidates: list[tuple[ProductProfile, ReferenceBank]], settings: IdentificationSettings):
        self.candidates, self.settings = candidates, settings

    def _match(self, image: np.ndarray, profile: ProductProfile, bank: ReferenceBank) -> ProductMatch:
        try:
            return backend_for(profile).match(image, profile, bank)
        except Exception as e:  # noqa: BLE001 - one product's failure is evidence, not a crash
            return ProductMatch(profile.product_id, profile.variant_id, "ERROR", reason=f"{type(e).__name__}: {e}")

    def identify(self, image: np.ndarray) -> Identification:
        s = self.settings
        usable, skipped = [], []
        for profile, bank in self.candidates:
            if not profile.auto_identification:
                skipped.append({"product_id": profile.product_id, "reason": "automatic identification disabled"})
            elif not bank.can_identify or backend_for(profile) is None:
                skipped.append({"product_id": profile.product_id, "reason": f"no golden reference ({bank.status}: {bank.reason})"})
            else:
                usable.append((profile, bank))
        base = dict(method="auto_reference_registration", not_identifiable=skipped)
        if not usable:
            return Identification(UNKNOWN, None, None, reason="no product has golden references for automatic "
                                  "identification; select the product", **base)
        matches = [self._match(image, p, b) for p, b in usable]
        candidates = [m.to_dict() for m in matches]
        if all(m.status == INVALID_FRAME for m in matches):
            return Identification(INVALID_FRAME, None, None, reason=matches[0].reason, candidates=candidates, **base)
        if all(m.status in (NO_PCB, INVALID_FRAME) for m in matches):
            return Identification(NO_PCB, None, None, reason=next(m.reason for m in matches if m.status == NO_PCB),
                                  candidates=candidates, **base)
        # POSITION_INVALID = registered well but placed outside the allowed position / rotation: it IS this product;
        # the inspection's alignment gate then reports the placement (REVIEW), not "unknown product".
        passing = sorted((m for m in matches if m.status in RECOGNISED and m.inliers >= s.min_inliers
                          and m.inlier_ratio >= s.min_inlier_ratio), key=lambda m: -m.inlier_ratio)
        if not passing:
            best = max(matches, key=lambda m: m.inlier_ratio)
            return Identification(UNKNOWN, None, None, reason=(
                f"matches no known product (best: {best.product_id}, {best.status}, {best.inliers} inliers, ratio "
                f"{best.inlier_ratio:.2f}; needed OK, >= {s.min_inliers} inliers, ratio >= {s.min_inlier_ratio:.2f})"),
                candidates=candidates, **base)
        best = passing[0]
        if len(passing) > 1 and best.inlier_ratio - passing[1].inlier_ratio < s.min_ratio_margin:
            return Identification(AMBIGUOUS, None, None, reason=(
                f"{best.product_id} ({best.inlier_ratio:.2f}) and {passing[1].product_id} "
                f"({passing[1].inlier_ratio:.2f}) match about equally well"), candidates=candidates, **base)
        return Identification(IDENTIFIED, best.product_id, best.variant_id, reason=(
            f"registered to the {best.product_id} golden reference: {best.inliers} inliers, ratio {best.inlier_ratio:.2f}"),
            score=best.inlier_ratio, candidates=candidates, match=best, **base)

    def select(self, product_id: str, image: np.ndarray | None = None) -> Identification:
        """Operator selection. The product's registration is computed (if it has a reference) for reuse and shown
        as evidence; a mismatch is decided by the inspection's alignment gate (REVIEW), not here."""
        for profile, bank in self.candidates:
            if profile.product_id == product_id:
                match = None
                if image is not None and bank.can_identify and backend_for(profile) is not None:
                    match = self._match(image, profile, bank)
                return Identification(SELECTED, profile.product_id, profile.variant_id, "operator_selected",
                                      reason=f"selected by the operator: {profile.display_name}",
                                      score=match.inlier_ratio if match else None,
                                      candidates=[match.to_dict()] if match else [], match=match)
        known = ", ".join(p.product_id for p, _ in self.candidates)
        return Identification(UNKNOWN, None, None, "operator_selected",
                              reason=f"selected product {product_id!r} has no profile (known: {known})")
