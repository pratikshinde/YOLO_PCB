"""Golden reference bank: the GOOD boards of ONE product variant that inspections are compared with.

Not a single golden image. The bank holds every physical GOOD board that the product's board registry puts in
its reference split, with all of their kept images. Status:

    READY           enough physical GOOD boards (golden_bank.yaml min_boards): inspections can decide
    INSUFFICIENT    golden reference present, too few GOOD boards: identification possible, every inspection REVIEW
    NOT_CONFIGURED  no golden reference / regions / board size: neither identification nor inspection
    BROKEN          configured but could not be loaded (missing file, region map drawn on another image, ...): ERROR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

READY, INSUFFICIENT, NOT_CONFIGURED, BROKEN = "READY", "INSUFFICIENT", "NOT_CONFIGURED", "BROKEN"


@dataclass
class ReferenceBank:
    product_id: str
    variant_id: str
    engine: str | None
    status: str
    reason: str
    boards: list[str] = field(default_factory=list)          # physical GOOD boards in the bank
    images: int = 0
    min_boards: int | None = None
    geometric_reference: str | None = None
    data_roots: list[str] = field(default_factory=list)
    regions: int = 0
    unverified_regions: int = 0
    unverified_boards: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)          # GOOD images that could not be aligned into the bank
    data: Any = field(default=None, repr=False)               # backend-specific loaded references

    @property
    def can_inspect(self) -> bool:
        return (self.status == READY or (self.status == INSUFFICIENT and len(self.boards) == 1)) and self.data is not None

    @property
    def can_identify(self) -> bool:
        return self.status in (READY, INSUFFICIENT) and self.data is not None

    def summary(self) -> dict[str, Any]:
        return {"product_id": self.product_id, "variant_id": self.variant_id, "engine": self.engine,
                "status": self.status, "reason": self.reason, "boards": self.boards, "images": self.images,
                "min_boards": self.min_boards, "geometric_reference": self.geometric_reference,
                "regions": self.regions, "unverified_regions": self.unverified_regions,
                "unverified_boards": self.unverified_boards, "skipped": self.skipped}
