"""Multi-board statistical GOOD baseline representing normal inter-board variation.

Key Principles:
1. Physical PCB Independence:
   GOOD_001 is NOT the only good board. We can have GOOD_001, GOOD_002, GOOD_003, ...
   of the same product (dimensions come from the product profile, never assumed).
2. Normal Board-to-Board Variation:
   Different physical boards exhibit natural tolerances:
   - Solder meniscus reflectance & wetting angle
   - Silkscreen printing slight offsets and thickness differences
   - Passive component body shade variations
   - Laminate / soldermask natural color variation
   - Lighting & perspective variations
3. Tolerance Map:
   The baseline computes a per-pixel acceptable normal variation map T(x, y).
   Only deviations that significantly exceed normal inter-board variation become defect candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..dataset.discovery import DiscoveredBoard, GoodDatasetReport, discover_good_dataset
from ..preprocessing.board import normalise_illumination, to_lab


@dataclass
class GoodBaseline:
    """Statistical model of verified GOOD physical PCBs in canonical coordinates."""
    board_dimensions_mm: tuple[float, float]     # from the product profile (either order)
    px_per_mm: float
    canonical_size: tuple[int, int]              # (width_px, height_px)
    board_ids: list[str]                         # physical boards included in this baseline
    median_canonical_bgr: np.ndarray             # median appearance across physical boards (uint8)
    median_canonical_lab: np.ndarray             # median in CIE L*a*b* (float32)
    tolerance_map: np.ndarray                    # acceptable per-pixel variation map (float32, Delta E scale)
    board_mask_canonical: np.ndarray             # binary mask of board interior (uint8 0/255)
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    samples_per_board: dict[str, int] = field(default_factory=dict)
    board_colour_hues: list[int] = field(default_factory=list)
    board_medians_bgr: dict[str, np.ndarray] = field(default_factory=dict, repr=False)   # per physical board
    min_delta_e_tolerance: float = 18.0
    k_sigma: float = 2.5

    def without_board(self, exclude_board_id: str | None) -> GoodBaseline:
        """Leave-one-board-out: the baseline rebuilt from the OTHER physical boards only.

        Raises ValueError if the board is the only one (an image of it cannot be judged against itself)."""
        if not exclude_board_id or exclude_board_id not in self.board_ids:
            return self
        remaining = [(b, img) for b, img in sorted(self.board_medians_bgr.items()) if b != exclude_board_id]
        if not remaining:
            raise ValueError(f"{exclude_board_id} is the only GOOD board of the baseline: no other board to compare with")
        return build_good_baseline(remaining, board_dimensions_mm=self.board_dimensions_mm, px_per_mm=self.px_per_mm,
                                   min_delta_e_tolerance=self.min_delta_e_tolerance, k_sigma=self.k_sigma)

    def compute_anomaly(self, test_canonical_bgr: np.ndarray, illumination_mode: str = "global"
                        ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute pixel-wise normalized deviation from the multi-board baseline.

        Returns:
            normalized_deviation: (H, W) float32 ratio = Delta_E / Tolerance
            difference_mask: (H, W) uint8 binary mask of pixels exceeding normal variation
            overall_anomaly_score: float summary score
        """
        inside = self.board_mask_canonical > 0
        test_lab = to_lab(test_canonical_bgr)
        # Global illumination normalization against baseline median
        normalized_test_lab = normalise_illumination(test_lab, self.median_canonical_lab,
                                                     self.board_mask_canonical, illumination_mode)

        # CIE76 Delta E against multi-board median
        delta_e = np.sqrt(np.sum((self.median_canonical_lab - normalized_test_lab) ** 2, axis=2))
        delta_e[~inside] = 0.0

        # Normalized by per-pixel tolerance map: ratio = Delta_E / Tolerance
        safe_tolerance = np.maximum(self.tolerance_map, 1.0)
        normalized_deviation = np.zeros_like(delta_e, dtype=np.float32)
        normalized_deviation[inside] = delta_e[inside] / safe_tolerance[inside]

        # Significant deviation when Delta_E exceeds tolerance (ratio > 1.0)
        difference_mask = ((normalized_deviation > 1.0) & inside).astype(np.uint8) * 255

        # Anomaly score: 99th percentile of normalized deviation inside board
        score = float(np.percentile(normalized_deviation[inside], 99)) if inside.any() else 0.0
        return normalized_deviation, difference_mask, score


def build_good_baseline(
    canonical_images: Sequence[tuple[str, np.ndarray]],  # list of (board_id, canonical_bgr)
    board_dimensions_mm: tuple[float, float] | None = None,   # from the product profile; required
    px_per_mm: float = 20.0,
    min_delta_e_tolerance: float = 18.0,
    k_sigma: float = 2.5,
) -> GoodBaseline:
    """Builds a multi-board GoodBaseline from rectified canonical images."""
    if not canonical_images:
        raise ValueError("At least one canonical image is required to build a GoodBaseline")
    if board_dimensions_mm is None or None in board_dimensions_mm:
        raise ValueError("board_dimensions_mm is required (product profile board size); it is never assumed")

    short_mm, long_mm = sorted(board_dimensions_mm)
    w_mm, h_mm = long_mm, short_mm
    canonical_w = round(w_mm * px_per_mm)
    canonical_h = round(h_mm * px_per_mm)
    canonical_size = (canonical_w, canonical_h)

    # Group images by board ID
    by_board: dict[str, list[np.ndarray]] = {}
    board_ids = []
    resized_images = []

    for b_id, img in canonical_images:
        if b_id not in by_board:
            by_board[b_id] = []
            board_ids.append(b_id)
        if img.shape[1] != canonical_w or img.shape[0] != canonical_h:
            img = cv2.resize(img, (canonical_w, canonical_h), interpolation=cv2.INTER_AREA)
        by_board[b_id].append(img)
        resized_images.append(img)

    stack_bgr = np.stack(resized_images, axis=0)  # (N, H, W, 3)
    median_bgr = np.median(stack_bgr, axis=0).astype(np.uint8)
    median_lab = to_lab(median_bgr)

    # Compute inter-board variation (standard deviation across physical boards)
    # First compute per-board mean/median to represent each physical board equally
    board_representations = []
    board_medians: dict[str, np.ndarray] = {}
    for b_id, imgs in by_board.items():
        board_rep = np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)
        board_medians[b_id] = board_rep
        board_representations.append(to_lab(board_rep))

    board_stack_lab = np.stack(board_representations, axis=0)  # (B, H, W, 3)

    if len(board_representations) > 1:
        # Standard deviation in Delta E across physical boards
        delta_es = np.sqrt(np.sum((board_stack_lab - median_lab[None, ...]) ** 2, axis=3))  # (B, H, W)
        std_delta_e = np.std(delta_es, axis=0)  # (H, W)
    else:
        std_delta_e = np.zeros((canonical_h, canonical_w), dtype=np.float32)

    # Per-pixel tolerance: T(x, y) = max(T_min, k * std(x, y))
    tolerance_map = np.maximum(min_delta_e_tolerance, k_sigma * std_delta_e).astype(np.float32)

    # Board interior mask (excluding border margin)
    margin_px = max(1, round(1.5 * px_per_mm))
    mask = np.zeros((canonical_h, canonical_w), dtype=np.uint8)
    mask[margin_px:-margin_px, margin_px:-margin_px] = 255

    samples_per_board = {b_id: len(imgs) for b_id, imgs in by_board.items()}

    return GoodBaseline(
        board_dimensions_mm=board_dimensions_mm,
        px_per_mm=px_per_mm,
        canonical_size=canonical_size,
        board_ids=sorted(board_ids),
        median_canonical_bgr=median_bgr,
        median_canonical_lab=median_lab,
        tolerance_map=tolerance_map,
        board_mask_canonical=mask,
        sample_count=len(resized_images),
        samples_per_board=samples_per_board,
        board_medians_bgr=board_medians,
        min_delta_e_tolerance=min_delta_e_tolerance,
        k_sigma=k_sigma,
    )
