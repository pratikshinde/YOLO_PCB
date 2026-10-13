"""Difference analysis between a reference board image and an aligned test image.

Both images are in the canonical board frame (same size, px_per_mm known).
Several complementary representations are computed:

* colour difference: CIE76 delta E on L*a*b* (after optional illumination normalisation);
* structural dissimilarity: (1 - SSIM) / 2 on the L channel (Gaussian window);
* edge mismatch: edges present in one image without an edge nearby in the other;
* candidate regions: connected areas where delta E or dissimilarity exceed thresholds.

Candidate regions are NOT defects. On two good boards every candidate region is
by definition a false candidate; measuring them is the point of Experiment 1.
Verified defect boxes, if any, come only from human ground truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..preprocessing.board import normalise_illumination, to_lab


@dataclass(frozen=True)
class ComparisonParams:
    illumination: str = "global"       # none | global | l_only (see preprocessing.board.normalise_illumination)
    blur_sigma_mm: float = 0.05        # light noise suppression before comparing
    border_margin_mm: float = 1.5      # ignore a strip along the board edge
    ssim_sigma_px: float = 1.5
    edge_tolerance_mm: float = 0.25
    delta_e_threshold: float = 20.0    # experiment parameter, not a validated limit
    dissimilarity_threshold: float = 0.25
    min_region_mm2: float = 0.5
    open_kernel_mm: float = 0.2


@dataclass
class CandidateRegion:
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    area_mm2: float
    mean_delta_e: float
    max_dissimilarity: float

    def overlaps(self, box_mm: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = box_mm
        return self.x_mm < x1 and x0 < self.x_mm + self.width_mm and self.y_mm < y1 and y0 < self.y_mm + self.height_mm


@dataclass
class ComparisonResult:
    delta_e: np.ndarray = field(repr=False)
    dissimilarity: np.ndarray = field(repr=False)
    edge_mismatch: np.ndarray = field(repr=False)
    candidates_mask: np.ndarray = field(repr=False)
    compared: np.ndarray = field(repr=False)        # mask of pixels that were compared
    regions: list[CandidateRegion]
    metrics: dict[str, float]
    seconds: float


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float, data_range: float) -> np.ndarray:
    """Per-pixel SSIM (Wang et al. 2004) with a Gaussian window."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def _edges(l_channel: np.ndarray) -> np.ndarray:
    u8 = np.clip(l_channel * 2.55, 0, 255).astype(np.uint8)
    median = float(np.median(u8))
    return cv2.Canny(u8, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))


def compare(reference_bgr: np.ndarray, aligned_bgr: np.ndarray, valid: np.ndarray, px_per_mm: float,
            params: ComparisonParams = ComparisonParams()) -> ComparisonResult:
    start = time.perf_counter()
    margin = max(1, round(params.border_margin_mm * px_per_mm))
    compared = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1)))
    compared[:margin, :] = compared[-margin:, :] = 0
    compared[:, :margin] = compared[:, -margin:] = 0
    inside = compared > 0

    ref_lab, test_lab = to_lab(reference_bgr), to_lab(aligned_bgr)
    sigma = params.blur_sigma_mm * px_per_mm
    if sigma > 0.3:
        ref_lab = cv2.GaussianBlur(ref_lab, (0, 0), sigma)
        test_lab = cv2.GaussianBlur(test_lab, (0, 0), sigma)
    test_lab = normalise_illumination(test_lab, ref_lab, compared, params.illumination)

    delta_e = np.sqrt(np.sum((ref_lab - test_lab) ** 2, axis=2))
    dissimilarity = np.clip((1.0 - ssim_map(ref_lab[..., 0], test_lab[..., 0], params.ssim_sigma_px, 100.0)) / 2, 0, 1)
    delta_e[~inside] = 0
    dissimilarity[~inside] = 0

    tol = max(1, round(params.edge_tolerance_mm * px_per_mm))
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ref_edges, test_edges = _edges(ref_lab[..., 0]), _edges(test_lab[..., 0])
    edge_mismatch = ((test_edges > 0) & (cv2.dilate(ref_edges, near) == 0)) | ((ref_edges > 0) & (cv2.dilate(test_edges, near) == 0))
    edge_mismatch &= inside

    raw = ((delta_e > params.delta_e_threshold) | (dissimilarity > params.dissimilarity_threshold)) & inside
    k = max(1, round(params.open_kernel_mm * px_per_mm))
    candidates = cv2.morphologyEx(raw.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    regions: list[CandidateRegion] = []
    keep = np.zeros_like(candidates)
    px2 = px_per_mm ** 2
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area / px2 < params.min_region_mm2:
            continue
        member = labels == i
        keep[member] = 255
        regions.append(CandidateRegion(
            x_mm=x / px_per_mm, y_mm=y / px_per_mm, width_mm=w / px_per_mm, height_mm=h / px_per_mm,
            area_mm2=area / px2, mean_delta_e=float(delta_e[member].mean()),
            max_dissimilarity=float(dissimilarity[member].max()),
        ))
    regions.sort(key=lambda r: r.area_mm2, reverse=True)

    compared_area_mm2 = inside.sum() / px2
    de, ds = delta_e[inside], dissimilarity[inside]
    ssim_inside = 1.0 - 2.0 * ds
    metrics = {
        "compared_area_mm2": float(compared_area_mm2),
        "ssim_mean": float(ssim_inside.mean()),
        "delta_e_mean": float(de.mean()),
        "delta_e_p50": float(np.percentile(de, 50)),
        "delta_e_p95": float(np.percentile(de, 95)),
        "delta_e_p99": float(np.percentile(de, 99)),
        "dissimilarity_p99": float(np.percentile(ds, 99)),
        "fraction_delta_e_over_10": float((de > 10).mean()),
        "fraction_delta_e_over_20": float((de > 20).mean()),
        "fraction_delta_e_over_30": float((de > 30).mean()),
        "edge_mismatch_fraction": float(edge_mismatch.sum() / max(1, ((ref_edges > 0) & inside).sum())),
        "candidate_regions": len(regions),
        "candidate_area_mm2": float(sum(r.area_mm2 for r in regions)),
        "candidate_area_fraction": float(sum(r.area_mm2 for r in regions) / max(compared_area_mm2, 1e-9)),
        "largest_candidate_mm2": float(regions[0].area_mm2) if regions else 0.0,
    }
    return ComparisonResult(
        delta_e=delta_e, dissimilarity=dissimilarity, edge_mismatch=edge_mismatch, candidates_mask=keep,
        compared=compared, regions=regions, metrics=metrics, seconds=time.perf_counter() - start,
    )


def verified_defect_overlap(regions: list[CandidateRegion], boxes_mm: list[tuple[float, float, float, float]]) -> list[dict]:
    """For each human-verified defect box (canonical mm), list the candidate regions that overlap it.

    Used only to compare against ground truth supplied by the operator / production;
    nothing is labelled automatically.
    """
    return [{"box_mm": list(box), "overlapping_candidates": [i for i, r in enumerate(regions) if r.overlaps(box)]}
            for box in boxes_mm]
