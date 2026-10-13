"""Feature-based registration of a test board image to a reference board image.

Keypoints are detected only inside the board masks, so the board is aligned and
not the background. A single homography is fitted with RANSAC; it models the
board as a plane (camera position, rotation, scale, perspective). Tall
components are not on that plane, so they keep some parallax after alignment;
the tile residuals below make that visible instead of hiding it.

No fiducials are assumed (the board's holes are not confirmed as fiducials, PCB-17).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..preprocessing.board import downscale


@dataclass(frozen=True)
class RegistrationParams:
    detector: str = "sift"              # sift | akaze | orb
    working_max_side_px: int = 2400     # features are computed on a downscaled copy
    max_features: int = 8000
    ratio_test: float = 0.75
    ransac_threshold_px: float = 3.0    # at working resolution
    mask_erosion_fraction: float = 0.01  # shrink the board mask so background next to the edge is ignored
    min_inliers: int = 30


@dataclass
class RegistrationResult:
    success: bool
    homography: np.ndarray | None        # test full-res px -> reference full-res px
    keypoints_reference: int = 0
    keypoints_test: int = 0
    matches: int = 0
    inliers: int = 0
    inlier_rmse_px: float | None = None  # in reference full-res px
    seconds: float = 0.0
    message: str = ""
    inlier_points: np.ndarray | None = field(default=None, repr=False)  # reference full-res px, for plots

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


def _detector(params: RegistrationParams):
    if params.detector == "sift":
        return cv2.SIFT_create(nfeatures=params.max_features), cv2.NORM_L2
    if params.detector == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if params.detector == "orb":
        return cv2.ORB_create(nfeatures=params.max_features), cv2.NORM_HAMMING
    raise ValueError(f"unknown detector {params.detector!r}")


def _features(gray: np.ndarray, mask: np.ndarray, params: RegistrationParams):
    small, scale = downscale(gray, params.working_max_side_px)
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    erode = max(1, round(params.mask_erosion_fraction * max(small.shape)))
    small_mask = cv2.erode(small_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    detector, norm = _detector(params)
    keypoints, descriptors = detector.detectAndCompute(small, small_mask)
    points = np.float32([k.pt for k in keypoints]) / scale if keypoints else np.zeros((0, 2), np.float32)
    return points, descriptors, norm, scale


def register(reference_bgr: np.ndarray, reference_mask: np.ndarray, test_bgr: np.ndarray, test_mask: np.ndarray,
             params: RegistrationParams = RegistrationParams()) -> RegistrationResult:
    """Estimate the homography mapping the test image onto the reference image (full resolution)."""
    start = time.perf_counter()
    ref_pts, ref_desc, norm, ref_scale = _features(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY), reference_mask, params)
    test_pts, test_desc, _, _ = _features(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY), test_mask, params)
    result = RegistrationResult(False, None, keypoints_reference=len(ref_pts), keypoints_test=len(test_pts))
    if ref_desc is None or test_desc is None or len(ref_pts) < 4 or len(test_pts) < 4:
        result.message, result.seconds = "too few keypoints on the board", time.perf_counter() - start
        return result

    pairs = cv2.BFMatcher(norm).knnMatch(test_desc, ref_desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < params.ratio_test * p[1].distance]
    result.matches = len(good)
    if len(good) < 4:
        result.message, result.seconds = "too few feature matches", time.perf_counter() - start
        return result

    src = test_pts[[m.queryIdx for m in good]]
    dst = ref_pts[[m.trainIdx for m in good]]
    # RANSAC threshold given at working resolution; points are in full-res px
    homography, inlier_mask = cv2.findHomography(
        src, dst, cv2.RANSAC, params.ransac_threshold_px / ref_scale, maxIters=10000, confidence=0.999
    )
    if homography is None:
        result.message, result.seconds = "homography estimation failed", time.perf_counter() - start
        return result
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src[inliers].reshape(-1, 1, 2), homography).reshape(-1, 2)
    result.homography = homography
    result.inliers = int(inliers.sum())
    result.inlier_rmse_px = float(np.sqrt(np.mean(np.sum((projected - dst[inliers]) ** 2, axis=1))))
    result.inlier_points = dst[inliers]
    result.success = result.inliers >= params.min_inliers
    result.message = "ok" if result.success else f"only {result.inliers} inliers (< {params.min_inliers})"
    result.seconds = time.perf_counter() - start
    return result


@dataclass
class TileResiduals:
    """Local misalignment left after registration, measured by phase correlation per tile."""

    tile_px: int
    centres: np.ndarray        # (N, 2) tile centres in canonical px
    shifts_px: np.ndarray      # (N, 2) measured (dx, dy) shift of the test relative to the reference
    reliable: np.ndarray       # (N,) bool: enough texture and a clear correlation peak

    def magnitudes_px(self) -> np.ndarray:
        return np.hypot(self.shifts_px[:, 0], self.shifts_px[:, 1])


def tile_residuals(reference_gray: np.ndarray, aligned_gray: np.ndarray, valid: np.ndarray, tile_px: int,
                   min_std: float = 4.0, min_response: float = 0.15) -> TileResiduals:
    """Measure residual shifts on a grid of tiles fully inside the valid region.

    This is independent of the feature matches used to fit the homography, so it
    is an honest estimate of the remaining misalignment (including parallax of
    tall parts and any non-planar effects). Windowed phase correlation slightly
    underestimates shifts (about 5-10 %); the values are for judging alignment,
    not for metrology.
    """
    window = cv2.createHanningWindow((tile_px, tile_px), cv2.CV_32F)
    centres, shifts, reliable = [], [], []
    h, w = reference_gray.shape
    for y in range(0, h - tile_px + 1, tile_px):
        for x in range(0, w - tile_px + 1, tile_px):
            if valid[y:y + tile_px, x:x + tile_px].min() == 0:
                continue
            a = reference_gray[y:y + tile_px, x:x + tile_px].astype(np.float32)
            b = aligned_gray[y:y + tile_px, x:x + tile_px].astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(a, b, window)
            centres.append((x + tile_px / 2, y + tile_px / 2))
            shifts.append((dx, dy))
            reliable.append(bool(a.std() >= min_std and b.std() >= min_std and response >= min_response))
    return TileResiduals(
        tile_px=tile_px,
        centres=np.float32(centres).reshape(-1, 2),
        shifts_px=np.float32(shifts).reshape(-1, 2),
        reliable=np.array(reliable, bool),
    )
