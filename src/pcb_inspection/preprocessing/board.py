"""Board segmentation, canonical board frame and illumination normalisation (classical, no learning).

The canonical board frame is a fronto-parallel image of the board at a fixed
scale (px per mm), so that comparisons and region sizes are in millimetres and
the background is cropped away. Board corners come from the board's own colour
(the same dominant-hue idea as the image-quality check); they define only the
reference's rectification, so small corner errors affect both compared images
equally and do not disturb the comparison itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..image_quality.checks import dominant_central_hue

WORKING_MAX_SIDE_PX = 1600


@dataclass(frozen=True)
class BoardSegmentationParams:
    saturation_min: int = 60
    value_min: int = 40
    hue_tolerance: int = 12   # Pillow 0-255 hue scale


def downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Area-downscale so that the longer side is at most max_side; returns (image, scale)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def board_mask(bgr: np.ndarray, params: BoardSegmentationParams = BoardSegmentationParams()) -> np.ndarray:
    """Binary mask (uint8 0/255) of the board, same size as the input image; empty if no board found."""
    small, scale = downscale(bgr, WORKING_MAX_SIDE_PX)
    hsv = np.asarray(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("HSV")).astype(np.int16)
    hue = hsv[..., 0]
    saturated = (hsv[..., 1] >= params.saturation_min) & (hsv[..., 2] >= params.value_min)
    board_hue = dominant_central_hue(hue, saturated)
    mask = np.zeros(small.shape[:2], np.uint8)
    if board_hue is not None:
        distance = np.abs(hue - board_hue)
        mask[saturated & (np.minimum(distance, 256 - distance) <= params.hue_tolerance)] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask[:] = 0
        if contours:  # largest region, holes (components) filled
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if scale < 1.0:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def board_quad(mask: np.ndarray) -> np.ndarray | None:
    """Four board corners (float32, shape 4x2) ordered top-left, top-right, bottom-right, bottom-left."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    points = approx.reshape(-1, 2).astype(np.float32) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(np.asarray(points, np.float32))


def order_corners(points: np.ndarray) -> np.ndarray:
    s, d = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.float32([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]])


@dataclass(frozen=True)
class CanonicalFrame:
    """Mapping from an image to the canonical board frame."""

    homography: np.ndarray       # image px -> canonical px
    size: tuple[int, int]        # (width, height) in canonical px
    px_per_mm: float
    source_px_per_mm: float      # approximate scale of the board in the source image


def canonical_frame(quad: np.ndarray, board_mm: tuple[float, float], px_per_mm: float) -> CanonicalFrame:
    """Frame that maps the board quad to an upright rectangle; board_mm = (short side, long side).

    The canonical frame keeps the board's orientation as photographed: a board
    whose long side is vertical in the photo stays portrait, otherwise landscape.
    """
    tl, tr, br, bl = quad
    width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    short_mm, long_mm = sorted(board_mm)
    w_mm, h_mm = (short_mm, long_mm) if height_px >= width_px else (long_mm, short_mm)
    size = (round(w_mm * px_per_mm), round(h_mm * px_per_mm))
    target = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
    homography = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    source_ppm = (width_px / w_mm + height_px / h_mm) / 2
    return CanonicalFrame(homography=homography, size=size, px_per_mm=px_per_mm, source_px_per_mm=float(source_ppm))


def warp_to_canonical(image: np.ndarray, homography: np.ndarray, size: tuple[int, int],
                      source_px_per_mm: float, px_per_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp image into the canonical frame; returns (warped image, validity mask 0/255).

    Large images are first area-downscaled to about 1.5x the target scale to avoid
    aliasing, then warped with bicubic interpolation.
    """
    factor = min(1.0, 1.5 * px_per_mm / max(source_px_per_mm, 1e-6))
    if factor < 1.0:
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        homography = homography @ np.diag([1 / factor, 1 / factor, 1.0])
    warped = cv2.warpPerspective(image, homography, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), homography, size,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """CIE L*a*b* as float32 (L 0-100, a/b roughly -128..127)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def normalise_illumination(test_lab: np.ndarray, reference_lab: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """Make the test image's global brightness / colour statistics match the reference within mask.

    method:
      "none"    no change;
      "l_only"  lightness gain and offset (exposure);
      "global"  lightness gain and offset plus an a*/b* offset (white balance).
    Colour channels are only shifted, never scaled: scaling chroma to match the
    reference could erase a genuine colour difference (e.g. a wrong or missing part).
    """
    if method == "none":
        return test_lab
    if method not in ("global", "l_only"):
        raise ValueError(f"unknown illumination normalisation {method!r}")
    out = test_lab.copy()
    inside = mask > 0
    t, r = test_lab[..., 0][inside], reference_lab[..., 0][inside]
    out[..., 0] = (test_lab[..., 0] - t.mean()) * (r.std() / max(float(t.std()), 1e-6)) + r.mean()
    if method == "global":
        for c in (1, 2):
            out[..., c] = test_lab[..., c] - test_lab[..., c][inside].mean() + reference_lab[..., c][inside].mean()
    return out
