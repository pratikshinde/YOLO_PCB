"""Burned-in camera-app watermark (DroidCam free edition: "using droidcam.app"): find it, ignore it.

Measured on 2026-10-05 (2560 x 1440 frames):
  * the watermark is the same text at the same size (glyph block 598 x 36 px) in every frame;
  * its position is fixed WITHIN one DroidCam streaming session (31 curated frames 12:49-12:56 at
    x 376-975, y 343-381; 436 live frames 13:59-14:21 at x 784-1382, y 58-94) but it MOVES between sessions.
So a fixed mask is wrong; the watermark is located in every frame by template matching on its letter
shapes, and only the letter pixels (dilated by a few px) are ignored / inpainted. A location is accepted only
when the match is clearly better than anywhere else in the frame; otherwise nothing is masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WatermarkParams:
    enabled: bool = True
    template: Path | None = None          # 0/255 glyph mask of the watermark text, measured at template_frame_width
    template_frame_width: int = 2560      # frame width the template was measured at; scaled to other widths
    search_region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)   # normalised x0, y0, x1, y1
    min_score: float = 0.45               # normalised correlation of the glyphs with the frame's bright details
    min_margin: float = 0.10              # best match must beat the best match elsewhere by this much
    dilate_px: int = 2                    # cover the soft edges of the letters


@dataclass
class WatermarkMatch:
    found: bool
    box_px: tuple[int, int, int, int] | None   # x0, y0, x1, y1 of the glyph block in the frame
    score: float
    margin: float
    mask: np.ndarray | None = None              # frame-sized bool mask of the ignored (letter) pixels

    @property
    def normalised_box(self) -> tuple[float, ...] | None:
        return None if self.box_px is None or self.mask is None else tuple(
            round(v / s, 4) for v, s in zip(self.box_px, (self.mask.shape[1], self.mask.shape[0]) * 2))


def load_template(params: WatermarkParams) -> np.ndarray | None:
    if not params.enabled or params.template is None:
        return None
    template = cv2.imread(str(params.template), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"watermark template not found: {params.template}")
    return template


def _bright_details(gray: np.ndarray) -> np.ndarray:
    """White top-hat: thin bright structures (letters) stand out, smooth bright areas do not."""
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).astype(np.float32)


def locate_watermark(bgr: np.ndarray, template: np.ndarray | None, params: WatermarkParams) -> WatermarkMatch:
    """Find the watermark text in a frame. Not found -> nothing is masked."""
    h, w = bgr.shape[:2]
    if template is None or not params.enabled:
        return WatermarkMatch(False, None, 0.0, 0.0)
    scale = w / params.template_frame_width
    glyphs = template if abs(scale - 1) < 1e-3 else cv2.resize(
        template, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    nx0, ny0, nx1, ny1 = params.search_region
    sx0, sy0, sx1, sy1 = round(nx0 * w), round(ny0 * h), round(nx1 * w), round(ny1 * h)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    region = _bright_details(gray[sy0:sy1, sx0:sx1])
    th, tw = glyphs.shape
    if region.shape[0] < th or region.shape[1] < tw:
        return WatermarkMatch(False, None, 0.0, 0.0)
    scores = cv2.matchTemplate(region, (glyphs > 0).astype(np.float32), cv2.TM_CCORR_NORMED)
    _, best, _, (bx, by) = cv2.minMaxLoc(scores)
    others = scores.copy()
    others[max(0, by - th):by + th, max(0, bx - tw // 2):bx + tw // 2] = 0     # exclude the peak itself
    margin = float(best - others.max()) if others.size else float(best)
    box = (sx0 + bx, sy0 + by, sx0 + bx + tw, sy0 + by + th)
    if best < params.min_score or margin < params.min_margin:
        return WatermarkMatch(False, box, float(best), margin)
    mask = np.zeros((h, w), np.uint8)
    mask[box[1]:box[3], box[0]:box[2]] = glyphs
    if params.dilate_px > 0:
        k = 2 * params.dilate_px + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return WatermarkMatch(True, box, float(best), margin, mask > 0)


def remove_watermark(bgr: np.ndarray, match: WatermarkMatch) -> np.ndarray:
    """Inpaint the letter pixels from their immediate surroundings (a few px); the rest of the frame is untouched."""
    if not match.found or match.mask is None:
        return bgr
    return cv2.inpaint(bgr, match.mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def measure_template(frames: list[np.ndarray], box_px: tuple[int, int, int, int], min_level: int = 170) -> np.ndarray:
    """Glyph mask from many frames of ONE streaming session: pixels inside box that are bright in every frame
    (the scene changes between frames, the overlay does not). Cropped to the glyph block."""
    x0, y0, x1, y1 = box_px
    darkest = None
    for f in frames:
        g = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f[y0:y1, x0:x1]
        darkest = g if darkest is None else np.minimum(darkest, g)
    glyphs = (darkest >= min_level).astype(np.uint8) * 255
    ys, xs = np.nonzero(glyphs)
    if xs.size == 0:
        raise ValueError("no pixel is bright in every frame: no static overlay in this box")
    return glyphs[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
