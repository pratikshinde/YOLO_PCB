"""Capture-time PCB check and canonical PCB image (clean data collection, docs/clean_data_collection.md).

    camera frame -> PCB region (board colour) -> 4 board corners -> crop/warp ONLY the PCB -> canonical PCB image

The canonical PCB image is the board alone, rectified to an upright rectangle of the profile's board size at a
fixed scale (canonical_px_per_mm). Background is cut away. The DroidCam free-edition watermark ("using
droidcam.app", burned into every frame) is located in the frame and must lie OUTSIDE the board, so the canonical
image contains no watermark pixels. That image, never the full frame, is what a golden-bank comparison or an
inspection uses.

Checks. A definite problem gives RECAPTURE:
    NO_PCB              no board-coloured region of plausible size
    PCB_INCOMPLETE      board at / beyond the frame edge (part of the PCB is not in the picture)
    PCB_OUTSIDE_ROI     board not completely inside the inspection ROI (the dashed guide in the preview)
    WATERMARK_OVER_PCB  watermark over the board, or closer to it than watermark_clearance_mm
    BAD_CROP            board outline is not a 4-sided board of the profile's shape (outline aspect, and how well
                        the board region fills the 4-corner outline, measured on its convex hull so that parts on
                        the board edge do not count as a broken outline), or a large occlusion of the board
    PCB_TOO_SMALL       board below min_source_px_per_mm in the frame: camera too far, too little detail
    BLURRY              PCB image less sharp than min_pcb_sharpness (PROVISIONAL), measured at a FIXED scale
                        (sharpness_px_per_mm) so that the camera distance does not change the score
    OVEREXPOSED / UNDEREXPOSED   too many clipped pixels on the PCB image
What cannot be confirmed automatically gives REVIEW:
    WATERMARK_NOT_LOCATED   watermark-like text ON the PCB area that is not clearly located: confirm by eye
    BOARD_SIZE_UNKNOWN      the profile has no board size: shape and crop cannot be checked
The watermark need not be visible at all: when it is not found in the frame (outside the view, cut off at the
image edge, faint on a light background), the PCB area is searched on its own; no match there at the detection
threshold means no watermark on the PCB (every one of 94 watermarks over a PCB in the archived frames matched).
Otherwise OK. A check never deletes or changes the original; it is evidence for the operator's KEEP / REJECT.
All limits are PROVISIONAL (set from the setup, not validated on many sessions).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..image_quality.checks import exposure, sharpness
from ..preprocessing.board import BoardSegmentationParams, board_mask, board_quad, canonical_frame, warp_to_canonical
from ..preprocessing.watermark import WatermarkParams, load_template, locate_watermark

OK, REVIEW, RECAPTURE = "OK", "REVIEW", "RECAPTURE"
NO_PCB, PCB_INCOMPLETE, PCB_OUTSIDE_ROI = "NO_PCB", "PCB_INCOMPLETE", "PCB_OUTSIDE_ROI"
WATERMARK_OVER_PCB, BAD_CROP, BLURRY = "WATERMARK_OVER_PCB", "BAD_CROP", "BLURRY"
PCB_TOO_SMALL, PCB_ROTATED = "PCB_TOO_SMALL", "PCB_ROTATED"
OVEREXPOSED, UNDEREXPOSED = "OVEREXPOSED", "UNDEREXPOSED"
WATERMARK_NOT_LOCATED, BOARD_SIZE_UNKNOWN = "WATERMARK_NOT_LOCATED", "BOARD_SIZE_UNKNOWN"
LESS_SHARP_THAN_SESSION, PCB_IMAGE_ERROR = "LESS_SHARP_THAN_SESSION", "PCB_IMAGE_ERROR"   # set by the store

MESSAGES = {
    NO_PCB: "No PCB found. Place the board inside the dashed frame.",
    PCB_INCOMPLETE: "PCB not completely in the picture. Move it away from the image edge.",
    PCB_OUTSIDE_ROI: "PCB not completely inside the dashed inspection frame. Move it inside.",
    WATERMARK_OVER_PCB: "Camera watermark over (or too close to) the PCB. Move the board so the watermark is outside it.",
    BAD_CROP: "Board outline not clean (hand / object over the edge, board tilted, or wrong board?).",
    BLURRY: "PCB out of focus. Refocus (tap the PCB in the DroidCam app) and capture again.",
    PCB_TOO_SMALL: "PCB too small in the picture (camera too far): move the camera closer.",
    PCB_ROTATED: "PCB placed the other way round (rotated 180 deg): turn it so it lies like the reference board.",
    OVEREXPOSED: "Glare / overexposure on the PCB. Reduce light or remove reflections.",
    UNDEREXPOSED: "PCB too dark. Add light.",
    WATERMARK_NOT_LOCATED: "Watermark-like text on the PCB that is not clearly located: confirm that no overlay "
                           "covers the PCB.",
    BOARD_SIZE_UNKNOWN: "Board size unknown in the product profile: crop shape not checked.",
    LESS_SHARP_THAN_SESSION: "Clearly less sharp than the other captures of this session: check the focus.",
    PCB_IMAGE_ERROR: "The PCB image could not be produced / saved: check the original.",
}
RECAPTURE_CODES = (NO_PCB, PCB_INCOMPLETE, PCB_OUTSIDE_ROI, WATERMARK_OVER_PCB, BAD_CROP, PCB_TOO_SMALL, PCB_ROTATED,
                   BLURRY, OVEREXPOSED, UNDEREXPOSED)
ORIENTATION_PX_PER_MM = 4.0      # PCB images are compared with the reference PCB image at this scale (blurred)

WORKING_MAX_SIDE_PX = 1280     # resolution of the geometric tests (watermark distance), not of the crop


@dataclass(frozen=True)
class PcbCheckParams:
    board_mm: tuple[float, float] | None             # (long side, short side); None = unknown
    roi_fraction: float = 0.8                        # inspection ROI = board-shaped, centred, this fraction of the frame
    min_board_area_fraction: float = 0.10            # smaller board-colour region -> NO_PCB
    edge_margin_fraction: float = 0.01               # board must stay this far from the frame edge (of the short side)
    max_aspect_error: float = 0.08                   # board outline vs profile board shape
    min_outline_fill: float = 0.93                   # convex hull of the board region / 4-corner outline area
    max_outline_fill: float = 1.10
    min_pixel_fill: float = 0.75                     # board pixels / outline area: only a LARGE occlusion fails
    watermark_clearance_mm: float = 3.0              # watermark must stay at least this far from the board
    canonical_px_per_mm: float = 20.0                # canonical PCB image scale (= Experiment 1 / 2 pipeline)
    pcb_border_mm: float = 1.0                       # exposure / sharpness ignore this rim (board edge, background)
    min_source_px_per_mm: float | None = None        # board scale in the frame; smaller -> PCB_TOO_SMALL
    sharpness_px_per_mm: float = 8.0                 # the PCB image is measured at this fixed scale (focus score)
    min_pcb_sharpness: float | None = None           # Laplacian variance at sharpness_px_per_mm; None = not checked
    orientation_reference: Path | None = None        # PCB image of the inspection reference: a PCB image that matches
    min_orientation_margin: float = 0.02             # it better turned 180 deg (by this) -> PCB_ROTATED; None = off
    near_black_level: int = 5
    near_white_level: int = 250
    max_near_white_fraction: float | None = 0.02
    max_near_black_fraction: float | None = 0.05
    segmentation: BoardSegmentationParams = BoardSegmentationParams()
    watermark: WatermarkParams = WatermarkParams(enabled=False)

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any], board_mm: tuple[float, float] | None,
                     project_root: Path) -> PcbCheckParams:
        values = {k: v for k, v in section.items() if k not in ("segmentation", "watermark", "preview_every_s")}
        seg = BoardSegmentationParams(**(section.get("segmentation") or {}))
        wm = dict(section.get("watermark") or {"enabled": False})
        if wm.get("template"):
            wm["template"] = project_root / wm["template"]
        if values.get("orientation_reference"):
            values["orientation_reference"] = project_root / values["orientation_reference"]
        if "search_region" in wm:
            wm["search_region"] = tuple(wm["search_region"])
        return cls(board_mm=board_mm, segmentation=seg, watermark=WatermarkParams(**wm), **values)


@dataclass
class PcbCheck:
    status: str                                      # OK | REVIEW | RECAPTURE
    problems: list[str] = field(default_factory=list)          # RECAPTURE codes
    review: list[str] = field(default_factory=list)            # REVIEW codes
    frame_size: tuple[int, int] = (0, 0)             # (width, height)
    roi_px: tuple[int, int, int, int] | None = None
    quad_px: list[list[float]] | None = None          # board corners: top-left, top-right, bottom-right, bottom-left
    board_area_fraction: float | None = None
    edge_margin_px: int | None = None
    aspect_error: float | None = None
    fill_fraction: float | None = None               # board pixels / outline area (only a large occlusion fails)
    outline_fill: float | None = None                # convex hull of the board region / outline area (checked)
    source_px_per_mm: float | None = None
    watermark: dict[str, Any] = field(default_factory=dict)
    pcb: dict[str, Any] = field(default_factory=dict)          # canonical PCB image measurements
    canonical: np.ndarray | None = field(default=None, repr=False)

    @property
    def messages(self) -> list[str]:
        return [MESSAGES.get(c, c) for c in (*self.problems, *self.review)]

    def add(self, code: str) -> None:
        target = self.problems if code in RECAPTURE_CODES else self.review
        if code not in target:
            target.append(code)
        self.status = RECAPTURE if self.problems else (REVIEW if self.review else OK)

    def to_record(self) -> dict[str, Any]:
        r4 = lambda v: None if v is None else round(float(v), 4)  # noqa: E731
        return {
            "status": self.status, "problems": list(self.problems), "review": list(self.review),
            "messages": self.messages, "frame_size": list(self.frame_size),
            "roi_px": list(self.roi_px) if self.roi_px else None,
            "quad_px": self.quad_px, "board_area_fraction": r4(self.board_area_fraction),
            "edge_margin_px": self.edge_margin_px, "aspect_error": r4(self.aspect_error),
            "fill_fraction": r4(self.fill_fraction), "outline_fill": r4(self.outline_fill),
            "source_px_per_mm": r4(self.source_px_per_mm),
            "watermark": dict(self.watermark), "pcb": dict(self.pcb),
        }


def inspection_roi(width: int, height: int, board_mm: tuple[float, float] | None,
                   fraction: float) -> tuple[int, int, int, int]:
    """The inspection ROI (x0, y0, x1, y1): a board-shaped rectangle, centred, `fraction` of the frame.
    The SAME definition draws the dashed guide in the live preview."""
    long_mm, short_mm = board_mm or (1.0, 1.0)      # board size unknown: square guide
    bh = fraction * min(height, width)
    bw = min(bh * long_mm / short_mm, fraction * max(height, width))
    if height > width:
        bw, bh = bh, bw
    return int((width - bw) / 2), int((height - bh) / 2), int((width + bw) / 2), int((height + bh) / 2)


def _scaled(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


class PcbChecker:
    """Loads the watermark template once; check() runs on every capture (and on preview frames)."""

    def __init__(self, params: PcbCheckParams):
        self.params = params
        self.template = load_template(params.watermark)
        self.orientation_ref = None
        if params.orientation_reference is not None:
            ref = cv2.imread(str(params.orientation_reference), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                raise ValueError(f"orientation reference not found: {params.orientation_reference}")
            self.orientation_ref = _orientation_image(ref, params.canonical_px_per_mm)

    def check(self, bgr: np.ndarray, make_canonical: bool = True) -> PcbCheck:
        p = self.params
        h, w = bgr.shape[:2]
        result = PcbCheck(status=OK, frame_size=(w, h))
        result.roi_px = inspection_roi(w, h, p.board_mm, p.roi_fraction)
        if p.board_mm is None:
            result.add(BOARD_SIZE_UNKNOWN)

        mask = board_mask(bgr, p.segmentation)
        area = float(np.count_nonzero(mask)) / (w * h)
        result.board_area_fraction = area
        quad = board_quad(mask) if area >= p.min_board_area_fraction else None
        if quad is None:
            result.add(NO_PCB)
            self._watermark(bgr, None, result)
            return result
        result.quad_px = [[round(float(x), 1), round(float(y), 1)] for x, y in quad]

        # complete PCB in the picture: board pixels and corners away from the frame edge
        ys, xs = np.nonzero(mask)
        margin = int(min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max(),
                         quad[:, 0].min(), quad[:, 1].min(), w - 1 - quad[:, 0].max(), h - 1 - quad[:, 1].max()))
        result.edge_margin_px = margin
        if margin < p.edge_margin_fraction * min(w, h):
            result.add(PCB_INCOMPLETE)

        # inside the inspection ROI
        x0, y0, x1, y1 = result.roi_px
        if not ((quad[:, 0] >= x0).all() and (quad[:, 0] <= x1).all() and (quad[:, 1] >= y0).all()
                and (quad[:, 1] <= y1).all()):
            result.add(PCB_OUTSIDE_ROI)

        # clean outline of the right shape
        tl, tr, br, bl = quad
        width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        quad_area = float(cv2.contourArea(quad.astype(np.float32)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = float(cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea))))
        result.fill_fraction = float(np.count_nonzero(mask)) / quad_area if quad_area > 0 else 0.0
        result.outline_fill = hull_area / quad_area if quad_area > 0 else 0.0
        if p.board_mm is not None:
            long_mm, short_mm = p.board_mm
            aspect = max(width_px, height_px) / max(min(width_px, height_px), 1e-6)
            result.aspect_error = float(abs(aspect / (long_mm / short_mm) - 1.0))
            result.source_px_per_mm = float(max(width_px, height_px) / long_mm)
            if result.aspect_error > p.max_aspect_error:
                result.add(BAD_CROP)
            if p.min_source_px_per_mm is not None and result.source_px_per_mm < p.min_source_px_per_mm:
                result.add(PCB_TOO_SMALL)
        if not p.min_outline_fill <= result.outline_fill <= p.max_outline_fill or result.fill_fraction < p.min_pixel_fill:
            result.add(BAD_CROP)

        self._watermark(bgr, quad, result)
        if make_canonical and p.board_mm is not None:
            self._canonical(bgr, quad, result)
        return result

    # --- watermark ---------------------------------------------------------------------------
    def _watermark(self, bgr: np.ndarray, quad: np.ndarray | None, result: PcbCheck) -> None:
        p = self.params
        if not p.watermark.enabled or self.template is None:
            result.watermark = {"checked": False, "reason": "watermark check disabled in the capture config"}
            return
        match = locate_watermark(bgr, self.template, p.watermark)
        result.watermark = {"checked": True, "found": match.found, "box_px": list(match.box_px) if match.box_px else None,
                            "score": round(match.score, 3), "margin": round(match.margin, 3)}
        if match.found:
            result.watermark["zone_px"] = self._zone(match.box_px, result.source_px_per_mm, bgr.shape[:2])
        if quad is None:
            return
        # The PCB area (+ clearance) is ALWAYS searched on its own as well: the frame-wide match may be absent (outside
        # the view, cut off at the image edge, faint on a light background) or be another, clearer patch elsewhere.
        ppm = result.source_px_per_mm or 0.0
        pad = p.watermark_clearance_mm * ppm + self.template.shape[0] * bgr.shape[1] / p.watermark.template_frame_width
        h, w = bgr.shape[:2]
        region = (max(0.0, (quad[:, 0].min() - pad) / w), max(0.0, (quad[:, 1].min() - pad) / h),
                  min(1.0, (quad[:, 0].max() + pad) / w), min(1.0, (quad[:, 1].max() + pad) / h))
        on_pcb = locate_watermark(bgr, self.template, replace(p.watermark, search_region=region))
        result.watermark.update(on_pcb_best_score=round(on_pcb.score, 3),
                                not_on_pcb=not on_pcb.found and on_pcb.score < p.watermark.min_score)
        candidates = [m for m in (match, on_pcb) if m.found]
        if not candidates:
            if on_pcb.score >= p.watermark.min_score:
                result.add(WATERMARK_NOT_LOCATED)      # watermark-like text on the PCB, ambiguous: a person decides
            return
        measured = [(*_box_to_board_distance(m.box_px, quad, bgr.shape[:2]), m) for m in candidates]
        distance_px, overlap, nearest = min(measured, key=lambda t: (not t[1], t[0]))   # worst case first
        if nearest is not match:                       # the one on / at the PCB is what matters: show that one
            result.watermark.update(found=True, box_px=list(nearest.box_px), score=round(nearest.score, 3),
                                    margin=round(nearest.margin, 3),
                                    zone_px=self._zone(nearest.box_px, result.source_px_per_mm, bgr.shape[:2]))
        ppm = result.source_px_per_mm
        result.watermark.update(over_pcb=overlap, distance_px=round(distance_px, 1),
                                distance_mm=None if not ppm else round(float(distance_px / ppm), 2))
        clearance_px = p.watermark_clearance_mm * ppm if ppm else 0.0
        if overlap or distance_px < clearance_px:
            result.add(WATERMARK_OVER_PCB)

    def _zone(self, box: tuple[int, int, int, int], ppm: float | None, shape: tuple[int, int]) -> list[int]:
        """The watermark box grown by the clearance: the area the PCB must stay out of (shown in the preview). Without
        a board in view the scale is estimated from the frame width (12 px/mm at 2560 px, as measured)."""
        ppm = ppm or 12.0 * shape[1] / 2560
        pad = round(self.params.watermark_clearance_mm * ppm)
        return [max(0, box[0] - pad), max(0, box[1] - pad), min(shape[1], box[2] + pad), min(shape[0], box[3] + pad)]

    # --- canonical PCB image -----------------------------------------------------------------
    def _canonical(self, bgr: np.ndarray, quad: np.ndarray, result: PcbCheck) -> None:
        p = self.params
        frame = canonical_frame(quad, p.board_mm, p.canonical_px_per_mm)
        warped, _ = warp_to_canonical(bgr, frame.homography, frame.size, frame.source_px_per_mm, p.canonical_px_per_mm)
        result.canonical = warped
        rim = int(round(p.pcb_border_mm * p.canonical_px_per_mm))
        inner = warped[rim:warped.shape[0] - rim, rim:warped.shape[1] - rim] if rim > 0 else warped
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean, near_black, near_white = exposure(gray, p.near_black_level, p.near_white_level)
        factor = min(1.0, p.sharpness_px_per_mm / p.canonical_px_per_mm)   # fixed scale: camera distance cancels out
        focus = gray if factor >= 1.0 else cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        result.pcb = {"width": frame.size[0], "height": frame.size[1], "px_per_mm": p.canonical_px_per_mm,
                      "orientation": "landscape" if frame.size[0] >= frame.size[1] else "portrait",
                      "sharpness": round(sharpness(focus), 2), "sharpness_px_per_mm": p.sharpness_px_per_mm,
                      "mean_brightness": round(mean, 2),
                      "near_white_fraction": round(near_white, 5), "near_black_fraction": round(near_black, 5),
                      "upscale": round(p.canonical_px_per_mm / max(frame.source_px_per_mm, 1e-6), 3)}
        if p.min_pcb_sharpness is not None and result.pcb["sharpness"] < p.min_pcb_sharpness:
            result.add(BLURRY)
        if self.orientation_ref is not None:
            small = _orientation_image(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), p.canonical_px_per_mm)
            if small.shape == self.orientation_ref.shape:      # same board shape (both landscape)
                same = float(cv2.matchTemplate(small, self.orientation_ref, cv2.TM_CCOEFF_NORMED)[0, 0])
                turned = float(cv2.matchTemplate(small, self.orientation_ref[::-1, ::-1].copy(),
                                                 cv2.TM_CCOEFF_NORMED)[0, 0])
                result.pcb.update(orientation_score=round(same, 3), orientation_score_180=round(turned, 3))
                if p.min_orientation_margin is not None and turned > same + p.min_orientation_margin:
                    result.add(PCB_ROTATED)
        if p.max_near_white_fraction is not None and near_white > p.max_near_white_fraction:
            result.add(OVEREXPOSED)
        if p.max_near_black_fraction is not None and near_black > p.max_near_black_fraction:
            result.add(UNDEREXPOSED)


def _orientation_image(gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """A PCB image reduced to ORIENTATION_PX_PER_MM and blurred: the coarse layout that tells 0 from 180 deg."""
    f = ORIENTATION_PX_PER_MM / px_per_mm
    small = cv2.resize(gray, (max(1, round(gray.shape[1] * f)), max(1, round(gray.shape[0] * f))),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    return cv2.GaussianBlur(small, (0, 0), 2)


def _box_to_board_distance(box: tuple[int, int, int, int], quad: np.ndarray,
                           shape: tuple[int, int]) -> tuple[float, bool]:
    """(distance in frame px from the watermark box to the board outline, whether they overlap), measured on a
    rasterised copy at WORKING_MAX_SIDE_PX (precision about one working pixel)."""
    h, w = shape
    scale = min(1.0, WORKING_MAX_SIDE_PX / max(h, w))
    sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
    board = np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(board, [np.round(quad * scale).astype(np.int32)], 255)
    x0, y0, x1, y1 = (int(np.floor(box[0] * scale)), int(np.floor(box[1] * scale)),
                      int(np.ceil(box[2] * scale)), int(np.ceil(box[3] * scale)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(sw, max(x1, x0 + 1)), min(sh, max(y1, y0 + 1))
    if np.count_nonzero(board[y0:y1, x0:x1]):
        return 0.0, True
    distance = cv2.distanceTransform(255 - board, cv2.DIST_L2, 5)
    return float(distance[y0:y1, x0:x1].min()) / scale, False


# --- drawing (preview overlay, verification sheet) -----------------------------------------------

STATUS_COLOURS = {OK: (60, 180, 60), REVIEW: (0, 190, 255), RECAPTURE: (40, 40, 230)}


def draw_check(image: np.ndarray, check: PcbCheck, scale: float = 1.0, dashed_roi: bool = True) -> np.ndarray:
    """Draw the ROI, the board outline and the watermark box of a check (made on a frame `1/scale` times the size
    of `image`) onto a copy of image. Display only: never applied to saved originals or PCB images."""
    out = image.copy()
    thick = max(1, round(2 * max(1.0, out.shape[1] / 1000)))
    if check.roi_px:
        x0, y0, x1, y1 = (round(v * scale) for v in check.roi_px)
        colour = (0, 220, 255)
        step = max(8, round(16 * out.shape[1] / 800))
        if dashed_roi:
            for x in range(x0, x1, step):
                cv2.line(out, (x, y0), (min(x + step // 2, x1), y0), colour, thick)
                cv2.line(out, (x, y1), (min(x + step // 2, x1), y1), colour, thick)
            for y in range(y0, y1, step):
                cv2.line(out, (x0, y), (x0, min(y + step // 2, y1)), colour, thick)
                cv2.line(out, (x1, y), (x1, min(y + step // 2, y1)), colour, thick)
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), colour, thick)
    if check.quad_px:
        pts = np.round(np.asarray(check.quad_px) * scale).astype(np.int32)
        cv2.polylines(out, [pts], True, STATUS_COLOURS[check.status], thick + 1)
    zone = check.watermark.get("zone_px")
    if zone and check.watermark.get("found"):        # the no-go zone: watermark + clearance, shaded red
        x0, y0, x1, y1 = (round(v * scale) for v in zone)
        shade = out.copy()
        cv2.rectangle(shade, (x0, y0), (x1, y1), (40, 40, 230), cv2.FILLED)
        out = cv2.addWeighted(shade, 0.35, out, 0.65, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 40, 230), thick)
        cv2.putText(out, "WATERMARK ZONE: keep the PCB out", (x0, min(out.shape[0] - 4, y1 + round(14 * thick / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, (40, 40, 230), max(1, thick // 2), cv2.LINE_AA)
    return out
