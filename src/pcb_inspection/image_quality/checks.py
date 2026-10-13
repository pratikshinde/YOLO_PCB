"""Basic image-quality measurements for captured board images.

Deliberately simple, classical checks (NumPy + Pillow) for Experiment 0:
readable file, resolution, sharpness, exposure, whether the board appears
complete in the frame, and image orientation. They flag images to re-take or
inspect; they do not judge defects.

Board detection is a heuristic. It takes the dominant saturated hue in the
centre of the image as the board colour (blue / red / green reported, PCB-18)
and keeps only pixels of that hue, so a background of a different colour is
ignored. It assumes the board is roughly centred, with its edges roughly
parallel to the image edges. Results are approximate and must be confirmed
visually; the colour name is an estimate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DETECTION_MAX_SIDE_PX = 1600  # board detection runs on a downscaled copy for speed


@dataclass(frozen=True)
class QualityParams:
    """Thresholds; None means 'measure and report only, do not flag'."""

    min_megapixels: float | None = None
    min_sharpness: float | None = None
    relative_sharpness_min: float | None = None
    near_black_level: int = 5
    near_white_level: int = 250
    max_near_white_fraction: float | None = None
    max_near_black_fraction: float | None = None
    saturation_min: int = 60
    value_min: int = 40
    hue_tolerance: int = 12          # on Pillow's 0-255 hue scale (about 17 degrees)
    line_fraction_min: float = 0.10
    min_margin_fraction: float = 0.01
    max_aspect_error: float | None = None

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> QualityParams:
        detection = section.get("board_detection") or {}
        values = {k: v for k, v in {**section, **detection}.items() if k != "board_detection" and v is not None}
        return cls(**values)


@dataclass(frozen=True)
class BoardExtent:
    """Axis-aligned bounding box (x0, y0, x1, y1), exclusive x1/y1, in full-image pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class ImageQuality:
    path: str
    readable: bool
    error: str = ""
    width: int = 0
    height: int = 0
    orientation: str = ""
    sharpness: float = 0.0
    mean_brightness: float = 0.0
    near_white_fraction: float = 0.0
    near_black_fraction: float = 0.0
    board: BoardExtent | None = None
    colour_estimate: str = ""
    board_min_margin_px: int | None = None
    board_aspect: float | None = None
    board_aspect_error: float | None = None
    px_per_mm: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array, applying the EXIF orientation as a viewer would."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian. Higher = sharper; only comparable for the same scene and setup."""
    g = gray.astype(np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def exposure(gray: np.ndarray, near_black: int, near_white: int) -> tuple[float, float, float]:
    """Mean brightness and the fractions of near-black and near-white (clipped) pixels."""
    return float(gray.mean()), float((gray <= near_black).mean()), float((gray >= near_white).mean())


def detect_board(rgb: np.ndarray, params: QualityParams) -> tuple[BoardExtent | None, int | None]:
    """Find the board by its own colour; returns (bounding box, board hue on the 0-255 scale)."""
    height, width = rgb.shape[:2]
    scale = min(1.0, DETECTION_MAX_SIDE_PX / max(height, width))
    img = Image.fromarray(rgb)
    if scale < 1.0:
        img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BOX)
    hsv = np.asarray(img.convert("HSV")).astype(np.int16)
    hue, saturated = hsv[..., 0], (hsv[..., 1] >= params.saturation_min) & (hsv[..., 2] >= params.value_min)
    board_hue = dominant_central_hue(hue, saturated)
    if board_hue is None:
        return None, None
    distance = np.abs(hue - board_hue)
    mask = saturated & (np.minimum(distance, 256 - distance) <= params.hue_tolerance)
    rows = _longest_run(mask.mean(axis=1) >= params.line_fraction_min)
    cols = _longest_run(mask.mean(axis=0) >= params.line_fraction_min)
    if rows is None or cols is None:
        return None, board_hue
    box = BoardExtent(
        x0=round(cols[0] / scale), y0=round(rows[0] / scale),
        x1=min(width, round(cols[1] / scale)), y1=min(height, round(rows[1] / scale)),
    )
    return box, board_hue


def dominant_central_hue(hue: np.ndarray, saturated: np.ndarray, min_fraction: float = 0.05) -> int | None:
    """Most frequent saturated hue in the central 40 % of the image (where the board should be),
    or in the whole image if the centre shows too little colour (board off-centre)."""
    h, w = hue.shape
    centre = (slice(int(0.3 * h), int(0.7 * h) + 1), slice(int(0.3 * w), int(0.7 * w) + 1))
    for region in (centre, (slice(None), slice(None))):
        values = hue[region][saturated[region]]
        if values.size >= min_fraction * hue[region].size:
            counts = np.bincount(values // 8, minlength=32)
            return int(np.argmax(counts)) * 8 + 4
    return None


def colour_name(board_hue: int | None) -> str:
    """Coarse colour name for a hue on the 0-255 scale; an estimate, not a measurement."""
    if board_hue is None:
        return ""
    degrees = board_hue * 360 / 256
    if degrees < 20 or degrees >= 330:
        return "red"
    if 75 <= degrees < 165:
        return "green"
    if 180 <= degrees < 260:
        return "blue"
    return "other"


def _longest_run(active: np.ndarray, gap_fraction: float = 0.02) -> tuple[int, int] | None:
    """Longest span of True values, bridging gaps up to gap_fraction of the length. Returns (start, end)."""
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return None
    max_gap = max(1, int(gap_fraction * active.size))
    breaks = np.flatnonzero(np.diff(indices) > max_gap + 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]])) + 1
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def check_image(path: Path, params: QualityParams, board_mm: tuple[float, float] | None) -> ImageQuality:
    """Measure one image. board_mm = (long side, short side) of the physical board in mm; None if the board size
    is unknown (the shape check and the mm scale are then skipped)."""
    result = ImageQuality(path=str(path), readable=False)
    if not path.is_file():
        result.error = "file not found"
        result.flags.append("missing")
        return result
    try:
        rgb = load_rgb(path)
    except (UnidentifiedImageError, OSError) as e:
        result.error = f"cannot read image: {e}"
        result.flags.append("unreadable")
        return result

    result.readable = True
    result.height, result.width = rgb.shape[:2]
    result.orientation = "landscape" if result.width > result.height else "portrait" if result.height > result.width else "square"
    gray = to_gray(rgb)
    result.sharpness = sharpness(gray)

    board, board_hue = detect_board(rgb, params)
    result.board = board
    result.colour_estimate = colour_name(board_hue)
    region = gray[board.y0:board.y1, board.x0:board.x1] if board else gray
    result.mean_brightness, result.near_black_fraction, result.near_white_fraction = exposure(
        region, params.near_black_level, params.near_white_level
    )

    if params.min_megapixels is not None and result.megapixels < params.min_megapixels:
        result.flags.append("low_resolution")
    if params.min_sharpness is not None and result.sharpness < params.min_sharpness:
        result.flags.append("blurry")
    if params.max_near_white_fraction is not None and result.near_white_fraction > params.max_near_white_fraction:
        result.flags.append("glare_or_overexposed")
    if params.max_near_black_fraction is not None and result.near_black_fraction > params.max_near_black_fraction:
        result.flags.append("underexposed")

    if board is None:
        result.flags.append("board_not_detected")
        return result
    result.board_min_margin_px = min(board.x0, board.y0, result.width - board.x1, result.height - board.y1)
    min_margin = params.min_margin_fraction * min(result.width, result.height)
    if result.board_min_margin_px < min_margin:
        result.flags.append("board_at_image_edge_possible_crop")
    long_px, short_px = max(board.width, board.height), min(board.width, board.height)
    result.board_aspect = long_px / short_px if short_px else None
    if board_mm is None or None in board_mm:
        return result
    long_mm, short_mm = board_mm
    if result.board_aspect is not None:
        result.board_aspect_error = abs(result.board_aspect / (long_mm / short_mm) - 1.0)
        if params.max_aspect_error is not None and result.board_aspect_error > params.max_aspect_error:
            result.flags.append("board_shape_mismatch")
    result.px_per_mm = long_px / long_mm
    return result


def flag_relative_sharpness(results: Mapping[str, list[ImageQuality]], ratio: float | None) -> None:
    """Flag images clearly less sharp than the median of their group (e.g. one capture session = one setup)."""
    if ratio is None:
        return
    for images in results.values():
        values = [r.sharpness for r in images if r.readable]
        if len(values) < 2:
            continue
        reference = median(values)
        for r in images:
            if r.readable and r.sharpness < ratio * reference:
                r.flags.append("less_sharp_than_other_captures")
