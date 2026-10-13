"""Images for reference-comparison experiments. Heat maps use fixed scales so runs are comparable."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..alignment.registration import TileResiduals
from ..comparison.difference import ComparisonResult

DELTA_E_SCALE = 40.0          # delta E mapped to the full colour range
DISSIMILARITY_SCALE = 0.5
RESIDUAL_SCALE_MM = 0.5       # residual shift mapped to the full colour range
CANDIDATE_COLOUR = (0, 255, 255)   # BGR yellow
VERIFIED_COLOUR = (255, 128, 0)    # BGR blue: human-verified defect boxes


def heatmap(values: np.ndarray, scale: float, mask: np.ndarray) -> np.ndarray:
    scaled = np.clip(values / scale * 255.0, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    coloured[mask == 0] = (40, 40, 40)
    return coloured


def checkerboard_overlay(reference: np.ndarray, aligned: np.ndarray, square_px: int) -> np.ndarray:
    """Alternate squares from both images: misalignment shows as broken lines at square borders."""
    yy, xx = np.indices(reference.shape[:2])
    use_ref = ((yy // square_px + xx // square_px) % 2 == 0)[..., None]
    return np.where(use_ref, reference, aligned)


def draw_candidates(image: np.ndarray, result: ComparisonResult, px_per_mm: float,
                    verified_boxes_mm: list[tuple[float, float, float, float]] = ()) -> np.ndarray:
    out = image.copy()
    tint = out.copy()
    tint[result.candidates_mask > 0] = CANDIDATE_COLOUR
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
    thickness = max(2, round(0.1 * px_per_mm))
    for r in result.regions:
        p0 = (round(r.x_mm * px_per_mm), round(r.y_mm * px_per_mm))
        p1 = (round((r.x_mm + r.width_mm) * px_per_mm), round((r.y_mm + r.height_mm) * px_per_mm))
        cv2.rectangle(out, p0, p1, CANDIDATE_COLOUR, thickness)
    for x0, y0, x1, y1 in verified_boxes_mm:
        cv2.rectangle(out, (round(x0 * px_per_mm), round(y0 * px_per_mm)),
                      (round(x1 * px_per_mm), round(y1 * px_per_mm)), VERIFIED_COLOUR, 2 * thickness)
    return out


def residual_map(aligned: np.ndarray, residuals: TileResiduals, px_per_mm: float) -> np.ndarray:
    """Tiles coloured by residual shift (mm); grey = not reliable (little texture / unclear peak)."""
    out = (aligned * 0.5).astype(np.uint8)
    half = residuals.tile_px // 2
    magnitudes = residuals.magnitudes_px() / px_per_mm
    for (cx, cy), (dx, dy), mm, ok in zip(residuals.centres, residuals.shifts_px, magnitudes, residuals.reliable):
        p0, p1 = (int(cx - half), int(cy - half)), (int(cx + half) - 1, int(cy + half) - 1)
        if ok:
            colour = cv2.applyColorMap(np.uint8([[min(255, mm / RESIDUAL_SCALE_MM * 255)]]), cv2.COLORMAP_INFERNO)[0, 0]
            cv2.rectangle(out, p0, p1, tuple(int(c) for c in colour), -1)
            end = (int(cx + dx * 10), int(cy + dy * 10))  # shift exaggerated 10x
            cv2.arrowedLine(out, (int(cx), int(cy)), end, (255, 255, 255), 1, tipLength=0.3)
        else:
            cv2.rectangle(out, p0, p1, (90, 90, 90), 1)
    return out


def mm_grid(image: np.ndarray, px_per_mm: float, major_mm: int = 10, minor_mm: int = 5) -> np.ndarray:
    """Board image with a labelled mm grid (origin = top-left board corner, x right, y down).

    Lets the operator read off the box of a verified defect (x0,y0,x1,y1 in mm)
    for defects.csv, without drawing on the image or the board.
    """
    out = image.copy()
    h, w = out.shape[:2]
    for mm in range(0, int(max(w, h) / px_per_mm) + 1, minor_mm):
        p = round(mm * px_per_mm)
        major = mm % major_mm == 0
        colour, thickness = ((255, 255, 255), 2) if major else ((200, 200, 200), 1)
        if p < w:
            cv2.line(out, (p, 0), (p, h - 1), colour, thickness)
            if major:
                _label(out, str(mm), (p + 4, 34))
        if p < h:
            cv2.line(out, (0, p), (w - 1, p), colour, thickness)
            if major and mm:
                _label(out, str(mm), (4, p - 6))
    return out


def _label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """Large yellow text with a black outline, readable on any board colour."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)


def labelled(image: np.ndarray, text: str, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    small = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    bar = np.full((28, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, small])


def panel(images: list[tuple[str, np.ndarray]], columns: int = 4, tile_width: int = 420) -> np.ndarray:
    tiles = [labelled(img, label, tile_width) for label, img in images]
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((height - t.shape[0], tile_width, 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % columns:
        tiles.append(np.full((height, tile_width, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not encode {path}")
    path.write_bytes(buffer.tobytes())
