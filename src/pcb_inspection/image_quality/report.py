"""Quality report table shared by the Experiment 0 scripts (one row per image)."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from .checks import ImageQuality

REPORT_COLUMNS = (
    "image_id", "session_id", "board_id", "readable", "width", "height", "megapixels", "orientation",
    "colour_estimate", "sharpness", "board_mean_brightness", "board_near_white_fraction",
    "board_near_black_fraction", "board_bbox", "board_min_margin_px", "board_aspect_error",
    "px_per_mm_approx", "flags", "error",
)


def quality_row(image_id: str, session_id: str, board_id: str, q: ImageQuality) -> dict[str, object]:
    bbox = f"{q.board.x0},{q.board.y0},{q.board.x1},{q.board.y1}" if q.board else ""
    return {
        "image_id": image_id, "session_id": session_id, "board_id": board_id, "readable": q.readable,
        "width": q.width, "height": q.height, "megapixels": round(q.megapixels, 2), "orientation": q.orientation,
        "colour_estimate": q.colour_estimate, "sharpness": round(q.sharpness, 2),
        "board_mean_brightness": round(q.mean_brightness, 1),
        "board_near_white_fraction": round(q.near_white_fraction, 4),
        "board_near_black_fraction": round(q.near_black_fraction, 4),
        "board_bbox": bbox,
        "board_min_margin_px": "" if q.board_min_margin_px is None else q.board_min_margin_px,
        "board_aspect_error": "" if q.board_aspect_error is None else round(q.board_aspect_error, 3),
        "px_per_mm_approx": "" if q.px_per_mm is None else round(q.px_per_mm, 1),
        "flags": ";".join(q.flags), "error": q.error,
    }


def merge_report(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write rows into the report at path, replacing earlier rows of the same image_id."""
    merged: dict[str, Mapping[str, object]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            merged = {row["image_id"]: row for row in csv.DictReader(f)}
    merged.update({str(row["image_id"]): row for row in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))


def print_table(rows: Iterable[Mapping[str, object]]) -> None:
    print(f"  {'image_id':<20} {'size':>11} {'colour':>7} {'px/mm':>6} {'sharpness':>10}  flags")
    for r in rows:
        size = f"{r['width']}x{r['height']}" if r["readable"] else "-"
        print(f"  {r['image_id']!s:<20} {size:>11} {r['colour_estimate'] or '-':>7} "
              f"{r['px_per_mm_approx'] or '-':>6} {r['sharpness']!s:>10}  {r['flags'] or 'ok'}")
