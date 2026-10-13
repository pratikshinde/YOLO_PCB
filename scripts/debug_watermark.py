"""DroidCam watermark: locate it in images and write debug views; or (re)measure its glyph template.

    python scripts/debug_watermark.py <image|folder> [...]            debug views for these images
    python scripts/debug_watermark.py --measure data/testing/live --box 774 48 1392 104
                                                                      measure the glyph template from frames of
                                                                      ONE DroidCam session (box = around the text)

Debug views (default experiments/experiment_002/watermark_debug/<image>/):
  1_original.jpg   region around the watermark, as captured
  2_mask.jpg       the ignored pixels (letters only, white) on black
  3_masked.jpg     the same region after inpainting the letter pixels; everything else unchanged
  overview.jpg     whole frame, ignored pixels in magenta (to see that no board area is masked wholesale)
Settings: watermark section of experiments/experiment_002/config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pcb_inspection.comparison.board_inspection import load_inspection_config  # noqa: E402
from pcb_inspection.comparison.pipeline import load_bgr  # noqa: E402
from pcb_inspection.preprocessing.watermark import (  # noqa: E402
    load_template,
    locate_watermark,
    measure_template,
    remove_watermark,
)

SUFFIXES = {".png", ".jpg", ".jpeg"}


def images(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        found += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate the DroidCam watermark and write debug views.")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_002" / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "experiment_002" / "watermark_debug")
    parser.add_argument("--measure", type=Path, help="folder of frames from ONE streaming session")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="region around the text")
    args = parser.parse_args()
    cfg = load_inspection_config(ROOT, args.config)

    if args.measure:
        if not args.box:
            parser.error("--measure needs --box")
        frames = [load_bgr(f) for f in images([args.measure])]
        template = measure_template(frames, tuple(args.box))
        cv2.imwrite(str(cfg.watermark.template), template)
        print(f"template {template.shape[1]} x {template.shape[0]} px from {len(frames)} frames -> {cfg.watermark.template}")
        return 0

    template = load_template(cfg.watermark)
    for path in images(args.images):
        bgr = load_bgr(path)
        match = locate_watermark(bgr, template, cfg.watermark)
        print(f"{path.name}: {'FOUND' if match.found else 'not found'}  box {match.box_px}  normalised "
              f"{match.normalised_box}  score {match.score:.2f}  margin {match.margin:.2f}"
              + (f"  ignored px {int(match.mask.sum())} ({match.mask.mean():.3%} of the frame)" if match.found else ""))
        if not match.found:
            continue
        out = args.output / path.stem
        out.mkdir(parents=True, exist_ok=True)
        x0, y0, x1, y1 = match.box_px
        m = 60
        ya, yb, xa, xb = max(0, y0 - m), min(bgr.shape[0], y1 + m), max(0, x0 - m), min(bgr.shape[1], x1 + m)
        cleaned = remove_watermark(bgr, match)
        cv2.imwrite(str(out / "1_original.jpg"), bgr[ya:yb, xa:xb], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "2_mask.jpg"), match.mask[ya:yb, xa:xb].astype(np.uint8) * 255)
        cv2.imwrite(str(out / "3_masked.jpg"), cleaned[ya:yb, xa:xb], [cv2.IMWRITE_JPEG_QUALITY, 95])
        overview = bgr.copy()
        overview[match.mask] = (255, 0, 255)
        cv2.rectangle(overview, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 0, 255), 2)
        cv2.imwrite(str(out / "overview.jpg"), cv2.resize(overview, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 0


if __name__ == "__main__":
    sys.exit(main())
