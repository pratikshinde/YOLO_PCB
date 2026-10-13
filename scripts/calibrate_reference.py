"""Natural GOOD variation of every component, per physical board (leave-one-board-out) -> per-component limits.

    python scripts/calibrate_reference.py

For every physical GOOD board B of the reference set (board registry, split 'reference'):
  every image of B is compared with the GOOD examples of the OTHER boards only (B left out completely,
  including the geometric reference image if it belongs to B) -> best similarity per component.
  Per component and board: the LOWEST value over B's images (the worst natural case seen for that board).
Per component: limit = lowest value over all boards - margin (config reference_set.margin).

Several photos of one board count once: the unit is the physical board. Only reference-split GOOD boards are
used; validation and test boards never influence a limit. The inspection uses these limits only when they come
from at least reference_set.min_boards boards and match the current reference set; otherwise it keeps the global
limit component_check.min_similarity. Output: experiments/experiment_002/reference_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pcb_inspection.comparison.board_inspection import load_gallery, load_inspection_config, load_reference_and_map  # noqa: E402
from pcb_inspection.comparison.components import check_components  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-component GOOD variation (leave-one-board-out).")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_002" / "config.yaml")
    args = parser.parse_args()
    cfg = load_inspection_config(ROOT, args.config)
    reference, cmap = load_reference_and_map(cfg, ROOT / "experiments" / cfg.experiment / "reference" / cfg.reference_file.stem)
    gallery = load_gallery(cfg, reference)
    boards = gallery.board_ids
    ppm = cfg.pipeline.px_per_mm
    print(f"GOOD reference set: {len(gallery.names)} images of {len(boards)} physical boards: "
          + ", ".join(f"{b} ({gallery.boards.count(b)} images)" for b in boards))
    for message in gallery.skipped:
        print(f"  GOOD image not used: {message}")
    if len(boards) < 2:
        print("At least 2 physical GOOD boards are needed for leave-one-board-out; nothing calibrated.")
        return 1

    per: dict[str, dict[str, float]] = {c.refdes: {} for c in cmap.components}
    for board in boards:
        others = gallery.without_board(board)
        tests = [(n, f) for n, b, f in zip(gallery.names[1:], gallery.boards[1:], gallery.files) if b == board]
        if reference.board_id == board:
            tests.append((reference.info["id"], None))
        for name, file in tests:
            aligned = reference.canonical if file is None else cv2.imread(str(file))
            usable = np.ones(aligned.shape[:2], bool)
            results = check_components(reference.canonical, aligned, usable, cmap, ppm, cfg.check, others.images,
                                       None, reference_is_example=board != reference.board_id)
            for r in results:
                if r.similarity is not None:
                    per[r.refdes][board] = min(per[r.refdes].get(board, 1.0), r.similarity)
        print(f"  {board}: {len(tests)} image(s) compared with {len(others.names) - (board == reference.board_id)} "
              "examples of the other boards")

    margin, active = cfg.calibration_margin, len(boards) >= cfg.calibration_min_boards
    components = {}
    for refdes, values in per.items():
        complete = len(values) == len(boards)
        low = min(values.values()) if complete else None
        components[refdes] = {"per_board": {b: round(v, 4) for b, v in sorted(values.items())},
                              "min": None if low is None else round(low, 4),
                              "mean": round(mean(values.values()), 4) if values else None,
                              "limit": None if low is None else round(low - margin, 4)}
    lows = [v["min"] for v in components.values() if v["min"] is not None]
    out = {
        "created": datetime.now().isoformat(timespec="seconds"), "reference_split": cfg.reference_split,
        "boards": boards, "images_per_board": {b: gallery.boards.count(b) for b in boards},
        "min_boards": cfg.calibration_min_boards, "active": active, "margin": margin,
        "method": "leave-one-board-out; per board the lowest best-similarity over its images; limit = min over boards - margin",
        "global_cross_board_min": round(min(lows), 4) if lows else None,
        "global_limit_from_this_calibration": round(min(lows) - margin, 4) if lows else None,
        "configured_global_limit": cfg.check.min_similarity,
        "components": components,
    }
    cfg.calibration_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{'component':<9} " + " ".join(f"{b:>10}" for b in boards) + f" {'min':>7} {'limit':>7}   (most variable first)")
    for refdes, v in sorted(components.items(), key=lambda kv: kv[1]["min"] if kv[1]["min"] is not None else 9)[:15]:
        print(f"{refdes:<9} " + " ".join(f"{v['per_board'].get(b, float('nan')):>10.3f}" for b in boards)
              + f" {v['min'] if v['min'] is not None else float('nan'):>7.3f} {v['limit'] if v['limit'] is not None else float('nan'):>7.3f}")
    print(f"... ({len(components)} components)\nLowest cross-board GOOD similarity: {out['global_cross_board_min']} "
          f"-> global limit by the same rule {out['global_limit_from_this_calibration']} (configured: {cfg.check.min_similarity})")
    print(f"Per-component limits {'ACTIVE' if active else 'NOT active'}: {len(boards)} physical GOOD boards "
          f"({'>=' if active else '<'} {cfg.calibration_min_boards} needed). Written: {cfg.calibration_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
