"""Inspect PCB images against the GOOD reference and name missing components (Experiment 2 prototype).

    python scripts/inspect_pcb.py                       every image in data/testing/input/
    python scripts/inspect_pcb.py <image|folder> [...]  given images / folders (recursive)

For every image: frame check -> watermark removed -> PCB present? -> alignment + alignment validation ->
per-component comparison with the GOOD reference set (experiments/experiment_002/config.yaml, components.yaml,
board registry data/dataset/boards.csv). An image of a board in the GOOD reference set is judged against the
OTHER boards only (leave-one-board-out). Prints the result and writes <output>/<image name>/report.txt,
result.json, overview.jpg and, when inspected, annotated.jpg and evidence.jpg (GOOD crop next to the test crop).
With several images, summary.csv lists all results.
States: PASS, FAIL, REVIEW, INSPECTION_ERROR, NO_PCB, INVALID_FRAME, PCB_POSITION_INVALID, ALIGNMENT_FAILED.

Deterministic image processing; nothing is trained. Thresholds are experiment parameters, not
validated limits; the component map may be an unverified draft (the report says so). Input images
are only read, never moved or changed. Exit code: 0 all PASS, 1 any FAIL, 2 anything else.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.comparison.board_inspection import (  # noqa: E402
    board_of,
    inspect_image,
    load_calibration,
    load_dataset_index,
    load_gallery,
    load_inspection_config,
    load_reference_and_map,
    write_outputs,
)
from pcb_inspection.comparison.components import FAIL, MISSING, NOT_CHECKED, PASS  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            found.append(p)
        else:
            print(f"  not found: {p}")
    return found


def shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Name missing PCB components by comparison with the GOOD reference.")
    parser.add_argument("images", nargs="*", type=Path, help="images or folders (default: the input folder)")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_002" / "config.yaml")
    parser.add_argument("--output", type=Path, help="output folder (default: output_dir in the config)")
    parser.add_argument("--quiet", action="store_true", help="print only the summary table")
    args = parser.parse_args()

    cfg = load_inspection_config(ROOT, args.config)
    out_root = args.output or cfg.output_dir
    if args.images:
        images = collect([p if p.is_absolute() else Path.cwd() / p for p in args.images])
    else:
        cfg.input_dir.mkdir(parents=True, exist_ok=True)
        images = collect([cfg.input_dir])
        if not images:
            print(f"No images in {shown(cfg.input_dir)}. Copy a PCB image there and run again.")
            return 0
    reference, cmap = load_reference_and_map(cfg, ROOT / "experiments" / cfg.experiment / "reference" / cfg.reference_file.stem)
    if any(img.resolve() == cfg.reference_file.resolve() for img in images):
        print(f"  note: {cfg.reference_file.name} is the GOOD reference itself; it is skipped")
        images = [i for i in images if i.resolve() != cfg.reference_file.resolve()]
    gallery = load_gallery(cfg, reference)
    limits, limits_note = load_calibration(cfg, gallery)
    index = load_dataset_index(cfg)[0]
    print(f"GOOD reference set: {len(gallery.names)} images of {len(gallery.board_ids)} physical boards "
          f"({', '.join(gallery.board_ids) or 'no registry'}); {limits_note}")
    for message in gallery.skipped:
        print(f"  GOOD image not used: {message}")

    rows, worst = [], 0
    for path in images:
        start = time.perf_counter()
        # a board of the GOOD reference set is judged as a NEW board: all its images are left out of the examples
        board = board_of(path.stem, index)
        exclude = board if board in gallery.board_ids else None
        result = inspect_image(path, reference, cmap, cfg, gallery, exclude, limits, limits_note)
        out_dir = out_root / path.stem
        write_outputs(result, reference, cmap, cfg, out_dir)
        seconds = time.perf_counter() - start
        if not args.quiet:
            print(result.report)
            print(f"Outputs: {shown(out_dir)}  ({seconds:.1f} s)\n")
        missing = [r for r in result.components if r.status == MISSING]
        present = [r for r in result.components if r.status == "present" and r.score is not None]
        rows.append({
            "image": shown(path), "board_id": board, "status": result.status,
            "pcb": result.stages.get("pcb", ""), "alignment": result.stages.get("alignment", ""),
            "missing": ";".join(r.refdes for r in missing),
            "scores": ";".join(f"{r.refdes}={r.score:.2f}" for r in missing),
            "not_checked": ";".join(r.refdes for r in result.components if r.status == NOT_CHECKED),
            "max_present_score": max((r.score for r in present), default=""),
            "max_present_component": max(((r.score, r.refdes) for r in present), default=(None, ""))[1],
            "reason": result.reason, "seconds": round(seconds, 1),
        })
        worst = max(worst, {PASS: 0, FAIL: 1}.get(result.status, 2))

    if len(rows) > 1:
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{'image':<46} {'board':<11} {'status':<21} {'missing (score)':<20} highest difference among present parts")
        for r in rows:
            name = Path(r["image"]).name[-45:]
            detail = r["scores"] or (r["reason"][:60] if r["status"] not in (PASS, FAIL) else "-")
            print(f"{name:<46} {r['board_id'] or '-':<11} {r['status']:<21} {detail:<20} "
                  f"{r['max_present_score'] if r['max_present_score'] != '' else '-'} {r['max_present_component']}")
        print(f"\nSummary: {shown(out_root / 'summary.csv')}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
