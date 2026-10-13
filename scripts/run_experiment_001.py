"""Experiment 1: align good-board images to a reference image and measure their natural differences.

This is a classical baseline (no learning). It is NOT a defect detector: on good
boards every candidate region is a false candidate, and measuring them is the goal.

    python scripts/run_experiment_001.py --reference S20261002-01_C003 \
        --test S20261002-01_C001 S20261002-01_C002 S20261002-02_C001 S20261002-03_C001

--reference / --test take Experiment 0 image IDs (looked up in the capture log)
or image file paths. Outputs go to --output (default experiments/experiment_001/):

    reference/<reference_id>/   canonical reference, mm grid, reference.json (experiment-only reference)
    runs/<reference_id>__<test_id>/
        reference.jpg  test.jpg  aligned.jpg  difference.jpg  dissimilarity.jpg
        edges.jpg  thresholded.jpg  overlay_checkerboard.jpg  residuals.jpg  panel.jpg  metrics.json
    results.csv                 one row per reference/test pair

--defect-box x0,y0,x1,y1 (board mm, canonical frame) marks a HUMAN-VERIFIED defect
location for the test board; the report only states which candidate regions
overlap it. Nothing is labelled automatically. The raw dataset is never modified.
Experiment 1B (scripts/run_experiment_001b.py) uses the same pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.comparison.pipeline import compare_pair, load_pipeline_config, merge_results, prepare_reference  # noqa: E402
from pcb_inspection.dataset.experiment0 import E0_CAPTURES, IMAGE_ID_PATTERN, sha256_of  # noqa: E402
from pcb_inspection.dataset.manifest import read_table  # noqa: E402

E0_ROOT = ROOT / "data" / "experiment_0"


def resolve(arg: str, captures: list[dict[str, str]]) -> dict[str, str]:
    """Image ID or path -> {'id', 'path', 'board_id', 'sha256'}."""
    path = Path(arg)
    if IMAGE_ID_PATTERN.fullmatch(arg):
        row = next((r for r in captures if r["image_id"] == arg), None)
        if row is None:
            raise SystemExit(f"image ID {arg} is not in {E0_ROOT / 'metadata' / 'captures.csv'}")
        return {"id": arg, "path": str(E0_ROOT / row["image_file"]), "board_id": row["board_id"], "sha256": row["sha256"]}
    if not path.is_file():
        raise SystemExit(f"{arg} is neither an Experiment 0 image ID nor an existing file")
    digest = sha256_of(path)
    row = next((r for r in captures if r["sha256"] == digest), None)
    return {"id": row["image_id"] if row else path.stem, "path": str(path),
            "board_id": row["board_id"] if row else "", "sha256": digest}


def parse_box(text: str) -> tuple[float, float, float, float]:
    values = [float(v) for v in text.split(",")]
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise argparse.ArgumentTypeError("--defect-box must be x0,y0,x1,y1 in mm with x0<x1, y0<y1")
    return tuple(values)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 1: alignment + reference comparison of good boards.")
    parser.add_argument("--reference", required=True, help="Experiment 0 image ID or image path")
    parser.add_argument("--test", required=True, nargs="+", help="one or more image IDs or paths")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "experiment_001")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_001" / "config.yaml")
    parser.add_argument("--defect-box", type=parse_box, action="append", default=[],
                        help="human-verified defect box x0,y0,x1,y1 in canonical board mm (applies to all tests)")
    args = parser.parse_args()

    cfg = load_pipeline_config(ROOT, args.config)
    captures_csv = E0_ROOT / "metadata" / E0_CAPTURES.filename
    captures = read_table(captures_csv, E0_CAPTURES)[0] if captures_csv.is_file() else []

    info = resolve(args.reference, captures)
    print(f"Reference: {info['id']} (board {info['board_id'] or 'unassigned'})")
    reference = prepare_reference(info, cfg, args.output / "reference" / info["id"], "Experiment 1")

    rows = []
    for test_arg in args.test:
        test = resolve(test_arg, captures)
        row = compare_pair(reference, test, cfg, args.output / "runs" / f"{info['id']}__{test['id']}",
                           verified_boxes_mm=args.defect_box)
        rows.append(row)
        if not row["registration_ok"]:
            print(f"  {test['id']}: registration FAILED")
            continue
        print(f"  {test['id']} (board {test['board_id'] or 'unassigned'}): inliers {row['inliers']}, "
              f"RMSE {row['inlier_rmse_mm']} mm, residual median {row['residual_median_mm']} mm, "
              f"SSIM {row['ssim_mean']}, candidates {row['candidate_regions']} ({row['candidate_area_mm2']} mm2), "
              f"{row['total_seconds']} s")

    merge_results(args.output / "results.csv", rows)
    print(f"\nResults: {args.output / 'results.csv'}  ;  per-pair images and metrics: {args.output / 'runs'}")
    print("Candidate regions on good boards are false candidates by definition; nothing here is a defect verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
