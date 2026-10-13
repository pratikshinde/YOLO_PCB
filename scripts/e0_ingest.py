"""Experiment 0: ingest phone photos from the inbox. Camera-first: nothing is typed per image.

    1. Copy the ORIGINAL phone photos into data/experiment_0/inbox/
       (optional: into inbox/GOOD_001/ etc., or use --board, to tag the physical board).
    2. Run:  python scripts/e0_ingest.py
       Optional session context, given once for all photos of this run:
           --stage after_smt_reflow|after_tht  --distance-mm 300  --lighting "ceiling LED"  --board GOOD_001

The script assigns session and image IDs, reads capture time / phone / zoom /
flash / exposure / ISO / focal length from EXIF, computes a checksum, estimates
the board colour, runs the image-quality check, copies each original byte-for-byte
(verified) to images/<board_id|unassigned>/<session_id>/, appends
metadata/captures.csv, updates reports/quality_report.csv and validates the log.
Staged inbox files are then parked in inbox/_ingested/<session_id>/ (never deleted).
Use --dry-run to see the plan without copying or writing anything.

Exit code 1 if any image is flagged or the capture log has issues.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.acquisition.ingest import commit_ingest, plan_ingest  # noqa: E402
from pcb_inspection.config import load_config, load_profile  # noqa: E402
from pcb_inspection.dataset.experiment0 import E0_CAPTURES, UNKNOWN_STAGE, validate_with_configs  # noqa: E402
from pcb_inspection.dataset.manifest import read_table  # noqa: E402
from pcb_inspection.dataset.naming import BOARD_ID_PATTERN  # noqa: E402
from pcb_inspection.image_quality.checks import QualityParams, check_image, flag_relative_sharpness  # noqa: E402
from pcb_inspection.image_quality.report import merge_report, print_table, quality_row  # noqa: E402


def shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Experiment 0 phone photos from the inbox.")
    parser.add_argument("--stage", help="inspection stage of the boards in this batch (default: unknown)")
    parser.add_argument("--distance-mm", type=float, help="lens-to-board distance in mm, if measured")
    parser.add_argument("--lighting", default="", help="short description of the lighting, e.g. 'ceiling LED'")
    parser.add_argument("--board", default="", help="board ID for photos directly in the inbox, e.g. GOOD_001 (inbox/<board>/ sub-folders take precedence)")
    parser.add_argument("--notes", default="", help="free text stored with every photo of this run")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; copy, move and write nothing")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    parser.add_argument("--capture-config", type=Path, default=ROOT / "configs" / "capture" / "mobile_phone_e0.yaml")
    args = parser.parse_args()

    project = load_config(args.config)
    capture = load_config(args.capture_config)
    profile = load_profile(ROOT / project["profiles"]["directory"], capture["profile"])
    root, inbox = ROOT / capture["data_dir"], ROOT / capture["inbox_dir"]
    stages = profile["inspection_stages"]["confirmed"]
    if args.stage and args.stage not in stages:
        parser.error(f"--stage must be one of {stages}")
    if args.board and not BOARD_ID_PATTERN.fullmatch(args.board):
        parser.error("--board must look like GOOD_001 or DEFECT_001")
    if args.distance_mm is not None and args.distance_mm <= 0:
        parser.error("--distance-mm must be positive")

    captures_path = root / "metadata" / E0_CAPTURES.filename
    existing = read_table(captures_path, E0_CAPTURES)[0] if captures_path.is_file() else []
    defaults = {
        "pcb_model": profile["pcb"]["model"],
        "variant": profile["pcb"]["variant_label"],
        "inspection_stage": args.stage or UNKNOWN_STAGE,
        "camera_distance_mm": f"{args.distance_mm:g}" if args.distance_mm is not None else "UNKNOWN",
        "lighting_condition": args.lighting,
        "board_id": args.board,
        "notes": args.notes,
    }
    plan = plan_ingest(inbox, existing, capture["allowed_image_formats"], defaults)
    for message in plan.skipped:
        print(f"  SKIPPED {message}")
    if not plan.captures:
        print(f"Nothing to ingest in {shown(inbox)} (copy the original phone photos there).")
        return 0

    print(f"Checking {len(plan.captures)} new photo(s)...")
    params = QualityParams.from_mapping(capture["quality_check"])
    dims = profile["pcb"]["dimensions_mm"]
    results, per_session = [], defaultdict(list)
    for planned in plan.captures:
        quality = check_image(planned.source, params, (dims["long_side"], dims["short_side"]))
        planned.row["colour"] = quality.colour_estimate
        results.append((planned, quality))
        per_session[planned.row["session_id"]].append(quality)
    flag_relative_sharpness(per_session, params.relative_sharpness_min)
    rows = [quality_row(p.row["image_id"], p.row["session_id"], p.row["board_id"], q) for p, q in results]

    print("\n== New photos ==")
    print(f"  {'original file':<30} -> image_id")
    for planned, _ in results:
        r = planned.row
        print(f"  {r['original_filename']:<30} -> {r['image_id']}  ({r['captured_at']}, {r['camera_device'] or 'device unknown'}"
              f"{', board ' + r['board_id'] if r['board_id'] else ''})")
    print()
    print_table(rows)
    if args.dry_run:
        print("\nDry run: nothing was copied, moved or written.")
        return 0

    commit_ingest(plan, root)
    report_path = root / "reports" / "quality_report.csv"
    merge_report(report_path, rows)
    validation = validate_with_configs(root, project, profile, capture)
    folders = sorted({str(Path(p.image_file).parent.as_posix()) for p in plan.captures})
    print(f"\nIngested {len(plan.captures)} photo(s) (byte-identical copies, checksum verified):")
    for folder in folders:
        print(f"  images -> {shown(root / folder)}")
    print(f"  log    -> {shown(captures_path)}")
    print(f"  report -> {shown(report_path)}")
    print(f"  staged originals parked in {shown(inbox / '_ingested')} (delete them yourself when no longer needed)")
    for issue in validation.issues:
        print(f"  ISSUE   {issue}")
    for warning in validation.warnings:
        print(f"  NOTE    {warning}")
    flagged = sum(1 for _, q in results if q.flags)
    print(f"\n{flagged} of {len(results)} new photo(s) flagged. Open flagged photos at full resolution "
          "and re-take them if needed. Flags are prompts, not a verdict on the board.")
    return 1 if validation.issues or flagged else 0


if __name__ == "__main__":
    sys.exit(main())
