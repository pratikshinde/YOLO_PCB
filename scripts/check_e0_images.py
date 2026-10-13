"""Re-check every ingested Experiment 0 photo and its capture log.

New photos are ingested (and checked) with scripts/e0_ingest.py. This script
re-validates the whole log (IDs, files, checksums, profile values) and re-runs
the image-quality check on every listed photo, e.g. after thresholds in
configs/capture/mobile_phone_e0.yaml were adjusted. It rewrites
data/experiment_0/reports/quality_report.csv.

Exit code 1 if there are log issues, missing / unreadable images, or flagged images.

Usage:  python scripts/check_e0_images.py [--capture-config configs/capture/mobile_phone_e0.yaml]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.config import load_config, load_profile  # noqa: E402
from pcb_inspection.dataset.experiment0 import E0_CAPTURES, validate_with_configs  # noqa: E402
from pcb_inspection.dataset.manifest import read_table  # noqa: E402
from pcb_inspection.image_quality.checks import QualityParams, check_image, flag_relative_sharpness  # noqa: E402
from pcb_inspection.image_quality.report import merge_report, print_table, quality_row  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-check Experiment 0 photos and capture log.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    parser.add_argument("--capture-config", type=Path, default=ROOT / "configs" / "capture" / "mobile_phone_e0.yaml")
    args = parser.parse_args()

    project = load_config(args.config)
    capture = load_config(args.capture_config)
    profile = load_profile(ROOT / project["profiles"]["directory"], capture["profile"])
    root = ROOT / capture["data_dir"]

    report = validate_with_configs(root, project, profile, capture)
    print("== Capture log ==")
    for table, count in report.counts.items():
        print(f"  {table:<14} {count} rows")
    for issue in report.issues:
        print(f"  ISSUE   {issue}")
    for warning in report.warnings:
        print(f"  NOTE    {warning}")

    captures_path = root / "metadata" / E0_CAPTURES.filename
    rows = [r for r in (read_table(captures_path, E0_CAPTURES)[0] if captures_path.is_file() else []) if r["image_file"]]
    if not rows:
        print("\nNo photos ingested yet: copy originals to data/experiment_0/inbox/ and run scripts/e0_ingest.py")
        return 1 if report.issues else 0

    params = QualityParams.from_mapping(capture["quality_check"])
    dims = profile["pcb"]["dimensions_mm"]
    results, per_session = [], defaultdict(list)
    for row in rows:
        quality = check_image(root / row["image_file"].replace("\\", "/"), params, (dims["long_side"], dims["short_side"]))
        results.append((row, quality))
        per_session[row["session_id"]].append(quality)
    flag_relative_sharpness(per_session, params.relative_sharpness_min)
    report_rows = [quality_row(r["image_id"], r["session_id"], r["board_id"], q) for r, q in results]

    report_path = root / "reports" / "quality_report.csv"
    merge_report(report_path, report_rows)
    print(f"\n== Photos ({len(results)}) ==")
    print_table(report_rows)
    flagged = sum(1 for _, q in results if q.flags)
    shown = report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path
    print(f"\n{flagged} of {len(results)} photo(s) flagged. Report: {shown}")
    print("Flags are prompts to look at the photo at full resolution; px/mm is approximate.")
    return 1 if report.issues or flagged else 0


if __name__ == "__main__":
    sys.exit(main())
