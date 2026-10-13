"""Validate data/metadata/*.csv against the naming rules and the files in data/raw/.

This is the production-representative dataset (frozen station setup). The
Experiment 0 phone images are checked with scripts/check_e0_images.py instead.
Run after every capture session (docs/data_collection_plan.md, section 9).
Exit code 1 if any issue is found.

Usage:  python scripts/validate_dataset.py [--config configs/project.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.config import load_config, load_profile  # noqa: E402
from pcb_inspection.dataset.manifest import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset metadata and raw files.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    parser.add_argument("--good", action="store_true", help="Discover and validate all GOOD PCB data across sources")
    args = parser.parse_args()

    config = load_config(args.config)
    profile = load_profile(ROOT / config["profiles"]["directory"], config["profiles"]["active"])
    data_root = ROOT / config["data"]["root"]

    if args.good:
        from pcb_inspection.dataset.discovery import discover_good_dataset, format_good_dataset_summary
        good_report = discover_good_dataset(data_root)
        print(format_good_dataset_summary(good_report))
        return 0 if not good_report.issues and not good_report.invalid_images else 1

    report = validate_dataset(
        data_root, config["defects"]["taxonomy"], profile["inspection_stages"]["confirmed"]
    )

    for table, count in report.counts.items():
        print(f"{table:<14} {count} rows")
    if report.ok:
        print("\nOK: no issues found.")
        return 0
    print(f"\n{len(report.issues)} issue(s):")
    for issue in report.issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
