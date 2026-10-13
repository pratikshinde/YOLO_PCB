"""Former operator inspection script: replaced (2026-10-05) by the product-profile station.

The earlier version of this script built its GOOD baseline from raw, unaligned photos, filled it with SYNTHETIC
boards labelled GOOD_001 / GOOD_002 / GOOD_003 when fewer than two boards were found, and, when the camera
failed, silently inspected SYNTHETIC frames and stored them as real captures (data/inspections/sessions/
S20261005_161630 holds two such frames; they are kept, and reported as SYNTHETIC by
`python scripts/pcb.py validate`). It also asked the operator for dataset type and board ID during inspection.

Now:
    live inspection (system decides, no operator input)   python scripts/station.py
    same, without a window                                 python scripts/station.py --headless --captures 5
    data collection (GOOD / DEFECT / TEST boards)          python scripts/acquire.py --product lg_indicator
    inspect stored images                                  python scripts/pcb.py inspect <image>

This wrapper forwards --camera-index and --count to the station (headless). --simulate no longer exists: a
camera failure is reported and logged, never replaced by a generated image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Replaced: forwards to scripts/station.py --headless.")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--count", type=int, default=1, help="number of automatic captures")
    parser.add_argument("--dataset-type", default=None, help="no longer used in inspection: use scripts/acquire.py")
    parser.add_argument("--board-id", default=None, help="no longer used in inspection: use scripts/acquire.py")
    parser.add_argument("--simulate", action="store_true", help="removed (synthetic frames are never stored as captures)")
    args, _ = parser.parse_known_args()
    if args.simulate:
        print("--simulate was removed: synthetic frames must never be stored or inspected as real captures. "
              "Tests use a mock camera (tests/test_product_architecture.py).")
        return 2
    if args.dataset_type or args.board_id:
        print("Dataset type and board ID are data-collection inputs: use  python scripts/acquire.py --product <id>. "
              "Live inspection needs no operator input.")
        return 2
    import station
    sys.argv = ["station.py", "--headless", "--captures", str(args.count)] + (
        ["--camera-index", str(args.camera_index)] if args.camera_index is not None else [])
    return station.main()


if __name__ == "__main__":
    sys.exit(main())
