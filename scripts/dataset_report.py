"""Dataset report: images vs PHYSICAL boards, splits, ground truth and leakage checks.

    python scripts/dataset_report.py

Reads the board registry (data/dataset/boards.csv, image_boards.csv) and every curated image of the
acquisition sessions. Prints, per condition and split, the number of images AND of physical boards, lists
problems (images without a board, boards missing from the registry, fault boards without recorded defects,
TEST images of boards that are not in the test split, ...) and writes the derived index
data/dataset/index.csv (one row per curated image: board, condition, split, defects, session, capture, stage,
lighting, camera, resolution). Read-only apart from index.csv. Exit code 1 if there are problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.dataset.registry import build_index, counts, load_registry, write_index  # noqa: E402

DATASET = ROOT / "data" / "dataset"


def main() -> int:
    registry = load_registry(DATASET / "boards.csv", DATASET / "image_boards.csv")
    rows, issues, warnings = build_index(ROOT / "data", registry)
    write_index(rows, DATASET / "index.csv")
    print(f"Curated images: {len(rows)}   physical boards in the registry: {len(registry.boards)}\n")
    print(f"  {'condition':<10} {'split':<11} {'images':>6} {'boards':>6}   boards")
    for c in counts(rows, registry):
        print(f"  {c['condition']:<10} {c['split']:<11} {c['images']:>6} {c['physical_boards']:>6}   {c['boards']}")
    print("\n  board        condition split       images  defects (ground truth)     identity")
    for b in sorted(registry.boards.values(), key=lambda b: b.board_id):
        n = sum(r.board_id == b.board_id for r in rows)
        defects = ", ".join(f"{c} {t}" for c, t in b.defects) or "-"
        print(f"  {b.board_id:<12} {b.condition:<9} {b.split:<11} {n:>6}  {defects:<26} "
              f"{'PROVISIONAL (inferred)' if b.provisional else (b.identity_source or 'recorded')}"
              f"{'' if not b.defects else ' | defects: ' + (b.defect_source or 'source not recorded')}")
    for title, items in (("PROBLEMS", issues), ("NOTES", warnings)):
        if items:
            print(f"\n{title} ({len(items)}):")
            for item in items[:40]:
                print(f"  - {item}")
            if len(items) > 40:
                print(f"  ... and {len(items) - 40} more")
    print(f"\nIndex: {(DATASET / 'index.csv').relative_to(ROOT)}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
