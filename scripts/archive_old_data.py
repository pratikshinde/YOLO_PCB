"""Archive (never delete) the old experimental image data before the clean collection phase.

    python scripts/archive_old_data.py            dry run: list what would be archived, nothing is moved
    python scripts/archive_old_data.py --apply    move it to archive/<date>_pre_collection_reset/

Moves the old GOOD / DEFECT / TEST captures, golden / reference images, synthetic frames and stale acquisition /
inspection data to archive/<date>_pre_collection_reset/<same relative path>, then writes there:
    manifest.csv   one row per archived file: original path, archived path, size, SHA-256, kind
    README.md      what was archived, when and why
Every file is checked after the move (present, same size); the counts are compared with the listing taken before.
Source code, configs, tests and documentation are never touched. Moves within the same disk: nothing is copied or
re-encoded. The production-track metadata templates (data/metadata/*.csv) are re-created empty (header only).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.dataset.experiment0 import sha256_of  # noqa: E402
from pcb_inspection.dataset.manifest import TABLES  # noqa: E402

# (path relative to the project root, why)
TARGETS = [
    ("data/acquisition", "Experiment 1B originals: old GOOD / DEFECT / TEST sessions (DroidCam watermark over the PCB)"),
    ("data/curated", "Experiment 1B KEEP copies of the same captures; held the watermarked golden reference"),
    ("data/quarantine", "Experiment 1B REJECT copies"),
    ("data/captures", "old station captures (S20261005-11 .. -16)"),
    ("data/experiment_0", "Experiment 0 phone photos (handheld, not from the live camera) + their logs"),
    ("data/inspections", "old operator-station sessions incl. 2 SYNTHETIC frames labelled GOOD_002"),
    ("data/testing", "Experiment 2 input / output and live_inspection.py captures (no board, no ground truth)"),
    ("data/products", "old station inspection results / reviews"),
    ("data/processed", "derived caches of the old data (aligned golden-bank images, checksums)"),
    ("data/dataset", "old physical-board registry (GOOD_001-003, GOOD_901/902, DEFECT_901/902): stale identities"),
    ("data/metadata", "production-track metadata with stale Experiment 0 rows (templates re-created empty)"),
    ("experiments/experiment_001/reference", "Experiment 1 reference image"),
    ("experiments/experiment_001/runs", "Experiment 1 result images of the old photos"),
    ("experiments/experiment_002/reference", "Experiment 2 golden reference (watermark over LED4)"),
    ("experiments/experiment_002/runs", "Experiment 2 results on the old images"),
    ("experiments/experiment_002/runs_v1_bare_board", "Experiment 2 run 1 results on the old images"),
    ("experiments/experiment_002/reference_calibration.json", "per-region limits measured on the watermarked bank"),
    ("experiments/experiment_002/droidcam_watermark_mask.png", "superseded fixed watermark mask of one old session"),
]
METADATA_DIR = "data/metadata"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RECORD_EXTS = {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml"}


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    return "image" if ext in IMAGE_EXTS else "record" if ext in RECORD_EXTS else "other"


def files_of(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []


def has_rows(table: Path) -> bool:
    """A metadata CSV with at least one line after its header (an empty template is not old data)."""
    with table.open(encoding="utf-8-sig") as f:
        return sum(1 for line in f if line.strip()) > 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive the old image data (never deletes).")
    parser.add_argument("--apply", action="store_true", help="really move the files (default: dry run)")
    parser.add_argument("--name", default=f"{date.today().isoformat()}_pre_collection_reset",
                        help="archive folder name under archive/")
    args = parser.parse_args()
    return run(ROOT, args.name, args.apply)


def run(root: Path, name: str, apply: bool) -> int:
    archive = root / "archive" / name
    plan = []
    for rel, why in TARGETS:
        src = root / rel
        if not src.exists():
            continue
        if rel == METADATA_DIR:     # file by file: only tables with rows (empty templates stay)
            plan += [(f.relative_to(root).as_posix(), why, f, [f]) for f in files_of(src)
                     if f.suffix.lower() != ".csv" or has_rows(f)]
        elif files_of(src):
            plan.append((rel, why, src, files_of(src)))
    total = Counter()
    print(f"{'Path':<58}{'files':>7}{'images':>8}{'MB':>9}")
    for rel, _, _, files in plan:
        n_img = sum(kind_of(f) == "image" for f in files)
        mb = sum(f.stat().st_size for f in files) / 1e6
        total.update(files=len(files), images=n_img, mb=mb)
        print(f"{rel:<58}{len(files):>7}{n_img:>8}{mb:>9.1f}")
    print(f"{'TOTAL':<58}{total['files']:>7}{total['images']:>8}{total['mb']:>9.1f}")
    if not plan:
        print("Nothing to archive.")
        return 0
    if not apply:
        print(f"\nDry run. To move all of this to {archive.relative_to(root).as_posix()}/ run:\n"
              "  python scripts/archive_old_data.py --apply")
        return 0
    if archive.exists() and any(archive.iterdir()):
        print(f"{archive} already exists and is not empty; choose another --name. Nothing moved.")
        return 1

    before = {f.relative_to(root).as_posix(): f.stat().st_size for _, _, _, files in plan for f in files}
    for rel, _, src, _ in plan:
        dst = archive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)                     # same disk: a rename, the content is not touched
    rows, problems = [], []
    for original, size in sorted(before.items()):
        moved = archive / original
        if not moved.is_file():
            problems.append(f"{original}: not found in the archive")
            continue
        if moved.stat().st_size != size:
            problems.append(f"{original}: size changed during the move")
        rows.append({"original_path": original, "archived_path": moved.relative_to(root).as_posix(),
                     "size_bytes": size, "sha256": sha256_of(moved), "kind": kind_of(moved)})
    with (archive / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["original_path"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = root / METADATA_DIR                  # production-track templates: header only, no stale rows
    metadata.mkdir(parents=True, exist_ok=True)
    for schema in TABLES:
        path = metadata / schema.filename
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(schema.columns)

    kinds = Counter(r["kind"] for r in rows)
    lines = [f"# Archive {name}", "",
             "Old experimental image data, moved here (never deleted) before the clean GOOD / FAULT collection phase",
             "(docs/clean_data_collection.md). **Do not use it as dataset or golden reference data:** the DroidCam",
             "watermark lies over the PCB in 94 of the 96 kept / station frames (measured 2026-10-06 with the new capture check),",
             "the golden reference hid LED4, two frames are synthetic, and the board identities are stale.", "",
             f"Archived: {len(rows)} files ({kinds['image']} images, {kinds['record']} records, {kinds['other']} other), "
             f"{sum(r['size_bytes'] for r in rows) / 1e6:.1f} MB. Every file: `manifest.csv` (original path, size, SHA-256).",
             "Paths below this folder are the original project-relative paths.", "",
             "| Original path | Files | Why |", "|---|---|---|"]
    for rel, why, _, files in plan:
        lines.append(f"| `{rel}` | {len(files)} | {why} |")
    lines += ["", "To restore a folder, move it back to its original path (nothing else refers to this archive)."]
    (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nArchived {len(rows)} files ({kinds['image']} images) to {archive.relative_to(root).as_posix()}/")
    print("Re-created empty production-track templates in data/metadata/.")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("Verified: every file is in the archive with its original size; manifest.csv lists its SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
