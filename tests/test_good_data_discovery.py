"""Regression tests for GOOD PCB dataset discovery, registration, and deduplication.

Since the 2026-10-06 reset the old real data lives in archive/<date>_pre_collection_reset/ with its original
project-relative paths, so the real-data tests below read that archive (read-only) as a project root. They test
the legacy discovery code on the legacy data; nothing here makes the archived images dataset data again. They are
skipped when no archive is present.

Tests all 12 critical requirements:
1. Existing GOOD_001 data is still detected.
2. A new GOOD_002 board is detected.
3. GOOD_001 + GOOD_002 are both included.
4. Multiple sessions for GOOD_001 are included.
5. Duplicate images are not counted twice.
6. A new image with a different checksum is detected.
7. TEST data is not included in GOOD.
8. DEFECT data is not included in GOOD.
9. Unknown/invalid board IDs are reported.
10. Existing Experiment 0 data remains valid.
11. Experiment 1B acquisition sessions are discoverable.
12. Board-level grouping remains correct.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pcb_inspection.dataset.experiment0 import sha256_of
from pcb_inspection.dataset.discovery import (
    GoodDatasetReport,
    discover_good_dataset,
    format_good_dataset_summary,
)
from pcb_inspection.dataset.registry import (
    build_index,
    counts,
    curated_images,
    load_registry,
    reference_images,
)
from pcb_inspection.dataset.splitting import assign_captures, find_leakage, split_boards
from run_experiment_001b import load_all_captures

LEGACY = next(iter(sorted((ROOT / "archive").glob("*_pre_collection_reset"))), None)   # archived pre-reset data
LEGACY_DATA = LEGACY / "data" if LEGACY else ROOT / "archive" / "missing"
legacy_data = pytest.mark.skipif(not (LEGACY_DATA / "curated").is_dir(),
                                 reason="pre-reset data not archived in this checkout (archive/*_pre_collection_reset)")


# =========================================================================
# 1. Existing GOOD_001 data is still detected
# =========================================================================
@legacy_data
def test_1_existing_good_001_detected():
    report = discover_good_dataset(LEGACY_DATA)
    assert "GOOD_001" in report.boards, "GOOD_001 must be detected in the good dataset"
    board = report.boards["GOOD_001"]
    assert len(board.images) >= 4, f"Expected at least 4 images for GOOD_001, got {len(board.images)}"
    assert board.condition == "good"
    assert board.split == "reference"


# =========================================================================
# 2. A new GOOD_002 board is detected
# =========================================================================
@legacy_data
def test_2_new_good_002_board_detected():
    report = discover_good_dataset(LEGACY_DATA)
    assert "GOOD_002" in report.boards, "GOOD_002 must be detected as a distinct physical board"
    board = report.boards["GOOD_002"]
    assert len(board.images) >= 1
    assert board.condition == "good"


# =========================================================================
# 3. GOOD_001 + GOOD_002 are both included
# =========================================================================
@legacy_data
def test_3_good_001_and_good_002_both_included():
    report = discover_good_dataset(LEGACY_DATA)
    assert "GOOD_001" in report.boards
    assert "GOOD_002" in report.boards
    assert "GOOD_003" in report.boards
    # Verify boards are independent physical entities
    assert report.boards["GOOD_001"] is not report.boards["GOOD_002"]
    assert report.total_good_boards >= 3


# =========================================================================
# 4. Multiple sessions for GOOD_001 are included
# =========================================================================
@legacy_data
def test_4_multiple_sessions_for_good_001_included():
    report = discover_good_dataset(LEGACY_DATA)
    board = report.boards["GOOD_001"]
    session_ids = set(board.sessions.keys())
    assert "S20261002-01" in session_ids, "Session 01 must be included for GOOD_001"
    assert "S20261002-02" in session_ids, "Session 02 must be included for GOOD_001"
    assert len(session_ids) >= 2, "GOOD_001 must have multiple sessions"


# =========================================================================
# 5. Duplicate images are not counted twice (SHA-256 deduplication)
# =========================================================================
def test_5_duplicate_images_not_counted_twice(tmp_path):
    # Create a synthetic dataset directory with an image and an identical duplicate
    data_dir = tmp_path / "data"
    curated_dir = data_dir / "curated" / "good" / "GOOD_001" / "S20261005-01"
    curated_dir.mkdir(parents=True)
    img_data = np.full((100, 100, 3), 120, dtype=np.uint8)
    img_file1 = curated_dir / "img1.png"
    cv2.imwrite(str(img_file1), img_data)

    # Identical copy in acquisition
    acq_dir = data_dir / "acquisition" / "good" / "GOOD_001" / "S20261005-01" / "images"
    acq_dir.mkdir(parents=True)
    img_file2 = acq_dir / "img1.png"
    cv2.imwrite(str(img_file2), img_data)

    # session.json in acquisition
    sess_json = data_dir / "acquisition" / "good" / "GOOD_001" / "S20261005-01" / "session.json"
    h = sha256_of(img_file1)
    sess_json.write_text(json.dumps({
        "session_id": "S20261005-01",
        "board_id": "GOOD_001",
        "stage": "after_tht",
        "images": [{
            "image_id": "img1",
            "saved_file": "curated/good/GOOD_001/S20261005-01/img1.png",
            "original_file": "acquisition/good/GOOD_001/S20261005-01/images/img1.png",
            "sha256": h,
            "review": {"status": "kept"},
        }]
    }), encoding="utf-8")

    # boards.csv
    dataset_dir = data_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "boards.csv").write_text(
        "board_id,condition,split,defects,defect_source,identity_source,verified_by,notes\n"
        "GOOD_001,good,reference,,,test,tester,\n",
        encoding="utf-8"
    )

    report = discover_good_dataset(data_dir)
    assert report.total_good_images == 1, "Duplicate image must only be counted once"
    assert report.duplicate_image_count >= 1, "Duplicate instance must be recorded"


# =========================================================================
# 6. A new image with a different checksum is detected
# =========================================================================
def test_6_new_image_with_different_checksum_detected(tmp_path):
    data_dir = tmp_path / "data"
    curated_dir = data_dir / "curated" / "good" / "GOOD_001" / "S20261005-01"
    curated_dir.mkdir(parents=True)

    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(curated_dir / "img1.png"), img1)
    cv2.imwrite(str(curated_dir / "img2.png"), img2)

    dataset_dir = data_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "boards.csv").write_text(
        "board_id,condition,split,defects,defect_source,identity_source,verified_by,notes\n"
        "GOOD_001,good,reference,,,test,tester,\n",
        encoding="utf-8"
    )

    report = discover_good_dataset(data_dir)
    assert report.total_good_images == 2, "Images with different checksums must both be registered"


# =========================================================================
# 7. TEST data is not included in GOOD
# =========================================================================
@legacy_data
def test_7_test_data_not_included_in_good():
    report = discover_good_dataset(LEGACY_DATA)
    for img in report.unique_images:
        assert img.dataset_type != "test", f"Test image {img.image_id} must not be in GOOD dataset"
        assert not img.image_id.startswith("TEST_"), f"TEST image {img.image_id} found in GOOD dataset"


# =========================================================================
# 8. DEFECT data is not included in GOOD
# =========================================================================
@legacy_data
def test_8_defect_data_not_included_in_good():
    report = discover_good_dataset(LEGACY_DATA)
    for img in report.unique_images:
        assert img.dataset_type != "defect", f"Defect image {img.image_id} must not be in GOOD dataset"
        assert not img.image_id.startswith("DEFECT_"), f"DEFECT image {img.image_id} found in GOOD dataset"
    assert "DEFECT_901" not in report.boards
    assert "DEFECT_902" not in report.boards


# =========================================================================
# 9. Unknown/invalid board IDs are reported
# =========================================================================
def test_9_unknown_or_invalid_board_ids_reported(tmp_path):
    data_dir = tmp_path / "data"
    curated_dir = data_dir / "curated" / "good" / "INVALID_NAME" / "S20261005-01"
    curated_dir.mkdir(parents=True)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(curated_dir / "test.png"), img)

    dataset_dir = data_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "boards.csv").write_text(
        "board_id,condition,split,defects,defect_source,identity_source,verified_by,notes\n"
        "GOOD_001,good,reference,,,test,tester,\n",
        encoding="utf-8"
    )

    report = discover_good_dataset(data_dir)
    assert report.missing_metadata_count > 0 or len(report.unknown_board_ids) > 0, \
        "Invalid board folder without registry mapping must be reported as missing metadata / unknown"


# =========================================================================
# 10. Existing Experiment 0 data remains valid
# =========================================================================
@legacy_data
def test_10_existing_experiment_0_data_remains_valid():
    captures_path = LEGACY_DATA / "experiment_0" / "metadata" / "captures.csv"
    assert captures_path.is_file(), "Experiment 0 captures.csv must exist"
    with captures_path.open(newline="", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 5, "Experiment 0 has 5 documented captures"
    for r in reader:
        img_file = LEGACY_DATA / "experiment_0" / r["image_file"]
        assert img_file.is_file(), f"Experiment 0 image file missing: {img_file}"
        digest = sha256_of(img_file)
        assert digest == r["sha256"], f"SHA256 mismatch for historical Experiment 0 image {r['image_id']}"


# =========================================================================
# 11. Experiment 1B acquisition sessions are discoverable
# =========================================================================
@legacy_data
def test_11_experiment_1b_acquisition_sessions_discoverable():
    captures = load_all_captures(LEGACY)
    session_ids = {c["session_id"] for c in captures}
    assert "S20261002-01" in session_ids, "Experiment 0 session must be discoverable"
    assert "S20261005-01" in session_ids, "Acquisition session S20261005-01 must be discoverable"
    assert "S20261005-05" in session_ids, "Acquisition session S20261005-05 must be discoverable"
    # Verify image files exist on disk
    for c in captures:
        assert Path(c["full_path"]).is_file(), f"Capture file {c['full_path']} does not exist"


# =========================================================================
# 12. Board-level grouping remains correct
# =========================================================================
@legacy_data
def test_12_board_level_grouping_remains_correct():
    report = discover_good_dataset(LEGACY_DATA)
    assert report.total_good_boards >= 3
    # Check that images for GOOD_001 all belong to GOOD_001
    for img in report.boards["GOOD_001"].images:
        assert img.board_id == "GOOD_001"
    # Check that images for GOOD_003 all belong to GOOD_003
    for img in report.boards["GOOD_003"].images:
        assert img.board_id == "GOOD_003"

    # Verify splitting by board works properly and avoids data leakage
    board_strata = {b: "good" for b in report.boards}
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    board_split = split_boards(board_strata, ratios, seed=42)

    # Map each discovered image to split via its physical board ID
    capture_boards = {img.image_id: img.board_id for img in report.unique_images if img.board_id}
    capture_splits = assign_captures(capture_boards, board_split)
    leakage = find_leakage(capture_splits, capture_boards)
    assert not leakage, f"Data leakage detected across splits: {leakage}"
