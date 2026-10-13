"""Camera-first ingest. Small JPEGs with EXIF are generated in a temp inbox (test inputs, not PCB images)."""

import csv
import os
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

import pcb_inspection.acquisition.ingest as ingest
from pcb_inspection.acquisition.ingest import commit_ingest, plan_ingest, read_capture_info
from pcb_inspection.dataset.experiment0 import E0_CAPTURES, sha256_of, validate_experiment0
from pcb_inspection.dataset.manifest import read_table

FORMATS = ["jpg", "jpeg", "png", "tif", "tiff"]
DEFAULTS = {"pcb_model": "LG Indicator", "variant": "1", "inspection_stage": "unknown", "camera_distance_mm": "UNKNOWN"}
VALIDATE = dict(pcb_model="LG Indicator", variant="1", inspection_stages=["after_smt_reflow", "after_tht"],
                defect_types=["missing_component"], reported_colours=["red"], image_formats=FORMATS)


def photo(path: Path, taken: str | None = "2026:10:02 15:10:51", size=(60, 40), orientation=None, colour=(200, 30, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, colour)
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Xiaomi", "Redmi 12 5G"
    if orientation:
        exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    if taken:
        sub[0x9003] = taken
    sub[0xA404], sub[0x9209] = 1.0, 16      # digital zoom 1x; flash did not fire
    sub[0x829A], sub[0x8827], sub[0x920A] = 0.01, 100, 5.5  # exposure 1/100 s, ISO 100, focal length 5.5 mm
    img.save(path, "JPEG", exif=exif)
    return path


def test_read_capture_info(tmp_path):
    info = read_capture_info(photo(tmp_path / "a.jpg", orientation=6))
    assert info["captured_at"] == "2026-10-02T15:10:51" and info["captured_at_source"] == "exif"
    assert info["camera_device"] == "Xiaomi Redmi 12 5G"
    assert (info["width"], info["height"]) == ("40", "60")  # EXIF rotation applied
    assert info["digital_zoom"] == "1" and info["flash_fired"] == "no"
    assert (info["exposure_time_s"], info["iso"], info["focal_length_mm"]) == ("0.01", "100", "5.5")


def test_missing_exif_falls_back_to_file_time(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (30, 20), (0, 0, 200)).save(path)  # no EXIF at all
    stamp = datetime(2026, 10, 3, 9, 30, 0).timestamp()
    os.utime(path, (stamp, stamp))
    info = read_capture_info(path)
    assert info["captured_at"] == "2026-10-03T09:30:00" and info["captured_at_source"] == "file_time"
    assert info["camera_device"] == "" and info["iso"] == ""


def test_plan_ids_in_capture_order_and_board_folders(tmp_path):
    inbox = tmp_path / "inbox"
    photo(inbox / "IMG_late.jpg", taken="2026:10:02 15:10:51", colour=(10, 200, 20))
    photo(inbox / "IMG_early.JPG", taken="2026:10:02 15:08:36", colour=(10, 20, 200))
    photo(inbox / "GOOD_001" / "IMG_x.jpg", taken="2026:10:02 15:09:00")
    photo(inbox / "IMG_next_day.jpg", taken="2026:10:03 08:00:00", colour=(90, 90, 200))
    plan = plan_ingest(inbox, [], FORMATS, DEFAULTS)
    assert plan.skipped == []
    assert [(c.row["original_filename"], c.row["image_id"], c.image_file) for c in plan.captures] == [
        ("IMG_early.JPG", "S20261002-01_C001", "images/unassigned/S20261002-01/S20261002-01_C001.jpg"),
        ("IMG_x.jpg", "S20261002-01_C002", "images/GOOD_001/S20261002-01/S20261002-01_C002.jpg"),
        ("IMG_late.jpg", "S20261002-01_C003", "images/unassigned/S20261002-01/S20261002-01_C003.jpg"),
        ("IMG_next_day.jpg", "S20261003-01_C001", "images/unassigned/S20261003-01/S20261003-01_C001.jpg"),
    ]
    rows = [c.row for c in plan.captures]
    assert [r["board_id"] for r in rows] == ["", "GOOD_001", "", ""]
    assert all(r["pcb_model"] == "LG Indicator" and r["inspection_stage"] == "unknown" for r in rows)
    assert all(r["sha256"] == sha256_of(c.source) for r, c in zip(rows, plan.captures))


def test_board_option_applies_to_inbox_root_but_folders_take_precedence(tmp_path):
    inbox = tmp_path / "inbox"
    photo(inbox / "root.jpg", colour=(1, 2, 200))
    photo(inbox / "DEFECT_001" / "sub.jpg", colour=(200, 2, 1))
    plan = plan_ingest(inbox, [], FORMATS, {**DEFAULTS, "board_id": "GOOD_001"})
    assert {c.row["original_filename"]: c.row["board_id"] for c in plan.captures} == {
        "root.jpg": "GOOD_001", "sub.jpg": "DEFECT_001"}
    assert {c.row["original_filename"]: c.image_file.split("/")[1] for c in plan.captures} == {
        "root.jpg": "GOOD_001", "sub.jpg": "DEFECT_001"}


def test_plan_skips_unsupported_duplicates_bad_and_reserved_folders(tmp_path):
    inbox = tmp_path / "inbox"
    first = photo(inbox / "a.jpg")
    (inbox / "b.jpg").write_bytes(first.read_bytes())
    (inbox / "c.heic").write_bytes(b"not supported")
    (inbox / "d.jpg").write_text("not an image")
    photo(inbox / "my photos" / "e.jpg", colour=(5, 5, 250))
    photo(inbox / "_ingested" / "S20261002-01" / "old.jpg", colour=(5, 250, 5))  # reserved: ignored silently
    plan = plan_ingest(inbox, [], FORMATS, DEFAULTS)
    assert [c.row["original_filename"] for c in plan.captures] == ["a.jpg"]
    text = "\n".join(plan.skipped)
    assert "identical to an image already ingested" in text
    assert "c.heic: format" in text
    assert "d.jpg: cannot read image" in text
    assert "my photos/: sub-folder name is not a board ID" in text
    assert "_ingested" not in text


def test_commit_copies_unchanged_parks_staged_file_and_appends_log(tmp_path):
    inbox, root = tmp_path / "inbox", tmp_path / "e0"
    original = photo(inbox / "GOOD_001" / "IMG_1.jpg")
    original_bytes, original_mtime = original.read_bytes(), original.stat().st_mtime
    plan = plan_ingest(inbox, [], FORMATS, DEFAULTS)
    commit_ingest(plan, root)

    stored = root / "images" / "GOOD_001" / "S20261002-01" / "S20261002-01_C001.jpg"
    assert stored.read_bytes() == original_bytes                         # byte-identical copy
    assert abs(stored.stat().st_mtime - original_mtime) < 2              # timestamps preserved
    parked = inbox / "_ingested" / "S20261002-01" / "GOOD_001" / "IMG_1.jpg"
    assert parked.read_bytes() == original_bytes and not original.exists()  # staged file kept, not deleted
    rows, issues = read_table(root / "metadata" / "captures.csv", E0_CAPTURES)
    assert issues == [] and rows[0]["board_id"] == "GOOD_001" and rows[0]["original_filename"] == "IMG_1.jpg"

    # second run, same day: new session, appended; a re-copied photo is recognised and skipped
    photo(inbox / "IMG_2.jpg", taken="2026:10:02 16:00:00", colour=(0, 0, 255))
    (inbox / "copy_of_1.jpg").write_bytes(original_bytes)
    plan2 = plan_ingest(inbox, rows, FORMATS, DEFAULTS)
    assert [c.row["image_id"] for c in plan2.captures] == ["S20261002-02_C001"]
    assert any("identical" in s for s in plan2.skipped)
    commit_ingest(plan2, root)
    with open(root / "metadata" / "captures.csv", newline="", encoding="utf-8") as f:
        assert [r["image_id"] for r in csv.DictReader(f)] == ["S20261002-01_C001", "S20261002-02_C001"]

    report = validate_experiment0(root, **VALIDATE)
    assert report.issues == []


def test_corrupted_copy_is_removed_and_not_logged(tmp_path, monkeypatch):
    inbox, root = tmp_path / "inbox", tmp_path / "e0"
    source = photo(inbox / "IMG_1.jpg")
    plan = plan_ingest(inbox, [], FORMATS, DEFAULTS)
    monkeypatch.setattr(ingest.shutil, "copy2", lambda src, dst: Path(dst).write_bytes(b"corrupted"))
    with pytest.raises(OSError, match="does not match the original"):
        commit_ingest(plan, root)
    assert source.exists()                                               # staged file untouched
    assert not (root / plan.captures[0].image_file).exists()
    rows, _ = read_table(root / "metadata" / "captures.csv", E0_CAPTURES)
    assert rows == []


def test_existing_target_is_never_overwritten(tmp_path):
    inbox, root = tmp_path / "inbox", tmp_path / "e0"
    photo(inbox / "IMG_1.jpg")
    plan = plan_ingest(inbox, [], FORMATS, DEFAULTS)
    target = root / plan.captures[0].image_file
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")
    with pytest.raises(FileExistsError):
        commit_ingest(plan, root)
    assert target.read_bytes() == b"already here"
