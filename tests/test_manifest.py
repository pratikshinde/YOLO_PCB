"""Metadata validation. Builds tiny datasets in a temp dir; image files are empty stand-ins."""

import csv
from pathlib import Path

import pytest

from pcb_inspection.dataset.manifest import BOARDS, CAPTURES, DEFECTS, SESSIONS, TABLES, read_table, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ["missing_component", "damaged_pcb"]
STAGES = ["after_smt_reflow", "after_tht"]

GOOD = "raw/good/GOOD_001/GOOD_001_TOP_S20261002-01_C001.png"
DEFECTIVE = "raw/defective/DEFECT_001/DEFECT_001_TOP_S20261002-01_C001.png"
CALIBRATION = "raw/calibration/S20261002-01/S20261002-01_GEOM_01.png"


def board(board_id, **kw):
    return {"board_id": board_id, "pcb_model": "UNKNOWN", "pcb_variant": "UNKNOWN", "pcb_revision": "UNKNOWN",
            "source": "production", "verified_by": "QA-1", **kw}


def valid_tables():
    return {
        BOARDS: [board("GOOD_001"), board("DEFECT_001", source="production_reject")],
        SESSIONS: [{"session_id": "S20261002-01", "date": "2026-10-02", "station_id": "PILOT-1",
                    "camera_model": "UNKNOWN", "camera_serial": "UNKNOWN", "lens": "UNKNOWN",
                    "working_distance_mm": "UNKNOWN", "lighting_setup": "L1", "exposure_us": "UNKNOWN",
                    "gain_db": "UNKNOWN", "white_balance": "UNKNOWN", "image_format": "png", "bit_depth": "8",
                    "calibration_files": CALIBRATION, "operator_id": "OP-1"}],
        CAPTURES: [
            {"capture_file": GOOD, "condition": "good", "inspection_stage": "after_tht",
             "captured_at": "2026-10-02T10:15:00"},
            {"capture_file": DEFECTIVE, "condition": "defective", "inspection_stage": "after_smt_reflow",
             "captured_at": "2026-10-02T10:20:00", "defect_ids": "DEF_001", "reseated_before_capture": "yes"},
        ],
        DEFECTS: [{"defect_id": "DEF_001", "board_id": "DEFECT_001", "side": "TOP", "defect_type": "missing_component",
                   "component_refdes": "C12", "location_description": "C12", "origin": "production",
                   "verified_by": "QA-1", "visible_in_images": "yes"}],
    }


def write_dataset(root: Path, tables, files=(GOOD, DEFECTIVE, CALIBRATION)):
    (root / "metadata").mkdir(parents=True)
    for schema, rows in tables.items():
        with open(root / "metadata" / schema.filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=schema.columns)
            writer.writeheader()
            writer.writerows(rows)
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).touch()


def issues_for(tmp_path, tables=None, files=(GOOD, DEFECTIVE, CALIBRATION)):
    write_dataset(tmp_path, tables or valid_tables(), files)
    return validate_dataset(tmp_path, TAXONOMY, STAGES).issues


def test_shipped_templates_match_schema():
    for schema in TABLES:
        rows, issues = read_table(ROOT / "data" / "metadata" / schema.filename, schema)
        assert issues == [], schema.filename


def test_valid_dataset_has_no_issues(tmp_path):
    write_dataset(tmp_path, valid_tables())
    report = validate_dataset(tmp_path, TAXONOMY, STAGES)
    assert report.issues == []
    assert report.counts["captures.csv"] == 2


def test_missing_metadata_file_is_reported(tmp_path):
    tmp_path.joinpath("metadata").mkdir()
    assert any("not found" in i for i in validate_dataset(tmp_path, TAXONOMY, STAGES).issues)


def test_excel_bom_is_accepted(tmp_path):
    write_dataset(tmp_path, valid_tables())
    path = tmp_path / "metadata" / "boards.csv"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert validate_dataset(tmp_path, TAXONOMY, STAGES).issues == []


def test_windows_backslashes_in_paths_are_accepted(tmp_path):
    tables = valid_tables()
    tables[CAPTURES][0]["capture_file"] = GOOD.replace("/", "\\")
    assert issues_for(tmp_path, tables) == []


def mutate(table, index, **changes):
    tables = valid_tables()
    tables[table][index].update(changes)
    return tables


@pytest.mark.parametrize(
    "tables, expected",
    [
        (mutate(BOARDS, 0, pcb_revision=""), "'pcb_revision' is required"),
        (mutate(BOARDS, 0, source="from_a_friend"), "source"),
        (mutate(SESSIONS, 0, date="2026-10-03"), "does not match session_id"),
        (mutate(CAPTURES, 0, condition="defective", defect_ids="DEF_001"), "expected path"),
        (mutate(CAPTURES, 0, captured_at="2026-10-05T09:00:00"), "not on the date"),
        (mutate(CAPTURES, 1, defect_ids=""), "must list defect_ids"),
        (mutate(CAPTURES, 0, defect_ids="DEF_001"), "must not list defect_ids"),
        (mutate(CAPTURES, 1, defect_ids="DEF_099"), "DEF_099 is not in"),
        (mutate(DEFECTS, 0, side="BOT"), "belongs to DEFECT_001 BOT"),
        (mutate(DEFECTS, 0, defect_type="scratchy"), "taxonomy"),
        (mutate(DEFECTS, 0, origin="synthetic"), "origin"),
        (mutate(CAPTURES, 0, inspection_stage="final_assembly"), "inspection_stage"),
        (mutate(CAPTURES, 0, inspection_stage=""), "'inspection_stage' is required"),
    ],
)
def test_inconsistencies_are_reported(tmp_path, tables, expected):
    issues = issues_for(tmp_path, tables)
    assert any(expected in i for i in issues), issues


def test_unknown_board_and_duplicates(tmp_path):
    tables = valid_tables()
    tables[BOARDS] = [board("GOOD_001"), board("GOOD_001")]
    issues = issues_for(tmp_path, tables)
    assert any("duplicate board_id" in i for i in issues)
    assert any("DEFECT_001 is not in boards.csv" in i for i in issues)


def test_missing_and_unlisted_files(tmp_path):
    extra = "raw/good/GOOD_001/GOOD_001_TOP_S20261002-01_C002.png"
    issues = issues_for(tmp_path, files=(GOOD, extra))
    assert any(f"file not found: {DEFECTIVE}" in i for i in issues)
    assert any("calibration file not found" in i for i in issues)
    assert any(f"{extra}: file is not listed" in i for i in issues)


def test_lossy_capture_is_rejected(tmp_path):
    jpg = GOOD.replace(".png", ".jpg")
    tables = mutate(CAPTURES, 0, capture_file=jpg)
    issues = issues_for(tmp_path, tables, files=(jpg, DEFECTIVE, CALIBRATION))
    assert any("lossless" in i for i in issues)
