"""Experiment 0 capture-log validation. Tiny logs in a temp dir; the image files are small byte stand-ins."""

import csv
from pathlib import Path

import pytest

from pcb_inspection.dataset.experiment0 import (
    E0_BOARDS,
    E0_CAPTURES,
    E0_DEFECTS,
    E0_TABLES,
    parse_defect_box,
    sha256_of,
    validate_experiment0,
)
from pcb_inspection.dataset.manifest import read_table

ROOT = Path(__file__).resolve().parents[1]
KWARGS = dict(
    pcb_model="LG Indicator", variant="1", inspection_stages=["after_smt_reflow", "after_tht"],
    defect_types=["missing_component", "damaged_pcb"], reported_colours=["blue", "red", "green"],
    image_formats=["jpg", "jpeg", "png", "tif", "tiff"], board_size_mm=(83, 116),
)


def capture(n, board="", **kw):
    image_id = f"S20261002-01_C{n:03d}"
    folder = board or "unassigned"
    return {"image_id": image_id, "image_file": f"images/{folder}/S20261002-01/{image_id}.jpg", "session_id": "S20261002-01",
            "captured_at": f"2026-10-02T15:10:{n:02d}", "captured_at_source": "exif", "camera_device": "phone X",
            "width": "6144", "height": "8160", "digital_zoom": "1", "flash_fired": "no",
            "exposure_time_s": "0.01", "iso": "100", "focal_length_mm": "5.5",
            "original_filename": f"IMG_{n}.jpg", "sha256": "", "pcb_model": "LG Indicator", "variant": "1",
            "colour": "red", "board_id": board, "inspection_stage": "after_tht", "camera_distance_mm": "300",
            "lighting_condition": "ceiling LED", "orientation": "", "notes": "", **kw}


def valid_tables():
    return {
        E0_BOARDS: [{"board_id": "DEFECT_001", "condition": "defective", "colour": "red", "verified_by": "OP-1",
                     "ground_truth_source": "reject tag"}],
        E0_CAPTURES: [capture(1), capture(2), capture(3, board="DEFECT_001")],
        E0_DEFECTS: [{"defect_id": "DEF_001", "board_id": "DEFECT_001", "defect_type": "missing_component",
                      "defect_location": "upper right", "defect_description": "resistor missing",
                      "ground_truth_source": "reject tag", "verified_by": "OP-1", "visible_in_images": ""}],
    }


def write_log(root: Path, tables, create_files=True):
    if create_files:  # each stand-in file gets distinct bytes, and its checksum goes into the log
        for row in tables[E0_CAPTURES]:
            path = root / row["image_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row["image_id"].encode())
            row["sha256"] = row["sha256"] or sha256_of(path)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    for schema, rows in tables.items():
        with open(root / "metadata" / schema.filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=schema.columns)
            writer.writeheader()
            writer.writerows(rows)


def run(tmp_path, tables=None, create_files=True):
    write_log(tmp_path, tables or valid_tables(), create_files)
    return validate_experiment0(tmp_path, **KWARGS)


def mutate(table, index, **changes):
    tables = valid_tables()
    tables[table][index].update(changes)
    return tables


def test_shipped_templates_match_schema():
    # 2026-10-06: Experiment 0 (handheld phone photos) is closed; its data incl. these tables is archived
    # (archive/*_pre_collection_reset/data/experiment_0/). The tables are checked wherever they are.
    folders = [ROOT / "data" / "experiment_0" / "metadata",
               *sorted((ROOT / "archive").glob("*/data/experiment_0/metadata"))]
    folders = [f for f in folders if f.is_dir()]
    if not folders:
        pytest.skip("no Experiment 0 metadata in this checkout")
    for folder in folders:
        for schema in E0_TABLES:
            _, issues = read_table(folder / schema.filename, schema)
            assert issues == [], f"{folder}/{schema.filename}"


def test_defect_box_is_optional_and_parsed(tmp_path):
    tables = valid_tables()
    tables[E0_DEFECTS][0]["defect_box_mm"] = "30.5,40,36,46.5"
    assert run(tmp_path, tables).issues == []
    assert parse_defect_box("30.5,40,36,46.5") == (30.5, 40.0, 36.0, 46.5)


def test_valid_log(tmp_path):
    report = run(tmp_path)
    assert report.issues == []
    assert report.counts["captures.csv"] == 3


def test_optional_fields_may_be_empty(tmp_path):
    tables = valid_tables()
    tables[E0_CAPTURES][2] = capture(3)  # stored under images/unassigned/
    for row in tables[E0_CAPTURES]:
        row.update(board_id="", inspection_stage="unknown", camera_distance_mm="UNKNOWN", lighting_condition="",
                   camera_device="", digital_zoom="", flash_fired="", colour="", orientation="", notes="")
    report = run(tmp_path, tables)
    assert report.issues == []
    text = "\n".join(report.warnings)
    assert "3 image(s): inspection_stage unknown" in text
    assert "3 image(s): no board_id" in text


@pytest.mark.parametrize(
    "tables, expected",
    [
        (mutate(E0_CAPTURES, 0, image_id="GOOD_001_C001"), "must look like S20261002-01_C001"),
        (mutate(E0_CAPTURES, 0, session_id="S20261002-02"), "does not match image_id"),
        (mutate(E0_CAPTURES, 0, image_file="images/GOOD_001.jpg"), "image_file must be"),
        (mutate(E0_CAPTURES, 0, image_file="images/S20261002-01/S20261002-01_C001.jpg"), "image_file must be"),
        (mutate(E0_CAPTURES, 2, board_id="DEFECT_002"), "image_file must be images/DEFECT_002/"),
        (mutate(E0_CAPTURES, 0, iso="high"), "iso"),
        (mutate(E0_CAPTURES, 0, captured_at="2026-10-04T09:00:00"), "not on the date of session"),
        (mutate(E0_CAPTURES, 0, pcb_model="Other PCB"), "pcb_model"),
        (mutate(E0_CAPTURES, 0, variant="2"), "variant"),
        (mutate(E0_CAPTURES, 0, inspection_stage="final"), "inspection_stage"),
        (mutate(E0_CAPTURES, 0, camera_distance_mm="35 cm"), "camera_distance_mm"),
        (mutate(E0_CAPTURES, 0, board_id="BOARD_1"), "board_id"),
        (mutate(E0_CAPTURES, 0, orientation="upside down"), "orientation"),
        (mutate(E0_CAPTURES, 0, width="wide"), "width"),
        (mutate(E0_CAPTURES, 0, sha256="abc"), "64-character"),
        (mutate(E0_CAPTURES, 0, original_filename=""), "'original_filename' is required"),
        (mutate(E0_DEFECTS, 0, defect_type="scratchy"), "taxonomy"),
        (mutate(E0_DEFECTS, 0, defect_box_mm="30,40,20,50"), "defect_box_mm"),
        (mutate(E0_DEFECTS, 0, defect_box_mm="30,40"), "defect_box_mm"),
        (mutate(E0_DEFECTS, 0, defect_box_mm="70,100,90,130"), "outside the board"),
        (mutate(E0_BOARDS, 0, condition="broken"), "condition"),
    ],
)
def test_issues(tmp_path, tables, expected):
    issues = run(tmp_path, tables).issues
    assert any(expected in i for i in issues), issues


def test_changed_image_is_detected(tmp_path):
    write_log(tmp_path, valid_tables())
    (tmp_path / "images" / "unassigned" / "S20261002-01" / "S20261002-01_C001.jpg").write_bytes(b"edited")
    issues = validate_experiment0(tmp_path, **KWARGS).issues
    assert any("has changed since it was ingested" in i for i in issues)


def test_duplicate_image_is_detected(tmp_path):
    tables = valid_tables()
    write_log(tmp_path, tables)
    tables[E0_CAPTURES][1]["sha256"] = tables[E0_CAPTURES][0]["sha256"]
    write_log(tmp_path, tables, create_files=False)
    assert any("same image file as" in i for i in validate_experiment0(tmp_path, **KWARGS).issues)


def test_missing_and_not_ingested_images(tmp_path):
    write_log(tmp_path, valid_tables(), create_files=False)
    stray = tmp_path / "images" / "GOOD_001.jpg"
    stray.parent.mkdir(parents=True)
    stray.touch()
    issues = validate_experiment0(tmp_path, **KWARGS).issues
    assert any("file not found" in i for i in issues)
    assert any("images/GOOD_001.jpg: not ingested" in i for i in issues)


def test_board_added_later_to_unassigned_image_needs_no_file_move(tmp_path):
    tables = valid_tables()
    write_log(tmp_path, tables)                       # file written under images/unassigned/
    tables[E0_CAPTURES][0]["board_id"] = "DEFECT_001"  # board assigned afterwards in captures.csv only
    write_log(tmp_path, tables, create_files=False)
    assert validate_experiment0(tmp_path, **KWARGS).issues == []


def test_missing_metadata_files_count_as_empty(tmp_path):
    assert validate_experiment0(tmp_path, **KWARGS).issues == []
    (tmp_path / "metadata").mkdir()
    with open(tmp_path / "metadata" / "captures.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=E0_CAPTURES.columns).writeheader()
    assert validate_experiment0(tmp_path, **KWARGS).issues == []   # no boards.csv / defects.csv


def test_loose_duplicate_is_a_note_but_unregistered_file_is_an_issue(tmp_path):
    write_log(tmp_path, valid_tables())
    registered = tmp_path / "images" / "unassigned" / "S20261002-01" / "S20261002-01_C001.jpg"
    (tmp_path / "images" / "copy_of_c001.jpg").write_bytes(registered.read_bytes())
    (tmp_path / "images" / "unknown.jpg").write_bytes(b"never ingested")
    report = validate_experiment0(tmp_path, **KWARGS)
    assert [i for i in report.issues if "copy_of_c001" in i] == []
    assert any("copy_of_c001.jpg: byte-identical duplicate of registered image S20261002-01_C001" in w
               for w in report.warnings)
    assert any("unknown.jpg: not ingested" in i for i in report.issues)


def test_board_condition_and_defects_must_agree(tmp_path):
    tables = valid_tables()
    tables[E0_BOARDS].append({"board_id": "GOOD_001", "condition": "good", "colour": "red", "verified_by": "OP-1",
                              "ground_truth_source": "operator"})
    tables[E0_DEFECTS][0]["board_id"] = "GOOD_001"
    issues = run(tmp_path, tables).issues
    assert any("defective board DEFECT_001 has no row" in i for i in issues)
    assert any("good board GOOD_001 has defects" in i for i in issues)


def test_session_setup_and_capture_warnings(tmp_path):
    tables = valid_tables()
    tables[E0_CAPTURES][1].update(camera_distance_mm="500", digital_zoom="2", flash_fired="yes",
                                  captured_at_source="file_time", colour="black")
    tables[E0_CAPTURES][0]["board_id"] = "GOOD_009"  # assigned later to an unassigned image
    report = run(tmp_path, tables)
    assert report.issues == []
    text = "\n".join(report.warnings)
    for expected in ("camera_distance_mm differs", "digital zoom used", "flash fired", "no EXIF capture time",
                     "not among the reported colours", "board GOOD_009 has no ground-truth row"):
        assert expected in text
