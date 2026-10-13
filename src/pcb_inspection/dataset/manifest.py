"""Dataset metadata tables (data/metadata/*.csv) and their validation.

The column definitions below are the single source of truth for the tables
described in docs/data_collection_plan.md, section 6.1. The templates shipped in
data/metadata/ contain exactly these headers and no rows.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path, PurePosixPath

from .naming import (
    BOARD_ID_PATTERN,
    CONDITIONS,
    SIDES,
    parse_capture_filename,
    raw_capture_path,
    session_date,
)

BOARD_SOURCES = ("production", "production_reject", "failed_archive", "engineering")
DEFECT_ORIGINS = ("production", "controlled")
VISIBILITY = ("yes", "no", "uncertain")
YES_NO = ("yes", "no")
DEFECT_ID_PATTERN = re.compile(r"DEF_\d{3}")


@dataclass(frozen=True)
class TableSchema:
    filename: str
    columns: tuple[str, ...]
    required: frozenset[str]


BOARDS = TableSchema(
    "boards.csv",
    columns=(
        "board_id", "serial_number", "pcb_model", "pcb_variant", "pcb_revision", "board_colour",
        "production_lot", "manufacture_date", "source", "electrical_test_result",
        "verified_by", "notes",
    ),
    required=frozenset({"board_id", "pcb_model", "pcb_variant", "pcb_revision", "source", "verified_by"}),
)

SESSIONS = TableSchema(
    "sessions.csv",
    columns=(
        "session_id", "date", "station_id", "camera_model", "camera_serial", "lens",
        "working_distance_mm", "lighting_setup", "exposure_us", "gain_db", "aperture_f",
        "white_balance", "image_format", "bit_depth", "calibration_files", "operator_id", "notes",
    ),
    required=frozenset({
        "session_id", "date", "station_id", "camera_model", "camera_serial", "lens",
        "working_distance_mm", "lighting_setup", "exposure_us", "gain_db", "white_balance",
        "image_format", "bit_depth", "calibration_files", "operator_id",
    }),
)

CAPTURES = TableSchema(
    "captures.csv",
    columns=(
        "capture_file", "condition", "inspection_stage", "captured_at", "defect_ids",
        "reseated_before_capture", "sha256", "notes",
    ),
    required=frozenset({"capture_file", "condition", "inspection_stage", "captured_at"}),
)

DEFECTS = TableSchema(
    "defects.csv",
    columns=(
        "defect_id", "board_id", "side", "defect_type", "component_refdes",
        "location_description", "severity", "origin", "verified_by", "visible_in_images",
        "electrical_test_result", "notes",
    ),
    required=frozenset({
        "defect_id", "board_id", "side", "defect_type", "location_description", "origin", "verified_by",
    }),
)

TABLES = (BOARDS, SESSIONS, CAPTURES, DEFECTS)

Row = dict[str, str]


@dataclass
class ValidationReport:
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def read_table(path: Path, schema: TableSchema) -> tuple[list[Row], list[str]]:
    """Read one metadata CSV. Returns (rows, header issues). Accepts Excel's UTF-8 BOM."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        issues = []
        missing = [c for c in schema.columns if c not in header]
        unexpected = [c for c in header if c not in schema.columns]
        if missing:
            issues.append(f"{schema.filename}: missing columns {missing}")
        if unexpected:
            issues.append(f"{schema.filename}: unexpected columns {unexpected}")
        rows = [{col: (row.get(col) or "").strip() for col in schema.columns} for row in reader]
    return rows, issues


def read_tables(metadata_dir: Path, schemas: Collection[TableSchema], report: ValidationReport) -> dict[str, list[Row]]:
    """Read several tables, recording missing files, header problems and empty required cells."""
    tables: dict[str, list[Row]] = {}
    for schema in schemas:
        path = metadata_dir / schema.filename
        if not path.is_file():
            report.issues.append(f"{schema.filename}: not found in {metadata_dir}")
            tables[schema.filename] = []
            continue
        rows, header_issues = read_table(path, schema)
        report.issues += header_issues
        for line, row in enumerate(rows, start=2):
            for col in schema.columns:
                if col in schema.required and not row[col]:
                    report.issues.append(f"{schema.filename}:{line}: '{col}' is required")
        tables[schema.filename] = rows
        report.counts[schema.filename] = len(rows)
    return tables


def validate_dataset(
    data_root: Path, defect_types: Collection[str], inspection_stages: Collection[str]
) -> ValidationReport:
    """Check metadata tables against each other, the naming rules, and the files in data/raw/."""
    report = ValidationReport()
    tables = read_tables(data_root / "metadata", TABLES, report)

    boards = index_rows(tables[BOARDS.filename], "board_id", BOARDS.filename, report)
    sessions = index_rows(tables[SESSIONS.filename], "session_id", SESSIONS.filename, report)
    defects = index_rows(tables[DEFECTS.filename], "defect_id", DEFECTS.filename, report)

    _check_boards(boards, report)
    _check_sessions(sessions, data_root, report)
    _check_defects(defects, boards, set(defect_types), report)
    _check_captures(tables[CAPTURES.filename], boards, sessions, defects, set(inspection_stages), data_root, report)
    return report


def index_rows(rows: list[Row], key: str, table: str, report: ValidationReport) -> dict[str, tuple[int, Row]]:
    index: dict[str, tuple[int, Row]] = {}
    for line, row in enumerate(rows, start=2):
        value = row[key]
        if not value:
            continue
        if value in index:
            report.issues.append(f"{table}:{line}: duplicate {key} {value} (first on line {index[value][0]})")
            continue
        index[value] = (line, row)
    return index


def _check_boards(boards: dict[str, tuple[int, Row]], report: ValidationReport) -> None:
    for board_id, (line, row) in boards.items():
        where = f"{BOARDS.filename}:{line}"
        if not BOARD_ID_PATTERN.fullmatch(board_id):
            report.issues.append(f"{where}: invalid board_id {board_id!r}, expected e.g. GOOD_001 or DEFECT_001")
        if row["source"] and row["source"] not in BOARD_SOURCES:
            report.issues.append(f"{where}: source {row['source']!r} must be one of {BOARD_SOURCES}")


def _check_sessions(sessions: dict[str, tuple[int, Row]], data_root: Path, report: ValidationReport) -> None:
    for session_id, (line, row) in sessions.items():
        where = f"{SESSIONS.filename}:{line}"
        try:
            expected_date = session_date(session_id)
        except ValueError as e:
            report.issues.append(f"{where}: {e}")
            continue
        if row["date"]:
            try:
                if date.fromisoformat(row["date"]) != expected_date:
                    report.issues.append(f"{where}: date {row['date']} does not match session_id {session_id}")
            except ValueError:
                report.issues.append(f"{where}: date {row['date']!r} is not YYYY-MM-DD")
        for rel in _split_list(row["calibration_files"]):
            if not (data_root / rel).is_file():
                report.issues.append(f"{where}: calibration file not found: {rel}")


def _check_defects(
    defects: dict[str, tuple[int, Row]],
    boards: dict[str, tuple[int, Row]],
    defect_types: set[str],
    report: ValidationReport,
) -> None:
    for defect_id, (line, row) in defects.items():
        where = f"{DEFECTS.filename}:{line}"
        if not DEFECT_ID_PATTERN.fullmatch(defect_id):
            report.issues.append(f"{where}: invalid defect_id {defect_id!r}, expected e.g. DEF_001")
        if row["board_id"] and row["board_id"] not in boards:
            report.issues.append(f"{where}: board_id {row['board_id']} is not in {BOARDS.filename}")
        if row["side"] and row["side"] not in SIDES:
            report.issues.append(f"{where}: side {row['side']!r} must be one of {SIDES}")
        if row["defect_type"] and row["defect_type"] not in defect_types:
            report.issues.append(f"{where}: defect_type {row['defect_type']!r} is not in the configured taxonomy")
        if row["origin"] and row["origin"] not in DEFECT_ORIGINS:
            report.issues.append(f"{where}: origin {row['origin']!r} must be one of {DEFECT_ORIGINS}")
        if row["visible_in_images"] and row["visible_in_images"] not in VISIBILITY:
            report.issues.append(f"{where}: visible_in_images must be one of {VISIBILITY} or empty")


def _check_captures(
    rows: list[Row],
    boards: dict[str, tuple[int, Row]],
    sessions: dict[str, tuple[int, Row]],
    defects: dict[str, tuple[int, Row]],
    inspection_stages: set[str],
    data_root: Path,
    report: ValidationReport,
) -> None:
    listed: set[str] = set()
    for line, row in enumerate(rows, start=2):
        where = f"{CAPTURES.filename}:{line}"
        rel = row["capture_file"].replace("\\", "/")
        if not rel:
            continue
        if rel in listed:
            report.issues.append(f"{where}: duplicate capture_file {rel}")
            continue
        listed.add(rel)
        try:
            name = parse_capture_filename(PurePosixPath(rel).name)
        except ValueError as e:
            report.issues.append(f"{where}: {e}")
            continue

        condition = row["condition"]
        if condition not in CONDITIONS:
            report.issues.append(f"{where}: condition {condition!r} must be one of {CONDITIONS}")
        elif PurePosixPath(rel) != raw_capture_path(condition, name):
            report.issues.append(f"{where}: expected path {raw_capture_path(condition, name)}, got {rel}")
        if name.board_id not in boards:
            report.issues.append(f"{where}: board {name.board_id} is not in {BOARDS.filename}")
        if name.session_id not in sessions:
            report.issues.append(f"{where}: session {name.session_id} is not in {SESSIONS.filename}")
        if not (data_root / rel).is_file():
            report.issues.append(f"{where}: file not found: {rel}")
        if row["inspection_stage"] and row["inspection_stage"] not in inspection_stages:
            report.issues.append(
                f"{where}: inspection_stage {row['inspection_stage']!r} must be one of {sorted(inspection_stages)}"
            )
        if row["captured_at"]:
            try:
                if datetime.fromisoformat(row["captured_at"]).date() != session_date(name.session_id):
                    report.issues.append(f"{where}: captured_at is not on the date of {name.session_id}")
            except ValueError:
                report.issues.append(f"{where}: captured_at {row['captured_at']!r} is not ISO 8601")
        if row["reseated_before_capture"] and row["reseated_before_capture"] not in YES_NO:
            report.issues.append(f"{where}: reseated_before_capture must be one of {YES_NO} or empty")

        defect_ids = _split_list(row["defect_ids"])
        if condition == "defective" and not defect_ids:
            report.issues.append(f"{where}: defective capture must list defect_ids")
        if condition == "good" and defect_ids:
            report.issues.append(f"{where}: good capture must not list defect_ids")
        for defect_id in defect_ids:
            if defect_id not in defects:
                report.issues.append(f"{where}: defect {defect_id} is not in {DEFECTS.filename}")
                continue
            defect = defects[defect_id][1]
            if (defect["board_id"], defect["side"]) != (name.board_id, name.side):
                report.issues.append(
                    f"{where}: defect {defect_id} belongs to {defect['board_id']} {defect['side']}, "
                    f"not {name.board_id} {name.side}"
                )

    for condition in CONDITIONS:
        base = data_root / "raw" / condition
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                rel = path.relative_to(data_root).as_posix()
                if rel not in listed:
                    report.issues.append(f"{rel}: file is not listed in {CAPTURES.filename}")


def _split_list(value: str) -> list[str]:
    return [item.strip().replace("\\", "/") for item in value.split(";") if item.strip()]
