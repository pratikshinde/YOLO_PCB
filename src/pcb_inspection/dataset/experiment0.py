"""Experiment 0 (mobile-phone visibility study): capture log and its validation.

DEVELOPMENT / DATASET MODE ONLY. The final operator workflow (place PCB ->
camera -> PASS / FAIL / REVIEW) needs none of this metadata to be typed.

Camera-first: photos are dropped into the inbox and ingested by
`scripts/e0_ingest.py` (pcb_inspection.acquisition.ingest), which fills the
capture log automatically. Layout (docs/experiment_0_camera_visibility.md):

    data/experiment_0/
        inbox/                                  drop original phone photos here (inbox/GOOD_001/ = board)
        inbox/_ingested/<session_id>/           staged files after a verified copy (nothing is deleted)
        images/<board_id|unassigned>/<session_id>/<image_id>.<ext>
                                                e.g. images/GOOD_001/S20261002-01/S20261002-01_C001.jpg
        reference/                              optional scale / orientation photos
        metadata/captures.csv                   one row per image (written by the ingest)
        metadata/boards.csv                     optional: board ground truth (good / defective)
        metadata/defects.csv                    one row per verified defect
        reports/quality_report.csv              written by the quality check

image_id = <session_id>_<capture_id> and does not depend on the board. A board
ID can be added later to an "unassigned" image by editing captures.csv only; the
file may stay in images/unassigned/.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import TableSchema, ValidationReport, index_rows, read_tables
from .naming import BOARD_ID_PATTERN, CONDITIONS, session_date

E0_BOARDS = TableSchema(
    "boards.csv",
    columns=("board_id", "condition", "colour", "verified_by", "ground_truth_source", "notes"),
    required=frozenset({"board_id", "condition", "verified_by", "ground_truth_source"}),
)

# Automatic columns are filled by the ingest; the rest are optional session-level context.
E0_AUTOMATIC_COLUMNS = (
    "image_id", "image_file", "session_id", "captured_at", "captured_at_source", "camera_device",
    "width", "height", "digital_zoom", "flash_fired", "exposure_time_s", "iso", "focal_length_mm",
    "original_filename", "sha256",
    "pcb_model", "variant", "colour",
)
E0_OPTIONAL_COLUMNS = (
    "board_id", "inspection_stage", "camera_distance_mm", "lighting_condition", "orientation", "notes",
)
E0_CAPTURES = TableSchema(
    "captures.csv",
    columns=E0_AUTOMATIC_COLUMNS + E0_OPTIONAL_COLUMNS,
    required=frozenset({
        "image_id", "image_file", "session_id", "captured_at", "captured_at_source", "width", "height",
        "original_filename", "sha256", "pcb_model", "variant",
    }),
)

E0_DEFECTS = TableSchema(
    "defects.csv",
    columns=(
        "defect_id", "board_id", "defect_type", "defect_location", "defect_description",
        "ground_truth_source", "verified_by", "visible_in_images", "defect_box_mm",
    ),
    required=frozenset({
        "defect_id", "board_id", "defect_type", "defect_location", "defect_description",
        "ground_truth_source", "verified_by",
    }),
)

E0_TABLES = (E0_BOARDS, E0_CAPTURES, E0_DEFECTS)

# defect_box_mm: optional, "x0,y0,x1,y1" in mm on the board, origin at the top-left board
# corner as the board appears in the reference orientation (read off reference_mm_grid.jpg).
# It is human ground truth; it is never derived from difference images.

ORIENTATIONS = ("ref", "rot90", "rot180", "rot270")  # relative to an orientation reference photo
UNKNOWN_STAGE = "unknown"
UNASSIGNED_BOARD_FOLDER = "unassigned"   # images/unassigned/<session_id>/: no board ID at ingest
VISIBILITY = ("yes", "partial", "no")
SESSION_SETUP_FIELDS = ("camera_device", "camera_distance_mm", "lighting_condition")

IMAGE_ID_PATTERN = re.compile(r"(?P<session_id>S\d{8}-\d{2})_(?P<capture_id>C\d{3})")
DEFECT_ID_PATTERN = re.compile(r"DEF_\d{3}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_experiment0(
    root: Path,
    *,
    pcb_model: str,
    variant: str,
    inspection_stages: Collection[str],
    defect_types: Collection[str],
    reported_colours: Collection[str],
    image_formats: Collection[str],
    verify_checksums: bool = True,
    board_size_mm: tuple[float, float] | None = None,
) -> ValidationReport:
    """Validate the Experiment 0 capture log against the profile, the image files and their checksums."""
    report = ValidationReport()
    metadata = root / "metadata"
    # all three tables are optional: a missing file means "nothing recorded yet", not an error
    present = [s for s in E0_TABLES if (metadata / s.filename).is_file()]
    tables = {s.filename: [] for s in E0_TABLES} | read_tables(metadata, present, report)
    boards = index_rows(tables[E0_BOARDS.filename], "board_id", E0_BOARDS.filename, report)
    defects = index_rows(tables[E0_DEFECTS.filename], "defect_id", E0_DEFECTS.filename, report)
    _check_boards_and_defects(boards, defects, defect_types, reported_colours, report, board_size_mm)

    stages, formats = set(inspection_stages), {f.lower() for f in image_formats}
    listed: set[str] = set()
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    counts: Counter[str] = Counter()
    setups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for line, row in enumerate(tables[E0_CAPTURES.filename], start=2):
        where = f"{E0_CAPTURES.filename}:{line}"
        image_id = row["image_id"]
        match = IMAGE_ID_PATTERN.fullmatch(image_id)
        if match is None:
            report.issues.append(f"{where}: image_id {image_id!r} must look like S20261002-01_C001 (generated by the ingest)")
            continue
        if image_id in seen_ids:
            report.issues.append(f"{where}: duplicate image_id {image_id}")
            continue
        seen_ids.add(image_id)
        session_id = match["session_id"]
        if row["session_id"] and row["session_id"] != session_id:
            report.issues.append(f"{where}: session_id {row['session_id']} does not match image_id {image_id}")

        rel = row["image_file"].replace("\\", "/")
        if rel:
            listed.add(rel)
            path = PurePosixPath(rel)
            folders = {UNASSIGNED_BOARD_FOLDER} | ({row["board_id"]} if row["board_id"] else set())
            if path.stem != image_id or path.parent.parent.parent != PurePosixPath("images")                     or path.parent.name != session_id or path.parent.parent.name not in folders:
                expected = f"images/{row['board_id'] or UNASSIGNED_BOARD_FOLDER}/{session_id}/{image_id}.<ext>"
                report.issues.append(f"{where}: image_file must be {expected}, got {rel}")
            if path.suffix.lower().lstrip(".") not in formats:
                report.issues.append(f"{where}: format {path.suffix!r} not allowed; use one of {sorted(formats)}")
            if not (root / rel).is_file():
                report.issues.append(f"{where}: file not found: {rel}")
            elif verify_checksums and SHA256_PATTERN.fullmatch(row["sha256"]):
                if sha256_of(root / rel) != row["sha256"]:
                    report.issues.append(f"{where}: {rel} has changed since it was ingested (checksum differs)")
        if row["sha256"]:
            if not SHA256_PATTERN.fullmatch(row["sha256"]):
                report.issues.append(f"{where}: sha256 is not a 64-character hex checksum")
            elif row["sha256"] in seen_hashes:
                report.issues.append(f"{where}: same image file as {seen_hashes[row['sha256']]}")
            else:
                seen_hashes[row["sha256"]] = image_id

        if row["captured_at"]:
            try:
                if datetime.fromisoformat(row["captured_at"]).date() != session_date(session_id):
                    report.issues.append(f"{where}: captured_at is not on the date of session {session_id}")
            except ValueError:
                report.issues.append(f"{where}: captured_at {row['captured_at']!r} is not ISO 8601")
        if row["captured_at_source"] == "file_time":
            counts["no EXIF capture time (file time used)"] += 1
        for name in ("width", "height"):
            if row[name] and not (row[name].isdigit() and int(row[name]) > 0):
                report.issues.append(f"{where}: {name} must be a positive integer")
        for name in ("exposure_time_s", "iso", "focal_length_mm"):
            if row[name]:
                try:
                    float(row[name])
                except ValueError:
                    report.issues.append(f"{where}: {name} {row[name]!r} is not a number")
        if row["digital_zoom"]:
            try:
                if float(row["digital_zoom"]) > 1.0:
                    counts["digital zoom used"] += 1
            except ValueError:
                report.issues.append(f"{where}: digital_zoom {row['digital_zoom']!r} is not a number")
        if row["flash_fired"] == "yes":
            counts["flash fired"] += 1

        if row["pcb_model"] and row["pcb_model"] != pcb_model:
            report.issues.append(f"{where}: pcb_model {row['pcb_model']!r} must be {pcb_model!r}")
        if row["variant"] and row["variant"] != variant:
            report.issues.append(f"{where}: variant {row['variant']!r} must be {variant!r}")
        if row["colour"] and row["colour"] not in reported_colours:
            counts[f"colour {row['colour']!r} not among the reported colours"] += 1

        board_id = row["board_id"]
        if not board_id:
            counts["no board_id (fine for visibility checks; needed later for board-level splits)"] += 1
        elif not BOARD_ID_PATTERN.fullmatch(board_id):
            report.issues.append(f"{where}: board_id {board_id!r} must look like GOOD_001 or DEFECT_001")
        elif board_id not in boards:
            counts[f"board {board_id} has no ground-truth row in {E0_BOARDS.filename}"] += 1

        stage = row["inspection_stage"]
        if stage in ("", UNKNOWN_STAGE):
            counts["inspection_stage unknown"] += 1
        elif stage not in stages:
            report.issues.append(f"{where}: inspection_stage {stage!r} must be one of {sorted(stages | {UNKNOWN_STAGE})}")
        _check_distance(row["camera_distance_mm"], where, report, counts)
        if row["orientation"] and row["orientation"] not in ORIENTATIONS:
            report.issues.append(f"{where}: orientation {row['orientation']!r} must be one of {ORIENTATIONS} or empty")
        for field_name in SESSION_SETUP_FIELDS:
            setups[session_id][field_name].add(row[field_name])

    for what, n in sorted(counts.items()):
        report.warnings.append(f"{n} image(s): {what}")
    for session_id, values in sorted(setups.items()):
        for field_name, distinct in values.items():
            if len(distinct) > 1:
                report.warnings.append(
                    f"session {session_id}: {field_name} differs between images {sorted(distinct)}; "
                    "one session should use one setup"
                )

    images_dir = root / "images"
    if images_dir.is_dir():
        for path in sorted(images_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                rel = path.relative_to(root).as_posix()
                if rel in listed:
                    continue
                registered_as = seen_hashes.get(sha256_of(path))
                if registered_as:  # resolved: the same bytes are already ingested
                    report.warnings.append(
                        f"{rel}: byte-identical duplicate of registered image {registered_as}; "
                        "it can be moved out of images/ (e.g. to set_aside/)"
                    )
                else:
                    report.issues.append(f"{rel}: not ingested; copy it into inbox/ and run scripts/e0_ingest.py")
    return report


def validate_with_configs(
    root: Path, project: Mapping[str, Any], profile: Mapping[str, Any], capture: Mapping[str, Any]
) -> ValidationReport:
    """validate_experiment0 with its settings taken from the project, profile and capture configs."""
    return validate_experiment0(
        root,
        pcb_model=profile["pcb"]["model"],
        variant=profile["pcb"]["variant_label"],
        inspection_stages=profile["inspection_stages"]["confirmed"],
        defect_types=project["defects"]["taxonomy"],
        reported_colours=profile["appearance_variations"]["board_colour"]["reported_values"],
        image_formats=capture["allowed_image_formats"],
        board_size_mm=(profile["pcb"]["dimensions_mm"]["short_side"], profile["pcb"]["dimensions_mm"]["long_side"]),
    )


def parse_defect_box(text: str) -> tuple[float, float, float, float]:
    """'x0,y0,x1,y1' (mm) -> tuple; raises ValueError if malformed or empty in either direction."""
    values = tuple(float(v) for v in text.split(","))
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3] or min(values) < 0:
        raise ValueError(f"defect_box_mm {text!r} must be x0,y0,x1,y1 in mm with 0 <= x0 < x1 and 0 <= y0 < y1")
    return values  # type: ignore[return-value]


def _check_boards_and_defects(boards, defects, defect_types, reported_colours, report: ValidationReport,
                              board_size_mm: tuple[float, float] | None = None) -> None:
    defects_per_board: Counter[str] = Counter()
    for defect_id, (line, row) in defects.items():
        where = f"{E0_DEFECTS.filename}:{line}"
        if not DEFECT_ID_PATTERN.fullmatch(defect_id):
            report.issues.append(f"{where}: invalid defect_id {defect_id!r}, expected e.g. DEF_001")
        if row["board_id"] and row["board_id"] not in boards:
            report.issues.append(f"{where}: board_id {row['board_id']} is not in {E0_BOARDS.filename}")
        defects_per_board[row["board_id"]] += 1
        if row["defect_type"] and row["defect_type"] not in defect_types:
            report.issues.append(f"{where}: defect_type {row['defect_type']!r} is not in the configured taxonomy")
        if row["visible_in_images"] and row["visible_in_images"] not in VISIBILITY:
            report.issues.append(f"{where}: visible_in_images must be one of {VISIBILITY} or empty")
        if row["defect_box_mm"]:
            try:
                x0, y0, x1, y1 = parse_defect_box(row["defect_box_mm"])
                if board_size_mm and (x1 > max(board_size_mm) or y1 > max(board_size_mm)):
                    report.issues.append(f"{where}: defect_box_mm {row['defect_box_mm']!r} lies outside the board")
            except ValueError as e:
                report.issues.append(f"{where}: {e}")

    for board_id, (line, row) in boards.items():
        where = f"{E0_BOARDS.filename}:{line}"
        if not BOARD_ID_PATTERN.fullmatch(board_id):
            report.issues.append(f"{where}: invalid board_id {board_id!r}, expected e.g. GOOD_001 or DEFECT_001")
            continue
        condition = row["condition"]
        if condition and condition not in CONDITIONS:
            report.issues.append(f"{where}: condition {condition!r} must be one of {CONDITIONS}")
        expected = "good" if board_id.startswith("GOOD_") else "defective"
        if condition in CONDITIONS and condition != expected:
            report.warnings.append(
                f"{where}: {board_id} is recorded as {condition}; keep the ID "
                "(the prefix records the intake pool, the condition column is the verified label)"
            )
        if row["colour"] and row["colour"] not in reported_colours:
            report.warnings.append(f"{where}: colour {row['colour']!r} is not among the reported colours")
        if condition == "defective" and not defects_per_board[board_id]:
            report.issues.append(f"{where}: defective board {board_id} has no row in {E0_DEFECTS.filename}")
        if condition == "good" and defects_per_board[board_id]:
            report.issues.append(f"{where}: good board {board_id} has defects in {E0_DEFECTS.filename}")


def _check_distance(value: str, where: str, report: ValidationReport, counts: Counter[str]) -> None:
    if value in ("", "UNKNOWN"):
        counts["camera distance not recorded"] += 1
        return
    try:
        if float(value) <= 0:
            raise ValueError
    except ValueError:
        report.issues.append(f"{where}: camera_distance_mm must be a positive number in mm (or UNKNOWN), got {value!r}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
