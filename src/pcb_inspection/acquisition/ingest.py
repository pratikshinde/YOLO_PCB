"""Camera-first ingest of photos for Experiment 0 (development / dataset mode).

Developer workflow: copy the original phone photos into the inbox and run
`python scripts/e0_ingest.py`. Nothing is typed per image. The software:

* assigns a session ID per capture date and ingest run, and an image ID per photo;
* reads capture time, phone make/model, zoom, flash, exposure, ISO and focal
  length from EXIF where the phone recorded them;
* records image size, original file name and a SHA-256 checksum;
* copies the original file byte-for-byte (verified by checksum) to
  images/<board_id or "unassigned">/<session_id>/<image_id>.<ext>;
* appends one row per photo to metadata/captures.csv;
* moves the staged inbox file to inbox/_ingested/<session_id>/ (nothing is deleted).

A board ID is optional and never typed per image: photos in a sub-folder named
after the board (inbox/GOOD_001/...) get that board ID; photos directly in the
inbox get the --board value, if one is given, and are "unassigned" otherwise.
Session-level context (stage, camera distance, lighting) is passed once per run.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ..dataset.experiment0 import E0_CAPTURES, IMAGE_ID_PATTERN, UNASSIGNED_BOARD_FOLDER, sha256_of
from ..dataset.naming import BOARD_ID_PATTERN

INGESTED_FOLDER = "_ingested"  # inbox/_ingested/<session_id>/: staged files after a verified copy

EXIF_IFD = 0x8769
TAG_MAKE, TAG_MODEL, TAG_ORIENTATION, TAG_DATETIME = 0x010F, 0x0110, 0x0112, 0x0132
TAG_DATETIME_ORIGINAL, TAG_FLASH, TAG_DIGITAL_ZOOM = 0x9003, 0x9209, 0xA404
TAG_EXPOSURE_TIME, TAG_ISO, TAG_FOCAL_LENGTH = 0x829A, 0x8827, 0x920A


@dataclass
class PlannedCapture:
    source: Path
    row: dict[str, str]

    @property
    def image_file(self) -> str:
        return self.row["image_file"]


@dataclass
class IngestPlan:
    inbox: Path
    captures: list[PlannedCapture] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def read_capture_info(path: Path) -> dict[str, str]:
    """Capture metadata from the file itself (EXIF where available, file time otherwise)."""
    with Image.open(path) as img:
        exif = img.getexif()
        sub = exif.get_ifd(EXIF_IFD)
        width, height = img.size
    if exif.get(TAG_ORIENTATION) in (5, 6, 7, 8):  # displayed rotated by 90 degrees
        width, height = height, width

    stamp = sub.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
    try:
        captured_at, source = datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S"), "exif"
    except ValueError:
        captured_at, source = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0), "file_time"

    make, model = str(exif.get(TAG_MAKE) or "").strip(), str(exif.get(TAG_MODEL) or "").strip()
    device = model if make and model.lower().startswith(make.lower()) else f"{make} {model}".strip()
    flash = _number(sub.get(TAG_FLASH))
    return {
        "captured_at": captured_at.isoformat(),
        "captured_at_source": source,
        "camera_device": device,
        "width": str(width),
        "height": str(height),
        "digital_zoom": _text(sub.get(TAG_DIGITAL_ZOOM)),
        "flash_fired": "" if flash is None else ("yes" if int(flash) & 1 else "no"),
        "exposure_time_s": _text(sub.get(TAG_EXPOSURE_TIME)),
        "iso": _text(sub.get(TAG_ISO)),
        "focal_length_mm": _text(sub.get(TAG_FOCAL_LENGTH)),
    }


def plan_ingest(
    inbox: Path,
    existing_rows: Iterable[Mapping[str, str]],
    image_formats: Iterable[str],
    defaults: Mapping[str, str],
) -> IngestPlan:
    """Work out IDs and metadata for every photo in the inbox, without copying or moving anything."""
    plan = IngestPlan(inbox=inbox)
    formats = {f.lower() for f in image_formats}
    existing = list(existing_rows)
    known_hashes = {row["sha256"] for row in existing if row.get("sha256")}
    sessions_per_date: dict[str, int] = defaultdict(int)
    for row in existing:
        match = IMAGE_ID_PATTERN.fullmatch(row.get("image_id", ""))
        if match:
            date, number = match["session_id"][1:9], int(match["session_id"][10:12])
            sessions_per_date[date] = max(sessions_per_date[date], number)

    found: list[tuple[str, Path, str, dict[str, str]]] = []
    for path, folder_board in _inbox_files(inbox, plan):
        rel = path.relative_to(inbox).as_posix()
        if path.suffix.lower().lstrip(".") not in formats:
            plan.skipped.append(f"{rel}: format {path.suffix!r} not supported (set the phone to JPEG); left in inbox")
            continue
        digest = sha256_of(path)
        if digest in known_hashes:
            plan.skipped.append(f"{rel}: identical to an image already ingested; left in inbox")
            continue
        try:
            info = read_capture_info(path)
        except (UnidentifiedImageError, OSError) as e:
            plan.skipped.append(f"{rel}: cannot read image ({e}); left in inbox")
            continue
        known_hashes.add(digest)
        found.append((folder_board or defaults.get("board_id", ""), path, digest, info))

    found.sort(key=lambda item: (item[3]["captured_at"], item[1].name))
    session_of_date: dict[str, str] = {}
    captures_in_session: dict[str, int] = defaultdict(int)
    for board_id, path, digest, info in found:
        date = info["captured_at"][:10].replace("-", "")
        if date not in session_of_date:
            sessions_per_date[date] += 1
            session_of_date[date] = f"S{date}-{sessions_per_date[date]:02d}"
        session_id = session_of_date[date]
        captures_in_session[session_id] += 1
        image_id = f"{session_id}_C{captures_in_session[session_id]:03d}"
        folder = board_id or UNASSIGNED_BOARD_FOLDER
        row = {column: "" for column in E0_CAPTURES.columns}
        row.update(defaults)
        row.update(info)
        row.update(
            image_id=image_id,
            image_file=f"images/{folder}/{session_id}/{image_id}{path.suffix.lower()}",
            session_id=session_id,
            original_filename=path.name,
            sha256=digest,
            board_id=board_id,
        )
        plan.captures.append(PlannedCapture(source=path, row=row))
    return plan


def commit_ingest(plan: IngestPlan, root: Path) -> None:
    """Copy each photo byte-for-byte, verify the copy, log it, then park the staged file in inbox/_ingested/.

    Rows are written only for verified copies. A copy whose checksum differs is
    removed again and an error is raised; the staged file then stays in the inbox.
    """
    captures_csv = root / "metadata" / E0_CAPTURES.filename
    for capture in plan.captures:
        target = root / capture.image_file
        if target.exists():
            raise FileExistsError(f"{target} already exists; nothing was overwritten")
    write_header = not captures_csv.is_file() or captures_csv.stat().st_size == 0
    captures_csv.parent.mkdir(parents=True, exist_ok=True)
    with captures_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=E0_CAPTURES.columns)
        if write_header:
            writer.writeheader()
        for capture in plan.captures:
            target = root / capture.image_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(capture.source, target)
            if sha256_of(target) != capture.row["sha256"]:
                target.unlink()
                raise OSError(f"copy of {capture.source} does not match the original; nothing logged for it")
            writer.writerow(capture.row)
            f.flush()
            parked = plan.inbox / INGESTED_FOLDER / capture.row["session_id"] / capture.source.relative_to(plan.inbox)
            parked.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(capture.source), str(_free_name(parked)))


def _inbox_files(inbox: Path, plan: IngestPlan) -> list[tuple[Path, str]]:
    """Files in the inbox root (no folder board ID) and in board-named sub-folders (board ID = folder name).

    Folders starting with "_" or "." (e.g. _ingested) are reserved and ignored.
    """
    files: list[tuple[Path, str]] = []
    if not inbox.is_dir():
        return files
    for entry in sorted(inbox.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_file():
            files.append((entry, ""))
        elif entry.is_dir():
            if not BOARD_ID_PATTERN.fullmatch(entry.name):
                plan.skipped.append(
                    f"{entry.name}/: sub-folder name is not a board ID (e.g. GOOD_001, DEFECT_001); left in inbox"
                )
                continue
            files += [(p, entry.name) for p in sorted(entry.iterdir()) if p.is_file() and not p.name.startswith(".")]
    return files


def _free_name(path: Path) -> Path:
    """path, or path with a numeric suffix if a file of that name already exists."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


def _number(value: Any) -> float | None:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _text(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6g}"
