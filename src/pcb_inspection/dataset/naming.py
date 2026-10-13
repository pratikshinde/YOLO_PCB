"""Identifiers, capture filenames and the raw directory layout.

Defined in docs/data_collection_plan.md, section 9:

    board_id    GOOD_001 | DEFECT_001   one ID per physical board, kept for life.
                                        The prefix records the pool the board came
                                        from at intake (known-good pool / rejected
                                        boards); it is never changed, even if the
                                        board's verified condition turns out different.
    side        TOP | BOT
    session_id  S20261002-01            capture date + sequence number on that day
    capture_id  C001                    sequence within (board, side, session)

    filename    GOOD_001_TOP_S20261002-01_C001.png
    path        raw/<condition>/<board_id>/<filename>   (relative to data/)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import PurePosixPath

SIDES = ("TOP", "BOT")
CONDITIONS = ("good", "defective")
LOSSLESS_EXTENSIONS = ("png", "tif", "tiff")

BOARD_POOLS = ("GOOD", "DEFECT")
BOARD_ID_PATTERN = re.compile(r"(GOOD|DEFECT)_\d{3}")
SESSION_ID_PATTERN = re.compile(r"S(\d{8})-\d{2}")
CAPTURE_ID_PATTERN = re.compile(r"C\d{3}")

_FILENAME_PATTERN = re.compile(
    r"(?P<board_id>[^_]+_[^_]+)_(?P<side>[^_]+)_(?P<session_id>[^_]+)_(?P<capture_id>[^_.]+)"
    r"\.(?P<ext>[A-Za-z0-9]+)"
)


def format_board_id(pool: str, number: int) -> str:
    if pool not in BOARD_POOLS:
        raise ValueError(f"pool must be one of {BOARD_POOLS}, got {pool!r}")
    if not 1 <= number <= 999:
        raise ValueError(f"board number must be in 1..999, got {number}")
    return f"{pool}_{number:03d}"


def session_date(session_id: str) -> date:
    match = SESSION_ID_PATTERN.fullmatch(session_id)
    if match is None:
        raise ValueError(f"invalid session_id {session_id!r}, expected e.g. S20261002-01")
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"session_id {session_id!r} does not contain a valid date") from None


@dataclass(frozen=True)
class CaptureName:
    board_id: str
    side: str
    session_id: str
    capture_id: str
    ext: str = "png"

    def __post_init__(self) -> None:
        if not BOARD_ID_PATTERN.fullmatch(self.board_id):
            raise ValueError(f"invalid board_id {self.board_id!r}, expected e.g. GOOD_001 or DEFECT_001")
        if self.side not in SIDES:
            raise ValueError(f"invalid side {self.side!r}, expected one of {SIDES}")
        session_date(self.session_id)
        if not CAPTURE_ID_PATTERN.fullmatch(self.capture_id):
            raise ValueError(f"invalid capture_id {self.capture_id!r}, expected e.g. C001")
        if self.ext not in LOSSLESS_EXTENSIONS:
            raise ValueError(
                f"extension {self.ext!r} not allowed: raw captures must be lossless "
                f"{LOSSLESS_EXTENSIONS} (lowercase)"
            )

    @property
    def filename(self) -> str:
        return f"{self.board_id}_{self.side}_{self.session_id}_{self.capture_id}.{self.ext}"


def parse_capture_filename(filename: str) -> CaptureName:
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"{filename!r} does not follow <board_id>_<side>_<session_id>_<capture_id>.<ext> "
            "(e.g. GOOD_001_TOP_S20261002-01_C001.png)"
        )
    return CaptureName(**match.groupdict())


def raw_capture_path(condition: str, name: CaptureName) -> PurePosixPath:
    """Expected location of a capture, relative to the data/ directory."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} must be one of {CONDITIONS}")
    return PurePosixPath("raw", condition, name.board_id, name.filename)
