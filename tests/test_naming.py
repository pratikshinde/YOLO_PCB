from datetime import date
from pathlib import PurePosixPath

import pytest

from pcb_inspection.dataset.naming import (
    CaptureName,
    format_board_id,
    parse_capture_filename,
    raw_capture_path,
    session_date,
)


def test_format_board_id():
    assert format_board_id("GOOD", 1) == "GOOD_001"
    assert format_board_id("DEFECT", 999) == "DEFECT_999"
    for pool, number in (("GOOD", 0), ("GOOD", 1000), ("BOARD", 1), ("good", 1)):
        with pytest.raises(ValueError):
            format_board_id(pool, number)


def test_filename_round_trip():
    name = CaptureName("GOOD_001", "TOP", "S20261002-01", "C001")
    assert name.filename == "GOOD_001_TOP_S20261002-01_C001.png"
    assert parse_capture_filename(name.filename) == name
    assert parse_capture_filename("DEFECT_042_BOT_S20261231-03_C010.tiff").ext == "tiff"


def test_session_date():
    assert session_date("S20261002-01") == date(2026, 10, 2)
    with pytest.raises(ValueError, match="valid date"):
        session_date("S20261340-01")
    with pytest.raises(ValueError):
        session_date("20261002-01")


@pytest.mark.parametrize(
    "filename, message",
    [
        ("GOOD_001_TOP_S20261002-01_C001.jpg", "lossless"),
        ("GOOD_001_TOP_S20261002-01_C001.PNG", "lossless"),
        ("GOOD_001_top_S20261002-01_C001.png", "side"),
        ("GOOD_1_TOP_S20261002-01_C001.png", "board_id"),
        ("BOARD_0001_TOP_S20261002-01_C001.png", "board_id"),
        ("GOOD_001_TOP_S20261002_C001.png", "session_id"),
        ("GOOD_001_TOP_S20261002-01_1.png", "capture_id"),
        ("GOOD_001_TOP_C001.png", "does not follow"),
        ("board photo.png", "does not follow"),
    ],
)
def test_invalid_filenames(filename, message):
    with pytest.raises(ValueError, match=message):
        parse_capture_filename(filename)


def test_raw_capture_path():
    name = CaptureName("DEFECT_007", "BOT", "S20261002-01", "C002")
    assert raw_capture_path("defective", name) == PurePosixPath(
        "raw/defective/DEFECT_007/DEFECT_007_BOT_S20261002-01_C002.png"
    )
    with pytest.raises(ValueError):
        raw_capture_path("unknown", name)
