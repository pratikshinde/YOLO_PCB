"""Image-quality checks on simple synthetic test patterns (a coloured rectangle on grey,
a checkerboard), generated in a temp dir. These are inputs for the check logic, not PCB images."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pcb_inspection.config import load_config
from pcb_inspection.image_quality.checks import QualityParams, check_image, flag_relative_sharpness

ROOT = Path(__file__).resolve().parents[1]
BOARD_MM = (116.0, 83.0)
PARAMS = QualityParams(max_near_white_fraction=0.02, max_near_black_fraction=0.05, max_aspect_error=0.10,
                       relative_sharpness_min=0.5)


def save(array, path: Path) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def board_pattern(width=900, height=700, x0=100, y0=150, w=348, h=249):
    """Neutral grey background with a blue rectangle of the board's 116:83 aspect (3 px/mm)."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    img[y0:y0 + h, x0:x0 + w] = (30, 60, 200)
    return img


def checkerboard(size=400, square=10):
    tiles = (np.indices((size, size)).sum(axis=0) // square) % 2
    return np.repeat((tiles * 200 + 20)[..., None], 3, axis=2)


def test_board_found_complete_with_scale(tmp_path):
    q = check_image(save(board_pattern(), tmp_path / "a.png"), PARAMS, BOARD_MM)
    assert q.readable and (q.width, q.height) == (900, 700)
    assert (q.board.x0, q.board.y0, q.board.x1, q.board.y1) == (100, 150, 448, 399)
    assert abs(q.px_per_mm - 3.0) < 0.01
    assert q.board_aspect_error < 0.01
    assert q.orientation == "landscape"
    assert q.flags == []


def test_coloured_background_strip_is_not_part_of_the_board(tmp_path):
    img = board_pattern()
    img[:, 0:150] = (120, 80, 40)                    # brown "wooden" strip over the full image height
    img[150:399, 100:448] = (200, 30, 40)            # red board overlapping the strip
    q = check_image(save(img, tmp_path / "wood.png"), PARAMS, BOARD_MM)
    assert q.colour_estimate == "red"
    assert (q.board.x0, q.board.y0, q.board.x1, q.board.y1) == (100, 150, 448, 399)
    assert q.flags == []


def test_colour_estimate(tmp_path):
    for rgb, name in (((30, 60, 200), "blue"), ((20, 150, 40), "green"), ((200, 30, 40), "red")):
        img = board_pattern()
        img[150:399, 100:448] = rgb
        assert check_image(save(img, tmp_path / f"{name}.png"), PARAMS, BOARD_MM).colour_estimate == name


def test_cropped_board_is_flagged(tmp_path):
    q = check_image(save(board_pattern(x0=700), tmp_path / "crop.png"), PARAMS, BOARD_MM)
    assert "board_at_image_edge_possible_crop" in q.flags


def test_missing_board_is_flagged(tmp_path):
    q = check_image(save(np.full((300, 400, 3), 128), tmp_path / "empty.png"), PARAMS, BOARD_MM)
    assert "board_not_detected" in q.flags


def test_overexposure_is_flagged(tmp_path):
    q = check_image(save(np.full((300, 400, 3), 255), tmp_path / "white.png"), PARAMS, BOARD_MM)
    assert q.near_white_fraction == 1.0
    assert "glare_or_overexposed" in q.flags


def test_blur_lowers_sharpness_and_relative_flag(tmp_path):
    sharp = save(checkerboard(), tmp_path / "sharp.png")
    blurred = tmp_path / "blurred.png"
    Image.open(sharp).filter(ImageFilter.GaussianBlur(4)).save(blurred)
    results = [check_image(p, PARAMS, BOARD_MM) for p in (sharp, sharp, blurred)]
    assert results[2].sharpness < 0.5 * results[0].sharpness
    flag_relative_sharpness({"GOOD_001": results}, PARAMS.relative_sharpness_min)
    assert "less_sharp_than_other_captures" in results[2].flags
    assert "less_sharp_than_other_captures" not in results[0].flags


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    img = Image.fromarray(np.full((200, 300, 3), 128, dtype=np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    img.save(path, exif=exif)
    q = check_image(path, PARAMS, BOARD_MM)
    assert (q.width, q.height) == (200, 300)
    assert q.orientation == "portrait"


def test_missing_and_unreadable_files(tmp_path):
    assert check_image(tmp_path / "nope.jpg", PARAMS, BOARD_MM).flags == ["missing"]
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    q = check_image(bad, PARAMS, BOARD_MM)
    assert not q.readable and q.flags == ["unreadable"]


def test_params_from_shipped_phone_config():
    capture = load_config(ROOT / "configs" / "capture" / "mobile_phone_e0.yaml")
    params = QualityParams.from_mapping(capture["quality_check"])
    assert params.min_megapixels is None
    assert params.relative_sharpness_min == capture["quality_check"]["relative_sharpness_min"]
    assert params.saturation_min == capture["quality_check"]["board_detection"]["saturation_min"]
