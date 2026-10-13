"""Alignment and comparison on generated test patterns (random texture, coloured rectangles) in memory.
These are inputs for checking the geometry and maths, not PCB images and not defect data."""

import cv2
import numpy as np
import pytest

from pcb_inspection.alignment.registration import RegistrationParams, register, tile_residuals
from pcb_inspection.comparison.difference import CandidateRegion, ComparisonParams, compare, verified_defect_overlap
from pcb_inspection.preprocessing.board import (
    board_mask,
    board_quad,
    canonical_frame,
    normalise_illumination,
    order_corners,
    to_lab,
    warp_to_canonical,
)


def texture(h=600, w=800, seed=0):
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2.0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def red_board_scene(angle_deg=0.0):
    """Grey background with a red 83 x 116 'board' (3 px/mm) rotated about the image centre."""
    img = np.full((600, 600, 3), 128, np.uint8)
    corners = np.float32([[-124.5, -174], [124.5, -174], [124.5, 174], [-124.5, 174]])
    a = np.deg2rad(angle_deg)
    rot = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad = corners @ rot.T + 300
    cv2.fillConvexPoly(img, quad.astype(np.int32), (40, 30, 200))
    return img, order_corners(quad)


def test_order_corners():
    pts = np.float32([[10, 90], [90, 10], [10, 10], [90, 90]])
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


@pytest.mark.parametrize("angle", [0.0, 4.0])
def test_board_quad_and_canonical_frame(angle):
    img, expected = red_board_scene(angle)
    quad = board_quad(board_mask(img))
    assert np.abs(quad - expected).max() < 4  # px, at 3 px/mm
    frame = canonical_frame(quad, (83, 116), px_per_mm=4)
    assert frame.size == (332, 464)          # portrait, like the board in the image
    assert abs(frame.source_px_per_mm - 3.0) < 0.1
    warped, valid = warp_to_canonical(img, frame.homography, frame.size, frame.source_px_per_mm, 4)
    inner = warped[20:-20, 20:-20]
    assert valid[20:-20, 20:-20].min() == 255
    assert np.abs(inner.astype(int) - (40, 30, 200)).max() < 10  # all board, no background


def test_registration_recovers_known_homography():
    reference = texture()
    true_h = np.array([[0.98, -0.05, 20.0], [0.04, 1.01, -15.0], [1e-5, -2e-5, 1.0]])
    test = cv2.warpPerspective(reference, np.linalg.inv(true_h), (800, 600))  # test = reference seen differently
    full = np.full((600, 800), 255, np.uint8)
    result = register(reference, full, test, full, RegistrationParams(mask_erosion_fraction=0.02))
    assert result.success and result.inliers > 100
    grid = np.float32([[x, y] for x in range(150, 651, 100) for y in range(150, 451, 100)]).reshape(-1, 1, 2)
    error = cv2.perspectiveTransform(grid, result.homography) - cv2.perspectiveTransform(grid, true_h)
    assert np.abs(error).max() < 0.5
    assert result.inlier_rmse_px < 1.0


def test_registration_fails_cleanly_without_texture():
    flat = np.full((300, 300, 3), 120, np.uint8)
    full = np.full((300, 300), 255, np.uint8)
    result = register(flat, full, flat, full)
    assert not result.success and result.homography is None


def test_tile_residuals_measure_a_known_shift():
    gray = cv2.cvtColor(texture(), cv2.COLOR_BGR2GRAY)
    shifted = np.roll(gray, 2, axis=1)  # 2 px to the right
    valid = np.full(gray.shape, 255, np.uint8)
    residuals = tile_residuals(gray, shifted, valid, tile_px=100)
    assert residuals.reliable.all()
    # windowed phase correlation underestimates slightly (documented); direction and size must be right
    assert np.allclose(residuals.shifts_px[:, 0], 2.0, atol=0.2) and np.allclose(residuals.shifts_px[:, 1], 0.0, atol=0.1)


def test_identical_images_give_no_candidates():
    img = texture()
    valid = np.full(img.shape[:2], 255, np.uint8)
    result = compare(img, img.copy(), valid, px_per_mm=10)
    assert result.regions == []
    assert result.metrics["ssim_mean"] > 0.999 and result.metrics["delta_e_mean"] < 1e-3


def test_global_brightness_change_is_normalised():
    img = texture()
    darker = np.clip(img.astype(np.float32) * 0.8 + 5, 0, 255).astype(np.uint8)
    valid = np.full(img.shape[:2], 255, np.uint8)
    raw = compare(img, darker, valid, px_per_mm=10, params=ComparisonParams(illumination="none"))
    normalised = compare(img, darker, valid, px_per_mm=10, params=ComparisonParams(illumination="global"))
    assert normalised.metrics["delta_e_mean"] < raw.metrics["delta_e_mean"] / 3


def test_changed_patch_becomes_a_candidate_region_in_mm():
    img = texture()
    changed = img.copy()
    changed[200:260, 300:380] = (20, 20, 220)  # 8 x 6 mm patch at 10 px/mm
    valid = np.full(img.shape[:2], 255, np.uint8)
    result = compare(img, changed, valid, px_per_mm=10)
    assert len(result.regions) == 1
    r = result.regions[0]
    assert abs(r.x_mm - 30) < 0.5 and abs(r.y_mm - 20) < 0.5 and abs(r.area_mm2 - 48) < 6
    assert verified_defect_overlap(result.regions, [(29, 19, 31, 21), (0, 0, 5, 5)]) == [
        {"box_mm": [29, 19, 31, 21], "overlapping_candidates": [0]},
        {"box_mm": [0, 0, 5, 5], "overlapping_candidates": []},
    ]


def test_border_margin_and_invalid_pixels_are_not_compared():
    img = texture()
    changed = img.copy()
    changed[:, :10] = 0                     # change inside the border margin
    valid = np.full(img.shape[:2], 255, np.uint8)
    valid[:, 600:] = 0                      # outside the warped test image
    changed[:, 650:] = 255
    result = compare(img, changed, valid, px_per_mm=10, params=ComparisonParams(border_margin_mm=2.0))
    assert result.regions == []
    assert result.compared[:, :20].max() == 0 and result.compared[:, 600:].max() == 0


def test_candidate_region_overlap():
    r = CandidateRegion(10, 10, 5, 5, 25, 0, 0)
    assert r.overlaps((14, 14, 20, 20)) and not r.overlaps((15, 0, 20, 9))


def test_illumination_method_is_validated():
    lab = to_lab(texture())
    with pytest.raises(ValueError):
        normalise_illumination(lab, lab, np.full(lab.shape[:2], 255, np.uint8), "magic")
