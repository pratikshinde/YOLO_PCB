"""Is a successful registration also a usable one? Gates on the existing SIFT/RANSAC homography.

The registration (alignment.registration.register) is unchanged. These checks only read its result:

  quality   inlier ratio and RANSAC fit error             -> ALIGNMENT_FAILED if poor
  geometry  where the reference board's corners land in the test frame (through the homography):
              all four inside the frame                  -> else PCB_POSITION_INVALID (board cut off)
              orientation vs the reference               -> else PCB_POSITION_INVALID (e.g. upside down: the
                                                            GOOD references are upright; lighting, shadows and
                                                            parallax of tall parts reverse)
              corner angles near 90 degrees              -> else PCB_POSITION_INVALID (seen at a steep angle)
              board size in the frame                    -> else PCB_POSITION_INVALID (too far / too small)
              convex quadrilateral                       -> else ALIGNMENT_FAILED (degenerate homography)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .registration import RegistrationResult

OK, FAILED, POSITION_INVALID = "OK", "FAILED", "POSITION_INVALID"


@dataclass(frozen=True)
class AlignmentValidationParams:
    min_inlier_ratio: float = 0.25
    max_fit_error_mm: float = 0.25
    frame_margin_fraction: float = 0.005
    max_rotation_deg: float = 30.0
    max_corner_angle_deviation_deg: float = 20.0
    min_board_width_fraction: float = 0.25


@dataclass
class AlignmentCheck:
    status: str                                  # OK | FAILED | POSITION_INVALID
    reason: str = ""
    corners_px: np.ndarray | None = None         # reference board corners in the test frame (tl, tr, br, bl)
    rotation_deg: float | None = None
    fit_error_mm: float | None = None
    inlier_ratio: float = 0.0
    corner_angle_deviation_deg: float | None = None
    board_width_fraction: float | None = None

    def summary(self) -> str:
        if self.fit_error_mm is None:
            return self.reason
        return (f"ratio {self.inlier_ratio:.2f}, fit {self.fit_error_mm:.2f} mm, rotation {self.rotation_deg:+.0f} deg, "
                f"corner angles +-{self.corner_angle_deviation_deg:.0f} deg, board {self.board_width_fraction:.0%} of frame width")


def _angle(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def validate_alignment(registration: RegistrationResult, reference_quad: np.ndarray, reference_px_per_mm: float,
                       frame_shape: tuple[int, ...], params: AlignmentValidationParams = AlignmentValidationParams()
                       ) -> AlignmentCheck:
    if not registration.success or registration.homography is None:
        return AlignmentCheck(FAILED, f"registration failed ({registration.message})")
    h, w = frame_shape[:2]
    fit_mm = (registration.inlier_rmse_px or 0.0) / reference_px_per_mm
    check = AlignmentCheck(OK, fit_error_mm=fit_mm, inlier_ratio=registration.inlier_ratio)
    try:
        corners = cv2.perspectiveTransform(reference_quad.reshape(-1, 1, 2).astype(np.float64),
                                           np.linalg.inv(registration.homography)).reshape(-1, 2)
    except np.linalg.LinAlgError:
        check.status, check.reason = FAILED, "degenerate homography"
        return check
    check.corners_px = corners
    sides = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
    angles = [180 - np.degrees(np.arccos(np.clip(np.dot(sides[i - 1], sides[i]) /
                                                 (np.linalg.norm(sides[i - 1]) * np.linalg.norm(sides[i]) + 1e-9), -1, 1)))
              for i in range(4)]
    check.corner_angle_deviation_deg = float(max(abs(a - 90) for a in angles))
    rotation = _angle(sides[0]) - _angle(reference_quad[1] - reference_quad[0])
    check.rotation_deg = float((rotation + 180) % 360 - 180)
    check.board_width_fraction = float(max(np.linalg.norm(sides[0]), np.linalg.norm(sides[1])) / w)

    if registration.inlier_ratio < params.min_inlier_ratio:
        check.status, check.reason = FAILED, f"inlier ratio {registration.inlier_ratio:.2f} < {params.min_inlier_ratio}"
    elif fit_mm > params.max_fit_error_mm:
        check.status, check.reason = FAILED, f"fit error {fit_mm:.2f} mm > {params.max_fit_error_mm} mm"
    elif not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        check.status, check.reason = FAILED, "the mapped board outline is not convex (wrong match)"
    else:
        margin = params.frame_margin_fraction * min(h, w)
        outside = [i for i, (x, y) in enumerate(corners) if not (margin <= x <= w - 1 - margin and margin <= y <= h - 1 - margin)]
        names = ("top-left", "top-right", "bottom-right", "bottom-left")   # corners of the board, not of the frame
        problems = []
        if outside:
            problems.append("board not completely in the frame (board " + ", ".join(names[i] for i in outside)
                            + " corner outside)")
        if abs(check.rotation_deg) > params.max_rotation_deg:
            problems.append(f"board rotated {check.rotation_deg:+.0f} deg relative to the GOOD reference "
                            f"(max {params.max_rotation_deg:g}); place it the same way round as the reference")
        if check.corner_angle_deviation_deg > params.max_corner_angle_deviation_deg:
            problems.append(f"board seen at a steep angle (corner angles off by {check.corner_angle_deviation_deg:.0f} deg, "
                            f"max {params.max_corner_angle_deviation_deg:g})")
        if check.board_width_fraction < params.min_board_width_fraction:
            problems.append(f"board too small in the frame ({check.board_width_fraction:.0%} of the width)")
        if problems:
            check.status, check.reason = POSITION_INVALID, "; ".join(problems)
    return check
