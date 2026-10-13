"""Golden-reference PCB comparison (desktop prototype of the Android GoldenComparator).

LEGACY NOTE (2026-10-12): what is still used from this file is the ALIGNMENT code (detect_board, align, warp_to_reference,
prepare_reference, comparison_grays, local_flow, region_shifts, load_regions ...): the training scripts in ml/presence and
predict_photo.py import it, and the Android app has the same steps in GoldenComparator.kt. The per-photo checks below
(check_missing colour test, compare_regions -> MISPLACED / DIFFERENT, difference map, heat map, inspect) are OBSOLETE: on real
removals they found 0 of 19 parts. The missing-part decision is made by the YOLO11n-cls model (ml/presence, android Presence.kt).
Those functions were removed from the Kotlin app; their tests and docs are in archive/2026-10-12_old_app_checks/.

Pure classical OpenCV, no learning, no training: the new board photo is rectified,
aligned to ONE golden reference image and compared component by component. The steps
(A-G) and parameters are the same as in the app:
android_app/modules/golden-comparator/android/src/main/java/expo/modules/goldencomparator/GoldenComparator.kt

    python scripts/golden_compare.py REFERENCE NEW --out OUT_DIR [--regions regions.json]
    python scripts/golden_compare.py REFERENCE --regions-only --out OUT_DIR

Writes OUT_DIR/annotated.jpg, aligned.png, heatmap.jpg, result.json (and regions.json,
reference_regions.jpg for the reference). Decision: PASS / FAIL / RETAKE.

The main test is MISSING components (bare board where the reference shows a part body); small position
differences are tolerated. --fail-on chooses which findings turn the decision into FAIL (the others are notes).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np



@dataclass
class Params:
    board_w_mm: float = 83.0       # canonical frame is PORTRAIT (as the golden reference)
    board_h_mm: float = 116.0
    px_per_mm: float = 10.0
    max_input_side: int = 2000
    # A. board detection + rectification
    hue_tol: int = 12              # OpenCV hue scale 0-180
    sat_min: int = 60
    val_min: int = 40
    aspect_tol: float = 0.08
    min_board_fraction: float = 0.08
    full_frame_fraction: float = 0.80   # board colour touching all 4 borders and covering this -> pre-cropped
    blur_min: float = 25.0         # variance of the Laplacian of the canonical gray; below -> RETAKE
                                   # (real board photos: 1st percentile 81, median 250; catches severe blur only)
    # B. fine alignment
    features: int = 5000
    ratio_test: float = 0.75
    ransac_px: float = 3.0
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    guide_radius_mm: float = 5.0   # matches must land within this distance of the corner-rectified position
    ecc_iterations: int = 50
    ecc_max_mm: float = 3.0        # ECC may move a board corner at most this far from the feature homography
    min_scale: float = 0.8         # plausible scale of the fine homography (both images are rectified)
    max_scale: float = 1.25
    min_coverage: float = 0.97     # share of the reference board that the photo must cover
    # C. normalisation (camera independence)
    sigma_base: float = 1.0
    sharpness_ratio: float = 1.15
    sharpen_sigma_max: float = 4.0
    text_open_mm: float = 0.4      # removes thin silkscreen text from the part mask
    # holes: "don't care" (they show the background behind the board)
    hole_value_max: int = 80
    hole_min_mm2: float = 0.3
    hole_max_mm2: float = 40.0
    hole_roundness: float = 0.72   # blob area / enclosing-circle area (square body = 0.64)
    hole_dilate_mm: float = 0.2
    # D. component regions
    close_mm: float = 0.3
    region_min_mm2: float = 0.4
    region_max_mm2: float = 400.0
    region_pad_mm: float = 0.5
    merge_overlap: float = 0.5
    split_long_mm: float = 15.0
    min_texture_std: float = 6.0
    min_valid_fraction: float = 0.9
    # E0. missing-component check (the main test): does the reference BODY ZONE of a part still show
    # bare board? Colours are judged against the board colour measured in a ring around the part.
    body_hue_tol: float = 12.0     # reference body zone: only clearly non-board pixels (hue off by more than this, or
    body_s_k: float = 0.60         # less saturated / darker than these shares of the local board colour) ...
    body_v_k: float = 0.60         # ... that are not silver (below) belong to the body zone
    board_hue_tol: float = 6.0     # new photo, bare board: hue within this of the local board hue ...
    board_s_k: float = 0.70        # ... saturation >= this x the local board saturation ...
    board_v_k: float = 0.75        # ... and brightness >= this x the local board brightness
    silver_s_k: float = 0.30       # pads / solder / glare: saturation < this x board saturation ...
    silver_v_k: float = 0.60       # ... and brightness >= this x board brightness (never part of a body zone)
    flow_win_mm: float = 5.0       # local alignment: dense optical flow of the board around the part (no body zones)
    flow_min_px: int = 150         # ... needs at least this many structure pixels, else no local correction
    flow_max_mm: float = 4.0       # ... and is limited to this shift
    flow_mad_max_mm: float = 0.8   # ... and must be consistent (median deviation of the flow vectors), else ignored
    glare_v_k: float = 1.25        # the board around a part is brighter than this x the photo's board brightness, or ...
    glare_s_k: float = 0.50        # ... less saturated than this x the photo's board saturation: glare, no check
    max_unsure_fraction: float = 0.20   # more parts than this share not checkable (glare) and nothing missing: RETAKE
    ring_inner_mm: float = 1.0     # local board colour: ring from this distance ...
    ring_outer_mm: float = 3.0     # ... to this distance around the part box
    ring_min_px: int = 200
    body_min_px: int = 60          # smaller reference body zone: the part cannot be checked for 'missing'
    expose_max: float = 0.40       # bare-board share of the body zone (best position within the tolerance) -> MISSING
    # E. per-region comparison
    t_present: float = 0.60
    local_search_mm: float = 0.5
    wide_search_mm: float = 4.0    # a part missing at its position is looked for this far away (-> MISPLACED)
    tol_position_mm: float = 1.0   # "a little misaligned is OK": offsets up to this are never reported
    misplaced_margin: float = 0.10  # the shifted match must beat the match at the position by this much
    extra_growth: float = 0.5       # part-mask fraction growth (relative) -> DIFFERENT
    extra_growth_abs: float = 0.15
    # G. decision: 0 = FAIL on missing / misplaced parts (other findings are notes), 1 = FAIL on every finding
    fail_on: int = 0
    # F. global difference map
    t_diff: float = 40.0
    min_diff_area_mm2: float = 0.3
    border_mm: float = 1.5
    diff_open_mm: float = 0.2
    diff_group_mm: float = 0.5
    align_tol_px: int = 2           # +-2 px (0.2 mm) tolerance for residual misalignment

    @staticmethod
    def with_sensitivity(level: str) -> "Params":
        p = Params()
        p.t_present, p.t_diff, p.min_diff_area_mm2, p.expose_max = SENSITIVITY[level]
        return p

    def px(self, mm: float) -> int:
        return max(1, int(round(mm * self.px_per_mm)))

    @property
    def canon_size(self) -> tuple[int, int]:   # (width, height)
        return int(round(self.board_w_mm * self.px_per_mm)), int(round(self.board_h_mm * self.px_per_mm))


# (t_present, t_diff, min_diff_area_mm2, expose_max): the last one is the missing-part threshold
SENSITIVITY = {"low": (0.50, 55.0, 0.6, 0.50), "normal": (0.60, 40.0, 0.3, 0.40), "high": (0.70, 30.0, 0.15, 0.34)}


class Retake(Exception):
    """Photo unusable: board not found / cut off / cannot align / blurred."""


# ---------------------------------------------------------------- helpers

def downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def dominant_hue(hsv: np.ndarray, p: Params, central: float = 0.5) -> int | None:
    """Dominant saturated hue (0-179) of the central area; circular histogram smoothed +-3 bins."""
    h, w = hsv.shape[:2]
    y0, y1 = int(h * (1 - central) / 2), int(h * (1 + central) / 2)
    x0, x1 = int(w * (1 - central) / 2), int(w * (1 + central) / 2)
    c = hsv[y0:y1, x0:x1].reshape(-1, 3)
    sat = c[(c[:, 1] >= p.sat_min) & (c[:, 2] >= p.val_min)]
    if len(sat) < 0.10 * len(c):
        return None
    hist = np.bincount(sat[:, 0].astype(np.int64), minlength=180)[:180].astype(np.float64)
    smooth = sum(np.roll(hist, k) for k in range(-3, 4))
    return int(np.argmax(smooth))


def hue_mask(hsv: np.ndarray, hue: int, p: Params) -> np.ndarray:
    """Board-colour pixels: circular hue distance <= hue_tol, S >= sat_min, V >= val_min."""
    d = np.abs(hsv[..., 0].astype(np.int16) - hue)
    d = np.minimum(d, 180 - d)
    m = (d <= p.hue_tol) & (hsv[..., 1] >= p.sat_min) & (hsv[..., 2] >= p.val_min)
    return m.astype(np.uint8) * 255


def order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def ellipse(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, px), max(1, px)))


def clahe_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def _overlap(a, b) -> float:
    """Intersection area / area of the smaller box (boxes x, y, w, h)."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1, min(a[2] * a[3], b[2] * b[3]))


# ---------------------------------------------------------------- A. board detection + rectification

@dataclass
class Board:
    canonical: np.ndarray        # BGR, canonical size (portrait)
    valid: np.ndarray            # 0/255: canonical pixels that came from inside the photo
    hue: int
    full_frame: bool             # board filled the whole image (pre-cropped, like the golden reference)
    sharpness: float


def detect_board(bgr: np.ndarray, p: Params) -> Board:
    img = downscale(bgr, p.max_input_side)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = dominant_hue(hsv, p)
    if hue is None:
        raise Retake("No board found (no board colour in the centre of the image)")
    mask = hue_mask(hsv, hue, p)
    k = ellipse(max(3, int(round(max(h, w) / 300)) | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Retake("No board found")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    area_frac = cv2.contourArea(contour) / (h * w)
    if area_frac < p.min_board_fraction:
        raise Retake("No board found (board colour region too small)")

    m = 2
    touch = [filled[:m].any(), filled[:, -m:].any(), filled[-m:].any(), filled[:, :m].any()]
    if all(touch) and area_frac > p.full_frame_fraction:
        # Pre-cropped image (the golden reference): the board fills the frame.
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        full_frame = True
    elif any(touch):
        raise Retake("Board not completely visible (touches the image edge) - retake")
    else:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        pts = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
        corners = order_corners(pts)
        full_frame = False

    tl, tr, br, bl = corners
    wpx = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    hpx = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    aspect = max(wpx, hpx) / max(1e-6, min(wpx, hpx))
    expected = max(p.board_w_mm, p.board_h_mm) / min(p.board_w_mm, p.board_h_mm)
    if abs(aspect / expected - 1) > p.aspect_tol:
        raise Retake(f"Board not completely visible (aspect {aspect:.2f}, expected {expected:.2f}) - retake")
    if wpx > hpx:   # landscape photo -> portrait by a 90 deg turn; step B resolves 180 deg
        corners = np.float32([tr, br, bl, tl])

    # Area-downscale first when the board is much larger than the canonical frame (no aliasing),
    # then warp with linear interpolation.
    cw, ch = p.canon_size
    f = min(1.0, 1.5 * cw / max(1.0, min(wpx, hpx)))
    if f < 1.0:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        corners = corners * f
    dst = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    H = cv2.getPerspectiveTransform(np.float32(corners), dst)
    canonical = cv2.warpPerspective(img, H, (cw, ch), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), H, (cw, ch), flags=cv2.INTER_NEAREST)
    sharp = float(cv2.Laplacian(cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharp < p.blur_min:
        raise Retake(f"Image too blurred (sharpness {sharp:.0f} < {p.blur_min:.0f}) - retake")
    return Board(canonical, valid, hue, full_frame, sharp)


# ---------------------------------------------------------------- B. fine alignment

@dataclass
class Alignment:
    H: np.ndarray                # new canonical -> reference canonical
    inliers: int
    matches: int
    inlier_ratio: float
    residual_mm: float
    rotation_deg: float
    method: str


def _features(gray: np.ndarray, mask: np.ndarray, method: str, n: int):
    det = cv2.SIFT_create(nfeatures=n) if method == "SIFT" else cv2.ORB_create(n)
    kps, desc = det.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
    return pts, desc


def _rot_about_centre(rot: int, w: int, h: int) -> np.ndarray:
    """3x3 matrix turning canonical points by rot degrees (clockwise) about the canvas centre."""
    a = math.radians(rot)
    c, s = round(math.cos(a)), round(math.sin(a))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return np.float64([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]])


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.int32)


def _descriptor_distances(q: np.ndarray, t: np.ndarray, binary: bool) -> np.ndarray:
    """Distance matrix len(q) x len(t): L2 for SIFT, Hamming for ORB."""
    if binary:
        return _POPCOUNT[np.bitwise_xor(q[:, None, :], t[None, :, :])].sum(axis=2).astype(np.float32)
    d2 = (q * q).sum(1)[:, None] + (t * t).sum(1)[None, :] - 2.0 * q @ t.T
    return np.sqrt(np.maximum(d2, 0.0))


def guided_matches(pts_n, dn, pts_r, dr, R, radius: float, ratio: float, binary: bool):
    """Ratio-test matching restricted to reference keypoints within `radius` px of the expected position.

    Both images are already rectified by their board corners, so after the turn R a true match lands
    within a few mm. Restricting the candidates removes the ambiguity of repeated parts (rows of
    identical resistors) that makes a plain ratio test throw away most true matches. Candidates are
    looked up in a grid of cell size `radius` (3 x 3 neighbouring cells), which is fast on a phone.
    """
    exp = cv2.perspectiveTransform(pts_n.reshape(-1, 1, 2).astype(np.float64), R).reshape(-1, 2)
    cell = lambda xy: (int(math.floor(xy[0] / radius)), int(math.floor(xy[1] / radius)))
    grid: dict[tuple[int, int], list[int]] = {}
    for j, xy in enumerate(pts_r):
        grid.setdefault(cell(xy), []).append(j)
    qcells: dict[tuple[int, int], list[int]] = {}
    for i, xy in enumerate(exp):
        qcells.setdefault(cell(xy), []).append(i)
    found = []
    for (cx, cy), qs in qcells.items():
        cand = [j for gx in (cx - 1, cx, cx + 1) for gy in (cy - 1, cy, cy + 1) for j in grid.get((gx, gy), [])]
        if not cand:
            continue
        cand = np.array(cand)
        dist = _descriptor_distances(dn[qs].astype(np.float32) if not binary else dn[qs],
                                     dr[cand].astype(np.float32) if not binary else dr[cand], binary)
        for row, i in enumerate(qs):
            near = np.hypot(*(pts_r[cand] - exp[i]).T) <= radius
            if not near.any():
                continue
            dd = np.where(near, dist[row], np.inf)
            order = np.argsort(dd)
            best = order[0]
            if near.sum() >= 2 and dd[best] >= ratio * dd[order[1]]:
                continue
            found.append((i, cand[best]))
    found.sort()   # query order (as the Kotlin port), so RANSAC sees the same point order
    src = [pts_n[i] for i, _ in found]
    dst = [pts_r[j] for _, j in found]
    return np.float32(src).reshape(-1, 2), np.float32(dst).reshape(-1, 2)


def align(new: Board, ref_canonical: np.ndarray, ref_valid: np.ndarray, p: Params) -> Alignment:
    rg, ng = clahe_gray(ref_canonical), clahe_gray(new.canonical)
    rmask = cv2.erode(ref_valid, ellipse(p.px(1.0)))
    nmask = cv2.erode(new.valid, ellipse(p.px(1.0)))
    cw, ch = p.canon_size
    best, best_pts = None, None
    for method in ("SIFT", "ORB"):
        pts_r, dr = _features(rg, rmask, method, p.features)
        pts_n, dn = _features(ng, nmask, method, p.features)
        if dr is None or dn is None or len(pts_r) < p.min_inliers or len(pts_n) < p.min_inliers:
            continue
        for rot in (0, 180, 90, 270):   # the board may lie turned
            R = _rot_about_centre(rot, cw, ch)
            src, dst = guided_matches(pts_n, dn, pts_r, dr, R, p.px(p.guide_radius_mm), p.ratio_test,
                                      method == "ORB")
            if len(src) < p.min_inliers:
                continue
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, p.ransac_px)
            if H is None:
                continue
            inl = inl.ravel().astype(bool)
            cand = Alignment(H / H[2, 2], int(inl.sum()), len(src), float(inl.mean()), 0.0, float(rot), method)
            if best is None or cand.inliers > best.inliers:
                best, best_pts = cand, (src[inl], dst[inl])
        if best is not None and best.inliers >= p.min_inliers and best.inlier_ratio >= p.min_inlier_ratio:
            break
    if best is None or best.inliers < p.min_inliers or best.inlier_ratio < p.min_inlier_ratio:
        got = f"{best.inliers} inliers, ratio {best.inlier_ratio:.2f}" if best else "no matches"
        raise Retake(f"Cannot align with the golden reference ({got}) - retake photo")
    best.H = refine_ecc(rg, ng, best.H, cv2.bitwise_and(rmask, cv2.warpPerspective(nmask, best.H, (cw, ch))), p)
    src, dst = best_pts
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), best.H).reshape(-1, 2)
    best.residual_mm = float(np.median(np.linalg.norm(proj - dst, axis=1))) / p.px_per_mm
    best.rotation_deg = float(math.degrees(math.atan2(best.H[1, 0], best.H[0, 0])))
    sx, sy = math.hypot(best.H[0, 0], best.H[1, 0]), math.hypot(best.H[0, 1], best.H[1, 1])
    if not (p.min_scale < sx < p.max_scale and p.min_scale < sy < p.max_scale):
        raise Retake(f"Cannot align (implausible scale {sx:.2f}/{sy:.2f}) - retake photo")
    return best


def refine_ecc(ref_gray: np.ndarray, new_gray: np.ndarray, H: np.ndarray, mask: np.ndarray, p: Params) -> np.ndarray:
    """Dense refinement of H (new -> ref) by ECC on blurred CLAHE gray, half then full scale.

    ECC maximises the correlation coefficient, so it is insensitive to contrast / brightness
    differences between the two cameras. Falls back to H if ECC fails or moves a corner > ecc_max_mm.
    """
    cw, ch = p.canon_size
    W = np.linalg.inv(H)            # ECC warp: ref coords -> new coords
    W /= W[2, 2]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p.ecc_iterations, 1e-5)
    try:
        for f in (0.5, 1.0):
            S = np.float64([[f, 0, 0], [0, f, 0], [0, 0, 1]])
            r = cv2.GaussianBlur(cv2.resize(ref_gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
            n = cv2.GaussianBlur(cv2.resize(new_gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
            m = cv2.resize(mask, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
            Ws = (S @ W @ np.linalg.inv(S)).astype(np.float32)
            _, Ws = cv2.findTransformECC(r.astype(np.float32), n.astype(np.float32), Ws,
                                         cv2.MOTION_HOMOGRAPHY, crit, m, 5)
            W = np.linalg.inv(S) @ Ws.astype(np.float64) @ S
    except cv2.error:
        return H
    He = np.linalg.inv(W)
    He /= He[2, 2]
    corners = np.float64([[0, 0], [cw, 0], [cw, ch], [0, ch]]).reshape(-1, 1, 2)
    move = np.linalg.norm(cv2.perspectiveTransform(corners, He) - cv2.perspectiveTransform(corners, H), axis=2)
    return He if move.max() / p.px_per_mm <= p.ecc_max_mm else H


def warp_to_reference(new: Board, al: Alignment, p: Params) -> tuple[np.ndarray, np.ndarray]:
    size = p.canon_size
    aligned = cv2.warpPerspective(new.canonical, al.H, size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(new.valid, al.H, size, flags=cv2.INTER_NEAREST)
    return aligned, valid


# ---------------------------------------------------------------- C. normalisation + part mask

def board_area(valid: np.ndarray, p: Params) -> np.ndarray:
    """Board pixels used for comparison: inside the photo, minus a border of border_mm."""
    m = np.zeros_like(valid)
    b = p.px(p.border_mm)
    m[b:-b, b:-b] = 255
    return cv2.bitwise_and(m, valid)


def _detail(g: np.ndarray, inside: np.ndarray) -> float:
    return float(cv2.Laplacian(g, cv2.CV_32F)[inside].var())


def comparison_grays(ref_clahe: np.ndarray, new_clahe: np.ndarray, area: np.ndarray, p: Params
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """CLAHE gray of both images at EQUAL SHARPNESS, new mean/std matched to the reference.

    Both get a Gaussian blur of sigma_base (1 px). The sharper image (e.g. a contrast-enhanced golden
    photo from another camera) is blurred further, sigma in steps of 0.25 up to sharpen_sigma_max,
    until its detail energy (variance of the Laplacian inside the board) is at most sharpness_ratio x
    the other one's. Comparing at equal sharpness removes false differences caused by the camera alone.
    """
    inside = area > 0
    r = cv2.GaussianBlur(ref_clahe.astype(np.float32), (0, 0), p.sigma_base)
    n = cv2.GaussianBlur(new_clahe.astype(np.float32), (0, 0), p.sigma_base)
    dr, dn = _detail(r, inside), _detail(n, inside)
    sharper_is_ref = dr > dn
    src = ref_clahe if sharper_is_ref else new_clahe
    out = r if sharper_is_ref else n
    target = min(dr, dn) * p.sharpness_ratio
    sig = p.sigma_base
    while _detail(out, inside) > target and sig < p.sharpen_sigma_max:
        sig += 0.25
        out = cv2.GaussianBlur(src.astype(np.float32), (0, 0), sig)
    if sharper_is_ref:
        r = out
    else:
        n = out
    rm, rs = float(r[inside].mean()), float(r[inside].std())
    nm, ns = float(n[inside].mean()), float(n[inside].std())
    n = np.clip((n - nm) * (rs / max(ns, 1e-6)) + rm, 0, 255).astype(np.float32)
    return r, n, {"sigma_ref": sig if sharper_is_ref else p.sigma_base,
                  "sigma_new": p.sigma_base if sharper_is_ref else sig,
                  "detail_ref": round(dr, 1), "detail_new": round(dn, 1)}


def part_mask(bgr: np.ndarray, hue: int, area: np.ndarray, p: Params) -> np.ndarray:
    """Pixels that are NOT board colour (bodies, pads, solder, holes); thin silkscreen text removed."""
    m = cv2.bitwise_not(hue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), hue, p))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(p.px(p.text_open_mm)))
    return cv2.bitwise_and(m, area)


def hole_mask(ref_bgr: np.ndarray, area: np.ndarray, p: Params) -> np.ndarray:
    """Through-holes of the reference: dark, round blobs (hole_min..hole_max mm^2), slightly dilated.

    A hole shows whatever lies behind the board, which differs from photo to photo, so hole interiors
    are "don't care" in the comparison. A part inserted into a hole also covers the pad ring around it
    and is still detected.
    """
    v = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    dark = ((v < p.hole_value_max) & (area > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = np.zeros_like(dark)
    amin, amax = p.hole_min_mm2 * p.px_per_mm ** 2, p.hole_max_mm2 * p.px_per_mm ** 2
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (amin <= a <= amax):
            continue
        blob = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, rad = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        if a / (math.pi * max(rad, 1e-6) ** 2) >= p.hole_roundness:
            out[y:y + h, x:x + w][blob > 0] = 255
    return cv2.dilate(out, ellipse(2 * p.px(p.hole_dilate_mm) + 1))


def board_hue_of(canonical: np.ndarray, p: Params) -> int:
    h = dominant_hue(cv2.cvtColor(canonical, cv2.COLOR_BGR2HSV), p, central=0.8)
    if h is None:
        raise Retake("No board colour found after alignment")
    return h


# ---------------------------------------------------------------- D. component regions

@dataclass
class Region:
    id: str
    name: str
    bbox_px: list[int]           # x, y, w, h in canonical px
    kind: str = ""               # "part" | "empty" (footprint / hole that is empty in the reference)
    part_fraction: float = 0.0
    texture_std: float = 0.0


EMPTY_NAME_PREFIXES = ("LED", "RED", "SW", "HOLE", "MH", "J", "PIN", "TP")


def kind_from_name(name: str) -> str:
    """Footprints that are EMPTY at the SMT stage (LEDs, switch, holes, header, *DNP*) -> "empty".

    Colour cues cannot tell an empty LED footprint from a capacitor reliably, so the kind follows the
    region name (the user names regions once; the app also lets the user toggle the kind).
    """
    n = name.upper()
    return "empty" if n.startswith(EMPTY_NAME_PREFIXES) or "DNP" in n else "part"


def detect_regions(ref_mask: np.ndarray, p: Params) -> list[Region]:
    """Connected components of the closed reference part mask -> padded, merged boxes (reading order)."""
    closed = cv2.morphologyEx(ref_mask, cv2.MORPH_CLOSE, ellipse(p.px(p.close_mm)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    amin, amax = p.region_min_mm2 * p.px_per_mm ** 2, p.region_max_mm2 * p.px_per_mm ** 2
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < amin:
            continue
        if a > amax or (max(w, h) > p.px(p.split_long_mm) and max(w, h) / max(1, min(w, h)) > 3):
            # very large / long merged blob: split into its separate parts (blobs of the unclosed mask)
            sub = ((lab[y:y + h, x:x + w] == i) & (ref_mask[y:y + h, x:x + w] > 0)).astype(np.uint8)
            sn, _, sst, _ = cv2.connectedComponentsWithStats(sub, 8)
            parts = [sst[j] for j in range(1, sn) if amin <= sst[j][4] <= amax]
            if len(parts) >= 2:
                boxes += [[x + int(s[0]), y + int(s[1]), int(s[2]), int(s[3])] for s in parts]
                continue
            if a > amax:
                continue
        boxes.append([x, y, w, h])
    pad = p.px(p.region_pad_mm)
    H, W = ref_mask.shape
    boxes = [[max(0, x - pad), max(0, y - pad), min(W, x + w + pad) - max(0, x - pad),
              min(H, y + h + pad) - max(0, y - pad)] for x, y, w, h in boxes]
    merged = True
    while merged:   # merge boxes overlapping > merge_overlap of the smaller one
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]) > p.merge_overlap:
                    a, b = boxes[i], boxes[j]
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    row = p.px(3)
    boxes.sort(key=lambda b: ((b[1] + b[3] // 2) // row, b[0]))   # reading order, 3 mm rows
    return [Region(f"C{k:03d}", f"C{k:03d}", b, kind_from_name(f"C{k:03d}")) for k, b in enumerate(boxes, 1)]


def measure_regions(regions: list[Region], ref_mask: np.ndarray, ref_gray: np.ndarray) -> None:
    """Reference part-mask fraction and texture std of every region (in place)."""
    for r in regions:
        x, y, w, h = r.bbox_px
        r.part_fraction = float((ref_mask[y:y + h, x:x + w] > 0).mean()) if w * h else 0.0
        r.texture_std = float(ref_gray[y:y + h, x:x + w].std()) if w * h else 0.0
        if not r.kind:
            r.kind = kind_from_name(r.name)


# ---------------------------------------------------------------- E0. missing-component check

# A part that was never placed leaves BARE BOARD where the reference shows its body. So every 'part' region has a
# body zone in the reference (pixels that are neither board colour nor silver pad / solder / glare). In the new photo
# the share of that zone which shows bare board ("exposure") is small for a present part (even when it sits a little
# off its position) and large for a missing one. "Bare board" is judged against the board colour measured in a ring
# around the part, so lighting gradients, white balance and a different camera barely matter.

def hsv_blur(bgr: np.ndarray, p: Params) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), p.sigma_base), cv2.COLOR_BGR2HSV)


def hue_rel(hsv: np.ndarray, hue0: int) -> np.ndarray:
    """Signed hue distance to hue0 (-90..89), int16 (hue is circular)."""
    return (hsv[..., 0].astype(np.int16) - hue0 + 90) % 180 - 90


def _median_colour(hsv: np.ndarray, sel: np.ndarray, hue0: int) -> tuple[float, float, float]:
    return (float(np.median(hue_rel(hsv, hue0)[sel])), float(np.median(hsv[..., 1][sel])),
            float(np.median(hsv[..., 2][sel])))


def global_board_colour(hsv: np.ndarray, hue0: int, area: np.ndarray, p: Params) -> tuple[float, float, float]:
    sel = (hue_mask(hsv, hue0, p) > 0) & (area > 0)
    return _median_colour(hsv, sel, hue0) if sel.any() else (0.0, 200.0, 200.0)


def box_window(box, m: int, shape) -> tuple[int, int, int, int]:
    """Box grown by m px, clipped to the image -> x0, y0, x1, y1."""
    x, y, w, h = box
    H, W = shape[:2]
    return max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)


def shifted_window(win, shift, shape) -> tuple[int, int, int, int]:
    """Window in reference coordinates, clipped so that the same window moved by `shift` stays inside the image."""
    x0, y0, x1, y1 = win
    dx, dy = shift
    H, W = shape[:2]
    return max(x0, -dx), max(y0, -dy), min(x1, W - dx), min(y1, H - dy)


def local_board_colour(hsv, hue0: int, box, core: np.ndarray, fallback, p: Params, shift=(0, 0)):
    """Median (hue offset, S, V) of the bare-board pixels (reference says: board) in a ring around the box.
    `hsv` is sampled at reference position + shift (local alignment of the new photo)."""
    x0, y0, x1, y1 = shifted_window(box_window(box, p.px(p.ring_outer_mm), hsv.shape), shift, hsv.shape)
    if x1 <= x0 or y1 <= y0:
        return fallback
    m = (core[y0:y1, x0:x1] > 0).copy()
    ix0, iy0, ix1, iy1 = box_window(box, p.px(p.ring_inner_mm), hsv.shape)
    ya, yb, xa, xb = max(iy0, y0) - y0, min(iy1, y1) - y0, max(ix0, x0) - x0, min(ix1, x1) - x0
    if yb > ya and xb > xa:
        m[ya:yb, xa:xb] = False
    if int(m.sum()) < p.ring_min_px:
        return fallback
    dx, dy = shift
    return _median_colour(hsv[y0 + dy:y1 + dy, x0 + dx:x1 + dx], m, hue0)


def classify_board(hsv_win: np.ndarray, hue0: int, col, p: Params, reference: bool = False
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(bare board, silver) bool maps of a window, relative to the local board colour col.

    The reference uses the loose body_* rule (only clearly non-board pixels become body zone); the new photo the
    tighter board_* rule: a part body that is only a little different from the board (tan capacitors, dark parts in a
    reddish photo) must still not count as bare board."""
    hd = np.abs(hue_rel(hsv_win, hue0) - col[0])
    s = hsv_win[..., 1].astype(np.float32)
    v = hsv_win[..., 2].astype(np.float32)
    ht, ks, kv = (p.body_hue_tol, p.body_s_k, p.body_v_k) if reference else (p.board_hue_tol, p.board_s_k, p.board_v_k)
    board = (hd <= ht) & (s >= ks * col[1]) & (v >= kv * col[2])
    silver = ~board & (s < p.silver_s_k * col[1]) & (v >= p.silver_v_k * col[2])
    return board, silver


def reference_body(canonical: np.ndarray, area: np.ndarray, holes: np.ndarray, regions: list[Region], p: Params):
    """Body zone of every 'part' region of the reference -> (body 255, board core 255, silver 255, {region id: px}).

    silver = pads, terminals, solder and silkscreen of the reference (whole board). In the new photo the same silver
    at the same place is NOT evidence of a body: a removed part leaves its pads (and the silkscreen) behind.
    """
    hsv = hsv_blur(canonical, p)
    hue0 = board_hue_of(canonical, p)
    core = cv2.bitwise_and(cv2.erode(hue_mask(hsv, hue0, p), np.ones((5, 5), np.uint8)), area)
    gcol = global_board_colour(hsv, hue0, area, p)
    _, silver_all = classify_board(hsv, hue0, gcol, p, reference=True)
    silver_all = silver_all.astype(np.uint8) * 255
    H, W = area.shape
    parts = [r for r in regions if r.kind == "part"]
    cover = np.zeros((H, W), np.uint8)
    for r in parts:
        x, y, w, h = r.bbox_px
        cover[y:y + h, x:x + w] += 1
    body = np.zeros((H, W), np.uint8)
    counts: dict[str, int] = {}
    k3 = np.ones((3, 3), np.uint8)
    for r in parts:
        x, y, w, h = r.bbox_px
        col = local_board_colour(hsv, hue0, r.bbox_px, core, gcol, p)
        board, silver = classify_board(hsv[y:y + h, x:x + w], hue0, col, p, reference=True)
        # pixels covered by two part boxes are ambiguous: they belong to no body zone
        z = (~board & ~silver & (cover[y:y + h, x:x + w] == 1) & (area[y:y + h, x:x + w] > 0)
             & (holes[y:y + h, x:x + w] == 0))
        z = cv2.erode(z.astype(np.uint8), k3) > 0
        counts[r.id] = int(z.sum())
        if counts[r.id] >= p.body_min_px:
            body[y:y + h, x:x + w][z] = 255
    return body, core, silver_all, counts


def _coverage(hsv, hue0: int, col, ref: "Reference", box, tpl: np.ndarray, radius: int, shift, p: Params):
    """Share of the reference body zone that is covered by the new photo's 'part' pixels at every offset within
    +-radius px of the local position (reference position + shift). Covering means: not bare board, and not the pad /
    silver / silkscreen the reference shows at the same place too (a removed part leaves those behind). Other parts'
    body zones never count: they belong to those parts.
    -> (cov, dx0, dy0): cov[i, j] is the share at offset (dx0 + j, dy0 + i) from the local position, or None at the
    image border."""
    x, y, w, h = box
    dx, dy = shift
    x0, y0, x1, y1 = shifted_window(box_window(box, radius, hsv.shape), shift, hsv.shape)
    if x0 > x or y0 > y or x1 < x + w or y1 < y + h:
        return None
    board, silver = classify_board(hsv[y0 + dy:y1 + dy, x0 + dx:x1 + dx], hue0, col, p)
    cand = ~board & ~(silver & (cv2.dilate(ref.silver[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0))
    others = ref.body[y0:y1, x0:x1] > 0
    others[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    cand &= ~others
    cov = cv2.matchTemplate(cand.astype(np.float32), tpl, cv2.TM_CCORR) / max(1.0, float(tpl.sum()))
    return cov, x0 - x, y0 - y


def _best_offset(cov_and_origin, radius: int):
    """Highest coverage within the disk |offset| <= radius (smallest offset among near-ties) -> (cov, dx, dy)."""
    cov, dx0, dy0 = cov_and_origin
    ii, jj = np.indices(cov.shape)
    dx, dy = dx0 + jj, dy0 + ii
    d2 = dx * dx + dy * dy
    inside = d2 <= radius * radius
    if not inside.any():
        return None
    masked = np.where(inside, cov, -1.0)
    best = float(masked.max())
    near = inside & (masked >= best - 0.02)
    k = int(np.argmin(np.where(near, d2, 1 << 30)))
    i, j = divmod(k, cov.shape[1])
    return best, int(dx[i, j]), int(dy[i, j])


def local_flow(ref_gray: np.ndarray, new_gray: np.ndarray) -> np.ndarray:
    """Dense flow (px) golden -> new photo of the equal-sharpness comparison grays (DIS optical flow, fast preset).

    The homography of step B cannot follow a board that is a little bent, or lens distortion: parts of the same photo
    can sit 1-3 mm away from the reference in one corner and exactly on it in another."""
    r8 = np.clip(np.rint(ref_gray), 0, 255).astype(np.uint8)
    n8 = np.clip(np.rint(new_gray), 0, 255).astype(np.uint8)
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(r8, n8, None)


def reference_structure(gray: np.ndarray, body: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Pixels of the reference with clear structure (pads, traces, silkscreen) that are not part of a body zone: what the
    local alignment of a part is measured on (a part that is missing must not move its own reference)."""
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    inside = area > 0
    near_body = cv2.dilate(body, np.ones((7, 7), np.uint8)) > 0
    return ((grad > float(grad[inside].mean())) & ~near_body & inside).astype(np.uint8) * 255


def region_shifts(flow: np.ndarray, ref: "Reference", regions: list[Region], p: Params) -> dict[str, tuple[int, int]]:
    """region id -> (dx, dy) px: where the board around the part lies in the new photo relative to the reference
    (median flow over the structure pixels in the window around the part). Missing entry: no local correction (too
    little structure, or the flow vectors disagree: glare, blur, an obstacle)."""
    out: dict[str, tuple[int, int]] = {}
    rad = p.px(p.flow_win_mm)
    lim = p.flow_max_mm * p.px_per_mm
    for r in regions:
        if r.kind != "part":
            continue
        x0, y0, x1, y1 = box_window(r.bbox_px, rad, flow.shape)
        m = ref.structure[y0:y1, x0:x1] > 0
        if int(m.sum()) < p.flow_min_px:
            continue
        f = flow[y0:y1, x0:x1][m]
        med = np.median(f, axis=0)
        if math.hypot(*np.median(np.abs(f - med), axis=0)) > p.flow_mad_max_mm * p.px_per_mm:
            continue
        dx = int(np.clip(round(float(med[0])), -lim, lim))
        dy = int(np.clip(round(float(med[1])), -lim, lim))
        if dx or dy:
            out[r.id] = (dx, dy)
    return out


def check_missing(regions: list[Region], ref: "Reference", hsv, hue0: int, area: np.ndarray,
                  shifts: dict[str, tuple[int, int]], p: Params) -> dict[str, dict]:
    """region id -> {"status": OK | MISSING | UNSURE, exposure, offset_px, body_px, ...} for every checkable 'part' region
    (UNSURE: glare around the part, no decision).

    The body zone is looked for at the local position of the part (reference position + shift of the board around it).
    exposure = share of the body zone that shows bare board at the best position within +-tol_position_mm of that
    (a part that sits a little off its position is still found there). MISSING when exposure >= expose_max.
    offset_px is the total displacement of the part against the reference (shift + best position)."""
    out: dict[str, dict] = {}
    tol = p.px(p.tol_position_mm)
    core = cv2.bitwise_and(ref.board_core, area)
    gcol = global_board_colour(hsv, hue0, area, p)
    for r in regions:
        if r.kind != "part":
            continue
        x, y, w, h = r.bbox_px
        tpl = (ref.body[y:y + h, x:x + w] > 0).astype(np.float32)
        n = int(tpl.sum())
        if n < p.body_min_px:
            continue
        shift = shifts.get(r.id, (0, 0))
        col = local_board_colour(hsv, hue0, r.bbox_px, core, gcol, p, shift)
        if col[2] > p.glare_v_k * gcol[2] or col[1] < p.glare_s_k * gcol[1]:
            # glare / a hot spot: the board around the part is washed out, so "bare board" cannot be told from a body
            out[r.id] = {"status": "UNSURE", "exposure": None, "offset_px": [shift[0], shift[1]], "body_px": n,
                         "detail": "glare around the part: not checked for missing"}
            continue
        near = _coverage(hsv, hue0, col, ref, r.bbox_px, tpl, tol, shift, p)
        best = _best_offset(near, tol) if near is not None else None
        if best is None:
            continue
        cov, ox, oy = best
        exposure = 1.0 - cov
        info = {"status": "OK", "exposure": round(exposure, 3), "offset_px": [shift[0] + ox, shift[1] + oy], "body_px": n}
        if exposure >= p.expose_max:
            info.update(status="MISSING", detail=f"bare board where the part should be ({exposure:.0%} of its body zone)")
        out[r.id] = info
    return out


# ---------------------------------------------------------------- E. per-region comparison

def _score_map(new_gray, new_mask, ref_patch, ref_mpatch, win):
    """score = 0.5 * TM_CCOEFF_NORMED(gray) + 0.5 * IoU(part masks), for every offset in the window."""
    x0, y0, x1, y1 = win
    g = new_gray[y0:y1, x0:x1]
    m = (new_mask[y0:y1, x0:x1] > 0).astype(np.float32)
    tm = (ref_mpatch > 0).astype(np.float32)
    s_gray = np.nan_to_num(cv2.matchTemplate(g, ref_patch, cv2.TM_CCOEFF_NORMED), nan=0.0)
    inter = cv2.matchTemplate(m, tm, cv2.TM_CCORR)
    sum_n = cv2.matchTemplate(m, np.ones_like(tm), cv2.TM_CCORR)
    union = float(tm.sum()) + sum_n - inter
    iou = np.where(union > 0.5, inter / np.maximum(union, 1e-6), 1.0)
    return 0.5 * s_gray + 0.5 * iou, s_gray, iou


def _search(new_gray, new_mask, ref_gray, ref_mask, box, radius):
    """Best score of the reference patch in the new image within +-radius px -> (score, dx, dy, s_gray, s_mask)."""
    x, y, w, h = box
    H, W = new_gray.shape
    win = (max(0, x - radius), max(0, y - radius), min(W, x + w + radius), min(H, y + h + radius))
    if win[2] - win[0] < w or win[3] - win[1] < h:
        return None
    sc, sg, iou = _score_map(new_gray, new_mask, ref_gray[y:y + h, x:x + w], ref_mask[y:y + h, x:x + w], win)
    _, best, _, loc = cv2.minMaxLoc(sc)
    return float(best), win[0] + loc[0] - x, win[1] + loc[1] - y, float(sg[loc[1], loc[0]]), float(iou[loc[1], loc[0]])


def compare_regions(regions, ref_gray, ref_mask, new_gray, new_mask, new_valid, missing: dict, p: Params) -> list[dict]:
    """One record per region: status OK / MISSING / MISPLACED / DIFFERENT / NOT_CHECKED, with scores.

    MISSING comes from the body check (E0, `missing`): bare board at the part's position. Only then the part is looked
    for further away with the appearance score (gray correlation + part-mask IoU): found there, unique and clearly
    better than at its position -> MISPLACED. Otherwise the appearance score (searched within the position tolerance)
    only decides DIFFERENT: a different / damaged part, or something present where the footprint is empty.
    """
    out = []
    local, wide = p.px(max(p.local_search_mm, p.tol_position_mm)), p.px(p.wide_search_mm)
    for r in regions:
        x, y, w, h = r.bbox_px
        rec = {"region_id": r.id, "name": r.name, "kind": r.kind, "bbox_px": list(r.bbox_px),
               "expected_px": list(r.bbox_px), "offset_px": [0, 0], "status": "NOT_CHECKED", "detail": "",
               "score": None, "s_gray": None, "s_mask": None, "exposure": None,
               "fraction_ref": round(r.part_fraction, 3), "fraction_new": None}
        out.append(rec)
        if w <= 0 or h <= 0 or float((new_valid[y:y + h, x:x + w] > 0).mean()) < p.min_valid_fraction:
            rec["detail"] = "outside the photo"
            continue
        body = missing.get(r.id)
        if body is not None and body["status"] == "UNSURE":
            rec["detail"] = body["detail"]        # decided by the appearance score below, if at all
            body = None
        if body is not None:
            rec.update(exposure=body["exposure"], offset_px=list(body["offset_px"]), status="OK")
            if body["status"] == "MISSING":
                rec.update(status="MISSING", detail=body["detail"])
                if r.texture_std >= p.min_texture_std:
                    near, far = (_search(new_gray, new_mask, ref_gray, ref_mask, r.bbox_px, rad) for rad in (local, wide))
                    if near is not None and far is not None:
                        s, s2, dx2, dy2 = near[0], far[0], far[1], far[2]
                        # uniqueness: a match sitting on ANOTHER region's own position is that region's part
                        claimed = any(o is not r and _overlap([x + dx2, y + dy2, w, h], o.bbox_px) > p.merge_overlap
                                      for o in regions)
                        if (s2 >= p.t_present and s2 - s >= p.misplaced_margin and not claimed
                                and math.hypot(dx2, dy2) / p.px_per_mm > p.tol_position_mm):
                            rec.update(status="MISPLACED", bbox_px=[x + dx2, y + dy2, w, h], offset_px=[dx2, dy2],
                                       score=round(s2, 3), s_gray=round(far[3], 3), s_mask=round(far[4], 3),
                                       detail="part found away from its position")
                continue
        if r.texture_std < p.min_texture_std:
            rec["detail"] = rec["detail"] or "too little texture"
            continue
        res = _search(new_gray, new_mask, ref_gray, ref_mask, r.bbox_px, local)
        if res is None:
            rec["detail"] = rec["detail"] or "at the board edge"
            continue
        s, dx, dy, sg, siou = res
        f_ref, f_new = r.part_fraction, float((new_mask[y:y + h, x:x + w] > 0).mean())
        rec.update(score=round(s, 3), s_gray=round(sg, 3), s_mask=round(siou, 3), fraction_new=round(f_new, 3))
        grew = f_new - f_ref > max(p.extra_growth * f_ref, p.extra_growth_abs)
        if s >= p.t_present and not grew:
            rec["status"] = "OK"
            continue
        rec.update(status="DIFFERENT", detail="something present where the reference is empty"
                   if r.kind == "empty" or grew else "different / wrong / damaged part")
    return out


# ---------------------------------------------------------------- F. global difference map

def difference_map(ref_gray, new_gray, ref_mask, new_mask, area, p: Params):
    """|new - ref| with +-align_tol_px tolerance AND XOR of the part masks (same tolerance), thresholded."""
    t = 2 * p.align_tol_px + 1
    k = np.ones((t, t), np.uint8)
    hi, lo = cv2.dilate(ref_gray, k), cv2.erode(ref_gray, k)
    d = np.maximum(new_gray - hi, 0) + np.maximum(lo - new_gray, 0)
    xor = ((new_mask > 0) & (cv2.dilate(ref_mask, k) == 0)) | ((ref_mask > 0) & (cv2.dilate(new_mask, k) == 0))
    cand = ((d > p.t_diff) & xor & (area > 0)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ellipse(p.px(p.diff_open_mm)))
    return d, cand


def difference_findings(cand, explained_boxes, p: Params) -> list[dict]:
    grouped = cv2.dilate(cand, ellipse(p.px(p.diff_group_mm)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    amin = p.min_diff_area_mm2 * p.px_per_mm ** 2
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        real = int(((lab[y:y + h, x:x + w] == i) & (cand[y:y + h, x:x + w] > 0)).sum())
        if real < amin or any(_overlap([x, y, w, h], b) > 0.5 for b in explained_boxes):
            continue
        out.append({"region_id": None, "name": None, "kind": None, "status": "DIFFERENCE", "bbox_px": [x, y, w, h],
                    "expected_px": None, "offset_px": [0, 0], "score": round(real / p.px_per_mm ** 2, 2),
                    "detail": "possible solder defect / bridge / extra part / debris (score = area mm^2)"})
    return out


# ---------------------------------------------------------------- drawing

COLOURS = {"MISSING": (0, 0, 255), "MISPLACED": (0, 140, 255), "DIFFERENT": (200, 0, 160),
           "DIFFERENCE": (0, 230, 255), "OK": (0, 200, 0), "NOT_CHECKED": (160, 160, 160)}
FINDING_TYPES = ("MISSING", "MISPLACED", "DIFFERENT", "DIFFERENCE")
FAILING = {0: ("MISSING", "MISPLACED"), 1: FINDING_TYPES}


def severity_of(status: str, p: Params) -> str:
    """"fail" findings decide FAIL; the others are notes (listed and drawn, but PASS stays PASS)."""
    return "fail" if status in FAILING[1 if p.fail_on >= 1 else 0] else "note"


def label_of(f: dict, p: Params) -> str:
    who = f["name"] or f["region_id"] or ""
    if f["status"] == "MISPLACED":
        dx, dy = (v / p.px_per_mm for v in f["offset_px"])
        return f"{who} MISPLACED {dx:+.1f}mm,{dy:+.1f}mm"
    return f"{who} {f['status']}".strip()


def draw(img, records, p: Params, scale: int = 2, show_ok: bool = False) -> np.ndarray:
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for f in sorted(records, key=lambda r: r.get("severity") == "fail"):   # failing findings on top
        if f["status"] not in FINDING_TYPES and not show_ok:
            continue
        c = COLOURS[f["status"]]
        x, y, w, h = (v * scale for v in f["bbox_px"])
        if f["status"] == "MISPLACED":
            ex, ey, ew, eh = (v * scale for v in f["expected_px"])
            cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), (160, 160, 160), 1)
            cv2.arrowedLine(out, (ex + ew // 2, ey + eh // 2), (x + w // 2, y + h // 2), c, 2, tipLength=0.3)
        if f["status"] not in FINDING_TYPES or f.get("severity") == "note":
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 1)    # OK regions and notes: thin box, small name
            cv2.putText(out, f["name"] or "", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
            continue
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
        label = label_of(f, p)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y - 4 if y - th - 6 > 0 else y + h + th + 4
        cv2.rectangle(out, (x, ty - th - 3), (x + tw + 4, ty + 3), c, -1)
        cv2.putText(out, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def heatmap(aligned, d, area) -> np.ndarray:
    dm = np.clip(d * 3, 0, 255).astype(np.uint8)
    dm[area == 0] = 0
    return cv2.addWeighted(aligned, 0.45, cv2.applyColorMap(dm, cv2.COLORMAP_JET), 0.55, 0)


# ---------------------------------------------------------------- reference + G. decision

@dataclass
class Reference:
    canonical: np.ndarray        # rectified reference board (BGR)
    valid: np.ndarray
    clahe: np.ndarray            # CLAHE gray (uint8)
    mask: np.ndarray             # part mask
    area: np.ndarray
    holes: np.ndarray            # through-hole interiors (don't care)
    regions: list[Region] = field(default_factory=list)
    body: np.ndarray | None = None        # body zones of the 'part' regions (255)
    board_core: np.ndarray | None = None  # bare board of the reference (255), for the local board colour
    silver: np.ndarray | None = None      # pads / terminals / solder / silkscreen of the reference (255)
    structure: np.ndarray | None = None   # structure outside the body zones (255): local alignment is measured on it
    body_px: dict = field(default_factory=dict)   # region id -> body zone pixels (below body_min_px: not checkable)


def reference_from_canonical(canonical: np.ndarray, p: Params, regions: list[Region] | None = None,
                             valid: np.ndarray | None = None) -> Reference:
    """Reference data from the rectified board; regions detected (D) unless given (e.g. user-edited)."""
    valid = np.full(canonical.shape[:2], 255, np.uint8) if valid is None else valid
    area = board_area(valid, p)
    clahe = clahe_gray(canonical)
    gray = cv2.GaussianBlur(clahe.astype(np.float32), (0, 0), p.sigma_base)
    mask = part_mask(canonical, board_hue_of(canonical, p), area, p)
    regions = detect_regions(mask, p) if regions is None else regions
    measure_regions(regions, mask, gray)
    holes = hole_mask(canonical, area, p)
    body, core, silver, counts = reference_body(canonical, area, holes, regions, p)
    structure = reference_structure(gray, body, area)
    return Reference(canonical, valid, clahe, mask, area, holes, regions, body, core, silver, structure, counts)


def prepare_reference(bgr: np.ndarray, p: Params, regions: list[Region] | None = None) -> Reference:
    board = detect_board(bgr, p)
    return reference_from_canonical(board.canonical, p, regions, board.valid)


def inspect(ref: Reference, new_bgr: np.ndarray, p: Params):
    """-> (result dict, images dict or None). RETAKE never yields PASS / FAIL."""
    t0 = time.time()
    base = {"params": asdict(p), "regions": len(ref.regions)}
    try:
        new = detect_board(new_bgr, p)
        al = align(new, ref.canonical, ref.valid, p)
        aligned, valid = warp_to_reference(new, al, p)
        cover = float((valid[ref.area > 0] > 0).mean())
        if cover < p.min_coverage:
            raise Retake(f"Board not completely visible ({cover:.0%} of the board in the photo) - retake")
        area = cv2.bitwise_and(ref.area, valid)
        new_hue = board_hue_of(aligned, p)
    except Retake as e:
        return {**base, "decision": "RETAKE", "reason": str(e), "findings": [], "regions_checked": [],
                "time_s": round(time.time() - t0, 2)}, None
    ref_gray, new_gray, sharp = comparison_grays(ref.clahe, clahe_gray(aligned), area, p)
    shifts = region_shifts(local_flow(ref_gray, new_gray), ref, ref.regions, p)   # before holes are filled
    new_mask = part_mask(aligned, new_hue, area, p)
    ref_mask = ref.mask.copy()
    holes = ref.holes > 0
    if holes.any():   # hole interiors: same constant gray in both, 'part' in both masks
        fill = float(np.median(ref_gray[holes]))
        ref_gray[holes] = fill
        new_gray[holes] = fill
        ref_mask[holes] = 255
        new_mask[holes] = 255
    missing = check_missing(ref.regions, ref, hsv_blur(aligned, p), new_hue, area, shifts, p)
    records = compare_regions(ref.regions, ref_gray, ref_mask, new_gray, new_mask, valid, missing, p)
    findings = [r for r in records if r["status"] in FINDING_TYPES]
    explained = [f["bbox_px"] for f in findings] + [f["expected_px"] for f in findings]
    d, cand = difference_map(ref_gray, new_gray, ref_mask, new_mask,
                             cv2.bitwise_and(area, cv2.bitwise_not(ref.holes)), p)
    findings += difference_findings(cand, explained, p)
    for f in findings:
        f["severity"] = severity_of(f["status"], p)
    failing = [f for f in findings if f["severity"] == "fail"]
    decision = "FAIL" if failing else "PASS"
    mm = p.px_per_mm

    def export(f: dict) -> dict:
        b, e = f["bbox_px"], f["expected_px"]
        return {"type": f["status"], "severity": f.get("severity"), "region_id": f["region_id"], "name": f["name"],
                "kind": f["kind"],
                "bbox_mm": [round(v / mm, 2) for v in b],
                "expected_bbox_mm": [round(v / mm, 2) for v in e] if e else None,
                "expected_centre_mm": [round((e[0] + e[2] / 2) / mm, 2), round((e[1] + e[3] / 2) / mm, 2)] if e else None,
                "actual_centre_mm": [round((b[0] + b[2] / 2) / mm, 2), round((b[1] + b[3] / 2) / mm, 2)],
                "offset_mm": [round(f["offset_px"][0] / mm, 2), round(f["offset_px"][1] / mm, 2)],
                "score": f["score"], "s_gray": f.get("s_gray"), "s_mask": f.get("s_mask"),
                "exposure": f.get("exposure"),
                "fraction_ref": f.get("fraction_ref"), "fraction_new": f.get("fraction_new"), "detail": f["detail"]}

    counts = {t: sum(f["status"] == t for f in findings) for t in FINDING_TYPES}
    counts |= {"ok": sum(r["status"] == "OK" for r in records),
               "not_checked": sum(r["status"] == "NOT_CHECKED" for r in records),
               "fail": len(failing), "notes": len(findings) - len(failing),
               "parts": sum(r.kind == "part" for r in ref.regions),
               "missing_checked": sum(v["status"] != "UNSURE" for v in missing.values()),
               "missing_unsure": sum(v["status"] == "UNSURE" for v in missing.values())}
    reason = ", ".join(f"{sum(f['status'] == t for f in failing)} {t.lower()}" for t in FINDING_TYPES
                       if any(f["status"] == t for f in failing))
    unsure, checkable = counts["missing_unsure"], counts["missing_checked"] + counts["missing_unsure"]
    if decision == "PASS" and unsure > p.max_unsure_fraction * checkable:
        # nothing was found, but too many parts could not be judged: a PASS would claim more than the photo shows
        return {**base, "decision": "RETAKE",
                "reason": f"Glare / reflections hide {unsure} of {checkable} parts - retake with softer light "
                          f"(no flash, no lamp shining straight onto the board)",
                "findings": [], "regions_checked": [], "time_s": round(time.time() - t0, 2)}, None
    result = {
        **base,
        "decision": decision,
        "reason": reason,
        "alignment": {"method": al.method, "inliers": al.inliers, "matches": al.matches,
                      "inlier_ratio": round(al.inlier_ratio, 3), "residual_mm": round(al.residual_mm, 3),
                      "rotation_deg": round(al.rotation_deg, 1), "coverage": round(cover, 3),
                      "full_frame_input": new.full_frame, "sharpness": round(new.sharpness, 1)},
        "normalisation": sharp,
        "local_alignment": {"regions_shifted": len(shifts),
                            "max_shift_mm": round(max((math.hypot(*s) for s in shifts.values()), default=0.0) / mm, 2)},
        "counts": counts,
        "findings": [export(f) for f in findings],
        "regions_checked": [export(r) for r in records],
        "time_s": round(time.time() - t0, 2),
    }
    # the annotated picture shows what decided the result; notes (other differences) stay in result.json
    images = {"aligned": aligned, "annotated": draw(aligned, failing, p), "heatmap": heatmap(aligned, d, area),
              "debug_blend": np.dstack([ref_gray, new_gray, ref_gray]).clip(0, 255).astype(np.uint8),
              "debug_masks": np.dstack([ref_mask, new_mask, np.zeros_like(new_mask)])}
    return result, images


# ---------------------------------------------------------------- files

def regions_json(ref: Reference, p: Params) -> dict:
    mm = p.px_per_mm
    return {"canonical_px": list(p.canon_size), "px_per_mm": mm, "board_mm": [p.board_w_mm, p.board_h_mm],
            "regions": [{"id": r.id, "name": r.name, "kind": r.kind,
                         "bbox_mm": [round(v / mm, 2) for v in r.bbox_px],
                         "centre_mm": [round((r.bbox_px[0] + r.bbox_px[2] / 2) / mm, 2),
                                       round((r.bbox_px[1] + r.bbox_px[3] / 2) / mm, 2)],
                         "part_fraction": round(r.part_fraction, 4), "texture_std": round(r.texture_std, 2),
                         "body_px": ref.body_px.get(r.id, 0)}
                        for r in ref.regions]}


def load_regions(path: Path, p: Params) -> list[Region]:
    """Regions from regions.json (bbox in mm), clamped to the canonical frame."""
    data = json.loads(path.read_text(encoding="utf-8"))
    W, H = p.canon_size
    out = []
    for r in data["regions"]:
        x, y, w, h = (int(round(v * p.px_per_mm)) for v in r["bbox_mm"])
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
        out.append(Region(r["id"], r.get("name") or r["id"], [x, y, w, h], r.get("kind", "")))
    return out


def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    _, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 92] if path.suffix == ".jpg" else [])
    buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("new", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--regions", type=Path, help="regions.json (e.g. with names); default: detect")
    ap.add_argument("--regions-only", action="store_true", help="only prepare the reference")
    ap.add_argument("--sensitivity", choices=list(SENSITIVITY), default="normal")
    ap.add_argument("--fail-on", type=int, choices=(0, 1), default=0,
                    help="0 = FAIL on missing / misplaced parts, other findings are notes (default); 1 = FAIL on every finding")
    ap.add_argument("--tolerance-mm", type=float, default=None, help="position tolerance (default 1.0 mm)")
    ap.add_argument("--debug", action="store_true", help="also write debug_blend.png and debug_masks.png")
    a = ap.parse_args()
    p = Params.with_sensitivity(a.sensitivity)
    p.fail_on = a.fail_on
    if a.tolerance_mm is not None:
        p.tol_position_mm = a.tolerance_mm
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        ref = prepare_reference(imread(a.reference), p, load_regions(a.regions, p) if a.regions else None)
    except Retake as e:
        raise SystemExit(f"reference unusable: {e}")
    (a.out / "regions.json").write_text(json.dumps(regions_json(ref, p), indent=1), encoding="utf-8")
    regs = [{"status": "OK", "bbox_px": r.bbox_px, "name": r.name, "region_id": r.id} for r in ref.regions]
    imwrite(a.out / "reference_regions.jpg", draw(ref.canonical, regs, p, show_ok=True))
    print(f"reference: {len(ref.regions)} regions")
    if a.regions_only or a.new is None:
        return
    result, images = inspect(ref, imread(a.new), p)
    result["reference"], result["input"] = str(a.reference), str(a.new)
    (a.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if images:
        imwrite(a.out / "aligned.png", images["aligned"])
        imwrite(a.out / "annotated.jpg", images["annotated"])
        imwrite(a.out / "heatmap.jpg", images["heatmap"])
        if a.debug:   # magenta/green = reference/new gray; blue/green = reference/new part mask
            imwrite(a.out / "debug_blend.png", images["debug_blend"])
            imwrite(a.out / "debug_masks.png", images["debug_masks"])
    print(f"{result['decision']}: {result.get('reason', '')} counts={result.get('counts')} "
          f"align={result.get('alignment')}")


if __name__ == "__main__":
    main()
