"""Component-level comparison against GOOD boards (Experiment 2 prototype; deterministic, nothing learned).

Pipeline (experiments/experiment_002/README.md):

    test image -> board segmentation -> SIFT/RANSAC homography to the GOOD reference   (existing Experiment 1 code)
               -> warp into the reference's canonical board frame (mm)                 (existing)
               -> for every named component box: compare with the GOOD examples        (this module)
               -> MISSING / present / not checked per component -> PASS / FAIL / REVIEW

GOOD examples ("gallery"): the reference image plus other GOOD images aligned into the same frame. A single
golden board bakes in that board's own placement quirks (on this board, Q9 sits offset on the reference
board only); a part is therefore "present" when it looks like the same part on ANY known-good example.

Per component box (canonical mm of the reference image):

  similarity            best normalised cross-correlation between a GOOD example's box patch and the test
                        (searched within +-search_mm for residual misalignment / parallax of tall parts),
                        maximised over all GOOD examples. Both images are blurred by similarity_blur_mm first,
                        so the coarse layout (body between terminals) counts, not the marking text, which
                        differs between component lots. Insensitive to brightness / contrast.
  difference (score)    1 - similarity.
  bare-board increase   test minus reference share of plain solder-mask colour in the box. Reported for
                        information; optional second condition (min_bare_board_increase). On this board an
                        empty footprint shows its pads and solder rather than solder mask, so it is weak.

A component is MISSING when its similarity stays below min_similarity (and, if configured, the bare-board
increase reaches min_bare_board_increase). Both are experiment parameters, not validated limits. A box not
sufficiently inside the aligned area, or covered by an ignored region (e.g. the DroidCam watermark letters),
is NOT CHECKED. The verdict is an experiment output, not a production decision (decision_engine thresholds
stay null until validated).

Component boxes come from a component map (YAML). Without a BOM / centroid file (PCB-11, PCB-12 UNKNOWN) they
are drawn on the reference image (scripts/define_components.py); every box records its source and whether a
person has verified it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from ..image_quality.checks import dominant_central_hue

REFDES_PATTERN = re.compile(r"[A-Z]{1,4}\d{1,3}[A-Z]?")
SOURCES = ("operator", "assistant_draft", "centroid_file")
PRESENT, MISSING, NOT_CHECKED = "present", "missing", "not_checked"
PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"


# --- component map ---------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    refdes: str
    box_mm: tuple[float, float, float, float]   # x0, y0, x1, y1 in the reference's canonical frame (origin top-left)
    source: str
    verified: bool
    note: str = ""


@dataclass
class ComponentMap:
    components: list[Component]
    reference: dict[str, Any]                    # image_id, file (relative to the project), sha256
    size_mm: tuple[float, float]                 # canonical frame (width, height)
    profile: str = ""
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def unverified(self) -> list[str]:
        return [c.refdes for c in self.components if not c.verified]

    def get(self, refdes: str) -> Component:
        return next(c for c in self.components if c.refdes == refdes)


def validate_components(components: Sequence[Component], size_mm: tuple[float, float]) -> list[str]:
    issues, seen = [], set()
    width, height = size_mm
    for c in components:
        if not REFDES_PATTERN.fullmatch(c.refdes):
            issues.append(f"{c.refdes!r}: reference designator must look like R37, LED4, U2, SW2")
        if c.refdes in seen:
            issues.append(f"{c.refdes}: defined twice")
        seen.add(c.refdes)
        x0, y0, x1, y1 = c.box_mm
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            issues.append(f"{c.refdes}: box {list(c.box_mm)} must satisfy 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}")
        if c.source not in SOURCES:
            issues.append(f"{c.refdes}: source {c.source!r} must be one of {SOURCES}")
    return issues


def load_component_map(path: Path) -> ComponentMap:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    size = tuple(float(v) for v in data["canonical_size_mm"])
    components = [
        Component(refdes=str(c["refdes"]), box_mm=tuple(float(v) for v in c["box_mm"]), source=str(c["source"]),
                  verified=bool(c["verified"]), note=str(c.get("note") or ""))
        for c in data.get("components") or []
    ]
    issues = validate_components(components, size)
    if issues:
        raise ValueError(f"{path}: invalid component map:\n  " + "\n  ".join(issues))
    header = {k: v for k, v in data.items() if k not in ("components", "profile", "reference", "canonical_size_mm")}
    return ComponentMap(components, dict(data["reference"]), size, str(data.get("profile") or ""), header)


def save_component_map(cmap: ComponentMap, path: Path, comment: str = "") -> None:
    issues = validate_components(cmap.components, cmap.size_mm)
    if issues:
        raise ValueError("invalid component map:\n  " + "\n  ".join(issues))
    data = {**cmap.header, "profile": cmap.profile, "reference": cmap.reference,
            "canonical_size_mm": [round(v, 3) for v in cmap.size_mm],
            "components": [{"refdes": c.refdes, "box_mm": [round(v, 2) for v in c.box_mm], "source": c.source,
                            "verified": c.verified, **({"note": c.note} if c.note else {})}
                           for c in sorted(cmap.components, key=_refdes_key)]}
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
    Path(path).write_text(comment + text, encoding="utf-8")


def _refdes_key(c: Component) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)([A-Z]?)", c.refdes)
    return (match.group(1), int(match.group(2))) if match else (c.refdes, 0)


# --- per-component comparison ----------------------------------------------------

@dataclass(frozen=True)
class ComponentCheckParams:
    search_mm: float = 0.5              # local shift search (residual misalignment, parallax of tall parts)
    search_step_mm: float = 0.1         # step of the bare-board shift search
    similarity_blur_mm: float = 0.2     # compare the coarse layout: blur away finer detail (marking text)
    min_similarity: float = 0.735       # experiment parameter: below this the part does not look like any GOOD example
    min_bare_board_increase: float | None = None  # optional second condition; None = information only
    min_usable_fraction: float = 0.6    # share of the box that must be comparable (aligned, not ignored)
    saturation_min: int = 60            # solder-mask colour, same definition as the board segmentation
    value_min: int = 40
    hue_tolerance: int = 12             # Pillow 0-255 hue scale


@dataclass
class ComponentResult:
    refdes: str
    status: str                         # present | missing | not_checked
    score: float | None                 # difference = 1 - similarity
    similarity: float | None            # best match over the GOOD examples
    best_example: int | None            # index of the best-matching GOOD example (0 = reference)
    bare_board_increase: float | None
    reference_fraction: float | None
    test_fraction: float | None
    box_px: tuple[int, int, int, int]
    verified: bool
    source: str
    reason: str = ""
    limit: float | None = None          # similarity limit applied (global or per-component calibration)


def board_colour_mask(bgr: np.ndarray, params: ComponentCheckParams, hue: int | None = None) -> tuple[np.ndarray, int | None]:
    """Pixels of the board's own solder-mask colour (bool), and the board hue used (Pillow 0-255 scale)."""
    hsv = np.asarray(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("HSV")).astype(np.int16)
    saturated = (hsv[..., 1] >= params.saturation_min) & (hsv[..., 2] >= params.value_min)
    if hue is None:
        hue = dominant_central_hue(hsv[..., 0], saturated)
    if hue is None:
        return np.zeros(bgr.shape[:2], bool), None
    distance = np.abs(hsv[..., 0] - hue)
    return saturated & (np.minimum(distance, 256 - distance) <= params.hue_tolerance), hue


def similarity_image(bgr: np.ndarray, params: ComponentCheckParams, px_per_mm: float) -> np.ndarray:
    """Grey image prepared for the similarity measure (blurred by similarity_blur_mm)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = params.similarity_blur_mm * px_per_mm
    return cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0.3 else gray


def box_to_px(box_mm: tuple[float, float, float, float], px_per_mm: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_mm
    return round(x0 * px_per_mm), round(y0 * px_per_mm), round(x1 * px_per_mm), round(y1 * px_per_mm)


def check_components(reference_bgr: np.ndarray, aligned_bgr: np.ndarray, usable: np.ndarray, cmap: ComponentMap,
                     px_per_mm: float, params: ComponentCheckParams = ComponentCheckParams(),
                     gallery: Sequence[np.ndarray] = (), limits: dict[str, float] | None = None,
                     reference_is_example: bool = True) -> list[ComponentResult]:
    """Compare every component box of the map.

    usable: pixels aligned in both images and not ignored (bool / 0-255).
    gallery: further GOOD examples, already aligned into the reference frame and prepared with
    similarity_image(); the reference itself is example 0 unless reference_is_example is False
    (leave-one-board-out when the inspected board is the reference's own board).
    limits: per-component similarity limits (calibrated on GOOD boards); missing entries use min_similarity.
    """
    usable = usable.astype(bool)
    ref_board, _ = board_colour_mask(reference_bgr, params)
    test_board, _ = board_colour_mask(aligned_bgr, params)   # own hue estimate: tolerates a white-balance shift
    examples = ([similarity_image(reference_bgr, params, px_per_mm)] if reference_is_example else []) + list(gallery)
    test_gray = similarity_image(aligned_bgr, params, px_per_mm)
    h, w = usable.shape
    reach = max(0, round(params.search_mm * px_per_mm))
    step = max(1, round(params.search_step_mm * px_per_mm))
    results = []
    for c in cmap.components:
        x0, y0, x1, y1 = box_to_px(c.box_mm, px_per_mm)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        base = dict(refdes=c.refdes, box_px=(x0, y0, x1, y1), verified=c.verified, source=c.source)
        use = usable[y0:y1, x0:x1]
        if use.size == 0 or use.mean() < params.min_usable_fraction:
            results.append(ComponentResult(**base, status=NOT_CHECKED, score=None, similarity=None, best_example=None,
                                           bare_board_increase=None, reference_fraction=None, test_fraction=None,
                                           reason=f"only {use.mean() if use.size else 0:.0%} of the box is comparable "
                                                  "(outside the aligned area or under an ignored region)"))
            continue
        ref_fraction = float(ref_board[y0:y1, x0:x1][use].mean())
        test_fraction = _lowest_board_fraction(test_board, usable, (x0, y0, x1, y1), reach, step, params)
        increase = None if test_fraction is None else test_fraction - ref_fraction
        similarity, best = _best_similarity(examples, test_gray, (x0, y0, x1, y1), reach)
        limit = (limits or {}).get(c.refdes, params.min_similarity)
        looks_different = similarity is None or similarity < limit
        more_board = params.min_bare_board_increase is None or (
            increase is not None and increase >= params.min_bare_board_increase)
        results.append(ComponentResult(
            **base, status=MISSING if looks_different and more_board else PRESENT,
            score=None if similarity is None else round(1.0 - similarity, 3),
            similarity=None if similarity is None else round(similarity, 3), best_example=best,
            bare_board_increase=None if increase is None else round(increase, 3),
            reference_fraction=round(ref_fraction, 3),
            test_fraction=None if test_fraction is None else round(test_fraction, 3), limit=round(limit, 3),
        ))
    return results


def _lowest_board_fraction(board: np.ndarray, usable: np.ndarray, box: tuple[int, int, int, int], reach: int,
                           step: int, params: ComponentCheckParams) -> float | None:
    """Bare-board share of the box at the local shift where it is lowest (the most favourable 'present' reading)."""
    x0, y0, x1, y1 = box
    h, w = usable.shape
    area = (x1 - x0) * (y1 - y0)
    best = None
    for dy in range(-reach, reach + 1, step):
        for dx in range(-reach, reach + 1, step):
            ya, yb, xa, xb = y0 + dy, y1 + dy, x0 + dx, x1 + dx
            if ya < 0 or xa < 0 or yb > h or xb > w:
                continue
            use = usable[ya:yb, xa:xb]
            if use.sum() < params.min_usable_fraction * area:
                continue
            fraction = float(board[ya:yb, xa:xb][use].mean())
            best = fraction if best is None else min(best, fraction)
    return best


def _best_similarity(examples: Sequence[np.ndarray], test_gray: np.ndarray, box: tuple[int, int, int, int],
                     reach: int) -> tuple[float | None, int | None]:
    x0, y0, x1, y1 = box
    h, w = test_gray.shape
    window = test_gray[max(0, y0 - reach):min(h, y1 + reach), max(0, x0 - reach):min(w, x1 + reach)]
    best, index = None, None
    for i, example in enumerate(examples):
        patch = example[y0:y1, x0:x1]
        if patch.std() < 1.0 or window.shape[0] < patch.shape[0] or window.shape[1] < patch.shape[1]:
            continue
        value = float(cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED).max())
        if best is None or value > best:
            best, index = value, i
    return best, index


# --- proposals (snap a drawn box to the part inside it) ---------------------------------

def part_mask(bgr: np.ndarray, px_per_mm: float, params: ComponentCheckParams = ComponentCheckParams(),
              stroke_mm: float = 0.35) -> np.ndarray:
    """Pixels that are not solder mask, with thin silkscreen strokes removed (uint8 0/255)."""
    board, _ = board_colour_mask(bgr, params)
    k = max(3, round(stroke_mm * px_per_mm) | 1)
    return cv2.morphologyEx((~board).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def snap_box(parts: np.ndarray, box_px: tuple[int, int, int, int], pad_px: int = 3) -> tuple[int, int, int, int] | None:
    """Tight box around the part pixels inside a roughly drawn box, or None if there are none."""
    x0, y0, x1, y1 = box_px
    ys, xs = np.nonzero(parts[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    h, w = parts.shape
    return (max(0, x0 + int(xs.min()) - pad_px), max(0, y0 + int(ys.min()) - pad_px),
            min(w, x0 + int(xs.max()) + 1 + pad_px), min(h, y0 + int(ys.max()) + 1 + pad_px))


# --- verdict and report -------------------------------------------------------------

def overall_status(results: Sequence[ComponentResult], aligned: bool) -> str:
    if not aligned:
        return REVIEW
    if any(r.status == MISSING for r in results):
        return FAIL
    if not results or any(r.status == NOT_CHECKED for r in results):
        return REVIEW
    return PASS


def format_report(image: str, status: str, results: Sequence[ComponentResult], cmap: ComponentMap, alignment: str,
                  reference_id: str, params: ComponentCheckParams, good_examples: int,
                  stage_lines: Sequence[str] = (), limits_note: str = "", reason: str = "") -> str:
    line = "=" * 48
    missing = sorted((r for r in results if r.status == MISSING), key=lambda r: -(r.score or 0))
    skipped = [r for r in results if r.status == NOT_CHECKED]
    checked = len(results) - len(skipped)
    lines = [line, "PCB INSPECTION RESULT", line, "",
             f"Image:      {image}",
             f"Reference:  {reference_id} + {good_examples - 1} more GOOD image(s)",
             *stage_lines,
             f"Alignment:  {alignment}",
             f"Components: {checked} of {len(cmap.components)} in the component map checked"]
    if limits_note:
        lines.append(f"Limits:     {limits_note}")
    if cmap.unverified:
        lines.append(f"            component map: {len(cmap.unverified)} of {len(cmap.components)} boxes are an "
                     "UNVERIFIED draft (verify with scripts/define_components.py)")
    lines += ["", f"Status: {status.replace('_', ' ')}", ""]
    if status == FAIL:
        lines.append("Missing/Defective Components:")
        lines += [f"{i}. {r.refdes}" for i, r in enumerate(missing, 1)]
        lines += ["", "Confidence/Score (difference from the closest GOOD example; FAIL above 1 - limit):"]
        lines += [f"{r.refdes}: {r.score:.2f}   (similarity {r.similarity:.2f}, limit {r.limit:.3f}"
                  f"{'' if r.verified else '; box not verified'})" for r in missing]
    elif status == PASS:
        lines.append(f"All expected components detected ({checked} components in the component map).")
    elif reason:
        lines += [reason, "No PASS / FAIL decision was made."]
    elif alignment.startswith("OK"):
        lines.append("Not decided automatically: some components could not be checked.")
    else:
        lines.append("Alignment to the GOOD reference failed: the image could not be inspected.")
    if skipped:
        lines += ["", "Not checked:"] + [f"  {r.refdes}: {r.reason}" for r in skipped]
    lines += ["", "Experiment 2 prototype: deterministic comparison with GOOD images. Thresholds are",
              "experiment parameters, not validated limits. Not a production verdict.", line]
    return "\n".join(lines)
