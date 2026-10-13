"""Inspect one image against GOOD boards, stage by stage, component by component (Experiment 2 prototype).

    frame valid?  -> watermark located + inpainted -> PCB present? -> registration (SIFT/RANSAC, Experiment 1)
    -> alignment validated (quality, board whole in frame, orientation, perspective) -> canonical warp
    -> per-component comparison with the GOOD reference set -> plausibility -> PASS / FAIL / REVIEW

Final states: INVALID_FRAME, NO_PCB, PCB_POSITION_INVALID, ALIGNMENT_FAILED (no inspection was made),
PASS, FAIL, REVIEW (a part could not be checked), INSPECTION_ERROR (implausible result: too many parts differ).
PASS is only possible after every stage succeeded.

Reuses unchanged: board segmentation, canonical frame and SIFT/RANSAC registration (preprocessing.board,
alignment.registration, comparison.pipeline) and the component comparison (comparison.components).
The GOOD reference set comes from the board registry (dataset.registry): curated GOOD images of the boards in
the reference split, aligned into the reference frame once and cached as derived data under data/processed/.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..alignment.registration import RegistrationResult, register
from ..alignment.validation import FAILED as ALIGN_FAILED
from ..alignment.validation import POSITION_INVALID as ALIGN_POSITION_INVALID
from ..alignment.validation import AlignmentCheck, AlignmentValidationParams, validate_alignment
from ..config import load_config
from ..dataset.experiment0 import sha256_of
from ..dataset.registry import IndexRow, build_index, load_registry, reference_images
from ..preprocessing.board import board_mask, warp_to_canonical
from ..preprocessing.presence import NO_PCB as PRESENCE_NO_PCB
from ..preprocessing.presence import VALID, PresenceParams, check_frame, detect_pcb
from ..preprocessing.watermark import WatermarkMatch, WatermarkParams, load_template, locate_watermark, remove_watermark
from ..visualization import panels
from .components import (
    FAIL,
    MISSING,
    NOT_CHECKED,
    PASS,
    REVIEW,
    ComponentCheckParams,
    ComponentMap,
    ComponentResult,
    check_components,
    format_report,
    load_component_map,
    overall_status,
    similarity_image,
)
from .pipeline import PipelineConfig, Reference, load_bgr, load_pipeline_config, prepare_reference

GOOD_REFERENCE_DIR = "data/curated/good/"
INVALID_FRAME, NO_PCB, POSITION_INVALID = "INVALID_FRAME", "NO_PCB", "PCB_POSITION_INVALID"
ALIGNMENT_FAILED, INSPECTION_ERROR = "ALIGNMENT_FAILED", "INSPECTION_ERROR"
NOT_INSPECTED = (INVALID_FRAME, NO_PCB, POSITION_INVALID, ALIGNMENT_FAILED)
STATES = (*NOT_INSPECTED, PASS, FAIL, REVIEW, INSPECTION_ERROR)


# --- configuration ---------------------------------------------------------------------

@dataclass(frozen=True)
class IgnoreRegion:
    box_px: tuple[int, int, int, int]       # x0, y0, x1, y1 in the source image
    image_size: tuple[int, int] | None       # applies only to images of this (width, height); None = all
    reason: str
    mask: np.ndarray | None = None           # optional pixel mask (0/255) for the box; None = the whole box


@dataclass
class InspectionConfig:
    pipeline: PipelineConfig
    check: ComponentCheckParams
    ignore: list[IgnoreRegion]
    watermark: WatermarkParams
    presence: PresenceParams
    alignment: AlignmentValidationParams
    max_plausible_defects: int
    reference_file: Path
    component_map_file: Path
    registry_file: Path | None
    image_boards_file: Path | None
    reference_split: str
    calibration_file: Path | None
    calibration_min_boards: int
    calibration_margin: float
    cache_dir: Path
    input_dir: Path
    output_dir: Path
    experiment: str
    project_root: Path
    data_root: Path | None = None                   # acquisition store of the GOOD boards; None = <project>/data
    extra_data_roots: tuple[Path, ...] = ()         # further acquisition stores of the same product
    acquisition_dir: str = "acquisition"            # session folder inside the store (clean collection: originals)


def load_inspection_config(project_root: Path, config_path: Path) -> InspectionConfig:
    raw = load_config(config_path)
    reference = raw["reference_image"].replace("\\", "/")
    if not reference.startswith(GOOD_REFERENCE_DIR):
        raise ValueError(f"reference_image must be a curated GOOD image under {GOOD_REFERENCE_DIR}, got {reference}")
    rs = raw.get("reference_set") or {}
    wm = dict(raw.get("watermark") or {"enabled": False})
    if wm.get("template"):
        wm["template"] = project_root / wm["template"]
    if wm.get("search_region"):
        wm["search_region"] = tuple(wm["search_region"])
    wm.pop("observed", None)
    path = lambda key: project_root / rs[key] if rs.get(key) else None  # noqa: E731
    return InspectionConfig(
        pipeline=load_pipeline_config(project_root, config_path),
        check=ComponentCheckParams(**raw["component_check"]),
        ignore=[_ignore_region(project_root, r) for r in raw.get("ignore_source_regions_px") or []],
        watermark=WatermarkParams(**wm),
        presence=PresenceParams(**(raw.get("pcb_presence") or {})),
        alignment=AlignmentValidationParams(**(raw.get("alignment_validation") or {})),
        max_plausible_defects=int((raw.get("inspection_decision") or {}).get("max_plausible_defects", 10 ** 6)),
        reference_file=project_root / reference,
        component_map_file=project_root / raw["component_map"],
        registry_file=path("registry"), image_boards_file=path("image_boards"),
        reference_split=rs.get("split", "reference"), calibration_file=path("calibration"),
        calibration_min_boards=int(rs.get("min_boards", 5)), calibration_margin=float(rs.get("margin", 0.05)),
        cache_dir=project_root / raw["cache_dir"],
        input_dir=project_root / raw["input_dir"],
        output_dir=project_root / raw["output_dir"],
        experiment=raw["experiment"],
        project_root=project_root,
    )


def _ignore_region(project_root: Path, r: dict[str, Any]) -> IgnoreRegion:
    box = tuple(int(v) for v in r["box"])
    mask = None
    if r.get("mask_file"):
        mask = cv2.imread(str(project_root / r["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (box[3] - box[1], box[2] - box[0]):
            raise ValueError(f"ignore mask {r['mask_file']} is missing or does not match its box {list(box)}")
    return IgnoreRegion(box, tuple(r["image_size"]) if r.get("image_size") else None, r["reason"], mask)


def static_overlay_mask(images: list[np.ndarray], box_px: tuple[int, int, int, int], min_level: int = 170,
                        dilate_px: int = 3) -> np.ndarray:
    """Pixels inside box that are bright in EVERY image: a burned-in overlay (e.g. a camera-app watermark)."""
    x0, y0, x1, y1 = box_px
    darkest = None
    for img in images:
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img[y0:y1, x0:x1]
        darkest = gray if darkest is None else np.minimum(darkest, gray)
    mask = (darkest >= min_level).astype(np.uint8) * 255
    if dilate_px:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    return mask


def ignore_mask(shape: tuple[int, int], regions: list[IgnoreRegion]) -> np.ndarray:
    """0/255 mask of the static ignored regions that apply to an image of this shape (height, width)."""
    mask = np.zeros(shape, np.uint8)
    for r in regions:
        if r.image_size is None or tuple(r.image_size) == (shape[1], shape[0]):
            x0, y0, x1, y1 = r.box_px
            mask[y0:y1, x0:x1] = 255 if r.mask is None else np.maximum(mask[y0:y1, x0:x1], r.mask)
    return mask


def _source_ignore(shape: tuple[int, int], cfg: InspectionConfig, match: WatermarkMatch | None) -> np.ndarray:
    mask = ignore_mask(shape, cfg.ignore) > 0
    if match is not None and match.found and match.mask is not None:
        mask |= match.mask
    return mask


def _warp_mask(mask: np.ndarray, homography: np.ndarray, size: tuple[int, int], source_ppm: float, ppm: float) -> np.ndarray:
    if not mask.any():
        return np.zeros((size[1], size[0]), bool)
    warped, _ = warp_to_canonical(mask.astype(np.uint8) * 255 if mask.dtype == bool else mask, homography, size,
                                  source_ppm, ppm)
    return warped > 0


# --- dataset, reference, gallery, calibration ---------------------------------------------------

def load_dataset_index(cfg: InspectionConfig) -> tuple[list[IndexRow], list[str], list[str]]:
    if cfg.registry_file is None or not cfg.registry_file.is_file():
        return [], [f"board registry not found: {cfg.registry_file}"], []
    registry = load_registry(cfg.registry_file, cfg.image_boards_file)
    return build_index(cfg.data_root or cfg.project_root / "data", registry, cfg.project_root, cfg.extra_data_roots,
                       cfg.acquisition_dir)


def board_of(image_id: str, rows: list[IndexRow]) -> str:
    return next((r.board_id for r in rows if r.image_id == image_id), "")


@dataclass
class InspectionReference(Reference):
    """The geometric GOOD reference, with the watermark removed, plus what the inspection needs about it."""

    ignore_canonical: np.ndarray | None = None   # ignored pixels (watermark letters, static regions) in the frame
    watermark: WatermarkMatch | None = None
    board_id: str = ""
    template: np.ndarray | None = None           # watermark glyph template


def load_reference_and_map(cfg: InspectionConfig, reference_out_dir: Path) -> tuple[InspectionReference, ComponentMap]:
    """Rectify the GOOD reference (watermark removed) and load the component map drawn on it (checked by checksum)."""
    cmap = load_component_map(cfg.component_map_file)
    digest = sha256_of(cfg.reference_file)
    if cmap.reference.get("sha256") != digest:
        raise ValueError(f"{cfg.component_map_file.name} was drawn on another reference image "
                         f"({cmap.reference.get('image_id')}); configured reference: {cfg.reference_file.name}")
    reference_out_dir.mkdir(parents=True, exist_ok=True)
    info = {"id": cfg.reference_file.stem, "path": str(cfg.reference_file), "board_id": "", "sha256": digest}
    base = prepare_reference(info, cfg.pipeline, reference_out_dir, cfg.experiment)
    expected = (round(cmap.size_mm[0] * cfg.pipeline.px_per_mm), round(cmap.size_mm[1] * cfg.pipeline.px_per_mm))
    if tuple(base.frame.size) != expected:
        raise ValueError(f"canonical frame {base.frame.size} px does not match the component map ({expected} px)")
    template = load_template(cfg.watermark)
    match = locate_watermark(base.image, template, cfg.watermark)
    clean = remove_watermark(base.image, match)
    frame, p = base.frame, cfg.pipeline
    canonical, _ = warp_to_canonical(clean, frame.homography, frame.size, frame.source_px_per_mm, p.px_per_mm)
    ignore = _warp_mask(_source_ignore(clean.shape[:2], cfg, match), frame.homography, frame.size,
                        frame.source_px_per_mm, p.px_per_mm)
    rows = load_dataset_index(cfg)[0]
    values = {f.name: getattr(base, f.name) for f in fields(Reference)}
    values.update(image=clean, canonical=canonical)
    return InspectionReference(**values, ignore_canonical=ignore, watermark=match,
                               board_id=board_of(cfg.reference_file.stem, rows), template=template), cmap


@dataclass
class GoodGallery:
    """GOOD examples aligned into the reference frame (the reference itself is handled separately)."""

    names: list[str]                      # image ids; names[0] = the reference
    boards: list[str]                     # physical board of each name
    images: list[np.ndarray]              # similarity_image() of each example except the reference
    files: list[Path] = field(default_factory=list)   # cached aligned BGR of each example except the reference
    skipped: list[str] = field(default_factory=list)

    @property
    def reference_board(self) -> str:
        return self.boards[0] if self.boards else ""

    @property
    def board_ids(self) -> list[str]:
        return sorted({b for b in self.boards if b})

    def without_board(self, board: str | None) -> GoodGallery:
        """Leave-one-board-out: drop every example of this physical board (and the reference, if it is that board)."""
        if not board:
            return self
        keep = [i for i, b in enumerate(self.boards[1:]) if b != board]
        return GoodGallery([self.names[0]] + [self.names[i + 1] for i in keep],
                           [self.boards[0]] + [self.boards[i + 1] for i in keep],
                           [self.images[i] for i in keep], [self.files[i] for i in keep] if self.files else [],
                           self.skipped)


def _cache_key(cfg: InspectionConfig, reference: InspectionReference) -> str:
    wm = cfg.watermark
    parts = [str(cfg.pipeline.px_per_mm), reference.info["sha256"], str(wm.enabled), str(wm.min_score),
             str(wm.min_margin), str(wm.dilate_px), str(wm.search_region),
             hashlib.sha256(reference.template.tobytes()).hexdigest() if reference.template is not None else "-",
             repr(cfg.alignment)]     # the alignment limits decide which GOOD images may enter the gallery
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def align_to_reference(image: np.ndarray, reference: InspectionReference, cfg: InspectionConfig
                       ) -> tuple[np.ndarray | None, str]:
    """Watermark removed, registered and warped into the reference frame; (None, reason) if not alignable."""
    p = cfg.pipeline
    match = locate_watermark(image, reference.template, cfg.watermark)
    clean = remove_watermark(image, match)
    registration = register(reference.image, reference.mask, clean, board_mask(clean, p.segmentation), p.registration)
    check = validate_alignment(registration, reference.quad, reference.frame.source_px_per_mm, clean.shape, cfg.alignment)
    if check.status != "OK":
        return None, check.reason
    to_canonical = reference.frame.homography @ registration.homography
    test_ppm = reference.frame.source_px_per_mm / float(np.sqrt(abs(np.linalg.det(registration.homography[:2, :2]))))
    aligned, _ = warp_to_canonical(clean, to_canonical, reference.frame.size, test_ppm, p.px_per_mm)
    return aligned, ""


def load_gallery(cfg: InspectionConfig, reference: InspectionReference) -> GoodGallery:
    """GOOD images of the reference boards (board registry), aligned into the reference frame; cached."""
    rows, issues, _ = load_dataset_index(cfg)
    gallery = GoodGallery([reference.info["id"]], [reference.board_id], [], [], [])
    if not rows:
        gallery.skipped += issues
        return gallery
    key = _cache_key(cfg, reference)
    cache = cfg.cache_dir / "gallery" / reference.info["id"] / key
    cache.mkdir(parents=True, exist_ok=True)
    for row in reference_images(rows, cfg.reference_split):
        path = cfg.project_root / row.file
        if path.resolve() == cfg.reference_file.resolve():
            continue
        digest = sha256_of(path)
        cached, meta = cache / f"{path.stem}.png", cache / f"{path.stem}.json"
        aligned = None
        if cached.is_file() and meta.is_file():
            if json.loads(meta.read_text(encoding="utf-8")).get("source_sha256") == digest:
                aligned = cv2.imread(str(cached))
        if aligned is None:
            aligned, reason = align_to_reference(load_bgr(path), reference, cfg)
            if aligned is None:
                gallery.skipped.append(f"{path.name}: {reason}")
                continue
            cv2.imwrite(str(cached), aligned)
            meta.write_text(json.dumps({"source": row.file, "source_sha256": digest, "board_id": row.board_id,
                                        "reference": reference.info["id"], "cache_key": key,
                                        "note": "derived: GOOD image, watermark removed, aligned into the reference frame"},
                                       indent=2), encoding="utf-8")
        gallery.names.append(path.stem)
        gallery.boards.append(row.board_id)
        gallery.images.append(similarity_image(aligned, cfg.check, cfg.pipeline.px_per_mm))
        gallery.files.append(cached)
    return gallery


def load_calibration(cfg: InspectionConfig, gallery: GoodGallery) -> tuple[dict[str, float] | None, str]:
    """Per-component limits if a current calibration from enough GOOD boards exists; else (None, why)."""
    fallback = f"global limit {cfg.check.min_similarity}"
    if cfg.calibration_file is None or not cfg.calibration_file.is_file():
        return None, f"{fallback} (no per-component calibration yet)"
    cal = json.loads(cfg.calibration_file.read_text(encoding="utf-8"))
    if sorted(cal.get("boards", [])) != gallery.board_ids:
        return None, f"{fallback} (calibration is for boards {cal.get('boards')}, reference set is {gallery.board_ids})"
    if len(cal["boards"]) < cfg.calibration_min_boards:
        return None, (f"{fallback} (per-component limits need >= {cfg.calibration_min_boards} GOOD boards; "
                      f"reference set has {len(cal['boards'])})")
    limits = {k: v["limit"] for k, v in cal["components"].items() if v.get("limit") is not None}
    return limits, f"per-component limits from {len(cal['boards'])} GOOD boards"


# --- inspection --------------------------------------------------------------------------

@dataclass
class InspectionResult:
    image: Path | None
    status: str
    stages: dict[str, str]
    reason: str
    report: str
    registration: RegistrationResult | None = None
    alignment_check: AlignmentCheck | None = None
    alignment_text: str = ""
    components: list[ComponentResult] = field(default_factory=list)
    aligned: np.ndarray | None = None
    usable: np.ndarray | None = None
    watermark: WatermarkMatch | None = None
    examples: list[str] = field(default_factory=list)
    limits_note: str = ""
    frame_shape: tuple[int, ...] = ()

    @property
    def missing(self) -> list[ComponentResult]:
        return [r for r in self.components if r.status == MISSING]

    @property
    def inspected(self) -> bool:
        return self.status not in NOT_INSPECTED


def _stage_lines(stages: dict[str, str]) -> list[str]:
    names = {"frame": "Frame", "watermark": "Watermark", "pcb": "PCB", "alignment": "Alignment check",
             "inspection": "Inspection"}
    return [f"{names[k] + ':':<12}{v}" for k, v in stages.items() if k in names]


def _early(path: Path | None, status: str, reason: str, stages: dict[str, str], **extra: Any) -> InspectionResult:
    line = "=" * 48
    name = path.name if path else "(camera frame)"
    report = "\n".join([line, "PCB INSPECTION RESULT", line, "", f"Image:      {name}", *_stage_lines(stages), "",
                        f"Status: {status.replace('_', ' ')}", reason, "No component inspection was made "
                        "(this is not a PASS).", line])
    return InspectionResult(path, status, stages, reason, report, **extra)


def inspect_image(image: Path | np.ndarray, reference: InspectionReference, cmap: ComponentMap, cfg: InspectionConfig,
                  gallery: GoodGallery | None = None, exclude_board: str | None = None,
                  limits: dict[str, float] | None = None, limits_note: str = "",
                  registration: RegistrationResult | None = None) -> InspectionResult:
    """All stages for one image (path or BGR array). exclude_board: physical board of the inspected image, if it
    is in the GOOD reference set (leave-one-board-out). registration: an earlier registration of this same image
    against this same reference (e.g. from product identification), reused instead of computed again."""
    p = cfg.pipeline
    path = image if isinstance(image, Path) else None
    bgr = load_bgr(path) if path is not None else image
    gallery = gallery or GoodGallery([reference.info["id"]], [reference.board_id], [], [], [])
    stages: dict[str, str] = {}

    frame = check_frame(bgr, cfg.presence)
    stages["frame"] = frame.status if frame.status == VALID else f"{frame.status}: {frame.reason}"
    if frame.status != VALID:
        return _early(path, INVALID_FRAME, frame.reason, stages, frame_shape=() if bgr is None else bgr.shape)

    match = locate_watermark(bgr, reference.template, cfg.watermark)
    stages["watermark"] = (f"found at x {match.box_px[0]}-{match.box_px[2]}, y {match.box_px[1]}-{match.box_px[3]} "
                           f"(score {match.score:.2f}); {int(match.mask.sum())} letter pixels inpainted"
                           if match.found else "not found (nothing masked)")
    clean = remove_watermark(bgr, match)
    presence = detect_pcb(clean, cfg.presence, p.segmentation)
    stages["pcb"] = (f"DETECTED (board colour {presence.area_fraction:.0%} of the frame)"
                     if presence.status != PRESENCE_NO_PCB else "NO PCB")
    if presence.status == PRESENCE_NO_PCB:
        return _early(path, NO_PCB, presence.reason, stages, watermark=match, frame_shape=bgr.shape)

    if registration is None:
        registration = register(reference.image, reference.mask, clean, presence.mask, p.registration)
    check = validate_alignment(registration, reference.quad, reference.frame.source_px_per_mm, bgr.shape, cfg.alignment)
    text = (f"{registration.inliers} inliers, {check.summary()}" if registration.success else registration.message)
    stages["alignment"] = f"{check.status}: {text}" + (f" -> {check.reason}" if check.reason and registration.success else "")
    extra = dict(registration=registration, alignment_check=check, watermark=match, frame_shape=bgr.shape)
    if check.status == ALIGN_FAILED:
        return _early(path, ALIGNMENT_FAILED, f"PCB not aligned: {check.reason}", stages, **extra)
    if check.status == ALIGN_POSITION_INVALID:
        return _early(path, POSITION_INVALID, check.reason, stages, **extra)

    to_canonical = reference.frame.homography @ registration.homography
    test_ppm = reference.frame.source_px_per_mm / float(np.sqrt(abs(np.linalg.det(registration.homography[:2, :2]))))
    aligned, valid = warp_to_canonical(clean, to_canonical, reference.frame.size, test_ppm, p.px_per_mm)
    usable = (valid > 0) & (reference.valid > 0)
    if reference.ignore_canonical is not None:
        usable &= ~reference.ignore_canonical
    usable &= ~_warp_mask(_source_ignore(bgr.shape[:2], cfg, match), to_canonical, reference.frame.size, test_ppm,
                          p.px_per_mm)
    examples = gallery.without_board(exclude_board)
    reference_is_example = not exclude_board or exclude_board != reference.board_id
    results = check_components(reference.canonical, aligned, usable, cmap, p.px_per_mm, cfg.check, examples.images,
                               limits, reference_is_example)
    status, reason = overall_status(results, True), ""
    missing = [r.refdes for r in results if r.status == MISSING]
    if len(missing) > cfg.max_plausible_defects:
        status = INSPECTION_ERROR
        reason = (f"{len(missing)} of {len(results)} components differ from every GOOD example: not a credible defect "
                  "pattern (hand or object over the board, wrong board / side / stage, or local misalignment). "
                  "Differing: " + ", ".join(missing))
    stages["inspection"] = status
    names = (examples.names if reference_is_example else examples.names[1:])
    alignment_text = f"OK ({registration.inliers} inliers, {check.summary()})"
    note = limits_note or f"global limit {cfg.check.min_similarity}"
    if exclude_board:
        note += f"; board {exclude_board} itself left out of the GOOD examples"
    report = format_report(path.name if path else "(camera frame)", status, results, cmap, alignment_text,
                           reference.info["id"], cfg.check, len(names),
                           [line for k, line in zip(stages, _stage_lines(stages)) if k in ("frame", "watermark", "pcb")],
                           note, reason)
    return InspectionResult(path, status, stages, reason, report, registration, check, alignment_text, results,
                            aligned, usable, match, names, note, bgr.shape)


# --- outputs -----------------------------------------------------------------------------------

COLOURS = {"present": (60, 180, 60), "missing": (40, 40, 230), "not_checked": (160, 160, 160)}


def annotate(result: InspectionResult, px_per_mm: float) -> np.ndarray:
    """Aligned test image with every component box: green present, red missing, grey not checked."""
    img = result.aligned.copy()
    img[~result.usable] = (img[~result.usable] * 0.5).astype(np.uint8)
    thick = max(2, round(0.12 * px_per_mm))
    for r in sorted(result.components, key=lambda r: r.status == MISSING):
        x0, y0, x1, y1 = r.box_px
        colour = COLOURS[r.status]
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, thick * (2 if r.status == MISSING else 1))
        label = r.refdes + ("?" if not r.verified else "")
        cv2.putText(img, label, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if r.status != MISSING else 0.9,
                    colour, 2 if r.status == MISSING else 1, cv2.LINE_AA)
    banner = f"{result.status}: " + (", ".join(f"{r.refdes} MISSING" for r in result.missing)
                                     if result.status in (FAIL, INSPECTION_ERROR)
                                     else ("all checked components present" if result.status == PASS else "see report"))
    banner = banner[:110]
    cv2.putText(img, banner, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(img, banner, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                COLOURS["missing"] if result.status in (FAIL, INSPECTION_ERROR) else COLOURS["present"], 3, cv2.LINE_AA)
    return img


def overview(bgr: np.ndarray, result: InspectionResult, scale: float = 0.4) -> np.ndarray:
    """The frame as captured (downscaled): watermark letters in magenta, board outline from the alignment, status."""
    img = bgr.copy()
    if result.watermark is not None and result.watermark.found:
        img[result.watermark.mask] = (255, 0, 255)
    check = result.alignment_check
    if check is not None and check.corners_px is not None:
        colour = (0, 200, 0) if check.status == "OK" else (0, 0, 255)
        cv2.polylines(img, [check.corners_px.astype(np.int32)], True, colour, max(3, img.shape[1] // 400))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for colour, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(img, result.status.replace("_", " "), (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, width, cv2.LINE_AA)
        if result.reason:
            cv2.putText(img, result.reason[:95], (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, max(1, width - 3),
                        cv2.LINE_AA)
    return img


def evidence(result: InspectionResult, reference: Reference, px_per_mm: float, margin_mm: float = 3.0,
             zoom: float = 3.0) -> np.ndarray | None:
    """For every missing component: GOOD reference crop | test crop (same canonical area), enlarged."""
    rows = []
    m = round(margin_mm * px_per_mm)
    for r in result.missing:
        x0, y0, x1, y1 = r.box_px
        h, w = result.aligned.shape[:2]
        ya, yb, xa, xb = max(0, y0 - m), min(h, y1 + m), max(0, x0 - m), min(w, x1 + m)
        crops = []
        for title, img in (("GOOD reference", reference.canonical), ("test", result.aligned)):
            crop = cv2.resize(img[ya:yb, xa:xb], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (round((x0 - xa) * zoom), round((y0 - ya) * zoom)),
                          (round((x1 - xa) * zoom), round((y1 - ya) * zoom)), COLOURS["missing"], 2)
            cv2.putText(crop, f"{title}: {r.refdes}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(crop, f"{title}: {r.refdes}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            crops.append(crop)
        rows.append(np.hstack([crops[0], np.full((crops[0].shape[0], 12, 3), 255, np.uint8), crops[1]]))
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 12), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])


def defect_location_overlay(result: InspectionResult, px_per_mm: float) -> np.ndarray | None:
    """Canonical PCB image with prominent defect bounding box, pointer marker, and callout badge."""
    if result.aligned is None:
        return None
    img = result.aligned.copy()
    h, w = img.shape[:2]

    missing = [c for c in result.components if c.status == MISSING]
    if not missing:
        return annotate(result, px_per_mm)

    # Dim non-defect regions slightly (30%) so the defect stands out vividly
    dimmed = (img.astype(np.float32) * 0.70).astype(np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    pad = max(10, round(2.0 * px_per_mm))
    for m in missing:
        x0, y0, x1, y1 = m.box_px
        mask[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = True
    dimmed[mask] = img[mask]
    out = dimmed

    for m in missing:
        x0, y0, x1, y1 = m.box_px
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        # 1. Translucent red fill over defect box
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 240), -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # 2. Bold double bounding box
        thick = max(2, round(0.18 * px_per_mm))
        cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 0, 0), thick + 3)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 30, 255), thick)

        # 3. Visual marker (crosshairs / target circle)
        radius = max(14, round(1.6 * px_per_mm))
        cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # 4. Pointer arrow
        arrow_len = max(35, round(4.0 * px_per_mm))
        if y0 > arrow_len + 25:
            arrow_start = (cx, y0 - arrow_len)
            arrow_end = (cx, y0 - 6)
            badge_y = y0 - arrow_len - 14
        else:
            arrow_start = (cx, y1 + arrow_len)
            arrow_end = (cx, y1 + 6)
            badge_y = y1 + arrow_len + 28

        cv2.arrowedLine(out, arrow_start, arrow_end, (0, 255, 255), 4, tipLength=0.35)
        cv2.arrowedLine(out, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.35)

        # 5. Callout badge with label and score
        label = f"DEFECT: {m.refdes} (missing)"
        sub = f"Sim: {m.similarity:.3f} | Limit: {m.limit:.3f}" if (m.similarity is not None and m.limit is not None) else ""

        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(label, font, 0.65, 2)
        (sw, sh), _ = cv2.getTextSize(sub, font, 0.52, 1) if sub else ((0, 0), 0)
        bw = max(lw, sw) + 20
        bh = lh + sh + 16 if sub else lh + 14

        bx0 = max(10, min(w - bw - 10, cx - bw // 2))
        by0 = max(10, min(h - bh - 10, badge_y - bh // 2))

        cv2.rectangle(out, (bx0, by0), (bx0 + bw, by0 + bh), (20, 20, 20), -1)
        cv2.rectangle(out, (bx0, by0), (bx0 + bw, by0 + bh), (0, 0, 255), 2)

        cv2.putText(out, label, (bx0 + 10, by0 + lh + 4), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        if sub:
            cv2.putText(out, sub, (bx0 + 10, by0 + lh + sh + 10), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    banner = f"DEFECT DETECTED: {', '.join(m.refdes for m in missing)} missing"
    cv2.putText(out, banner, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, banner, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
    return out



def write_outputs(result: InspectionResult, reference: Reference, cmap: ComponentMap, cfg: InspectionConfig,
                  out_dir: Path, gallery_names: list[str] | None = None, frame: np.ndarray | None = None) -> None:
    """report.txt, result.json, overview.jpg (if the frame is given or readable), annotated.jpg / evidence.jpg
    (if the board was inspected)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(result.report + "\n", encoding="utf-8")
    check = result.alignment_check
    payload: dict[str, Any] = {
        "image": str(result.image) if result.image else None,
        "image_sha256": sha256_of(result.image) if result.image and result.image.is_file() else None,
        "status": result.status, "stages": result.stages, "reason": result.reason,
        "missing_components": [r.refdes for r in result.missing],
        "not_checked": [r.refdes for r in result.components if r.status == NOT_CHECKED],
        "reference": reference.info, "alignment": result.alignment_text,
        "registration": None if result.registration is None else {
            k: v for k, v in asdict(result.registration).items() if k not in ("homography", "inlier_points")},
        "alignment_check": None if check is None else {
            "status": check.status, "reason": check.reason, "inlier_ratio": check.inlier_ratio,
            "fit_error_mm": check.fit_error_mm, "rotation_deg": check.rotation_deg,
            "corner_angle_deviation_deg": check.corner_angle_deviation_deg,
            "board_width_fraction": check.board_width_fraction,
            "board_corners_px": None if check.corners_px is None else check.corners_px.round(1).tolist()},
        "watermark": None if result.watermark is None else {
            "found": result.watermark.found, "box_px": result.watermark.box_px, "score": round(result.watermark.score, 3),
            "margin": round(result.watermark.margin, 3)},
        "component_map": {"file": str(cfg.component_map_file), "components": len(cmap.components),
                          "unverified": cmap.unverified},
        "good_examples": result.examples or gallery_names or [reference.info["id"]],
        "limits": result.limits_note,
        "parameters": asdict(cfg.check),
        "components": [asdict(r) for r in result.components],
        "note": "Experiment 2 prototype output; thresholds are experiment parameters, not validated limits.",
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if frame is None and result.image is not None and result.image.is_file():
        frame = load_bgr(result.image)
    if frame is not None and frame.size and result.status != INVALID_FRAME:
        panels.write_jpeg(out_dir / "overview.jpg", overview(frame, result))
    if result.aligned is not None:
        defect_img = defect_location_overlay(result, cfg.pipeline.px_per_mm)
        if defect_img is not None:
            panels.write_jpeg(out_dir / "defect_location.jpg", defect_img)
        panels.write_jpeg(out_dir / "annotated.jpg", annotate(result, cfg.pipeline.px_per_mm))
        proof = evidence(result, reference, cfg.pipeline.px_per_mm)
        if proof is not None:
            panels.write_jpeg(out_dir / "evidence.jpg", proof)
