"""Reference-comparison pipeline for one reference image and one test image (Experiments 1 and 1B).

Shared by scripts/run_experiment_001.py and scripts/run_experiment_001b.py so that
both use exactly the same code and parameters; only then are their results comparable.
Classical image processing only; nothing is learned and nothing is labelled.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..alignment.registration import RegistrationParams, register, tile_residuals
from ..config import load_config, load_profile
from ..preprocessing.board import (
    BoardSegmentationParams,
    CanonicalFrame,
    board_mask,
    board_quad,
    canonical_frame,
    warp_to_canonical,
)
from ..visualization import panels
from .difference import ComparisonParams, compare, verified_defect_overlap

RESULT_COLUMNS = (
    "group", "reference_id", "test_id", "test_board", "same_board", "registration_ok", "inliers", "inlier_ratio",
    "inlier_rmse_mm", "residual_median_mm", "residual_p90_mm", "residual_max_mm", "reliable_tiles",
    "ssim_mean", "delta_e_mean", "delta_e_p99", "fraction_delta_e_over_20", "edge_mismatch_fraction",
    "candidate_regions", "candidate_area_mm2", "candidate_area_fraction", "largest_candidate_mm2",
    "verified_defect_boxes", "defect_boxes_overlapped", "total_seconds",
)


@dataclass(frozen=True)
class PipelineConfig:
    px_per_mm: float
    board_mm: tuple[float, float]          # (short side, long side)
    segmentation: BoardSegmentationParams
    registration: RegistrationParams
    comparison: ComparisonParams
    residual_tile_mm: float
    residual_min_std: float
    residual_min_response: float
    raw: dict[str, Any]

    @classmethod
    def from_config(cls, cfg: dict[str, Any], board_mm: tuple[float, float]) -> PipelineConfig:
        return cls(
            px_per_mm=float(cfg["canonical_px_per_mm"]), board_mm=board_mm,
            segmentation=BoardSegmentationParams(**cfg["board_segmentation"]),
            registration=RegistrationParams(**cfg["registration"]),
            comparison=ComparisonParams(**cfg["comparison"]),
            residual_tile_mm=cfg["residuals"]["tile_mm"], residual_min_std=cfg["residuals"]["min_std"],
            residual_min_response=cfg["residuals"]["min_response"], raw=cfg,
        )


def load_pipeline_config(project_root: Path, config_path: Path) -> PipelineConfig:
    """Experiment config + the board size from the profile it names."""
    cfg = load_config(config_path)
    project = load_config(project_root / "configs" / "project.yaml")
    dims = load_profile(project_root / project["profiles"]["directory"], cfg["profile"])["pcb"]["dimensions_mm"]
    return PipelineConfig.from_config(cfg, (dims["short_side"], dims["long_side"]))


@dataclass
class Reference:
    info: dict[str, str]          # id, path, board_id, sha256
    image: np.ndarray
    mask: np.ndarray
    quad: np.ndarray
    frame: CanonicalFrame
    canonical: np.ndarray
    valid: np.ndarray
    prepare_seconds: float


def load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(img).convert("RGB")), cv2.COLOR_RGB2BGR)


def _prepare(path: str, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    start = time.perf_counter()
    image = load_bgr(Path(path))
    mask = board_mask(image, cfg.segmentation)
    return image, mask, board_quad(mask), time.perf_counter() - start


def prepare_reference(info: dict[str, str], cfg: PipelineConfig, out_dir: Path, experiment: str) -> Reference:
    """Load and rectify the reference; store it (and an mm grid for reading defect coordinates) in out_dir."""
    image, mask, quad, seconds = _prepare(info["path"], cfg)
    if quad is None:
        raise ValueError(f"board not found in reference image {info['id']}")
    frame = canonical_frame(quad, cfg.board_mm, cfg.px_per_mm)
    canonical, valid = warp_to_canonical(image, frame.homography, frame.size, frame.source_px_per_mm, cfg.px_per_mm)
    panels.write_jpeg(out_dir / "reference_canonical.jpg", canonical)
    panels.write_jpeg(out_dir / "reference_mm_grid.jpg", panels.mm_grid(canonical, cfg.px_per_mm))
    (out_dir / "reference.json").write_text(json.dumps({
        "status": f"experiment-only reference for {experiment}; NOT a production golden reference",
        **info, "canonical_px_per_mm": cfg.px_per_mm, "canonical_size_px": frame.size,
        "canonical_size_mm": [frame.size[0] / cfg.px_per_mm, frame.size[1] / cfg.px_per_mm],
        "board_quad_px": quad.round(1).tolist(), "source_px_per_mm": round(frame.source_px_per_mm, 2),
    }, indent=2), encoding="utf-8")
    return Reference(info, image, mask, quad, frame, canonical, valid, seconds)


def compare_pair(reference: Reference, test: dict[str, str], cfg: PipelineConfig, run_dir: Path, group: str = "",
                 verified_boxes_mm: list[tuple[float, float, float, float]] = (),
                 verified_defects: list[dict[str, str]] = ()) -> dict[str, Any]:
    """Register, compare and visualise one test image against the reference. Returns one results row.

    verified_boxes_mm / verified_defects are human ground truth (defects.csv); they are only
    drawn and checked for overlap, never derived from the images.
    """
    ppm = cfg.px_per_mm
    timings = {"reference_prepare_s": reference.prepare_seconds}
    t0 = time.perf_counter()
    test_img, test_mask, test_quad, timings["test_prepare_s"] = _prepare(test["path"], cfg)
    registration = register(reference.image, reference.mask, test_img, test_mask, cfg.registration)
    timings["registration_s"] = registration.seconds
    same_board = bool(test.get("board_id")) and test["board_id"] == reference.info.get("board_id")
    metrics: dict[str, Any] = {
        "group": group, "reference": reference.info, "test": test, "same_board": same_board, "config": cfg.raw,
        "software": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                     "machine": platform.processor() or platform.machine()},
        "registration": {k: v for k, v in asdict(registration).items() if k not in ("homography", "inlier_points")},
        "verified_defects_on_board": list(verified_defects),
    }
    metrics["registration"]["inlier_ratio"] = registration.inlier_ratio
    rmse_mm = registration.inlier_rmse_px / reference.frame.source_px_per_mm if registration.inlier_rmse_px is not None else None
    metrics["registration"]["inlier_rmse_mm"] = rmse_mm
    row: dict[str, Any] = {
        "group": group, "reference_id": reference.info["id"], "test_id": test["id"], "test_board": test.get("board_id", ""),
        "same_board": same_board, "registration_ok": registration.success, "inliers": registration.inliers,
        "inlier_ratio": round(registration.inlier_ratio, 3), "inlier_rmse_mm": "" if rmse_mm is None else round(rmse_mm, 4),
        "verified_defect_boxes": len(verified_boxes_mm),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if not registration.success:
        metrics["timings"] = timings
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        row["total_seconds"] = round(reference.prepare_seconds + time.perf_counter() - t0, 1)
        return row

    t = time.perf_counter()
    to_canonical = reference.frame.homography @ registration.homography
    # the homography scales test px to reference px by about sqrt(det); the test's own scale is ref / that
    test_ppm = reference.frame.source_px_per_mm / float(np.sqrt(abs(np.linalg.det(registration.homography[:2, :2]))))
    aligned, valid = warp_to_canonical(test_img, to_canonical, reference.frame.size, test_ppm, ppm)
    valid &= reference.valid
    timings["warp_s"] = time.perf_counter() - t

    t = time.perf_counter()
    gray_ref = cv2.cvtColor(reference.canonical, cv2.COLOR_BGR2GRAY)
    residuals = tile_residuals(gray_ref, cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), valid,
                               round(cfg.residual_tile_mm * ppm), cfg.residual_min_std, cfg.residual_min_response)
    timings["residuals_s"] = time.perf_counter() - t
    mm = residuals.magnitudes_px()[residuals.reliable] / ppm
    stat = lambda f: float(f(mm)) if mm.size else None  # noqa: E731
    metrics["residuals"] = {
        "tiles": int(len(residuals.reliable)), "reliable_tiles": int(residuals.reliable.sum()),
        "median_mm": stat(np.median), "p90_mm": stat(lambda m: np.percentile(m, 90)), "max_mm": stat(np.max),
        "fraction_over_0_1_mm": stat(lambda m: (m > 0.1).mean()), "fraction_over_0_25_mm": stat(lambda m: (m > 0.25).mean()),
    }

    result = compare(reference.canonical, aligned, valid, ppm, cfg.comparison)
    timings["comparison_s"] = result.seconds
    overlap = verified_defect_overlap(result.regions, list(verified_boxes_mm))
    metrics["comparison"] = result.metrics
    metrics["candidate_regions_mm"] = [asdict(r) for r in result.regions[:50]]
    metrics["verified_defect_boxes"] = [
        {**o, "overlapping_regions": [asdict(result.regions[i]) for i in o["overlapping_candidates"]],
         "largest_overlapping_rank": min(o["overlapping_candidates"]) + 1 if o["overlapping_candidates"] else None}
        for o in overlap
    ]

    t = time.perf_counter()
    quad_view = test_img.copy()
    if test_quad is not None:
        cv2.polylines(quad_view, [test_quad.astype(np.int32)], True, (0, 255, 0), max(3, test_img.shape[1] // 400))
    views = {
        "reference.jpg": reference.canonical,
        "test.jpg": cv2.resize(quad_view, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
        "aligned.jpg": aligned,
        "difference.jpg": panels.heatmap(result.delta_e, panels.DELTA_E_SCALE, result.compared),
        "dissimilarity.jpg": panels.heatmap(result.dissimilarity, panels.DISSIMILARITY_SCALE, result.compared),
        "edges.jpg": np.where(result.edge_mismatch[..., None], np.uint8([255, 255, 0]), (aligned * 0.35).astype(np.uint8)),
        "thresholded.jpg": panels.draw_candidates(aligned, result, ppm, list(verified_boxes_mm)),
        "overlay_checkerboard.jpg": panels.checkerboard_overlay(reference.canonical, aligned, round(5 * ppm)),
        "residuals.jpg": panels.residual_map(aligned, residuals, ppm),
    }
    for name, image in views.items():
        panels.write_jpeg(run_dir / name, image)
    title = f"candidates: {len(result.regions)}" + (f" | verified defect boxes (blue): {len(verified_boxes_mm)}" if verified_boxes_mm else "")
    panels.write_jpeg(run_dir / "panel.jpg", panels.panel([
        ("reference", views["reference.jpg"]), ("aligned test", aligned),
        ("checkerboard overlay", views["overlay_checkerboard.jpg"]), ("residual shift (0-0.5 mm)", views["residuals.jpg"]),
        ("colour difference dE (0-40)", views["difference.jpg"]), ("SSIM dissimilarity (0-0.5)", views["dissimilarity.jpg"]),
        ("edge mismatch", views["edges.jpg"]), (title, views["thresholded.jpg"]),
    ]))
    timings["visualisation_s"] = time.perf_counter() - t
    timings["total_s"] = reference.prepare_seconds + time.perf_counter() - t0
    metrics["timings"] = timings
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    r, c = metrics["residuals"], result.metrics
    rnd = lambda v, n: "" if v is None else round(v, n)  # noqa: E731
    row.update({
        "residual_median_mm": rnd(r["median_mm"], 4), "residual_p90_mm": rnd(r["p90_mm"], 4),
        "residual_max_mm": rnd(r["max_mm"], 4), "reliable_tiles": f"{r['reliable_tiles']}/{r['tiles']}",
        "ssim_mean": round(c["ssim_mean"], 4), "delta_e_mean": round(c["delta_e_mean"], 2),
        "delta_e_p99": round(c["delta_e_p99"], 2), "fraction_delta_e_over_20": round(c["fraction_delta_e_over_20"], 4),
        "edge_mismatch_fraction": round(c["edge_mismatch_fraction"], 4), "candidate_regions": c["candidate_regions"],
        "candidate_area_mm2": round(c["candidate_area_mm2"], 1), "candidate_area_fraction": round(c["candidate_area_fraction"], 4),
        "largest_candidate_mm2": round(c["largest_candidate_mm2"], 1),
        "defect_boxes_overlapped": sum(1 for o in overlap if o["overlapping_candidates"]),
        "total_seconds": round(timings["total_s"], 1),
        "_verified_boxes": metrics["verified_defect_boxes"],  # not written to CSV; used by summaries
    })
    return row


def merge_results(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a results CSV, replacing earlier rows of the same (reference, test) pair."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            merged = {(r["reference_id"], r["test_id"]): r for r in csv.DictReader(f)}
    merged.update({(str(r["reference_id"]), str(r["test_id"])): r for r in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged[k] for k in sorted(merged))
