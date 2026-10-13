"""Experiment 1B grouping / summaries, and an end-to-end pipeline run on a generated textured board scene
(test fixture for geometry and plumbing only: not a PCB image, no defect injected)."""

import json

import cv2
import numpy as np
import pytest

from pcb_inspection.comparison.experiment1b import (
    GROUP_DEFECTIVE,
    GROUP_OTHER_GOOD,
    GROUP_SAME,
    compare_to_baseline,
    defect_findings,
    plan_groups,
    summarise,
)
from pcb_inspection.comparison.pipeline import PipelineConfig, compare_pair, merge_results, prepare_reference

CFG = {
    "canonical_px_per_mm": 5,
    "board_segmentation": {"saturation_min": 60, "value_min": 40, "hue_tolerance": 12},
    "registration": {"detector": "sift", "working_max_side_px": 2400, "max_features": 4000, "ratio_test": 0.75,
                     "ransac_threshold_px": 3.0, "mask_erosion_fraction": 0.01, "min_inliers": 30},
    "residuals": {"tile_mm": 8.0, "min_std": 2.0, "min_response": 0.1},
    "comparison": {"illumination": "global", "blur_sigma_mm": 0.05, "border_margin_mm": 2.0, "ssim_sigma_px": 1.5,
                   "edge_tolerance_mm": 0.4, "delta_e_threshold": 20.0, "dissimilarity_threshold": 0.25,
                   "min_region_mm2": 0.5, "open_kernel_mm": 0.4},
}


def cap(image_id, board):
    return {"image_id": image_id, "board_id": board, "captured_at": "2026-10-12T10:00:00"}


BOARDS = [{"board_id": "GOOD_001", "condition": "good"}, {"board_id": "GOOD_002", "condition": "good"},
          {"board_id": "DEFECT_001", "condition": "defective"}, {"board_id": "DEFECT_002", "condition": "defective"},
          {"board_id": "DEFECT_003", "condition": "defective"}]
DEFECTS = [{"board_id": "DEFECT_001", "defect_box_mm": "30,40,36,46"}, {"board_id": "DEFECT_002", "defect_box_mm": ""}]


def test_plan_groups_uses_only_human_records():
    captures = [cap("S1_C001", "GOOD_001"), cap("S1_C002", "GOOD_001"), cap("S1_C003", "GOOD_002"),
                cap("S1_C004", "DEFECT_001"), cap("S1_C005", "DEFECT_002"), cap("S1_C006", "DEFECT_003"),
                cap("S1_C007", "GOOD_009"), cap("S1_C008", "")]
    planned, skipped = plan_groups(captures, BOARDS, DEFECTS, "S1_C001")
    assert [(p.capture["image_id"], p.group) for p in planned] == [
        ("S1_C002", GROUP_SAME), ("S1_C003", GROUP_OTHER_GOOD), ("S1_C004", GROUP_DEFECTIVE), ("S1_C005", GROUP_DEFECTIVE)]
    assert planned[2].boxes_mm == [(30.0, 40.0, 36.0, 46.0)] and planned[3].boxes_mm == []  # no box = no location truth
    text = "\n".join(skipped)
    assert "DEFECT_003 has no verified defect" in text
    assert "GOOD_009 has no verified condition" in text
    assert "S1_C008: no board_id" in text
    with pytest.raises(ValueError):
        plan_groups(captures, BOARDS, DEFECTS, "missing")


def row(group, cand, frac, ssim, same=None, ok=True, largest=1.0):
    return {"group": group, "registration_ok": ok, "candidate_regions": cand, "candidate_area_fraction": frac,
            "ssim_mean": ssim, "delta_e_mean": 4.0, "residual_median_mm": 0.04, "inlier_rmse_mm": 0.06,
            "edge_mismatch_fraction": 0.1, "largest_candidate_mm2": largest, "total_seconds": 2.0,
            "same_board": same if same is not None else group == GROUP_SAME, "test_id": f"{group}-{cand}", "test_board": "B"}


def test_summarise_and_baseline_comparison():
    controlled = [row(GROUP_SAME, 4, 0.001, 0.98), row(GROUP_SAME, 6, 0.002, 0.97), row(GROUP_SAME, 99, 0.5, 0.1, ok=False),
                  row(GROUP_OTHER_GOOD, 30, 0.05, 0.9)]
    summary = summarise(controlled)
    assert summary[GROUP_SAME]["pairs"] == 3 and summary[GROUP_SAME]["registered"] == 2
    assert summary[GROUP_SAME]["candidate_regions"] == {"median": 5.0, "max": 6.0}   # failed pair excluded
    handheld = [{"same_board": "True", "candidate_regions": "17"}, {"same_board": "True", "candidate_regions": "47"},
                {"same_board": "True", "candidate_regions": "72"}, {"same_board": "False", "candidate_regions": "258"}]
    change = compare_to_baseline(handheld, controlled, keys=("candidate_regions",))
    same = change["same board"]["candidate_regions"]
    assert (same["handheld"], same["controlled"]) == (47.0, 6.0)  # controlled median includes the failed row's value
    assert change["other good board"]["candidate_regions"]["handheld"] == 258.0
    assert change["same board"]["handheld_pairs"] == 3


def test_defect_findings_are_descriptive_and_need_a_box():
    good = row(GROUP_SAME, 5, 0.01, 0.97, largest=4.0)
    hit = {**row(GROUP_DEFECTIVE, 12, 0.02, 0.9), "test_id": "D1", "test_board": "DEFECT_001",
           "_verified_boxes": [{"box_mm": [30, 40, 36, 46], "largest_overlapping_rank": 1,
                                "overlapping_regions": [{"area_mm2": 9.5}]}]}
    no_box = {**row(GROUP_DEFECTIVE, 3, 0.01, 0.9), "test_id": "D2", "test_board": "DEFECT_002", "_verified_boxes": []}
    findings = defect_findings([good, hit, no_box])
    assert findings[0]["overlapped"] and findings[0]["larger_than_every_false_candidate"] is True
    assert findings[0]["largest_false_candidate_on_good_boards_mm2"] == 4.0
    assert findings[1]["ground_truth_location"] is False and "cannot be evaluated" in findings[1]["note"]


def textured_board_scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((800, 700, 3), 128, np.uint8)
    noise = cv2.GaussianBlur(rng.random((580, 415)).astype(np.float32), (0, 0), 1.2)
    ranks = noise.ravel().argsort().argsort().reshape(noise.shape) / noise.size  # evenly spread values: enough contrast for SIFT
    texture = 0.3 + 0.7 * ranks
    board = (np.float32([40, 30, 210])[None, None, :] * texture[..., None]).astype(np.uint8)  # red, varying brightness
    img[110:690, 142:557] = board
    return img


def test_pipeline_end_to_end_on_generated_scene(tmp_path):
    reference_img = textured_board_scene()
    h = np.array([[0.99, -0.03, 12.0], [0.025, 1.0, -8.0], [1e-5, 0.0, 1.0]])
    test_img = cv2.warpPerspective(reference_img, h, (700, 800), borderValue=(128, 128, 128))
    cv2.imwrite(str(tmp_path / "ref.png"), reference_img)
    cv2.imwrite(str(tmp_path / "test.png"), test_img)
    cfg = PipelineConfig.from_config(CFG, (83, 116))
    reference = prepare_reference({"id": "REF", "path": str(tmp_path / "ref.png"), "board_id": "GOOD_001", "sha256": ""},
                                  cfg, tmp_path / "out" / "reference", "test")
    assert reference.canonical.shape[:2] == (580, 415)
    result = compare_pair(reference, {"id": "TEST", "path": str(tmp_path / "test.png"), "board_id": "GOOD_001", "sha256": ""},
                          cfg, tmp_path / "out" / "runs" / "pair", group=GROUP_SAME)
    assert result["registration_ok"] and result["same_board"]
    assert float(result["residual_median_mm"]) < 0.1
    assert result["candidate_area_fraction"] < 0.01
    for name in ("panel.jpg", "aligned.jpg", "difference.jpg", "thresholded.jpg", "metrics.json"):
        assert (tmp_path / "out" / "runs" / "pair" / name).is_file()
    assert (tmp_path / "out" / "reference" / "reference_mm_grid.jpg").is_file()
    assert json.loads((tmp_path / "out" / "reference" / "reference.json").read_text())["status"].startswith("experiment-only")
    merge_results(tmp_path / "out" / "results.csv", [result])
    assert "_verified_boxes" not in (tmp_path / "out" / "results.csv").read_text()
