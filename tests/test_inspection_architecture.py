"""Comprehensive Evaluation of the PCB Inspection Architecture.

Tests compliance with requirements:
- Physical PCB independence: multiple physical GOOD boards (GOOD_001, GOOD_002, GOOD_003)
- Canonical coordinate rectification: fronto-parallel canonical coordinate system (83 x 116 mm)
- Separation of alignment vs inspection: alignment differences are NOT PCB defects
- Alignment Quality Gate: poor alignment / obscured / empty frame yields REVIEW / ALIGNMENT_FAILED
- Multi-Board Baseline Anomaly Detection: normal inter-board variation acceptable, true defects flagged
- Dual Storage: both RAW evidence and CANONICAL image stored automatically
- Traceability: machine-readable InspectionRecord with all 20 required fields

Controlled Evaluation Suite:
- TEST A: GOOD_001, original placement
- TEST B: GOOD_001, shifted
- TEST C: GOOD_001, rotated
- TEST D: GOOD_001, camera moved (scale variation)
- TEST E: GOOD_002, different physical GOOD board (MUST NOT be classified as DEFECT!)
- TEST F: GOOD_002, shifted
- TEST G: GOOD_003, different physical GOOD board
- TEST H: Defective PCB (missing component C14)
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pcb_inspection.engine.classical import ClassicalInspectionEngine, EngineConfig, sha256_of
from pcb_inspection.comparison.components import Component, ComponentMap
from pcb_inspection.engine.base import (
    FAIL,
    PASS,
    RECORD_COLUMNS,
    REVIEW,
    DefectCandidate,
    InspectionRecord,
    InspectionResult,
    InspectionState,
)
from pcb_inspection.engine.baseline import GoodBaseline, build_good_baseline
from pcb_inspection.alignment.registration import RegistrationParams
from pcb_inspection.preprocessing.board import BoardSegmentationParams
from pcb_inspection.preprocessing.presence import PresenceParams
from tests.synthetic_fixtures import embed_in_scene, make_test_pcb


@pytest.fixture(scope="module")
def board_fixtures():
    """Generates synthetic test fixtures for GOOD_001, GOOD_002, GOOD_003, and DEFECT_001."""
    b1 = make_test_pcb(seed=101)                                       # Nominal GOOD_001
    b2 = make_test_pcb(seed=202, color_shift=(4, 3, -4), component_offset=(1, 0))  # Physical GOOD_002
    b3 = make_test_pcb(seed=303, color_shift=(-3, 4, 3), component_offset=(0, 1))  # Physical GOOD_003
    b_defect = make_test_pcb(seed=404, missing_component="C14")        # Defective board: missing C14

    return {
        "GOOD_001": b1,
        "GOOD_002": b2,
        "GOOD_003": b3,
        "DEFECT_001": b_defect,
    }


@pytest.fixture(scope="module")
def multi_board_baseline(board_fixtures):
    """Builds a multi-board baseline from GOOD_001, GOOD_002, and GOOD_003."""
    samples = [
        ("GOOD_001", board_fixtures["GOOD_001"]),
        ("GOOD_002", board_fixtures["GOOD_002"]),
        ("GOOD_003", board_fixtures["GOOD_003"]),
    ]
    return build_good_baseline(samples, board_dimensions_mm=(116.0, 83.0), px_per_mm=10.0,
                               min_delta_e_tolerance=12.0, k_sigma=2.5)


@pytest.fixture(scope="module")
def inspection_engine(multi_board_baseline):
    """Configures ClassicalInspectionEngine with component map and tuned parameters."""
    cmap = ComponentMap(
        components=[
            Component(refdes="U1_MCU", box_mm=(32.0, 17.0, 48.0, 33.0), source="operator", verified=True),
            Component(refdes="C14", box_mm=(23.0, 18.5, 27.0, 21.5), source="operator", verified=True),
            Component(refdes="R5", box_mm=(23.2, 29.0, 26.8, 31.0), source="operator", verified=True),
        ],
        reference={"image_id": "SYNTH_REF", "file": "synth.png", "sha256": ""},
        size_mm=(116.0, 83.0),
    )
    cfg = EngineConfig(
        board_dimensions_mm=(116.0, 83.0),
        px_per_mm=10.0,
        min_defect_area_mm2=0.5,
        anomaly_threshold=1.5,
        segmentation=BoardSegmentationParams(saturation_min=20, value_min=20),
        registration=RegistrationParams(min_inliers=15, ratio_test=0.80),
        presence=PresenceParams(min_board_area_fraction=0.08),
    )

    return ClassicalInspectionEngine(baseline=multi_board_baseline, cmap=cmap, config=cfg)


# =========================================================================
# CONTROLLED EVALUATION SUITE: TESTS A THROUGH H
# =========================================================================

def test_suite_eval_a_good_001_original_placement(inspection_engine, board_fixtures):
    """TEST A: GOOD_001 with nominal centered placement -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(0, 0), rotation_deg=0.0)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"
    assert align.inliers >= 15

    res = inspection_engine.inspect(canon, align, board_id="GOOD_001")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS
    assert len(res.defects) == 0


def test_suite_eval_b_good_001_shifted(inspection_engine, board_fixtures):
    """TEST B: GOOD_001 shifted (+25px X, -20px Y) -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(25, -20), rotation_deg=0.0)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    res = inspection_engine.inspect(canon, align, board_id="GOOD_001")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS
    assert len(res.defects) == 0


def test_suite_eval_c_good_001_rotated(inspection_engine, board_fixtures):
    """TEST C: GOOD_001 rotated (+3.5 deg) -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(10, 10), rotation_deg=3.5)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"
    assert abs(abs(align.rotation_deg) - 3.5) < 1.0  # detected rotation magnitude matches applied rotation

    res = inspection_engine.inspect(canon, align, board_id="GOOD_001")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS
    assert len(res.defects) == 0


def test_suite_eval_d_good_001_camera_scale_change(inspection_engine, board_fixtures):
    """TEST D: GOOD_001 camera distance change (0.95 scale) -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(-15, 15), rotation_deg=-2.0, scale=0.95)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    res = inspection_engine.inspect(canon, align, board_id="GOOD_001")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS


def test_suite_eval_e_good_002_different_physical_board(inspection_engine, board_fixtures):
    """TEST E: GOOD_002 is a DIFFERENT physical board -> ALIGNMENT_OK, GOOD.

    CRITICAL ACCEPTANCE CRITERION:
    Must NOT be classified as DEFECT just because GOOD_002 != GOOD_001 pixel-for-pixel!
    """
    scene = embed_in_scene(board_fixtures["GOOD_002"], shift_xy=(0, 0), rotation_deg=0.0)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    # Evaluated against baseline (with leave-one-board-out)
    res = inspection_engine.inspect(canon, align, board_id="GOOD_002")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS
    assert len(res.defects) == 0, f"Expected 0 defects on verified physical GOOD_002, got: {res.defects}"


def test_suite_eval_f_good_002_shifted(inspection_engine, board_fixtures):
    """TEST F: GOOD_002 shifted (-30px X, +15px Y) and rotated (-2 deg) -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_002"], shift_xy=(-30, 15), rotation_deg=-2.0)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    res = inspection_engine.inspect(canon, align, board_id="GOOD_002")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS


def test_suite_eval_g_good_003_different_physical_board(inspection_engine, board_fixtures):
    """TEST G: GOOD_003 third physical good board -> ALIGNMENT_OK, GOOD."""
    scene = embed_in_scene(board_fixtures["GOOD_003"], shift_xy=(15, -15), rotation_deg=1.5)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    res = inspection_engine.inspect(canon, align, board_id="GOOD_003")
    assert res.state == InspectionState.GOOD
    assert res.result == PASS
    assert len(res.defects) == 0


def test_suite_eval_h_defective_board_missing_component(inspection_engine, board_fixtures):
    """TEST H: Known defective PCB with missing component C14 -> ALIGNMENT_OK, DEFECT.

    Must detect abnormality and localize it in canonical PCB coordinates.
    """
    scene = embed_in_scene(board_fixtures["DEFECT_001"], shift_xy=(10, -5), rotation_deg=1.0)
    canon, align = inspection_engine.align(scene)

    assert align.success is True
    assert align.status == "OK"

    res = inspection_engine.inspect(canon, align, board_id="DEFECT_001")
    assert res.state == InspectionState.DEFECT
    assert res.result == FAIL
    assert len(res.defects) > 0

    top = res.defects[0]
    # Check that canonical coordinates localize the missing component (around x=25mm, y=20mm)
    assert abs(top.x_mm - 23.0) < 6.0
    assert abs(top.y_mm - 18.5) < 6.0
    if top.refdes:
        assert top.refdes == "C14"


# =========================================================================
# ALIGNMENT QUALITY GATE TESTS
# =========================================================================

def test_alignment_quality_gate_empty_scene(inspection_engine):
    """Alignment failure on empty/no-PCB scene must result in REVIEW / ALIGNMENT_FAILED, NOT DEFECT."""
    empty_scene = np.full((1000, 1000, 3), 60, dtype=np.uint8)
    canon, align = inspection_engine.align(empty_scene)

    assert align.success is False
    assert align.status in ("NO_PCB", "REGISTRATION_FAILED")

    res = inspection_engine.inspect(canon, align)
    assert res.state == InspectionState.ALIGNMENT_FAILED
    assert res.result == REVIEW
    assert "Not treated as a board defect" in res.message


def test_alignment_quality_gate_obscured_scene(inspection_engine, board_fixtures):
    """Obscured board with inadequate SIFT features must trigger ALIGNMENT_FAILED -> REVIEW."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(0, 0))
    # Obscure 90% of the board with a heavy black rectangle
    scene[300:1100, 300:1500] = 0

    canon, align = inspection_engine.align(scene)
    assert align.success is False

    res = inspection_engine.inspect(canon, align)
    assert res.state == InspectionState.ALIGNMENT_FAILED
    assert res.result == REVIEW


# =========================================================================
# DUAL STORAGE AND MACHINE-READABLE RECORDS
# =========================================================================

def test_dual_storage_and_records(inspection_engine, board_fixtures, tmp_path):
    """Verifies that process_capture preserves RAW image, saves CANONICAL image, and writes 20-field record."""
    scene = embed_in_scene(board_fixtures["GOOD_001"], shift_xy=(10, -5), rotation_deg=0.5)
    capture_info = {
        "image_id": "CAP_20261005_120000_001",
        "session_id": "SESSION_TEST_DUAL",
        "board_id": "GOOD_001",
        "dataset_type": "good",
        "stage": "after_tht",
        "camera": "test_camera",
    }

    record = inspection_engine.process_capture(scene, capture_info, tmp_path)

    # 1. Raw image saved
    raw_path = Path(record.raw_path)
    assert raw_path.is_file()
    assert raw_path.name == "CAP_20261005_120000_001.png"

    # 2. SHA-256 match
    assert record.sha256 == sha256_of(raw_path)

    # 3. Canonical image saved
    aligned_path = Path(record.aligned_path)
    assert aligned_path.is_file()
    assert aligned_path.name == "CAP_20261005_120000_001_canonical.png"

    # 4. JSON record saved and contains all fields
    json_path = tmp_path / "records" / "CAP_20261005_120000_001.json"
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["image_id"] == "CAP_20261005_120000_001"
    assert data["board_id"] == "GOOD_001"
    assert data["result"] == PASS
    assert data["inspection_status"] == "GOOD"

    # 5. summary.csv appended with exactly 20 columns
    csv_path = tmp_path / "summary.csv"
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == RECORD_COLUMNS
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["image_id"] == "CAP_20261005_120000_001"
        assert rows[0]["result"] == PASS
