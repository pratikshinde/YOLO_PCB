from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from station import LiveStation, result_lines
from test_component_inspection import board, scene
from test_product_architecture import build_project

from pcb_inspection.camera import MockCamera
from pcb_inspection.inspection.types import DEFECT, GOOD, REVIEW, Defect, InspectionOutcome, Reason
from pcb_inspection.preprocessing.board import BoardSegmentationParams
from pcb_inspection.preprocessing.presence import PresenceParams
from pcb_inspection.station import AutoTrigger, InspectionStation, StationResult, TriggerSettings


def test_livestation_auto_inspection_loop(tmp_path):
    st = InspectionStation(build_project(tmp_path))
    good = scene(board(marking=5))
    results = []
    # Test auto_inspection=True with a fast 0.2s interval
    live = LiveStation(
        st, MockCamera([good] * 10), "AUTO", AutoTrigger(TriggerSettings(0.1, 6.0, 1.0)),
        PresenceParams(), BoardSegmentationParams(), 0.05,
        on_result=lambda r, m: results.append(r),
        auto_inspection=True, interval_s=0.2,
    )
    assert live.auto_inspection is True
    assert live.interval_s == 0.2

    live.camera.open()
    live.start_loop(every_s=0.02)

    # In auto mode, it should automatically trigger captures without manual SPACE
    deadline = time.monotonic() + 10.0
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)

    assert len(results) >= 2, f"Expected at least 2 automatic captures, got {len(results)}"
    assert results[0].decision in ("GOOD", "DEFECT", "REVIEW")
    assert results[1].decision in ("GOOD", "DEFECT", "REVIEW")

    # Test pause and resume
    live.stop_auto()
    assert live.auto_inspection is False
    count_paused = len(results)
    time.sleep(0.5)
    assert len(results) == count_paused, "No new captures should trigger while paused"

    live.set_interval(0.1)
    assert live.interval_s == 0.1
    live.start_auto()
    assert live.auto_inspection is True
    deadline = time.monotonic() + 5.0
    while len(results) <= count_paused and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(results) > count_paused, "Captures should resume after start_auto"

    live.close()


def test_result_lines_formatting():
    # 1. Defect result formatting
    defect = Defect(
        defect_id="D1",
        defect_type="missing_component",
        confidence=None,
        location={"frame": "canonical_mm", "box_mm": [10.0, 5.0, 15.0, 8.0], "center_mm": [12.5, 6.5]},
        region="R19",
        evidence={"similarity": 0.45, "limit": 0.85, "below_limit_by": 0.40, "difference": 0.55},
        severity="critical",
    )
    outcome_defect = InspectionOutcome(
        inspection_id="INSP-001",
        product_id="lg_indicator",
        variant_id="main",
        image_id="IMG-001",
        board_id="B01",
        alignment_status="OK",
        alignment_error={"inliers": 85, "fit_error_mm": 0.12},
        quality_status="PASS",
        overall_decision=DEFECT,
        confidence=None,
        defects=[defect],
        reasons=[Reason("DEFECT_FOUND", "R19 missing")],
        regions_checked=19,
        reference_bank={"boards": ["B01_GOLDEN"]},
    )
    res_defect = StationResult(outcome_defect, None, Path("dummy/result.json"), None, "LG INDICATOR")
    lines_def = result_lines(res_defect)
    text_def = "\n".join(lines_def)

    assert "PRODUCT: LG INDICATOR" in text_def
    assert "ALIGNMENT: PASS" in text_def
    assert "INSPECTION: COMPLETE" in text_def
    assert "FINAL RESULT: DEFECT" in text_def
    assert "DEFECT TYPE:\n  Missing Component" in text_def
    assert "COMPONENT:\n  R19" in text_def
    assert "LOCATION:\n  x: 12.5 mm, y: 6.5 mm" in text_def
    assert "REGION:\n  R19" in text_def
    assert "MEASURED SCORE:\n  similarity 0.450" in text_def
    assert "THRESHOLD:\n  limit 0.850" in text_def
    assert "REASON:\n  Component 'R19' appearance differs significantly" in text_def
    assert "REFERENCE:\n  B01_GOLDEN" in text_def
    assert "CONFIDENCE:\n  NOT CALIBRATED" in text_def

    # 2. Good result formatting
    outcome_good = InspectionOutcome(
        inspection_id="INSP-002",
        product_id="lg_indicator",
        variant_id="main",
        image_id="IMG-002",
        board_id="B01",
        alignment_status="OK",
        alignment_error={"inliers": 90, "fit_error_mm": 0.08},
        quality_status="PASS",
        overall_decision=GOOD,
        confidence=None,
        defects=[],
        reasons=[Reason("ALL_REGIONS_MATCH", "all 19 regions match")],
        regions_checked=19,
        reference_bank={"boards": ["B01_GOLDEN"]},
    )
    res_good = StationResult(outcome_good, None, Path("dummy/result.json"), None, "LG INDICATOR")
    lines_good = result_lines(res_good)
    text_good = "\n".join(lines_good)

    assert "FINAL RESULT: GOOD" in text_good
    assert "No region exceeded the configured defect threshold." in text_good
    assert "ALIGNMENT STATUS:\n  PASS" in text_good
    assert "REFERENCE USED:\n  B01_GOLDEN" in text_good
    assert "INSPECTION SCORE:\n  All 19 inspected regions matched reference" in text_good
    assert "CONFIDENCE:\n  NOT CALIBRATED" in text_good

    # 3. Review result formatting
    outcome_review = InspectionOutcome(
        inspection_id="INSP-003",
        product_id="lg_indicator",
        variant_id="main",
        image_id="IMG-003",
        board_id=None,
        alignment_status="FAILED",
        alignment_error={"reason": "too few inliers (12)"},
        quality_status="PASS",
        overall_decision=REVIEW,
        confidence=None,
        defects=[],
        reasons=[Reason("ALIGNMENT", "too few inliers (12)")],
        reference_bank={"boards": ["B01_GOLDEN"]},
    )
    res_rev = StationResult(outcome_review, None, Path("dummy/result.json"), None, "LG INDICATOR")
    lines_rev = result_lines(res_rev)
    text_rev = "\n".join(lines_rev)

    assert "FINAL RESULT: REVIEW" in text_rev
    assert "EXACT REVIEW REASON:" in text_rev
    assert "alignment uncertainty: too few inliers (12)" in text_rev
    assert "CONFIDENCE:\n  NOT CALIBRATED" in text_rev
