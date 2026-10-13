"""Clean GOOD / FAULT data collection (2026-10-06): capture-time PCB check, canonical PCB image, collection store,
summary / reference check, dataset validation and the archive tool.

Test frames are GENERATED in memory / tmp_path (a coloured board with parts on a grey background, the real
DroidCam watermark glyphs burned in): unit-test fixtures only, never written to data/ and never dataset images.
"""

from __future__ import annotations

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pcb_inspection.acquisition.camera import encode_png  # noqa: E402
from pcb_inspection.acquisition.pcb_view import (  # noqa: E402
    BAD_CROP, BLURRY, NO_PCB, OK, OVEREXPOSED, PCB_IMAGE_ERROR, PCB_INCOMPLETE, PCB_OUTSIDE_ROI, RECAPTURE, REVIEW,
    WATERMARK_NOT_LOCATED, WATERMARK_OVER_PCB, PcbChecker, PcbCheckParams, inspection_roi)
from pcb_inspection.acquisition.session import (  # noqa: E402
    AcquisitionStore, ReviewError, SessionConfig, audit, capture_check, make_defect_report, normalise_board_id,
    settings_from_product)
from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.preprocessing.watermark import WatermarkParams  # noqa: E402
from pcb_inspection.products import load_product_profile  # noqa: E402

TEMPLATE = ROOT / "experiments" / "experiment_002" / "droidcam_watermark_template.png"
COLLECTION_CFG = ROOT / "configs" / "capture" / "droidcam_collection.yaml"
W, H = 2560, 1440                      # DroidCam frame size measured on this PC
PPM = 12.0                             # board scale in the generated frames (px per mm)
BOARD = (116.0, 83.0)
CAMERA = {"name": "DroidCam Video", "index": 1, "backend": "dshow", "width": W, "height": H,
          "fps_reported": 30.0, "fps_measured": 30.0}


def frame(board_at=(W / 2, H / 2), watermark_at=(60, 60), colour=(40, 40, 200), board_ppm=PPM, seed=0,
          edge_parts=False, extra_watermarks=()):
    """Grey matte background, a board of 116 x 83 mm centred at board_at (None = no board) with dark / light parts,
    and the real watermark glyphs (bright) with their top-left corner at watermark_at (None = no watermark; a
    negative y cuts the text off at the top edge of the image). edge_parts: parts ON the board's top edge, as on
    the real LG board (they notch the board-colour outline)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 120, np.uint8)
    if board_at is not None:
        bw, bh = round(BOARD[0] * board_ppm), round(BOARD[1] * board_ppm)
        x0, y0 = round(board_at[0] - bw / 2), round(board_at[1] - bh / 2)
        cv2.rectangle(img, (x0, y0), (x0 + bw - 1, y0 + bh - 1), colour, cv2.FILLED)
        for _ in range(260):          # parts, away from the board edge
            px, py = rng.integers(x0 + 40, x0 + bw - 60), rng.integers(y0 + 40, y0 + bh - 40)
            v = int(rng.choice([25, 230]))
            cv2.rectangle(img, (int(px), int(py)), (int(px) + 18, int(py) + 9), (v, v, v), cv2.FILLED)
        for _ in range(220):          # silkscreen designators: the fine detail a real board shows
            px, py = rng.integers(x0 + 40, x0 + bw - 90), rng.integers(y0 + 50, y0 + bh - 30)
            cv2.putText(img, f"R{int(rng.integers(1, 60))}", (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (235, 235, 235), 2, cv2.LINE_AA)
        if edge_parts:                # a row of parts flush with the top edge, plus a larger cluster at the top left
            for px in range(x0 + 30, x0 + bw - 60, 90):
                cv2.rectangle(img, (px, y0), (px + 40, y0 + 30), (30, 30, 30), cv2.FILLED)
            cv2.rectangle(img, (x0, y0), (x0 + 420, y0 + 300), (215, 215, 215), cv2.FILLED)
    for wx, wy in ([watermark_at] if watermark_at is not None else []) + list(extra_watermarks):
        glyphs = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)[max(0, -wy):]
        region = img[max(0, wy):max(0, wy) + glyphs.shape[0], wx:wx + glyphs.shape[1]]
        region[glyphs[:region.shape[0], :region.shape[1]] > 0] = 245
    return img


def checker(**overrides) -> PcbChecker:
    values = dict(board_mm=BOARD, min_pcb_sharpness=300.0, min_source_px_per_mm=8.0,     # = the collection config
                  watermark=WatermarkParams(enabled=True, template=TEMPLATE))
    values.update(overrides)
    return PcbChecker(PcbCheckParams(**values))


# --- capture-time PCB check ---------------------------------------------------------------------------

def test_clean_board_gives_ok_and_a_canonical_pcb_image():
    c = checker().check(frame())
    assert c.status == OK and c.problems == [] and c.review == []
    assert c.watermark["found"] and not c.watermark["over_pcb"] and c.watermark["distance_mm"] > 3.0
    assert c.canonical.shape == (1660, 2320, 3)                 # 116 x 83 mm at 20 px/mm: the PCB only
    assert c.pcb["orientation"] == "landscape" and abs(c.source_px_per_mm - PPM) < 0.3
    corners = c.canonical[[5, 5, -6, -6], [5, -6, 5, -6]]
    assert (corners[:, 2] > 150).all() and (corners[:, 0] < 90).all()   # board colour, no grey background
    record = c.to_record()
    assert record["status"] == OK and "canonical" not in record


def test_watermark_over_the_pcb_is_recapture():
    c = checker().check(frame(watermark_at=(1000, 600)))
    assert c.status == RECAPTURE and WATERMARK_OVER_PCB in c.problems and c.watermark["over_pcb"]


def test_watermark_just_outside_the_board_but_inside_the_clearance_is_recapture():
    board_top = round(H / 2 - BOARD[1] * PPM / 2)
    near = checker().check(frame(watermark_at=(1000, board_top - 36 - 12)))     # 1 mm above the board edge
    assert WATERMARK_OVER_PCB in near.problems and not near.watermark["over_pcb"] and near.watermark["distance_mm"] < 3
    clear = checker().check(frame(watermark_at=(1000, board_top - 36 - 60)))    # 5 mm above
    assert clear.status == OK


def test_watermark_out_of_view_or_cut_off_is_ok_when_none_is_on_the_pcb():
    # 2026-10-06 real setup: the watermark half cut off at the top image edge, far from the PCB -> not matchable.
    for watermark in (None, (800, -18)):
        c = checker().check(frame(watermark_at=watermark))
        assert c.status == OK and c.review == [] and not c.watermark["found"], watermark
        assert c.watermark["not_on_pcb"] and c.watermark["on_pcb_best_score"] < 0.45
    off = checker(watermark=WatermarkParams(enabled=False)).check(frame(watermark_at=None))
    assert off.status == OK and off.watermark["checked"] is False


def test_watermark_on_the_pcb_is_caught_even_when_the_frame_match_is_ambiguous():
    # two copies (one on the board, one off it): globally no clear best match, but on the PCB area there is one
    c = checker().check(frame(watermark_at=(100, 40), extra_watermarks=[(1000, 600)]))
    assert c.status == RECAPTURE and WATERMARK_OVER_PCB in c.problems and c.watermark["over_pcb"]


def test_parts_on_the_board_edge_are_not_a_bad_crop():
    # the real LG board: parts flush with the top edge notch the board-colour region (raw fill 0.84 on real frames)
    c = checker().check(frame(edge_parts=True))
    assert c.status == OK and c.fill_fraction < 0.92 and c.outline_fill > 0.95   # 0.92: the old (raw) limit


def test_board_cut_by_the_image_edge_is_incomplete():
    c = checker().check(frame(board_at=(W - 500, H / 2)))
    assert c.status == RECAPTURE and PCB_INCOMPLETE in c.problems


def test_board_in_view_but_outside_the_inspection_roi():
    c = checker().check(frame(board_at=(W / 2 + 330, H / 2)))
    assert c.problems == [PCB_OUTSIDE_ROI]


def test_no_pcb_and_the_droidcam_start_screen():
    assert checker().check(frame(board_at=None)).problems == [NO_PCB]
    start_screen = np.zeros((H, W, 3), np.uint8)
    cv2.putText(start_screen, "Start DroidCam", (1100, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    c = checker().check(start_screen)
    assert c.status == RECAPTURE and NO_PCB in c.problems and c.canonical is None


def test_blur_and_glare_are_recapture():
    blurred = cv2.GaussianBlur(frame(), (0, 0), 6)
    assert BLURRY in checker().check(blurred).problems
    glare = frame()
    cv2.circle(glare, (W // 2, H // 2), 160, (255, 255, 255), cv2.FILLED)
    assert OVEREXPOSED in checker().check(glare).problems


def test_large_object_over_the_board_or_a_wrong_outline_is_a_bad_crop():
    img = frame()
    cv2.rectangle(img, (round(W / 2 - 500), round(H / 2 - BOARD[1] * PPM / 2) - 20), (round(W / 2 + 500), 640),
                  (120, 120, 120), cv2.FILLED)            # e.g. a hand / paper over a large part of the board
    assert BAD_CROP in checker().check(img).problems
    merged = frame()                                      # a second board-coloured object touching the board
    cv2.rectangle(merged, (1975, 500), (2250, 560), (40, 40, 200), cv2.FILLED)
    assert BAD_CROP in checker().check(merged).problems


def test_other_board_colours_are_found():
    for colour in ((200, 60, 30), (40, 160, 40)):         # blue, green (PCB-18: colour varies)
        assert checker().check(frame(colour=colour)).status == OK


def test_a_board_turned_180_degrees_is_recognised(tmp_path):
    reference = tmp_path / "ref_pcb.png"
    cv2.imwrite(str(reference), checker().check(frame(watermark_at=None)).canonical)   # the reference's PCB image
    c = checker(orientation_reference=reference)
    same = c.check(frame(watermark_at=None, seed=0))
    assert same.status == OK and same.pcb["orientation_score"] > 0.9 > same.pcb["orientation_score_180"]
    turned = c.check(np.ascontiguousarray(frame(watermark_at=None, seed=0)[::-1, ::-1]))
    assert turned.status == RECAPTURE and turned.problems == ["PCB_ROTATED"]


def test_inspection_roi_is_the_preview_guide():
    assert inspection_roi(2560, 1440, (116, 83), 0.8) == (474, 144, 2085, 1296)
    x0, y0, x1, y1 = inspection_roi(800, 450, (116, 83), 0.8)
    assert (y1 - y0) == 360 and abs((x1 - x0) / (y1 - y0) - 116 / 83) < 0.01
    px0, py0, px1, py1 = inspection_roi(450, 800, (116, 83), 0.8)
    assert (px1 - px0) < (py1 - py0)                       # portrait frame: portrait guide


# --- collection store ------------------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    """A copy of the real configs (product profile + collection config) in a temporary project."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / TEMPLATE.relative_to(ROOT)).parent.mkdir(parents=True)
    shutil.copy2(TEMPLATE, tmp_path / TEMPLATE.relative_to(ROOT))
    return tmp_path


def collection_config() -> dict:
    """The real collection config, without the orientation reference (a real board's PCB image: generated test
    boards cannot be compared with it; the orientation check has its own test)."""
    capture = load_config(COLLECTION_CFG)
    capture["pcb_check"]["orientation_reference"] = None
    return capture


@pytest.fixture
def store(project):
    return AcquisitionStore(settings_from_product(load_product_profile("lg_indicator", project_root=project),
                                                  collection_config()))


def start(store, dataset_type="good", board="1", report=None):
    config = SessionConfig(dataset_type=dataset_type, stage="after_tht", requested_image_count=3, capture_interval_s=3,
                           board_id=normalise_board_id(dataset_type, board), defect_report=report)
    return store.create_session(config, CAMERA, started_at=datetime(2026, 10, 6, 10, 0))


def capture(store, session, img):
    return store.save_capture(session, encode_png(img), datetime(2026, 10, 6, 10, 0, 5), CAMERA, frame_seq=1)


def test_collection_uses_the_clean_layout_and_only_good_and_fault(store, project):
    s = store.settings
    assert s.data_root == project / "data" / "collection" and s.dataset_types == ("good", "fault")
    assert s.pcb_checker is not None and s.keep_requires_capture_check
    assert store.acquisition_root == project / "data" / "collection" / "originals"
    for refused in ("test", "defect"):
        with pytest.raises(ValueError, match="collects only GOOD, FAULT"):
            store.create_session(SessionConfig(refused, "after_tht", 1, 3), CAMERA)
    assert normalise_board_id("fault", "7") == "FAULT_007"
    with pytest.raises(ValueError):
        normalise_board_id("good", "FAULT_001")            # a FAULT board can never be captured as GOOD


def test_the_clean_store_refuses_a_capture_config_without_the_pcb_check(project):
    legacy = load_config(ROOT / "configs" / "capture" / "droidcam_e1b.yaml")       # Experiment 1B: no pcb_check
    with pytest.raises(ValueError, match="requires the capture-time PCB check"):
        settings_from_product(load_product_profile("lg_indicator", project_root=project), legacy)


def test_capture_is_saved_first_then_cropped_checked_and_fully_described(store, project):
    session = start(store)
    r = capture(store, session, frame())
    root = project / "data" / "collection"
    assert (root / r["original_file"]).is_file() and r["original_file"].startswith("originals/good/GOOD_001/")
    assert (root / r["pcb_file"]).is_file() and r["pcb_file"].endswith("/pcb/GOOD_001_TOP_S20261006-01_C001_pcb.png")
    for key in ("image_id", "product_id", "physical_board_id", "session_id", "stage", "timestamp", "camera",
                "resolution", "path", "sha256", "quality", "review_status"):
        assert r[key] is not None, key
    assert (r["product_id"], r["physical_board_id"], r["resolution"], r["review_status"]) == \
        ("lg_indicator", "GOOD_001", [W, H], "pending")
    assert capture_check(r)["status"] == OK and r["path"] == r["original_file"]
    assert cv2.imread(str(root / r["pcb_file"])).shape == (1660, 2320, 3)


def test_recapture_cannot_be_kept_but_is_never_lost(store, project):
    session = start(store)
    r = capture(store, session, frame(watermark_at=(1000, 600)))
    assert capture_check(r)["status"] == RECAPTURE
    with pytest.raises(ReviewError, match="cannot be kept"):
        store.review(session, r["image_id"], "keep")
    store.review(session, r["image_id"], "reject", "watermark_over_pcb")
    root = project / "data" / "collection"
    assert r["saved_file"].startswith("quarantine/good/GOOD_001/") and (root / r["saved_file"]).is_file()
    assert (root / r["saved_pcb_file"]).is_file() and (root / r["original_file"]).is_file()
    assert r["review_status"] == "rejected" and audit(store) == []


def test_keep_puts_only_the_watermark_free_pcb_image_into_the_good_dataset(store, project):
    session = start(store)
    r = capture(store, session, frame())                  # the frame has the watermark (outside the PCB)
    store.review(session, r["image_id"], "keep")
    root = project / "data" / "collection"
    assert r["saved_file"] is None and (root / r["original_file"]).is_file()    # the full frame stays in originals/
    assert r["saved_pcb_file"] == f"good/GOOD_001/{session.id}/{r['image_id']}_pcb.png"
    assert (root / r["saved_pcb_file"]).read_bytes() == (root / r["pcb_file"]).read_bytes()
    assert [p.name for p in (root / "good").rglob("*.png")] == [f"{r['image_id']}_pcb.png"]
    assert r["review_status"] == "kept" and audit(store) == []


def review_check(checker_):
    """Wrap a checker so that its result needs REVIEW (ambiguous watermark-like text on the PCB)."""
    original = checker_.check

    def check(*a, **k):
        c = original(*a, **k)
        c.add(WATERMARK_NOT_LOCATED)
        return c
    return check


def test_review_capture_can_be_kept_after_the_operator_confirms(store, monkeypatch):
    session = start(store)
    monkeypatch.setattr(store.settings.pcb_checker, "check", review_check(store.settings.pcb_checker))
    r = capture(store, session, frame())
    assert capture_check(r)["status"] == REVIEW
    store.review(session, r["image_id"], "keep", note="checked: no overlay on the PCB")
    assert r["review_status"] == "kept" and r["review"]["decided_by"] == "operator"
    with pytest.raises(ValueError, match="automatic KEEP needs capture check OK"):    # REVIEW: the operator decides
        store.review(session, capture(store, session, frame(seed=4))["image_id"], "keep", by="capture_check")


def test_automatic_keep_and_its_undo(store, project):
    session = start(store)
    r = capture(store, session, frame())
    store.review(session, r["image_id"], "keep", note="automatic: capture check OK", by="capture_check")
    assert r["review"]["decided_by"] == "capture_check" and store.kept_count("good", "GOOD_001") == 1
    dirty = capture(store, session, frame(watermark_at=(1000, 600), seed=2))
    with pytest.raises(ValueError, match="automatic KEEP needs capture check OK"):
        store.review(session, dirty["image_id"], "keep", by="capture_check")
    kept_copy = project / "data" / "collection" / r["saved_pcb_file"]
    store.undo_auto_keep(session, r["image_id"], "obstruction", "hand over the board")   # operator override
    assert not kept_copy.exists() and store.kept_count("good", "GOOD_001") == 0
    assert r["review"]["status"] == "rejected" and r["review"]["decided_by"] == "operator"
    assert r["review_history"][0]["status"] == "kept" and r["saved_file"].startswith("quarantine/good/")
    assert (project / "data" / "collection" / r["original_file"]).is_file() and audit(store) == []
    store.review(session, dirty["image_id"], "reject", "watermark_over_pcb")
    with pytest.raises(ReviewError, match="not kept automatically"):
        store.undo_auto_keep(session, dirty["image_id"], "other")


def test_pending_captures_are_rechecked_with_the_current_limits(store, monkeypatch):
    session = start(store)
    real_check = store.settings.pcb_checker.check
    monkeypatch.setattr(store.settings.pcb_checker, "check", review_check(store.settings.pcb_checker))
    pending = capture(store, session, frame())
    kept = capture(store, session, frame(seed=2))
    store.review(session, kept["image_id"], "keep")
    monkeypatch.setattr(store.settings.pcb_checker, "check", real_check)       # e.g. limits corrected
    assert store.recheck_pending() == [(pending["image_id"], REVIEW, OK)]
    assert capture_check(pending)["status"] == OK and capture_check(pending)["rechecked_at"]
    assert pending["quality"]["capture_check_history"][0]["status"] == REVIEW  # the earlier result is kept
    assert capture_check(kept)["status"] == REVIEW                              # decided captures: untouched
    assert store.recheck_pending() == [] and audit(store) == []


def test_a_failing_check_never_loses_the_capture(store, monkeypatch):
    session = start(store)
    monkeypatch.setattr(store.settings.pcb_checker, "check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    r = capture(store, session, frame())
    assert capture_check(r)["status"] == REVIEW and PCB_IMAGE_ERROR in capture_check(r)["review"]
    assert (store.root / r["original_file"]).is_file() and r["pcb_file"] is None


def test_good_bank_candidates_hold_every_good_board_and_never_a_fault_board(store):
    for board in ("1", "2"):
        session = start(store, "good", board)
        for seed in (1, 2):
            r = capture(store, session, frame(seed=seed))
            store.review(session, r["image_id"], "keep")
    report = make_defect_report(True, store.settings.defect_types, "missing_component", "led4", "top left", "")
    fault = start(store, "fault", "1", report)
    r = capture(store, fault, frame(seed=3))
    store.review(fault, r["image_id"], "keep")
    assert fault.data["declared_condition"] == "defective"
    assert fault.data["defect_report"]["component_reference"] == "LED4" and not fault.data["defect_report"]["verified"]
    assert r["saved_pcb_file"].startswith("fault/FAULT_001/")
    bank = store.good_bank_candidates()
    assert list(bank) == ["GOOD_001", "GOOD_002"] and all(len(v) == 2 for v in bank.values())
    assert all(img["dataset_type"] == "good" for imgs in bank.values() for img in imgs)
    unknown = make_defect_report(False, store.settings.defect_types)
    assert unknown["defect_type"] is None and unknown["defect_known"] == "unknown"     # never a guessed label


def test_a_restarted_fault_board_keeps_its_fault_information(store):
    from pcb_inspection.acquisition.session import inherit_defect_report
    from collection_summary import board_fault_lines
    known = make_defect_report(True, store.settings.defect_types, "missing_component", "LED4")
    first = start(store, "fault", "3", known)
    unknown = make_defect_report(False, store.settings.defect_types)
    report, source = inherit_defect_report(store, "fault", "FAULT_003", unknown)       # restart with UNKNOWN
    assert source == first.id and report["defect_type"] == "missing_component"
    assert report["inherited_from_session"] == first.id and report["verified"] is False
    other, source = inherit_defect_report(store, "fault", "FAULT_004", unknown)       # never from another board
    assert source is None and other["defect_known"] == "unknown"
    explicit = make_defect_report(True, store.settings.defect_types, "solder_defect")
    assert inherit_defect_report(store, "fault", "FAULT_003", explicit) == (explicit, None)   # never overwritten
    second = start(store, "fault", "3", report)
    assert second.data["defect_report"]["inherited_from_session"] == first.id
    assert "recorded UNKNOWN" in board_fault_lines([(first.id, known), ("S2", unknown)])[0]
    assert board_fault_lines([(first.id, known), ("S2", explicit)])[0].startswith("CONFLICTING")


def test_reference_verification_lists_real_ids_and_picks_the_best(store, tmp_path):
    from collection_summary import verify_reference
    session = start(store)
    for seed in (1, 2):
        r = capture(store, session, frame(seed=seed))
        store.review(session, r["image_id"], "keep", by="capture_check")
    ok, lines = verify_reference(store, "GOOD_001_TOP_<session>_C001", tmp_path)
    assert not ok and "no <...> placeholders" in lines[0] and "GOOD_001 (2): GOOD_001_TOP_S20261006-01_C00" in lines[2]
    ok, lines = verify_reference(store, "best", tmp_path)
    assert ok and lines[0].startswith("Reference check of GOOD_001_TOP_S20261006-01_C00")


def test_board_registry_rows_are_the_capture_declarations(store):
    from setup_reference import registry_rows
    good = start(store, "good", "1")
    store.review(good, capture(store, good, frame())["image_id"], "keep")
    labelled = start(store, "fault", "3", make_defect_report(True, store.settings.defect_types, "missing_component",
                                                              "led4"))          # stopped before any image
    restarted = start(store, "fault", "3", make_defect_report(False, store.settings.defect_types))
    store.review(restarted, capture(store, restarted, frame(seed=2))["image_id"], "keep")
    empty = start(store, "fault", "4")                                          # no image: not registered
    rows = {r["board_id"]: r for r in registry_rows(store)}
    assert set(rows) == {"GOOD_001", "FAULT_003"} and labelled.id and empty.id
    assert (rows["GOOD_001"]["condition"], rows["GOOD_001"]["split"]) == ("good", "reference")
    assert (rows["FAULT_003"]["condition"], rows["FAULT_003"]["split"]) == ("fault", "validation")
    assert rows["FAULT_003"]["defects"] == "LED4:missing_component"      # from the board's labelled session
    assert rows["FAULT_003"]["defect_source"] == "operator_at_capture_unverified" and not rows["GOOD_001"]["defects"]


def test_audit_reports_stray_files_and_test_folders(store, project):
    session = start(store)
    capture(store, session, frame())
    assert audit(store) == []
    root = project / "data" / "collection"
    (root / "test" / "TEST_001").mkdir(parents=True)
    (root / "test" / "TEST_001" / "x.png").write_bytes(b"x")
    (session.folder / "images" / "stray.png").write_bytes(b"x")
    issues = "\n".join(audit(store))
    assert "test/: folder of a type this store does not collect" in issues
    assert "stray.png: file in a session folder without a record" in issues


def test_summary_counts_and_reference_verification(store, project, tmp_path):
    from collection_summary import format_summary, summarise, verify_reference, write_csv
    good = start(store, "good", "1")
    clean = capture(store, good, frame())
    store.review(good, clean["image_id"], "keep")
    dirty = capture(store, good, frame(watermark_at=(1000, 600), seed=5))
    store.review(good, dirty["image_id"], "reject", "watermark_over_pcb")
    fault = start(store, "fault", "1", make_defect_report(False, store.settings.defect_types))
    capture(store, fault, frame(seed=7))                      # pending review
    s = summarise(store, project)
    text = format_summary(store, s, project)
    assert "GOOD boards: 1    GOOD images: 1" in text and "FAULT boards: 0    FAULT images: 0" in text
    assert "TEST images: 0" in text and "Audit: consistent" in text and "fault type UNKNOWN" in text
    ok, lines = verify_reference(store, clean["image_id"], tmp_path / "sheets")
    assert ok, lines
    assert (tmp_path / "sheets" / f"{clean['image_id']}.jpg").is_file()
    bad, lines = verify_reference(store, dirty["image_id"], tmp_path / "sheets")
    assert not bad and any("[FAIL] no watermark over the PCB" in x for x in lines)
    assert write_csv(store, tmp_path / "meta.csv") == 3
    with (tmp_path / "meta.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["review_status"] for r in rows} == {"kept", "rejected", "pending"}


def test_dataset_validation_discovers_the_collection(store, project):
    from pcb_inspection.dataset.product_discovery import PENDING, RESOLVED, validate_all
    good = start(store, "good", "1")
    r = capture(store, good, frame())
    store.review(good, r["image_id"], "keep")
    fault = start(store, "fault", "2")
    capture(store, fault, frame(seed=3))
    lg = load_product_profile("lg_indicator", project_root=project)
    v = validate_all(project, [lg], capture_root=project / "data" / "captures", derived=["data/processed"])
    report = v.products["lg_indicator"]
    assert v.unclaimed == [] and report.orphans == []                      # PCB images are derived, accounted for
    by_pool = {(e.pool, e.status) for e in report.images}
    assert by_pool == {("GOOD", RESOLVED), ("FAULT", PENDING)}
    assert {"GOOD_001", "FAULT_002"} <= set(report.boards)


# --- archive tool -------------------------------------------------------------------------------------------

def test_archive_moves_old_data_with_a_verified_manifest(tmp_path, capsys):
    from archive_old_data import run
    old = tmp_path / "data" / "curated" / "good" / "GOOD_001" / "S20261005-01"
    old.mkdir(parents=True)
    (old / "a.png").write_bytes(b"png-bytes")
    (tmp_path / "data" / "metadata").mkdir(parents=True)
    (tmp_path / "data" / "metadata" / "boards.csv").write_text("board_id\nGOOD_001 (stale row)\n", encoding="utf-8")
    (tmp_path / "data" / "metadata" / "defects.csv").write_text("defect_id\n", encoding="utf-8")   # empty: stays
    (tmp_path / "experiments" / "experiment_002" / "reference").mkdir(parents=True)
    (tmp_path / "experiments" / "experiment_002" / "reference" / "r.jpg").write_bytes(b"jpg")
    assert run(tmp_path, "x", apply=False) == 0 and (old / "a.png").is_file()          # dry run moves nothing
    assert run(tmp_path, "x", apply=True) == 0
    archive = tmp_path / "archive" / "x"
    assert not (tmp_path / "data" / "curated").exists()
    assert (archive / "data/curated/good/GOOD_001/S20261005-01/a.png").read_bytes() == b"png-bytes"
    with (archive / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = {r["original_path"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"data/curated/good/GOOD_001/S20261005-01/a.png", "data/metadata/boards.csv",
                         "experiments/experiment_002/reference/r.jpg"}
    assert rows["data/curated/good/GOOD_001/S20261005-01/a.png"]["kind"] == "image"
    header = (tmp_path / "data" / "metadata" / "boards.csv").read_text(encoding="utf-8").strip()
    assert header.startswith("board_id,serial_number") and "\n" not in header        # empty template re-created
    assert (tmp_path / "data" / "metadata" / "defects.csv").read_text(encoding="utf-8") == "defect_id\n"
    assert run(tmp_path, "x", apply=True) == 0                                        # nothing left to archive
    assert "Nothing to archive" in capsys.readouterr().out


# --- the real repository after the reset (read-only) ---------------------------------------------------------

def test_real_collection_is_fully_discovered_and_holds_no_test_data():
    from pcb_inspection.dataset.product_discovery import validate_all
    lg = load_product_profile("lg_indicator", project_root=ROOT)
    reference = lg.golden["geometric_reference"]                      # unset, or a clean-collection GOOD capture
    assert lg.acquisition_root == ROOT / "data" / "collection"
    assert reference is None or reference.startswith("data/collection/originals/good/")
    v = validate_all(ROOT, [lg], capture_root=ROOT / "data" / "captures",
                     derived=list(load_config(ROOT / "configs" / "inspection.yaml")["storage"]["derived_paths"]),
                     include_unlabelled=False)
    report = v.products["lg_indicator"]
    paths = {e.path for e in report.images} | {c for e in report.images for c in e.copies}
    for f in (ROOT / "data" / "collection").rglob("*.png"):
        rel = f.relative_to(ROOT).as_posix()
        assert rel in paths or rel.endswith("_pcb.png"), f
    assert not [e for e in report.images if e.pool in ("TEST", "DEFECT")]
