"""Product-agnostic, profile-driven inspection architecture (docs/inspection_architecture.md).

Generated test patterns only (red 'alpha' board and green 'beta' board, random white marks for texture, grey pads
and dark bodies for parts), built in a temporary project: they check the architecture (storage, identity,
isolation, decisions), not inspection accuracy on real PCBs. The real-data tests at the end only READ the
repository (profiles, the archive of the pre-reset data) and write nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from pcb_inspection.camera import CapturedFrame, MockCamera
from pcb_inspection.comparison.components import Component, ComponentMap, save_component_map
from pcb_inspection.dataset.experiment0 import sha256_of
from pcb_inspection.dataset.product_discovery import (
    PENDING,
    RESOLVED,
    SYNTHETIC,
    UNRESOLVED,
    format_validation,
    validate_all,
)
from pcb_inspection.engine.baseline import build_good_baseline
from pcb_inspection.inspection import (
    DEFECT,
    ERROR,
    GOOD,
    INSUFFICIENT,
    NOT_CONFIGURED,
    READY,
    REVIEW,
    InspectionEngine,
    load_reference_bank,
)
from pcb_inspection.inspection.identification import AMBIGUOUS, IDENTIFIED, SELECTED, UNKNOWN
from pcb_inspection.products import ProfileError, list_products, load_all_profiles, load_product_profile
from pcb_inspection.station import (
    CONFIRM_GOOD,
    AutoTrigger,
    CaptureError,
    InspectionStation,
    TriggerSettings,
    record_review,
)
from test_component_inspection import PARTS, PPM, board, draw_part, scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GREEN = (60, 150, 40)
BETA_PARTS = {"C1": (20, 15), "R2": (60, 15), "U3": (40, 45), "R4": (75, 50)}
DERIVED = ["data/processed", "data/products/*/inspections", "data/products/*/cache"]


# --- generated boards ------------------------------------------------------------------------------

def beta_board(missing=(), marking=0):
    """A different product: 100 x 70 mm, green, its own texture and parts."""
    img = np.full((round(70 * PPM), round(100 * PPM), 3), GREEN, np.uint8)
    rng = np.random.default_rng(77)
    for _ in range(450):
        x, y = int(rng.integers(5, img.shape[1] - 5)), int(rng.integers(5, img.shape[0] - 5))
        if any(abs(x / PPM - (px + 1.6)) < 4 and abs(y / PPM - (py + 0.8)) < 3.5 for px, py in BETA_PARTS.values()):
            continue
        cv2.circle(img, (x, y), int(rng.integers(2, 6)), (230, 230, 230), 1)
    for name, (x, y) in BETA_PARTS.items():
        draw_part(img, x, y, PPM, present=name not in missing, marking=marking)
    return img


def components(parts, margin_mm=0.4):
    return [Component(n, (x - margin_mm, y - margin_mm, x + 3.2 + margin_mm, y + 1.6 + margin_mm), "operator", True)
            for n, (x, y) in parts.items()]


# --- product folders and data stores ---------------------------------------------------------------

def write_product(root: Path, pid: str, *, dims, parts=None, reference: str | None = None, min_boards=2,
                  side="top", auto=True) -> Path:
    folder = root / "configs" / "products" / pid
    for sub in ("variants", "golden", "regions"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    dump = lambda path, data: path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")  # noqa: E731
    dump(folder / "product.yaml", {
        "product_id": pid, "name": pid.title(), "revision": None, "default_variant": "variant_1", "variants": ["variant_1"],
        "board": {"length_mm": dims[0], "width_mm": dims[1], "thickness_mm": None, "inspection_side": side,
                  "orientation": None, "allowed_rotation_deg": 30},
        "stages": ["after_tht"], "appearance": {"colour_variants": None, "colour_is_variant": False},
        "storage": {"root": f"data/products/{pid}", "acquisition_root": None, "registry": None, "legacy_sources": []},
        "identification": {"auto": auto}})
    dump(folder / "variants" / "variant_1.yaml", {
        "variant_id": "variant_1", "variant_label": "1", "revision": None,
        "engine": {"type": "REFERENCE_BASELINE", "status": "EXPERIMENTAL", "config": "configs/engine_test.yaml"},
        "golden": "golden/golden_bank.yaml", "regions": "regions/regions.yaml", "rules": "inspection_rules.yaml",
        "alignment": {"min_inlier_ratio": None, "max_fit_error_mm": None},
        "quality": {"capture_config": "configs/capture/droidcam_e1b.yaml"}})
    dump(folder / "golden" / "golden_bank.yaml", {
        "geometric_reference": reference, "reference_split": "reference", "min_boards": min_boards, "calibration": None,
        "calibration_min_boards": 5, "cache_dir": None, "max_rotation_deg": 180})
    regions = None
    if parts is not None and reference is not None:
        regions = {"file": f"configs/products/{pid}/regions/components.yaml", "verified": True}
        m = ComponentMap(components(parts), {"image_id": Path(reference).stem, "file": reference,
                                             "sha256": sha256_of(root / reference)}, (float(dims[0]), float(dims[1])))
        save_component_map(m, folder / "regions" / "components.yaml")
    dump(folder / "regions" / "regions.yaml", {"component_regions": regions, "led_regions": None,
                                               "connector_regions": None, "trace_regions": None, "hole_regions": None})
    dump(folder / "inspection_rules.yaml", {
        "defect_types": {"missing_component": {"enabled": True, "severity": None}},
        "decision": {"uncertainty_band": 0.03, "min_region_visible_fraction": 0.95, "max_plausible_defects": 3,
                     "allow_good_with_unverified_regions": False, "confidence": None}})
    return folder


def add_session(store: Path, dataset_type: str, board_id: str | None, session_id: str, images: list[np.ndarray],
                product_id: str, review: str = "kept") -> list[str]:
    """An acquisition session as scripts/acquire.py writes it (originals + session.json + KEEP copies)."""
    folder = board_id or "UNASSIGNED"
    acq = store / "acquisition" / dataset_type / folder / session_id
    (acq / "images").mkdir(parents=True, exist_ok=True)
    records, ids = [], []
    for n, img in enumerate(images, 1):
        image_id = f"{board_id or dataset_type.upper() + '_UNASSIGNED'}_TOP_{session_id}_C{n:03d}"
        original = acq / "images" / f"{image_id}.png"
        cv2.imwrite(str(original), img)
        saved = None
        if review in ("kept", "rejected"):
            target = store / ("curated" if review == "kept" else "quarantine") / dataset_type / folder / session_id
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target / original.name)
            saved = (target / original.name).relative_to(store).as_posix()
        records.append({"image_id": image_id, "session_id": session_id, "dataset_type": dataset_type,
                        "board_id": board_id, "capture_id": f"C{n:03d}", "stage": "after_tht", "camera": "test",
                        "width": img.shape[1], "height": img.shape[0], "captured_at": "2026-10-05T10:00:00",
                        "original_file": original.relative_to(store).as_posix(), "saved_file": saved,
                        "sha256": sha256_of(original), "quality": {"status": "pass", "flags": []},
                        "review": {"status": review}})
        ids.append(image_id)
    (acq / "session.json").write_text(json.dumps({
        "session_id": session_id, "dataset_type": dataset_type, "board_id": board_id, "stage": "after_tht",
        "product_id": product_id, "variant_id": "variant_1", "profile_id": f"{product_id}/variant_1",
        "lighting": "lab", "resolution": [1200, 900], "images": records}), encoding="utf-8")
    return ids


def write_registry(store: Path, rows: list[tuple[str, str, str, str]]) -> None:
    folder = store / "dataset"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["board_id,condition,split,defects,defect_source,identity_source,verified_by,notes"]
    lines += [f"{b},{c},{s},{d},test,acquisition_board_id,test," for b, c, s, d in rows]
    (folder / "boards.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def curated_ref(pid: str, board_id: str, session_id: str) -> str:
    return f"data/products/{pid}/curated/good/{board_id}/{session_id}/{board_id}_TOP_{session_id}_C001.png"


def build_project(root: Path) -> Path:
    shutil.copytree(ROOT / "configs" / "capture", root / "configs" / "capture")
    shutil.copytree(ROOT / "configs" / "profiles", root / "configs" / "profiles")
    shutil.copy2(ROOT / "configs" / "project.yaml", root / "configs" / "project.yaml")
    shutil.copy2(ROOT / "configs" / "inspection.yaml", root / "configs" / "inspection.yaml")
    engine = yaml.safe_load((ROOT / "experiments" / "experiment_002" / "config.yaml").read_text(encoding="utf-8"))
    engine.update(canonical_px_per_mm=PPM, ignore_source_regions_px=[], watermark={"enabled": False})
    engine["registration"]["min_inliers"] = 20
    (root / "configs" / "engine_test.yaml").write_text(yaml.safe_dump(engine), encoding="utf-8")
    # alpha: GOOD_001 (two sessions), GOOD_002 (another physical board, other parts lot) -> golden bank
    a = root / "data" / "products" / "alpha"
    add_session(a, "good", "GOOD_001", "S20261005-01", [scene(board())], "alpha")
    add_session(a, "good", "GOOD_001", "S20261005-03", [scene(board(), angle=1.0, shift=(-12, 8))], "alpha")
    add_session(a, "good", "GOOD_002", "S20261005-02", [scene(board(marking=2), angle=2.0, shift=(15, -10))], "alpha")
    add_session(a, "defect", "DEFECT_001", "S20261005-04", [scene(board(missing=("R5",)), angle=-2.0)], "alpha")
    add_session(a, "test", "TEST_001", "S20261005-05", [scene(board(marking=3), angle=1.5)], "alpha")
    write_registry(a, [("GOOD_001", "good", "reference", ""), ("GOOD_002", "good", "reference", ""),
                       ("DEFECT_001", "fault", "validation", "R5:missing"), ("TEST_001", "unknown", "test", "")])
    write_product(root, "alpha", dims=(116, 83), parts=PARTS, reference=curated_ref("alpha", "GOOD_001", "S20261005-01"))
    # beta: a second product with its own golden bank
    b = root / "data" / "products" / "beta"
    add_session(b, "good", "GOOD_001", "S20261005-06", [scene(beta_board())], "beta")
    add_session(b, "good", "GOOD_002", "S20261005-07", [scene(beta_board(marking=4), angle=-2.0, shift=(10, 5))], "beta")
    write_registry(b, [("GOOD_001", "good", "reference", ""), ("GOOD_002", "good", "reference", "")])
    write_product(root, "beta", dims=(100, 70), parts=BETA_PARTS, reference=curated_ref("beta", "GOOD_001", "S20261005-06"))
    # the real Pump Controller profile, unchanged (identity only, everything else UNKNOWN)
    shutil.copytree(ROOT / "configs" / "products" / "pump_controller", root / "configs" / "products" / "pump_controller")
    return root


@pytest.fixture(scope="module")
def base_project(tmp_path_factory) -> Path:
    root = build_project(tmp_path_factory.mktemp("pcb_project"))
    InspectionStation(root)                    # align the golden banks once (cached, copied with the project)
    return root


@pytest.fixture
def project(base_project, tmp_path) -> Path:
    return Path(shutil.copytree(base_project, tmp_path / "project"))


@pytest.fixture
def station(project) -> InspectionStation:
    return InspectionStation(project)


def frame_of(image: np.ndarray):
    camera = MockCamera([image])
    camera.open()
    return camera.capture()


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1-4: profiles, products, golden bank of several boards / sessions ------------------------------

def test_profile_loading_and_unknowns_stay_null(project):
    p = load_product_profile("alpha", project_root=project)
    assert p.key == "alpha/variant_1" and p.board_mm == (116.0, 83.0) and p.side == "top"
    assert p.engine_type == "REFERENCE_BASELINE" and p.enabled_defect_types == ["missing_component"]
    assert p.storage_root == project / "data" / "products" / "alpha" and p.registry_dir == p.storage_root / "dataset"
    assert "product.board.thickness_mm" in p.unknowns()          # null = UNKNOWN, never filled in
    with pytest.raises(ProfileError, match="unknown product"):
        load_product_profile("nonexistent", project_root=project)
    with pytest.raises(ProfileError, match="invalid product id"):
        load_product_profile("LG Indicator", project_root=project)
    with pytest.raises(ProfileError, match="variant"):
        load_product_profile("alpha", "variant_9", project_root=project)


def test_multiple_products_load_with_separate_storage(project):
    profiles, problems = load_all_profiles(project)
    assert [p.product_id for p in profiles] == ["alpha", "beta", "pump_controller"] and problems == []
    assert len({p.storage_root for p in profiles}) == 3
    beta = next(p for p in profiles if p.product_id == "beta")
    (beta.folder / "product.yaml").write_text((beta.folder / "product.yaml").read_text(encoding="utf-8").replace(
        "data/products/beta", "data/products/alpha"), encoding="utf-8")
    _, problems = load_all_profiles(project)
    assert any("share the storage folder" in p for p in problems)   # product data would mix: reported


def test_golden_bank_holds_multiple_good_boards_and_sessions(station):
    bank = station.banks["alpha"]
    assert bank.status == READY and bank.boards == ["GOOD_001", "GOOD_002"]
    assert bank.images == 3                                       # GOOD_001 from two sessions + GOOD_002
    assert bank.data.gallery.boards.count("GOOD_001") == 2
    assert station.banks["beta"].status == READY and station.banks["beta"].boards == ["GOOD_001", "GOOD_002"]


def test_too_few_good_boards_means_review(project):
    folder = project / "configs" / "products" / "alpha" / "golden" / "golden_bank.yaml"
    folder.write_text(folder.read_text(encoding="utf-8").replace("min_boards: 2", "min_boards: 3"), encoding="utf-8")
    st = InspectionStation(project)
    assert st.banks["alpha"].status == INSUFFICIENT
    r = st.process(frame_of(scene(board(marking=5), angle=-2.0)), st.start_session(), "alpha")
    assert r.decision == REVIEW and r.outcome.reasons[0].code == "INSUFFICIENT_GOLDEN_REFERENCES"


def test_single_good_board_allows_experimental_inspection(project):
    # Only 1 physical GOOD board (GOOD_001) in registry
    registry = project / "data" / "products" / "alpha" / "dataset" / "boards.csv"
    registry.write_text(
        "board_id,condition,split,defects,defect_source,identity_source,verified_by,notes\n"
        "GOOD_001,good,reference,,,acquisition_board_id,test,\n",
        encoding="utf-8",
    )
    st = InspectionStation(project)
    assert st.banks["alpha"].status == INSUFFICIENT
    assert len(st.banks["alpha"].boards) == 1
    assert st.banks["alpha"].can_inspect

    # Good board should pass inspection under single-reference experimental mode
    r_good = st.process(frame_of(scene(board(marking=5), angle=-2.0)), st.start_session(), "alpha")
    assert r_good.decision == GOOD
    assert r_good.outcome.alignment_status == "OK"
    assert r_good.outcome.details.get("reference_mode") == "EXPERIMENTAL / SINGLE_REFERENCE"
    assert any("EXPERIMENTAL / SINGLE_REFERENCE" in lim for lim in r_good.outcome.limitations)

    # Defect board should be correctly detected under single-reference mode
    r_defect = st.process(frame_of(scene(board(missing=("R5",)), angle=-2.0)), st.start_session(), "alpha")
    assert r_defect.decision == DEFECT
    assert r_defect.outcome.alignment_status == "OK"
    assert len(r_defect.outcome.defects) >= 1
    assert r_defect.outcome.defects[0].region == "R5"



# --- 5-10: identification, alignment, quality, normal GOOD variation ---------------------------------

def test_auto_identification_picks_the_right_product(station):
    session = station.start_session()
    a = station.process(frame_of(scene(board(marking=5), angle=-3.0, scale=0.95, shift=(-20, 25))), session)
    b = station.process(frame_of(scene(beta_board(marking=6), angle=2.0)), session)
    assert a.identification.status == IDENTIFIED and a.outcome.product_id == "alpha"
    assert b.identification.status == IDENTIFIED and b.outcome.product_id == "beta"
    assert a.identification.confidence is None and a.identification.score > 0.4       # measured, not a probability
    assert {c["product_id"] for c in a.identification.candidates} == {"alpha", "beta"}
    assert any(n["product_id"] == "pump_controller" for n in a.identification.not_identifiable)


def test_unknown_product_is_review_and_never_compared(station, project):
    gamma = np.full((600, 900, 3), (160, 60, 40), np.uint8)       # blue board, no profile
    rng = np.random.default_rng(5)
    for _ in range(400):
        x, y = int(rng.integers(5, 890)), int(rng.integers(5, 590))
        cv2.rectangle(gamma, (x, y), (x + int(rng.integers(3, 12)), y + int(rng.integers(3, 12))), (220, 220, 220), 1)
    r = station.process(frame_of(scene(gamma)), station.start_session())
    assert r.decision == REVIEW and r.outcome.reasons[0].code == "UNKNOWN_PRODUCT"
    assert r.outcome.product_id is None and r.outcome.defects == [] and r.outcome.evidence_files == {}
    assert r.result_path.is_relative_to(project / "data" / "products" / "_unresolved")


def test_two_products_matching_equally_is_ambiguous_review(project):
    twin = project / "data" / "products" / "alpha_twin"
    shutil.copytree(project / "data" / "products" / "alpha", twin)
    ref = curated_ref("alpha_twin", "GOOD_001", "S20261005-01")
    write_product(project, "alpha_twin", dims=(116, 83), parts=PARTS, reference=ref)
    st = InspectionStation(project)
    r = st.process(frame_of(scene(board(marking=5), angle=-2.0)), st.start_session())
    assert r.identification.status == AMBIGUOUS and r.decision == REVIEW
    assert r.outcome.reasons[0].code == "AMBIGUOUS_PRODUCT" and r.outcome.product_id is None


def test_alignment_within_tolerance_is_inspected(station):
    r = station.process(frame_of(scene(board(marking=5), angle=-3.0, scale=0.95, shift=(-20, 25))),
                        station.start_session())
    assert r.outcome.alignment_status == "OK" and r.outcome.alignment_error["fit_error_mm"] < 0.25
    assert r.outcome.regions_checked == len(PARTS)


def test_alignment_failure_is_review_not_defect(station):
    upside_down = scene(board(missing=("R5",)), angle=180.0)    # board outside the allowed rotation (30 deg)
    r = station.process(frame_of(upside_down), station.start_session(), "alpha")
    assert r.decision == REVIEW and r.outcome.alignment_status == "FAILED"
    assert r.outcome.reasons[0].code == "ALIGNMENT" and r.outcome.defects == []
    auto = station.process(frame_of(upside_down), station.start_session())
    assert auto.identification.status == IDENTIFIED and auto.outcome.product_id == "alpha"   # recognised, misplaced
    assert auto.decision == REVIEW and auto.outcome.reasons[0].code == "ALIGNMENT"


def test_image_quality_failure_is_review_not_defect(station):
    dark = np.full((900, 1200, 3), 2, np.uint8)
    r = station.process(frame_of(dark), station.start_session(), "alpha")
    assert r.decision == REVIEW and r.outcome.reasons[0].code == "IMAGE_QUALITY"
    assert r.outcome.quality_status == "FAIL" and r.capture.path.is_file()   # stored all the same


def test_new_good_board_is_not_a_defect(station):
    # GOOD_003: never seen, another parts lot (marking), placed differently -> judged against the BANK
    r = station.process(frame_of(scene(board(marking=9), angle=2.5, scale=1.02, shift=(18, -14))),
                        station.start_session())
    assert r.decision == GOOD, r.outcome.reason_text
    assert r.outcome.confidence is None                             # never invented


def test_bank_board_is_judged_against_the_other_boards(station):
    # an image of GOOD_002 itself: leave-one-board-out, so it is compared with GOOD_001 only, and still GOOD
    r = station.process(frame_of(scene(board(marking=2), angle=-1.0)), station.start_session(), "alpha", "GOOD_002")
    assert r.decision == GOOD and r.outcome.details["left_out_board"] == "GOOD_002"
    assert all("GOOD_002" not in name for name in r.outcome.details["good_examples"])


# --- 11-13: defect evidence, every capture stored, duplicates ------------------------------------------

def test_defect_is_located_with_evidence_and_stored(station, project):
    r = station.process(frame_of(scene(board(missing=("C3",)), angle=-3.0, scale=0.95, shift=(-20, 25))),
                        station.start_session())
    o = r.outcome
    assert o.overall_decision == DEFECT and [d.region for d in o.defects] == ["C3"]
    d = o.defects[0]
    assert d.defect_type == "missing_component" and d.confidence is None and d.status == "unconfirmed"
    x0, y0, x1, y1 = d.location["box_mm"]
    assert x0 <= PARTS["C3"][0] + 1.6 <= x1 and y0 <= PARTS["C3"][1] + 0.8 <= y1   # where the part is (mm)
    assert d.evidence["similarity"] < d.evidence["limit"] - d.evidence["uncertainty_band"]
    folder = r.result_path.parent
    assert folder.is_relative_to(project / "data" / "products" / "alpha" / "inspections")
    assert {"result.json", "annotated.jpg", "aligned.jpg", "evidence.jpg", "overview.jpg"} <= {p.name for p in folder.iterdir()}
    stored = json.loads(r.result_path.read_text(encoding="utf-8"))
    assert stored["overall_decision"] == DEFECT and stored["defects"][0]["region"] == "C3"
    assert stored["image"]["sha256"] == sha256_of(r.capture.path)


def test_every_capture_is_stored_first_whatever_the_result(station, project, monkeypatch):
    session = station.start_session()
    images = [scene(board(marking=5)), np.full((900, 1200, 3), 2, np.uint8), scene(beta_board())]
    results = [station.process(frame_of(img), session) for img in images]
    monkeypatch.setattr(station.identifier, "identify", lambda image: (_ for _ in ()).throw(RuntimeError("boom")))
    results.append(station.process(frame_of(scene(board())), session))
    assert [r.decision for r in results][-1] == ERROR and "boom" in results[-1].outcome.reason_text
    records = sorted((project / "data" / "captures" / session).glob("*_C*.json"))
    assert len(records) == 4
    for r, record_file in zip(results, records):
        record = json.loads(record_file.read_text(encoding="utf-8"))
        assert (project / record["file_path"]).is_file() and sha256_of(project / record["file_path"]) == record["sha256"]
        assert record["decision"] == r.decision and record["inspection_id"] == r.outcome.inspection_id
        assert record["image_id"] == r.capture.image_id and record["capture_timestamp"] and record["camera_id"].startswith("mock:")
        assert record["synthetic"] is True                       # a mock-camera image can never pass for a real capture
        assert r.result_path is not None and r.result_path.is_file()
        for key in ("image_id", "inspection_id", "product_id", "variant_id", "physical_board_id", "session_id",
                    "capture_timestamp", "camera_id", "resolution", "stage", "file_path", "sha256", "quality_status",
                    "inspection_status", "decision", "decision_confidence"):
            assert key in record


def test_capture_is_never_overwritten_and_reinspection_gets_a_new_id(station, project):
    session = station.start_session()
    first = station.process(frame_of(scene(board(marking=5))), session)
    original = sha_bytes(first.capture.path)
    again = station.inspect_file(first.capture.path)
    assert again.capture.image_id == first.capture.image_id and again.outcome.inspection_id.endswith("_I02")
    assert sha_bytes(first.capture.path) == original and first.result_path.is_file()
    with pytest.raises(CaptureError):
        station.captures.save(session, CapturedFrame(np.zeros((0, 0, 3), np.uint8), datetime.now(), "mock:x"))


def test_duplicate_checksums_are_counted_once(project):
    a = project / "data" / "products" / "alpha"
    original = next((a / "acquisition" / "good" / "GOOD_002").rglob("*.png"))
    shutil.copy2(original, a / "acquisition" / "good" / "GOOD_002" / "S20261005-02" / "images" / "stray_copy.png")
    profiles, _ = load_all_profiles(project)
    v = validate_all(project, profiles, capture_root=project / "data" / "captures", derived=DERIVED)
    alpha = v.products["alpha"]
    entries = [e for e in alpha.images if e.sha256 == sha256_of(original)]
    assert len(entries) == 1 and len(entries[0].copies) == 2      # curated copy + stray copy: one sample
    assert any("stray_copy.png" in p for p, _ in alpha.duplicates)
    assert any("stray_copy.png" in p for p, _ in alpha.orphans)   # a file without a record is reported, not ignored


# --- 14-17: isolation, splits, discovery of new sessions ------------------------------------------------

def test_test_and_defect_boards_never_enter_the_golden_bank(station):
    bank = station.banks["alpha"]
    assert "TEST_001" not in bank.boards and "DEFECT_001" not in bank.boards
    assert all("TEST" not in n and "DEFECT" not in n for n in bank.data.gallery.names)


def test_product_isolation(station, project):
    alpha, beta = station.profile("alpha"), station.profile("beta")
    image = scene(beta_board())
    refused = InspectionEngine().inspect(image, alpha, station.banks["beta"])
    assert refused.overall_decision == ERROR and "refused" in refused.reason_text
    r = station.process(frame_of(image), station.start_session(), "beta")
    assert r.outcome.product_id == "beta" and r.decision == GOOD
    assert r.result_path.is_relative_to(beta.storage_root) and not r.result_path.is_relative_to(alpha.storage_root)
    # alpha selected for a beta board: alpha's own references only -> not alignable -> REVIEW, never a defect
    wrong = station.process(frame_of(image), station.start_session(), "alpha")
    assert wrong.decision == REVIEW and wrong.outcome.defects == []


def test_splits_are_by_physical_board(station):
    bank = station.banks["alpha"]
    reg = (station.profile("alpha").registry_dir / "boards.csv").read_text(encoding="utf-8")
    assert reg.count("GOOD_001,") == 1                             # one row = one split per physical board
    gallery = bank.data.gallery
    assert set(gallery.without_board("GOOD_001").boards[1:]) == {"GOOD_002"}   # every image of the board leaves
    assert set(gallery.boards) == set(bank.boards)


def test_new_sessions_and_unreviewed_images_are_discovered(project):
    a = project / "data" / "products" / "alpha"
    add_session(a, "good", "GOOD_004", "S20261006-01", [scene(board(marking=8))], "alpha", review="pending")
    add_session(a, "good", None, "S20261006-02", [scene(board(marking=8), angle=1)], "alpha")
    profiles, _ = load_all_profiles(project)
    v = validate_all(project, profiles, capture_root=project / "data" / "captures", derived=DERIVED)
    alpha = v.products["alpha"]
    new = {e.session_id: e for e in alpha.images if e.session_id.startswith("S20261006")}
    assert new["S20261006-01"].status == PENDING and new["S20261006-02"].status == UNRESOLVED
    assert "GOOD_004" in alpha.boards and not alpha.boards["GOOD_004"].registered
    assert any("GOOD_004" in w and "no row in boards.csv" in w for w in alpha.warnings)
    text = format_validation(v)
    assert "Pending review" in text and "S20261006-02" in text
    assert v.unclaimed == []


def test_unclaimed_images_and_station_captures_are_reported(station, project):
    stray = project / "data" / "somewhere" / "photo.png"
    stray.parent.mkdir(parents=True)
    cv2.imwrite(str(stray), board())
    station.process(frame_of(np.full((900, 1200, 3), 2, np.uint8)), station.start_session())   # no product
    station.process(frame_of(scene(board(marking=5))), station.start_session())               # alpha
    v = validate_all(project, station.profiles, capture_root=project / "data" / "captures", derived=DERIVED)
    assert v.unclaimed == ["data/somewhere/photo.png"]
    assert len(v.unresolved_captures) == 1
    assert v.products["alpha"].station["captures_by_decision"] == {GOOD: 1}
    synthetic = [e for e in v.products["alpha"].images if e.source == "station_capture"]
    assert synthetic and synthetic[0].status == SYNTHETIC         # mock-camera captures never count as data


# --- 18-22: future products, Pump Controller, camera, review ------------------------------------------

def test_future_product_needs_configuration_only(project):
    g = project / "data" / "products" / "gamma"
    gamma = lambda **kw: scene(beta_board(**kw)[:, ::-1].copy(), angle=1.0)   # noqa: E731  (another layout)
    add_session(g, "good", "GOOD_001", "S20261007-01", [gamma()], "gamma")
    add_session(g, "good", "GOOD_002", "S20261007-02", [gamma(marking=3)], "gamma")
    write_registry(g, [("GOOD_001", "good", "reference", ""), ("GOOD_002", "good", "reference", "")])
    parts = {n: (100 - 3.2 - x, y) for n, (x, y) in BETA_PARTS.items()}           # mirrored layout
    write_product(project, "gamma", dims=(100, 70), parts=parts, reference=curated_ref("gamma", "GOOD_001", "S20261007-01"))
    st = InspectionStation(project)
    assert "gamma" in list_products(project / "configs" / "products") and st.banks["gamma"].status == READY
    r = st.process(frame_of(gamma(missing=("U3",))), st.start_session())
    assert r.outcome.product_id == "gamma" and r.decision == DEFECT and r.outcome.defects[0].region == "U3"


def test_core_code_names_no_product():
    pattern = re.compile(r"lg[_ ]?indicator|pump|116\.0|83\.0", re.IGNORECASE)
    for package in ("inspection", "station", "products", "camera"):
        for path in (ROOT / "src" / "pcb_inspection" / package).glob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), f"{path.name} names a product / its size"


def test_pump_controller_profile(station, project):
    p = station.profile("pump_controller")
    assert p.name == "Pump Controller" and p.board_mm is None and p.stages == [] and p.enabled_defect_types == []
    assert station.banks["pump_controller"].status == NOT_CONFIGURED
    r = station.process(frame_of(scene(beta_board())), station.start_session(), "pump_controller")
    assert r.identification.status == SELECTED and r.decision == REVIEW
    assert r.outcome.reasons[0].code == "NO_GOLDEN_REFERENCES" and r.outcome.defects == []
    assert r.result_path.is_relative_to(project / "data" / "products" / "pump_controller")
    auto = station.process(frame_of(scene(beta_board())), station.start_session())
    assert auto.outcome.product_id == "beta"                      # never identified as Pump Controller by guesswork


def test_mock_camera_interface_and_no_frame_means_no_capture(project):
    from station import LiveStation
    from pcb_inspection.preprocessing.board import BoardSegmentationParams
    from pcb_inspection.preprocessing.presence import PresenceParams
    st = InspectionStation(project)
    good = scene(board(marking=5))
    camera = MockCamera([good, good, good])
    assert camera.open() and camera.get_info()["width"] == 1200 and camera.read() is good
    live = LiveStation(st, camera, "AUTO", AutoTrigger(TriggerSettings(settle_s=1.0, still_max_mean_abs_diff=6.0,
                                                                         removed_after_s=1.0)),
                       PresenceParams(), BoardSegmentationParams(), presence_every_s=0.0)
    for t in (0.0, 0.5, 1.0, 1.6, 2.2, 3.0):                     # board in view and still -> exactly one capture
        live.observe(t)
        while live.busy:
            time.sleep(0.05)
    live.observe(3.5)
    assert len(live.results) == 1 and live.results[0].decision == GOOD
    empty = LiveStation(st, MockCamera([]), "AUTO", AutoTrigger(TriggerSettings()), PresenceParams(),
                        BoardSegmentationParams(), 0.0)
    empty.camera.open()
    empty.capture_now()
    empty.future = empty.executor.submit(empty._capture_and_inspect)
    empty.close()
    session = st.captures.session(empty.session_id)
    assert empty.results == [] and session["captures"] == 0 and "no frame" in session["failures"][0]["reason"]
    assert not list((project / "data" / "captures" / empty.session_id).glob("*.png"))   # nothing invented
    live.close()
    assert camera.closed


def test_human_review_never_changes_the_ai_result(station, project):
    r = station.process(frame_of(scene(board(marking=5))), station.start_session())
    before = sha_bytes(r.result_path)
    row = record_review(r.result_path, CONFIRM_GOOD, "QA Lead", "checked by eye")
    assert sha_bytes(r.result_path) == before
    assert row["ai_decision"] == GOOD and row["human_decision"] == CONFIRM_GOOD and row["agrees_with_ai"] is True
    assert row["reviewer"] == "QA Lead" and row["review_timestamp"]
    reviews = (project / "data" / "products" / "alpha" / "reviews" / "reviews.jsonl").read_text(encoding="utf-8")
    assert r.outcome.inspection_id in reviews
    with pytest.raises(ValueError):
        record_review(r.result_path, CONFIRM_GOOD, "  ")


def test_auto_trigger_one_capture_per_board():
    trig = AutoTrigger(TriggerSettings(settle_s=1.0, still_max_mean_abs_diff=6.0, removed_after_s=1.0))
    a, b = scene(board()), scene(beta_board())
    fired = [trig.update(t, True, a) for t in (0, 0.5, 1.0, 1.5, 2.0, 3.0)]
    assert fired.count(True) == 1                                  # one board, one capture
    assert not trig.update(3.2, False, None) and not trig.update(4.5, False, None)   # removed
    fired = [trig.update(t, True, b) for t in (5.0, 5.5, 6.0, 6.6)]
    assert fired.count(True) == 1                                  # next board captured again
    trig.manual()
    assert trig.update(7.0, False, None)                           # SPACE: capture now


def test_leave_one_board_out_is_real_in_the_classical_baseline():
    a, b = np.full((83, 116, 3), 100, np.uint8), np.full((83, 116, 3), 140, np.uint8)
    base = build_good_baseline([("GOOD_001", a), ("GOOD_002", b)], board_dimensions_mm=(116, 83), px_per_mm=1.0)
    other = base.without_board("GOOD_001")
    assert other.board_ids == ["GOOD_002"] and int(other.median_canonical_bgr[40, 50, 0]) == 140
    single = build_good_baseline([("GOOD_001", a)], board_dimensions_mm=(116, 83), px_per_mm=1.0)
    with pytest.raises(ValueError, match="only GOOD board"):
        single.without_board("GOOD_001")
    with pytest.raises(ValueError, match="required"):
        build_good_baseline([("GOOD_001", a)], px_per_mm=1.0)


# --- real repository (read-only) ---------------------------------------------------------------------

def test_real_lg_profile_matches_the_verified_facts():
    lg = load_product_profile("lg_indicator", project_root=ROOT)
    facts = yaml.safe_load((ROOT / "configs" / "profiles" / "lg_indicator" / "variant_1.yaml").read_text(encoding="utf-8"))
    pcb = facts["pcb"]
    assert lg.board_mm == (pcb["dimensions_mm"]["long_side"], pcb["dimensions_mm"]["short_side"])
    assert lg.side == pcb["inspection_sides"][0] and lg.stages == facts["inspection_stages"]["confirmed"]
    assert lg.colour_variants == facts["appearance_variations"]["board_colour"]["reported_values"]
    assert lg.name == pcb["model"] and lg.board["thickness_mm"] == pcb["thickness_mm"]
    pump = load_product_profile("pump_controller", project_root=ROOT)
    assert pump.board_mm is None and pump.golden["geometric_reference"] is None and pump.stages == []
    profiles, problems = load_all_profiles(ROOT)
    assert problems == [] and {"lg_indicator", "pump_controller"} <= {p.product_id for p in profiles}


def test_archived_pre_reset_data_is_complete():
    # 2026-10-06: the Experiment 0 / 1B / 2 and station images were archived (scripts/archive_old_data.py); the LG
    # profile now covers only the clean collection (tests/test_clean_collection.py). No silent loss: every file
    # listed in an archive manifest is still there with its recorded size.
    manifests = sorted((ROOT / "archive").glob("*/manifest.csv"))
    if not manifests:
        pytest.skip("no archive in this checkout")
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, manifest
        for r in rows:      # (new files may exist again at an original path, e.g. new station results)
            file = ROOT / r["archived_path"]
            assert file.is_file() and file.stat().st_size == int(r["size_bytes"]), r["archived_path"]
