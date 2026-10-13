"""Acquisition sessions: IDs, files, metadata, review and type separation.

Images are small generated test inputs (a coloured rectangle on a light
background), not PCB images. No camera is needed.
"""

import hashlib
import io
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pcb_inspection.acquisition.session import (
    AcquisitionStore,
    audit,
    CaptureError,
    ReviewError,
    SessionConfig,
    allocate_session_id,
    make_defect_report,
    make_image_id,
    normalise_board_id,
    settings_from_configs,
    split_groups,
)
from pcb_inspection.config import load_config
from pcb_inspection.dataset.splitting import assign_captures, find_leakage, split_boards

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_CFG = load_config(ROOT / "configs" / "capture" / "droidcam_e1b.yaml")
CAMERA = {"name": "DroidCam Video", "index": 1, "backend": "dshow", "width": 320, "height": 180,
          "fps_reported": 30.0, "fps_measured": 29.8}
START = datetime(2026, 10, 5, 10, 0, 0)


def png_bytes(seed: int = 0, colour=(200, 30, 40), size=(320, 180)) -> bytes:
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0], 3), 235, np.int16)
    h = int(size[1] * 0.7)
    w = int(h * 116 / 83)
    y0, x0 = (size[1] - h) // 2, (size[0] - w) // 2
    img[y0:y0 + h, x0:x0 + w] = colour
    img = np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path, active_profile):
    return AcquisitionStore(settings_from_configs(tmp_path / "data", active_profile, CAPTURE_CFG))


def new_session(store, dataset_type="good", board="GOOD_001", count=3, **kw):
    config = SessionConfig(dataset_type=dataset_type, stage=kw.pop("stage", "after_tht"), requested_image_count=count,
                           capture_interval_s=3, board_id=board, **kw)
    return store.create_session(config, CAMERA, started_at=START)


def capture(store, session, seed=0):
    return store.save_capture(session, png_bytes(seed), datetime(2026, 10, 5, 10, 0, seed + 3), CAMERA, frame_seq=seed)


# --- identifiers -------------------------------------------------------------

def test_session_id_generation():
    day = date(2026, 10, 5)
    assert allocate_session_id(day, []) == "S20261005-01"
    used = ["S20261005-01", "S20261005-03", "S20261004-07", "junk", "S20261005_09"]
    assert allocate_session_id(day, used) == "S20261005-04"
    assert allocate_session_id(date(2026, 10, 4), used) == "S20261004-08"
    with pytest.raises(ValueError):
        allocate_session_id(day, ["S20261005-99"])


def test_store_session_ids_are_unique_across_types_and_skip_reserved(tmp_path, active_profile):
    store = AcquisitionStore(settings_from_configs(tmp_path, active_profile, CAPTURE_CFG),
                             reserved_session_ids={"S20261005-01"})   # e.g. an Experiment 0 ingest session
    ids = [new_session(store, "good").id, new_session(store, "defect", "DEFECT_001").id,
           new_session(store, "test", "TEST_001").id]
    assert ids == ["S20261005-02", "S20261005-03", "S20261005-04"]
    # a restarted application sees the sessions on disk
    again = AcquisitionStore(settings_from_configs(tmp_path, active_profile, CAPTURE_CFG))
    assert new_session(again, "good").id == "S20261005-05"


@pytest.mark.parametrize("dataset_type, text, expected", [
    ("good", "", None), ("good", "   ", None), ("good", "1", "GOOD_001"), ("good", "001", "GOOD_001"),
    ("good", "good_7", "GOOD_007"), ("good", "GOOD-012", "GOOD_012"), ("defect", "3", "DEFECT_003"),
    ("defect", "DEFECT_001", "DEFECT_001"), ("test", "42", "TEST_042"), ("test", "test_001", "TEST_001"),
])
def test_board_id_normalisation(dataset_type, text, expected):
    assert normalise_board_id(dataset_type, text) == expected


@pytest.mark.parametrize("dataset_type, text", [
    ("good", "DEFECT_001"),   # a board of another pool: ambiguous label, refused
    ("defect", "GOOD_001"),
    ("test", "GOOD_001"),     # a test board never carries a GOOD / DEFECT label
    ("good", "TEST_001"),
    ("good", "0"), ("good", "1000"), ("good", "board one"), ("good", "GOOD_0001"),
])
def test_board_id_refused(dataset_type, text):
    with pytest.raises(ValueError):
        normalise_board_id(dataset_type, text)


def test_image_id_generation():
    assert make_image_id("good", "GOOD_001", "TOP", "S20261005-01", 1) == "GOOD_001_TOP_S20261005-01_C001"
    assert make_image_id("defect", "DEFECT_002", "TOP", "S20261005-02", 12) == "DEFECT_002_TOP_S20261005-02_C012"
    assert make_image_id("test", None, "TOP", "S20261005-03", 5) == "TEST_UNASSIGNED_TOP_S20261005-03_C005"
    for bad in (dict(number=0), dict(number=1000), dict(side="LEFT"), dict(session_id="S001")):
        args = {"dataset_type": "good", "board_id": "GOOD_001", "side": "TOP", "session_id": "S20261005-01",
                "number": 1, **bad}
        with pytest.raises(ValueError):
            make_image_id(**args)


# --- sessions ----------------------------------------------------------------

def test_create_session_directories_and_metadata(store):
    session = new_session(store, count=20)
    assert session.folder == store.root / "acquisition" / "good" / "GOOD_001" / "S20261005-01"
    assert (session.folder / "images").is_dir()
    data = json.loads((session.folder / "session.json").read_text(encoding="utf-8"))
    assert data["session_id"] == "S20261005-01" and data["dataset_type"] == "good"
    assert data["board_id"] == "GOOD_001" and data["split_group"] == "GOOD_001"
    assert data["stage"] == "after_tht" and data["side"] == "TOP"
    assert data["start_time"] == "2026-10-05T10:00:00" and data["end_time"] is None
    assert data["camera"]["name"] == "DroidCam Video" and data["resolution"] == [320, 180] and data["fps"] == 29.8
    assert data["capture_interval_s"] == 3 and data["requested_image_count"] == 20
    assert data["status"] == "running" and data["declared_condition"] == "good"
    assert (data["pcb_model"], data["variant"], data["profile_id"]) == ("LG Indicator", "1", "lg_indicator/variant_1")
    assert data["capture_config_id"] == "droidcam_e1b"
    assert data["counts"] == {"successful_captures": 0, "failed_captures": 0, "kept_count": 0,
                              "rejected_count": 0, "pending_review_count": 0}


def test_phone_model_recorded_only_for_the_droidcam_stream(store):
    phone = new_session(store)                                       # CAMERA name "DroidCam Video"
    assert (phone.data["device_model"], phone.data["device_connection"]) == ("Xiaomi Redmi 12 5G", "DroidCam")
    webcam = store.create_session(SessionConfig("good", "after_tht", 3, 3, "GOOD_001"),
                                  {**CAMERA, "name": "Integrated Camera", "index": 0}, started_at=START)
    assert webcam.data["device_model"] is None and webcam.data["device_connection"] is None
    assert webcam.data["camera"]["name"] == "Integrated Camera"


def test_session_without_board_is_its_own_split_group(store):
    session = new_session(store, "test", None)
    assert session.folder.parent.name == "UNASSIGNED"
    assert session.data["split_group"] == f"UNASSIGNED:{session.id}"
    assert session.data["declared_condition"] is None and session.data["condition_source"] is None


@pytest.mark.parametrize("change", [
    dict(stage="after_aoi"), dict(stage="other"), dict(count=0), dict(count=1000), dict(board="GOOD_1"),
    dict(board="DEFECT_001"), dict(review_mode="sometimes"),
    dict(defect_report={"defect_known": "unknown"}),   # defect information only in DEFECT sessions
])
def test_session_validation(store, change):
    with pytest.raises(ValueError):
        new_session(store, **change)


def test_other_stage_needs_a_note(store):
    session = new_session(store, stage="other", notes="after conformal coating trial")
    assert session.data["stage"] == "other" and session.data["notes"] == "after conformal coating trial"


# --- captures ----------------------------------------------------------------

def test_capture_metadata_and_checksum(store, tmp_path, active_profile):
    session = new_session(store)
    data = png_bytes(0)
    record = store.save_capture(session, data, datetime(2026, 10, 5, 10, 0, 3, 250000), CAMERA, frame_seq=91)
    path = store.root / record["original_file"]
    assert record["original_file"] == \
        "acquisition/good/GOOD_001/S20261005-01/images/GOOD_001_TOP_S20261005-01_C001.png"
    assert path.read_bytes() == data
    assert record["sha256"] == hashlib.sha256(data).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["file_size_bytes"] == len(data) == path.stat().st_size
    assert record["image_id"] == "GOOD_001_TOP_S20261005-01_C001" and record["capture_id"] == "C001"
    assert record["sequence"] == 1 and record["frame_seq"] == 91
    assert record["captured_at"] == "2026-10-05T10:00:03.250"
    assert (record["width"], record["height"], record["fps"]) == (320, 180, 29.8)
    assert record["camera"] == "DroidCam Video" and record["stage"] == "after_tht" and record["board_id"] == "GOOD_001"
    assert record["quality"]["status"] in ("pass", "flagged") and record["quality"]["readable"] is True
    assert record["colour_estimate"] == "red" and record["colour_source"] == "automatic_estimate"
    assert record["review"]["status"] == "pending" and record["saved_file"] is None
    assert record["annotation"] == {"status": "not_started", "defects": []}
    # persisted: a fresh store (application restart) reads the same record
    reloaded = AcquisitionStore(settings_from_configs(store.root, active_profile, CAPTURE_CFG))
    assert reloaded.sessions()[0].images == [record]


def test_quality_flags_are_recorded_not_enforced(store):
    session = new_session(store)
    dark = io.BytesIO()
    Image.new("RGB", (320, 180), (2, 2, 2)).save(dark, "PNG")
    record = store.save_capture(session, dark.getvalue(), START, CAMERA)
    assert record["quality"]["status"] == "flagged" and "board_not_detected" in record["quality"]["flags"]
    assert (store.root / record["original_file"]).is_file() and record["review"]["status"] == "pending"


def test_sequential_naming_never_overwrites(store):
    session = new_session(store, count=10)
    names = [Path(capture(store, session, i)["original_file"]).name for i in range(3)]
    assert names == [f"GOOD_001_TOP_S20261005-01_C00{i}.png" for i in (1, 2, 3)]
    squatter = session.folder / "images" / "GOOD_001_TOP_S20261005-01_C004.png"
    squatter.write_bytes(b"not ours")
    record = capture(store, session, 3)
    assert record["capture_id"] == "C005" and squatter.read_bytes() == b"not ours"


def test_failed_capture_creates_no_image_record(store):
    session = new_session(store)
    with pytest.raises(CaptureError):
        store.save_capture(session, b"", START, CAMERA)
    with pytest.raises(CaptureError, match="cannot be read back"):
        store.save_capture(session, b"definitely not an image", START, CAMERA)
    assert session.images == [] and list((session.folder / "images").iterdir()) == []
    store.record_failure(session, "camera not live (disconnected)")
    capture(store, session)
    store.record_failure(session, "no new frame from the camera within 2 s")
    assert session.counts["successful_captures"] == 1 and session.counts["failed_captures"] == 2
    assert [f["attempt"] for f in session.data["failures"]] == [1, 3]
    on_disk = json.loads(session.path.read_text(encoding="utf-8"))
    assert on_disk["failures"][0]["reason"] == "camera not live (disconnected)"


# --- review ------------------------------------------------------------------

def test_keep_copies_to_curated_and_preserves_original(store):
    session = new_session(store)
    record = capture(store, session)
    original = store.root / record["original_file"]
    store.review(session, record["image_id"], "keep", at=datetime(2026, 10, 5, 10, 1))
    curated = store.root / "curated" / "good" / "GOOD_001" / session.id / original.name
    assert curated.read_bytes() == original.read_bytes()             # verified copy
    assert original.is_file()                                        # the original stays in acquisition/
    assert record["saved_file"] == curated.relative_to(store.root).as_posix()
    assert record["review"] == {"status": "kept", "decided_at": "2026-10-05T10:01:00", "reason": None, "note": None,
                                "decided_by": "operator"}
    assert session.counts["kept_count"] == 1 and session.counts["pending_review_count"] == 0


def test_reject_needs_a_reason_and_goes_to_quarantine(store):
    session = new_session(store, "defect", "DEFECT_001")
    record = capture(store, session)
    for reason in (None, "", "looks odd"):
        with pytest.raises(ValueError):
            store.review(session, record["image_id"], "reject", reason)
    assert record["review"]["status"] == "pending"
    store.review(session, record["image_id"], "reject", "reflection", note="glare on U1")
    quarantined = store.root / "quarantine" / "defect" / "DEFECT_001" / session.id / f"{record['image_id']}.png"
    assert quarantined.is_file() and (store.root / record["original_file"]).is_file()
    assert not (store.root / "curated").exists()
    assert record["review"]["reason"] == "reflection" and record["review"]["note"] == "glare on U1"
    assert session.counts["rejected_count"] == 1
    on_disk = json.loads(session.path.read_text(encoding="utf-8"))
    assert on_disk["images"][0]["review"]["reason"] == "reflection"


def test_review_is_final_and_checks_the_original(store):
    session = new_session(store)
    first, second = capture(store, session, 0), capture(store, session, 1)
    store.review(session, first["image_id"], "keep")
    with pytest.raises(ReviewError, match="already reviewed"):
        store.review(session, first["image_id"], "reject", "blurry")
    with pytest.raises(ValueError):
        store.review(session, second["image_id"], "maybe")
    (store.root / second["original_file"]).write_bytes(png_bytes(9))   # original altered after capture
    with pytest.raises(ReviewError, match="changed"):
        store.review(session, second["image_id"], "keep")
    assert second["review"]["status"] == "pending"


def test_good_defect_test_separation(store):
    sessions = {t: new_session(store, t, b) for t, b in
                (("good", "GOOD_001"), ("defect", "DEFECT_001"), ("test", "TEST_001"))}
    for i, (dataset_type, session) in enumerate(sessions.items()):
        record = capture(store, session, i)
        assert record["dataset_type"] == dataset_type
        assert record["original_file"].startswith(f"acquisition/{dataset_type}/")
        store.review(session, record["image_id"], "keep")
        assert record["saved_file"].startswith(f"curated/{dataset_type}/")
    curated = store.root / "curated"
    assert {p.relative_to(curated).parts[0] for p in curated.rglob("*.png")} == {"good", "defect", "test"}
    assert not any(p.name.startswith("TEST_") for d in ("good", "defect") for p in (curated / d).rglob("*"))
    # the dataset view excludes TEST unless it is asked for explicitly
    assert {r["dataset_type"] for r in store.records()} == {"good", "defect"}
    assert [r["board_id"] for r in store.records(("test",))] == ["TEST_001"]


def test_pending_reviews_across_sessions(store):
    a, b = new_session(store), new_session(store, "test", "TEST_001")
    ra, rb = capture(store, a, 0), capture(store, b, 1)
    assert [(s.id, r["image_id"]) for s, r in store.pending_reviews()] == \
        [(a.id, ra["image_id"]), (b.id, rb["image_id"])]
    store.review(a, ra["image_id"], "keep")
    assert [r["image_id"] for _, r in store.pending_reviews()] == [rb["image_id"]]


# --- session end -------------------------------------------------------------

def test_session_stop_keeps_captured_images(store):
    session = new_session(store, count=5)
    capture(store, session, 0)
    capture(store, session, 1)
    store.finish(session, "stopped", at=datetime(2026, 10, 5, 10, 5))
    data = json.loads(session.path.read_text(encoding="utf-8"))
    assert data["status"] == "stopped" and data["end_time"] == "2026-10-05T10:05:00"
    assert data["requested_image_count"] == 5 and data["counts"]["successful_captures"] == 2
    assert len(list((session.folder / "images").iterdir())) == 2
    with pytest.raises(CaptureError):
        capture(store, session, 2)                 # no captures after the end
    store.review(session, data["images"][0]["image_id"], "keep")   # review still possible
    store.finish(session, "completed")             # already finished: unchanged
    assert session.status == "stopped"
    with pytest.raises(ValueError):
        store.finish(new_session(store), "exploded")


def test_session_completion(store):
    session = new_session(store, count=2)
    capture(store, session, 0)
    store.record_failure(session, "no new frame")
    store.finish(session, "completed")
    assert session.status == "completed"
    assert session.counts["successful_captures"] + session.counts["failed_captures"] == 2


def test_sessions_left_running_are_marked_interrupted(store, active_profile):
    crashed = new_session(store)
    capture(store, crashed)
    assert store.mark_interrupted() == []          # this store's own running session is not touched
    restarted = AcquisitionStore(settings_from_configs(store.root, active_profile, CAPTURE_CFG))
    assert restarted.mark_interrupted() == [crashed.id]
    session = restarted.sessions()[0]
    assert session.status == "interrupted" and len(session.images) == 1
    restarted.review(session, session.images[0]["image_id"], "keep")   # images remain reviewable


# --- defect information ------------------------------------------------------

def test_defect_report(store):
    types = store.settings.defect_types
    unknown = make_defect_report(False, types, notes="operator saw something near U2")
    assert unknown["defect_known"] == "unknown" and unknown["defect_type"] is None
    assert unknown["verified"] is False and unknown["source"] == "operator_at_capture"
    known = make_defect_report(True, types, "missing_component", " c14 ", "top left", "C14 physically missing")
    assert known["taxonomy_defect_type"] == "missing_component" and known["component_reference"] == "C14"
    assert make_defect_report(True, types, "component_misalignment")["taxonomy_defect_type"] == "misaligned_component"
    assert make_defect_report(True, types, "broken_component")["taxonomy_defect_type"] is None  # not mapped by guess
    with pytest.raises(ValueError):
        make_defect_report(True, types)                       # known, but no type
    with pytest.raises(ValueError):
        make_defect_report(False, types, "missing_component")  # unknown, but a type
    session = new_session(store, "defect", "DEFECT_001", defect_report=known)
    record = capture(store, session)
    assert json.loads(session.path.read_text(encoding="utf-8"))["defect_report"]["component_reference"] == "C14"
    assert record["review"]["status"] == "pending"   # defect information does not skip the review


def test_configured_defect_mapping_uses_the_project_taxonomy(project_config):
    taxonomy = set(project_config["defects"]["taxonomy"])
    assert {v for v in CAPTURE_CFG["defect_types"].values() if v is not None} <= taxonomy


# --- audit -----------------------------------------------------------------------

def test_audit_detects_leaked_changed_and_stray_files(store):
    test_session = new_session(store, "test", "TEST_001")
    good_session = new_session(store)
    t, g = capture(store, test_session, 0), capture(store, good_session, 1)
    store.review(test_session, t["image_id"], "keep")
    store.review(good_session, g["image_id"], "reject", "blurry")
    assert audit(store) == []
    # a TEST image copied into curated/good/ by hand: not referenced -> reported
    leaked = store.root / "curated" / "good" / "GOOD_001" / Path(t["saved_file"]).name
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes((store.root / t["saved_file"]).read_bytes())
    # a record pointing a TEST image at curated/good/ -> reported
    t["saved_file"] = store.rel(leaked)
    (store.root / g["original_file"]).write_bytes(b"changed")
    issues = "\n".join(audit(store))
    assert "is not under curated/test/" in issues
    assert "original missing or changed" in issues
    assert "TEST_001_TOP" in issues and "not referenced" in issues


# --- board tracking / leakage --------------------------------------------------

def test_split_groups_keep_each_board_in_one_split(store):
    s1, s2 = new_session(store), new_session(store)                    # same board, two sessions
    s3, s4 = new_session(store, board="GOOD_002"), new_session(store, board=None)
    for i, s in enumerate((s1, s1, s2, s3, s4)):
        r = capture(store, s, i)
        store.review(s, r["image_id"], "keep")
    groups = split_groups(store.records())
    assert sorted(set(groups.values())) == ["GOOD_001", "GOOD_002", f"UNASSIGNED:{s4.id}"]
    assert sum(1 for g in groups.values() if g == "GOOD_001") == 3
    boards = split_boards({g: "good" for g in set(groups.values())}, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1)
    splits = assign_captures(groups, boards)
    assert find_leakage(splits, groups) == {}
    assert len({splits[r["image_id"]] for r in store.records() if r["board_id"] == "GOOD_001"}) == 1
