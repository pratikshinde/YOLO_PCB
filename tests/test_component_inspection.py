"""Experiment 2: component map, per-component comparison, report and the end-to-end inspection.

All images are generated test patterns (a red 'board' with grey pads and dark bodies, random white marks
for texture). They check the geometry and the decision logic; they are not PCB images or defect data.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from pcb_inspection.comparison.board_inspection import (
    IgnoreRegion,
    ignore_mask,
    inspect_image,
    load_gallery,
    load_inspection_config,
    load_reference_and_map,
    static_overlay_mask,
)
from pcb_inspection.comparison.components import (
    FAIL,
    MISSING,
    NOT_CHECKED,
    PASS,
    PRESENT,
    REVIEW,
    Component,
    ComponentCheckParams,
    ComponentMap,
    check_components,
    format_report,
    load_component_map,
    overall_status,
    part_mask,
    save_component_map,
    similarity_image,
    snap_box,
    validate_components,
)
from pcb_inspection.dataset.experiment0 import sha256_of

ROOT = Path(__file__).resolve().parents[1]
RED, PAD, BODY = (40, 30, 200), (185, 185, 180), (30, 30, 30)
PPM = 8                                          # px per mm of the generated canonical boards
PARTS = {"R1": (10, 10), "R2": (30, 10), "C3": (50, 10), "LED4": (10, 40), "R5": (40, 40), "U6": (80, 50)}


def draw_part(img, x_mm, y_mm, ppm=PPM, present=True, marking=0, dx_mm=0.0):
    """A 2-terminal part at (x, y) mm: pads 1.2 x 1.6 mm, body 3.2 x 1.6 mm between them."""
    x, y = round((x_mm + dx_mm) * ppm), round(y_mm * ppm)
    w, h, p = round(3.2 * ppm), round(1.6 * ppm), round(1.2 * ppm)
    cv2.rectangle(img, (x, y), (x + p, y + h), PAD, -1)
    cv2.rectangle(img, (x + w - p, y), (x + w, y + h), PAD, -1)
    if present:
        cv2.rectangle(img, (x + p // 2, y), (x + w - p // 2, y + h), BODY, -1)
        if marking:   # fine marking text, different per 'lot'
            rng = np.random.default_rng(marking)
            for _ in range(4):
                cx, cy = x + w // 2 + int(rng.integers(-4, 5)), y + h // 2 + int(rng.integers(-3, 4))
                cv2.circle(img, (cx, cy), 1, (230, 230, 230), -1)


def board(ppm=PPM, missing=(), seed=1, marking=0, shifted=None):
    """116 x 83 mm board image (board content only) with texture and the PARTS."""
    img = np.full((round(83 * ppm), round(116 * ppm), 3), RED, np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(500):   # silkscreen-like marks: texture for the feature matching
        x, y = int(rng.integers(5, img.shape[1] - 5)), int(rng.integers(5, img.shape[0] - 5))
        if any(abs(x / ppm - (px + 1.6)) < 4 and abs(y / ppm - (py + 0.8)) < 3.5 for px, py in PARTS.values()):
            continue
        cv2.line(img, (x, y), (x + int(rng.integers(-12, 13)), y + int(rng.integers(-12, 13))), (235, 235, 235), 2)
    for name, (x, y) in PARTS.items():
        draw_part(img, x, y, ppm, present=name not in missing, marking=marking,
                  dx_mm=(shifted or {}).get(name, 0.0))
    return img


def boxes(margin_mm=0.4):
    return [Component(n, (x - margin_mm, y - margin_mm, x + 3.2 + margin_mm, y + 1.6 + margin_mm), "operator", True)
            for n, (x, y) in PARTS.items()]


def cmap(components=None):
    return ComponentMap(components or boxes(), {"image_id": "ref", "file": "ref.png", "sha256": "0" * 64}, (116.0, 83.0))


def run(test, reference=None, usable=None, gallery=(), **params):
    reference = board() if reference is None else reference
    usable = np.ones(reference.shape[:2], bool) if usable is None else usable
    p = ComponentCheckParams(**params)
    return {r.refdes: r for r in check_components(reference, test, usable, cmap(), PPM, p,
                                                   [similarity_image(g, p, PPM) for g in gallery])}


# --- component map -------------------------------------------------------------

def test_component_map_roundtrip_and_validation(tmp_path):
    path = tmp_path / "components.yaml"
    m = cmap(boxes() + [Component("SW2", (90, 60, 100, 70), "assistant_draft", False, "draft")])
    save_component_map(m, path, "# header\n")
    loaded = load_component_map(path)
    assert path.read_text(encoding="utf-8").startswith("# header\n")
    assert [c.refdes for c in loaded.components] == ["C3", "LED4", "R1", "R2", "R5", "SW2", "U6"]   # sorted
    assert loaded.get("SW2").box_mm == (90.0, 60.0, 100.0, 70.0) and loaded.unverified == ["SW2"]
    assert loaded.reference == m.reference and loaded.size_mm == (116.0, 83.0)
    bad = [Component("R1", (1, 1, 2, 2), "operator", True), Component("R1", (1, 1, 2, 2), "operator", True),
           Component("r 7", (1, 1, 2, 2), "operator", True), Component("C9", (5, 5, 4, 6), "operator", True),
           Component("C10", (110, 80, 117, 82), "operator", True), Component("D1", (1, 1, 2, 2), "guess", True)]
    issues = "\n".join(validate_components(bad, (116, 83)))
    for text in ("R1: defined twice", "'r 7'", "C9: box", "C10: box", "source 'guess'"):
        assert text in issues
    with pytest.raises(ValueError):
        save_component_map(cmap(bad), tmp_path / "bad.yaml")


def test_draft_map_in_the_repository_is_valid_and_marked_unverified():
    m = load_component_map(ROOT / "experiments" / "experiment_002" / "components.yaml")
    assert len(m.components) >= 50 and {"R37", "LED4", "U2"} <= {c.refdes for c in m.components}
    drafts = [c for c in m.components if c.source == "assistant_draft"]
    assert all(not c.verified for c in drafts)     # a draft is never silently marked as verified


# --- per-component decision ------------------------------------------------------------

def test_identical_board_all_present():
    results = run(board())
    assert {r.status for r in results.values()} == {PRESENT}
    assert all(r.similarity > 0.99 and r.score < 0.01 for r in results.values())


def test_missing_part_is_named():
    results = run(board(missing=("C3",)))
    assert [n for n, r in results.items() if r.status == MISSING] == ["C3"]
    assert results["C3"].score > 0.3 and results["C3"].best_example == 0


def test_small_misalignment_and_lighting_are_tolerated():
    shifted = np.roll(board(), (2, 2), axis=(0, 1))                    # 0.25 mm
    darker = cv2.convertScaleAbs(board(), alpha=0.8, beta=10)
    assert all(r.status == PRESENT for r in run(shifted).values())
    assert all(r.status == PRESENT for r in run(darker).values())


def test_marking_text_differences_are_blurred_away():
    other_lot = board(marking=7)                                        # same parts, different fine markings
    assert all(r.status == PRESENT for r in run(other_lot, reference=board(marking=3)).values())


def test_gallery_covers_placement_variation_of_the_reference_board():
    reference = board(shifted={"R5": 0.9})                              # reference board: R5 placed off-centre
    test = board()                                                      # other boards: R5 centred
    alone = run(test, reference=reference)
    assert alone["R5"].status == MISSING                                # one golden board: false alarm
    with_gallery = run(test, reference=reference, gallery=[board(seed=2)])
    assert with_gallery["R5"].status == PRESENT and with_gallery["R5"].best_example == 1
    still_missing = run(board(missing=("R5",)), reference=reference, gallery=[board(seed=2)])
    assert still_missing["R5"].status == MISSING                        # but a missing part stays missing


def test_unusable_box_is_not_checked():
    usable = np.ones((83 * PPM, 116 * PPM), bool)
    x, y = PARTS["LED4"]
    usable[(y - 1) * PPM:(y + 3) * PPM, (x - 1) * PPM:(x + 5) * PPM] = False
    results = run(board(missing=("LED4",)), usable=usable)
    assert results["LED4"].status == NOT_CHECKED and "comparable" in results["LED4"].reason


def test_optional_bare_board_condition():
    test = board(missing=("C3",))
    assert run(test, min_bare_board_increase=0.99)["C3"].status == PRESENT   # second condition not met
    assert run(test, min_bare_board_increase=None)["C3"].status == MISSING


# --- status and report -------------------------------------------------------------

def test_overall_status_and_report():
    m = cmap(boxes() + [Component("D7", (90, 10, 95, 14), "assistant_draft", False)])
    p = ComponentCheckParams()
    results = check_components(board(), board(missing=("R2",)), np.ones((83 * PPM, 116 * PPM), bool), m, PPM, p)
    assert overall_status(results, aligned=True) == FAIL
    report = format_report("x.png", FAIL, results, m, "OK (100 inliers)", "ref", p, 3)
    assert "Status: FAIL" in report and "1. R2" in report and "R2: 0." in report
    assert "UNVERIFIED draft" in report and "Not a production verdict" in report
    ok = check_components(board(), board(), np.ones((83 * PPM, 116 * PPM), bool), cmap(), PPM, p)
    assert overall_status(ok, aligned=True) == PASS
    assert "All expected components detected (6 components" in format_report("x.png", PASS, ok, cmap(), "OK", "ref", p, 1)
    assert overall_status(ok, aligned=False) == REVIEW
    assert overall_status([], aligned=True) == REVIEW
    ok[0].status = NOT_CHECKED
    assert overall_status(ok, aligned=True) == REVIEW


def test_snap_box_to_the_part():
    clean = np.full((83 * PPM, 116 * PPM, 3), RED, np.uint8)
    x, y = PARTS["R2"]
    draw_part(clean, x, y)
    parts = part_mask(clean, PPM)
    rough = (round((x - 2) * PPM), round((y - 2) * PPM), round((x + 5) * PPM), round((y + 3.5) * PPM))
    snapped = snap_box(parts, rough, pad_px=0)
    assert abs(snapped[0] - x * PPM) <= 2 and abs(snapped[2] - (x + 3.2) * PPM) <= 2   # 3.2 mm = 25.6 px, drawn as 26
    assert snap_box(parts, (5, 600, 15, 610)) is None                    # nothing but board there


# --- ignored regions ---------------------------------------------------------------

def test_static_overlay_mask_finds_only_the_overlay():
    text = np.zeros((60, 200), np.uint8)
    cv2.putText(text, "app", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    frames = []
    for i in range(6):   # the scene changes between frames; the overlay stays
        img = np.random.default_rng(i).integers(0, 200, (60, 200, 3)).astype(np.uint8)
        img[text > 0] = 250
        frames.append(img)
    mask = static_overlay_mask(frames, (0, 0, 200, 60), min_level=240, dilate_px=0)
    assert (mask[text > 0] > 0).all() and not (mask[text == 0] > 0).any()
    assert (static_overlay_mask(frames, (0, 0, 200, 60), min_level=240, dilate_px=2) > 0).sum() > (text > 0).sum()


def test_ignore_mask_applies_only_to_matching_image_size():
    patch = np.zeros((10, 20), np.uint8)
    patch[2:5, 3:9] = 255
    regions = [IgnoreRegion((100, 50, 120, 60), (400, 300), "watermark", patch), IgnoreRegion((0, 0, 5, 5), None, "corner")]
    m = ignore_mask((300, 400), regions)
    assert m[52:55, 103:109].min() == 255 and m[50:60, 100:120].sum() == 255 * 18 and m[:5, :5].min() == 255
    assert ignore_mask((200, 400), regions)[50:60, 100:120].max() == 0   # other resolution: not applied


# --- end to end: real alignment on generated scenes ---------------------------------------------

def scene(content, angle=0.0, scale=1.0, shift=(0, 0)):
    """Place the board content on a grey background, rotated / scaled about the image centre."""
    out = np.full((900, 1200, 3), 128, np.uint8)
    h, w = content.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (600 - w / 2 + shift[0], 450 - h / 2 + shift[1])
    cv2.warpAffine(content, m, (1200, 900), dst=out, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_LINEAR)
    return out


def add_session(root: Path, dataset_type: str, board_id: str | None, session_id: str, images: dict[str, np.ndarray]):
    """A minimal acquisition session (session.json + curated copies) for generated test images."""
    folder = board_id or "UNASSIGNED"
    curated = root / "data" / "curated" / dataset_type / folder / session_id
    curated.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (image_id, img) in enumerate(images.items(), 1):
        cv2.imwrite(str(curated / f"{image_id}.png"), img)
        records.append({"image_id": image_id, "session_id": session_id, "dataset_type": dataset_type,
                        "capture_id": f"C{n:03d}", "stage": "after_tht", "camera": "test", "width": img.shape[1],
                        "height": img.shape[0], "captured_at": "2026-10-05T10:00:00",
                        "saved_file": f"curated/{dataset_type}/{folder}/{session_id}/{image_id}.png",
                        "review": {"status": "kept"}})
    acq = root / "data" / "acquisition" / dataset_type / folder / session_id
    acq.mkdir(parents=True, exist_ok=True)
    (acq / "session.json").write_text(json.dumps({"session_id": session_id, "dataset_type": dataset_type,
                                                  "board_id": board_id, "stage": "after_tht", "lighting": "lab",
                                                  "resolution": [1200, 900], "images": records}), encoding="utf-8")


def write_registry(root: Path, boards: list[tuple[str, str, str, str]]):
    folder = root / "data" / "dataset"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["board_id,condition,split,defects,defect_source,identity_source,verified_by,notes"]
    lines += [f"{b},{c},{s},{d},test,acquisition_board_id,test," for b, c, s, d in boards]
    (folder / "boards.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    add_session(tmp_path, "good", "GOOD_001", "S20261005-01", {"ref": scene(board())})
    add_session(tmp_path, "good", "GOOD_002", "S20261005-02", {"good2": scene(board(marking=2), angle=2.0, shift=(15, -10))})
    write_registry(tmp_path, [("GOOD_001", "good", "reference", ""), ("GOOD_002", "good", "reference", "")])
    ref_file = "data/curated/good/GOOD_001/S20261005-01/ref.png"
    cfg = yaml.safe_load((ROOT / "experiments" / "experiment_002" / "config.yaml").read_text(encoding="utf-8"))
    cfg.update(reference_image=ref_file, component_map="components.yaml", cache_dir="cache", input_dir="in",
               output_dir="out", canonical_px_per_mm=PPM, ignore_source_regions_px=[], watermark={"enabled": False})
    cfg["reference_set"].update(calibration="calibration.json")
    cfg["registration"]["min_inliers"] = 20
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    m = cmap()
    m.reference = {"image_id": "ref", "file": ref_file, "sha256": sha256_of(tmp_path / ref_file)}
    save_component_map(m, tmp_path / "components.yaml")
    return tmp_path


def test_end_to_end_inspection_names_the_missing_part(project):
    cfg = load_inspection_config(project, project / "config.yaml")
    reference, m = load_reference_and_map(cfg, project / "refout")
    assert reference.board_id == "GOOD_001"
    gallery = load_gallery(cfg, reference)
    assert gallery.names == ["ref", "good2"] and gallery.boards == ["GOOD_001", "GOOD_002"] and gallery.skipped == []
    assert list((project / "cache" / "gallery" / "ref").glob("*/good2.png"))      # derived data, cached
    good_view = project / "good_view.png"
    cv2.imwrite(str(good_view), scene(board(), angle=-3.0, scale=0.95, shift=(-20, 25)))
    fault_view = project / "fault_view.png"
    cv2.imwrite(str(fault_view), scene(board(missing=("R5",)), angle=-3.0, scale=0.95, shift=(-20, 25)))
    good = inspect_image(good_view, reference, m, cfg, gallery)
    assert good.status == PASS, good.report
    fault = inspect_image(fault_view, reference, m, cfg, gallery)
    assert fault.status == FAIL and [r.refdes for r in fault.components if r.status == MISSING] == ["R5"]
    assert "MISSING" not in good.report and "1. R5" in fault.report
    assert load_gallery(cfg, reference).names == gallery.names                      # second load from the cache


def test_map_drawn_on_another_reference_is_refused(project):
    cfg = load_inspection_config(project, project / "config.yaml")
    m = load_component_map(project / "components.yaml")
    m.reference["sha256"] = "f" * 64
    save_component_map(m, project / "components.yaml")
    with pytest.raises(ValueError, match="drawn on another reference image"):
        load_reference_and_map(cfg, project / "refout")


def test_reference_must_be_curated_good(project):
    cfg = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    cfg["reference_image"] = "data/curated/test/TEST_001/S4/test.png"
    (project / "bad.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="curated GOOD"):
        load_inspection_config(project, project / "bad.yaml")
