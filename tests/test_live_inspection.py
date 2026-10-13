"""Live-inspection wrapper (scripts/live_inspection.py) around the unchanged Experiment 2 pipeline.

Uses the generated test boards of test_component_inspection (not PCB images) and a scripted camera;
the real inspection code runs. No physical camera is needed.
"""

import csv
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pcb_inspection.acquisition.camera import FrameGrabber
from pcb_inspection.comparison.board_inspection import load_gallery, load_inspection_config, load_reference_and_map
from test_component_inspection import board, project, scene  # noqa: F401  (project is a fixture)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_inspection as live  # noqa: E402

WHEN = datetime(2026, 10, 5, 13, 45, 1)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def inspector(project):  # noqa: F811
    cfg = load_inspection_config(project, project / "config.yaml")
    reference, cmap = load_reference_and_map(cfg, project / "refout")
    gallery = load_gallery(cfg, reference)
    return live.LiveInspector(cfg, reference, cmap, gallery, project / "live", project / "live_output",
                              lambda: {"name": "test camera", "index": 9})


def summary(inspector):
    with (inspector.output_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- timing ------------------------------------------------------------------------

def test_cadence_one_inspection_per_interval_never_overlapping():
    clock = Clock()
    c = live.Cadence(3.0, clock)
    assert not c.due() and c.seconds_left() == 3.0          # time to place the board
    clock.t += 3.0
    assert c.due()
    c.started()                                             # capture at t = 103
    clock.t += 1.4
    assert not c.due()                                      # busy
    c.finished()
    assert not c.due() and abs(c.seconds_left() - 1.6) < 1e-9   # 3 s from capture start to capture start
    clock.t += 1.6
    assert c.due()
    c.started()
    clock.t += 5.0                                          # an inspection slower than the interval
    assert not c.due()
    c.finished()
    assert c.due()                                          # next one at once, no overlap


def test_cadence_space_and_reset():
    clock = Clock()
    c = live.Cadence(3.0, clock)
    c.now()
    assert c.due()                                          # SPACE: immediately
    c.started()
    c.now()
    assert not c.due()                                      # still busy
    c.finished()
    assert c.due()                                          # ... right after the running inspection
    c.reset()
    assert not c.due() and c.seconds_left() == 3.0
    with pytest.raises(ValueError):
        live.Cadence(0)


# --- capture -> existing inspection --------------------------------------------------------

def test_pass_result_outputs_and_summary(inspector):
    r = inspector.inspect(scene(board(), angle=-2.0, shift=(10, 5)), WHEN)
    assert r.status == "PASS" and r.defects == [] and r.message == ""
    assert r.image.name == "LIVE_20261005_134501.png" and r.image.is_file()
    for name in ("report.txt", "result.json", "annotated.jpg", "capture.json"):
        assert (r.output_dir / name).is_file()
    assert '"timestamp": "2026-10-05T13:45:01.000"' in (r.output_dir / "capture.json").read_text(encoding="utf-8")
    row = summary(inspector)[0]
    assert (row["status"], row["defect_count"], row["defect_components"]) == ("PASS", "0", "")
    assert row["image"].endswith("LIVE_20261005_134501.png") and row["timestamp"] == "2026-10-05T13:45:01.000"
    assert "PCB STATUS: PASS\nAll expected components detected." in live.console_text(r)


def test_fail_names_the_component_and_never_overwrites(inspector):
    inspector.inspect(scene(board()), WHEN)
    r = inspector.inspect(scene(board(missing=("R5",)), angle=1.5), WHEN)          # same second
    assert r.status == "FAIL" and [c for c, _ in r.defects] == ["R5"] and r.score == r.defects[0][1]
    assert r.image.name == "LIVE_20261005_134501_2.png"                              # not overwritten
    assert (r.output_dir / "evidence.jpg").is_file()
    rows = summary(inspector)
    assert [x["status"] for x in rows] == ["PASS", "FAIL"]
    assert (rows[1]["defect_count"], rows[1]["defect_components"]) == ("1", "R5") and float(rows[1]["score"]) > 0.265
    text = live.console_text(r)
    assert "PCB STATUS: FAIL\nDefects detected:\n\n1. R5\n\nScore: 0." in text


def test_errors_never_report_pass(inspector):
    no_board = (np.random.default_rng(0).integers(0, 255, (900, 1200, 3))).astype(np.uint8)   # nothing to align
    results = [inspector.inspect(None, WHEN), inspector.inspect(np.full((900, 1200, 3), 3, np.uint8), WHEN),
               inspector.inspect(no_board, WHEN)]
    assert [r.status for r in results] == ["INSPECTION ERROR"] * 3
    assert "no new frame" in results[0].message and results[0].image is None
    assert "blank" in results[1].message and results[1].image is None                # rejected, not saved
    assert "not aligned" in results[2].message and results[2].image.is_file()        # saved, then not aligned
    assert all("PASS" not in live.console_text(r).replace("PASS / FAIL", "") for r in results)
    assert [x["status"] for x in summary(inspector)] == ["INSPECTION ERROR"] * 3


def test_overlay_texts():
    r = live.LiveResult(WHEN, "FAIL", [("R37", 0.27), ("LED4", 0.30)])
    lines = [t for t, _ in live.overlay_lines("1", 3, "FAIL", r, 1.2)]
    assert lines[:3] == ["PCB INSPECTION", "Camera: 1", "Interval: 3 sec"]
    assert lines[4:9] == ["Status: FAIL", "DEFECT DETECTED", "Components:", "- R37", "- LED4"]
    one = [t for t, _ in live.overlay_lines("1", 3, "FAIL", live.LiveResult(WHEN, "FAIL", [("R37", 0.27)]), 0)]
    assert "Component: R37" in one
    waiting = [t for t, _ in live.overlay_lines("1", 3, "WAITING", None, 2.0, "BLANK PICTURE")]
    assert "CAMERA: BLANK PICTURE" in waiting and "Next inspection in 2.0 s" in waiting
    shown = live.draw_overlay(np.zeros((540, 960, 3), np.uint8), live.overlay_lines("1", 3, "PASS", None, 1))
    assert shown.shape == (540, 960, 3) and shown.any()


# --- the live window with a scripted camera -------------------------------------------------

class ReplayCamera:
    """Scripted camera: plays generated board scenes at ~30 fps; each frame differs slightly (sensor noise)."""

    name, index, backend = "replay", 9, "dshow"

    def __init__(self, frames):
        self.frames, self.i, self.released = frames, 0, 0
        self.opened = 0

    def open(self):
        self.opened += 1
        return True

    def read(self):
        time.sleep(0.03)
        img = self.frames[min(self.i // 40, len(self.frames) - 1)].copy()
        img[0, 0] = self.i % 256
        self.i += 1
        return img

    def release(self):
        self.released += 1
        return True

    def properties(self):
        return {"fps_reported": 30.0}


def test_live_window_interval_keys_and_shutdown(inspector):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    camera = ReplayCamera([scene(board()), scene(board(missing=("R2",)))])
    grabber = FrameGrabber(camera)
    grabber.start()
    results = []
    app = live.LiveApp(root, grabber, inspector, "9", interval=1.0, on_result=results.append)
    ticks, start = 0, time.monotonic()
    while time.monotonic() - start < 6.0 and len(results) < 3:
        root.update()
        ticks += 1
        time.sleep(0.01)
    assert len(results) >= 2, [r.message for r in results]
    assert ticks > 100                                       # the preview loop kept running during inspections
    assert {r.status for r in results} <= {"PASS", "FAIL"}
    times = [r.timestamp for r in results]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    assert all(g >= 0.9 for g in gaps)                       # one inspection per interval, never more often
    before = len(results)
    app.inspect_now()                                        # SPACE
    deadline = time.monotonic() + 4
    while len(results) == before and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert len(results) == before + 1
    app.close()                                              # Q / ESC
    assert camera.released >= 1 and not any(t.name == "frame-grabber" for t in threading.enumerate())
    assert app.closing and len(summary(inspector)) == len(results)
