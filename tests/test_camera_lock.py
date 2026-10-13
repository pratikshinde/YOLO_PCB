"""One program at a time per camera (acquisition.camera.CameraLock), and the station's background live loop."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from pcb_inspection.acquisition.camera import CameraBusyError, CameraLock, OpenCVCamera

ROOT = Path(__file__).resolve().parents[1]
HOLDER = """
import sys, time
sys.path.insert(0, {src!r})
from pcb_inspection.acquisition.camera import CameraLock
lock = CameraLock("dshow", 7)
lock.acquire()
print("held", flush=True)
sys.stdin.readline()
"""


def test_second_program_is_refused_and_told_who_has_the_camera():
    holder = subprocess.Popen([sys.executable, "-c", HOLDER.format(src=str(ROOT / "src")), "station.py"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(CameraBusyError, match=rf"camera 7 is in use by .*pid {holder.pid}"):
            CameraLock("dshow", 7).acquire()
        cam = OpenCVCamera(7, "DroidCam Video")
        assert cam.open() is False and "in use by" in cam.last_error      # refused before touching the device
        assert CameraLock("dshow", 8).acquire() is None                      # other cameras are not affected
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=20)
    lock = CameraLock("dshow", 7)
    lock.acquire()                                     # the OS released it when the holder ended
    lock.release()


def test_lock_is_shared_inside_one_program_and_released_by_the_last_user():
    a, b = CameraLock("dshow", 9), CameraLock("dshow", 9)
    a.acquire()
    b.acquire()                                        # same program: allowed (objects of one program)
    a.release()
    check = subprocess.run([sys.executable, "-c", (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); from pcb_inspection.acquisition.camera import "
        "CameraLock, CameraBusyError\ntry:\n    CameraLock('dshow', 9).acquire(); print('free')\n"
        "except CameraBusyError:\n    print('busy')")], capture_output=True, text=True, timeout=30)
    assert check.stdout.strip() == "busy"             # still held by b
    b.release()
    check = subprocess.run(check.args, capture_output=True, text=True, timeout=30)
    assert check.stdout.strip() == "free"


def test_live_loop_runs_in_the_background_and_captures_once(tmp_path):
    from test_component_inspection import board, scene
    from test_product_architecture import build_project

    sys.path.insert(0, str(ROOT / "scripts"))
    from station import LiveStation

    from pcb_inspection.camera import MockCamera
    from pcb_inspection.preprocessing.board import BoardSegmentationParams
    from pcb_inspection.preprocessing.presence import PresenceParams
    from pcb_inspection.station import AutoTrigger, InspectionStation, TriggerSettings

    st = InspectionStation(build_project(tmp_path))
    good = scene(board(marking=5))
    results = []
    live = LiveStation(st, MockCamera([good] * 3), "AUTO", AutoTrigger(TriggerSettings(0.3, 6.0, 1.0)),
                       PresenceParams(), BoardSegmentationParams(), 0.05, on_result=lambda r, m: results.append(r))
    live.camera.open()
    live.start_loop(every_s=0.02)
    deadline = time.monotonic() + 60
    while not results and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)                                    # same board still in view: no second capture
    live.close()
    assert len(results) == 1 and results[0].decision == "GOOD"
    assert st.captures.session(live.session_id)["ended_at"] is not None
