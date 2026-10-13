"""Camera layer with a mock camera / mocked cv2.VideoCapture: no physical camera is needed."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from pcb_inspection.acquisition.camera import (
    STATUS_BLANK,
    STATUS_DISCONNECTED,
    STATUS_FROZEN,
    STATUS_LIVE,
    FrameGrabber,
    OpenCVCamera,
    VideoDevice,
    check_camera,
    device_at,
    encode_png,
    frame_signature,
    is_blank,
    list_video_devices,
)
from pcb_inspection.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeCamera:
    """Scripted camera: open() results and read() frames (None = read failure) are consumed in order."""

    name, index, backend = "DroidCam Video", 1, "dshow"

    def __init__(self, frames=None, opens=None):
        self.frames = list(frames or [])
        self.opens = list(opens or [])
        self.open_calls = self.release_calls = 0
        self.counter = 0

    def open(self) -> bool:
        self.open_calls += 1
        return self.opens.pop(0) if self.opens else True

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        self.counter += 1           # default: a live stream, every frame different
        return frame(self.counter)

    def release(self) -> None:
        self.release_calls += 1

    def properties(self):
        return {"fps_reported": 30.0}


def frame(value: int, shape=(36, 64, 3)) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    img[::8, ::8] = value % 255 + 1   # visible to frame_signature; never a uniform (blank) frame
    return img


def grabber(camera, clock, **kw) -> FrameGrabber:
    return FrameGrabber(camera, clock=clock, failures_before_disconnect=kw.pop("failures", 3),
                        frozen_after_s=kw.pop("frozen", 3.0), reconnect_every_s=kw.pop("reconnect", 2.0), **kw)


def test_list_video_devices_by_name_or_probe():
    names = list_video_devices(names_provider=lambda: ["Integrated Camera", "DroidCam Video"])
    assert names == [VideoDevice(0, "Integrated Camera"), VideoDevice(1, "DroidCam Video")]
    probed = list_video_devices(names_provider=lambda: [], probe=lambda i: i in (0, 2), max_probe=4)
    assert probed == [VideoDevice(0, "Camera 0"), VideoDevice(2, "Camera 2")]


def test_live_stream_and_camera_info():
    clock, cam = FakeClock(), FakeCamera()
    g = grabber(cam, clock)
    for i in range(5):
        clock.t = i / 30
        assert g.step()
    assert g.status == STATUS_LIVE and g.is_live and g.latest().seq == 5
    info = g.info()
    assert (info["name"], info["width"], info["height"], info["fps_reported"]) == ("DroidCam Video", 64, 36, 30.0)
    assert info["fps_measured"] == 30.0 and info["backend"] == "dshow"


def test_disconnect_is_detected_and_recovered():
    clock = FakeClock()
    cam = FakeCamera(frames=[frame(1), None, None, None], opens=[True, False])
    g = grabber(cam, clock)
    assert g.step() and g.is_live
    for _ in range(3):
        assert not g.step()
    assert g.status == STATUS_DISCONNECTED and "disconnected" in g.message.lower()
    assert cam.release_calls == 1
    clock.t = 1.0
    assert not g.step() and cam.open_calls == 1            # waits before re-opening
    clock.t = 2.0
    assert not g.step() and cam.open_calls == 2            # re-open attempt fails
    assert g.status == STATUS_DISCONNECTED
    clock.t = 4.0
    assert g.step() and g.is_live and cam.open_calls == 3  # recovered


def test_camera_that_cannot_be_opened_is_reported():
    cam = FakeCamera(opens=[False])
    g = grabber(cam, FakeClock())
    assert not g.step()
    assert g.status == STATUS_DISCONNECTED and g.latest() is None


def test_frozen_stream_is_not_live():
    clock = FakeClock()
    cam = FakeCamera(frames=[frame(7)] * 5)
    g = grabber(cam, clock)
    for t in (0.0, 1.0, 2.0):
        clock.t = t
        g.step()
    assert g.is_live
    clock.t = 3.5
    g.step()
    assert g.status == STATUS_FROZEN and not g.is_live
    clock.t = 3.6
    g.step()                       # still the same picture
    assert g.status == STATUS_FROZEN
    clock.t = 3.7
    g.step()                       # FakeCamera now delivers changing frames again
    assert g.is_live


def test_frozen_detection_can_be_switched_off():
    clock = FakeClock()
    g = grabber(FakeCamera(frames=[frame(7)] * 3), clock, frozen=None)
    for t in (0.0, 5.0, 60.0):
        clock.t = t
        g.step()
    assert g.is_live


def test_camera_properties_are_read_once_per_open():
    cam = FakeCamera()
    calls = []
    cam.properties = lambda: calls.append(1) or {"fps_reported": 25.0}
    g = grabber(cam, FakeClock())
    g.step()
    for _ in range(3):
        assert g.info()["fps_reported"] == 25.0
    assert len(calls) == 1


def test_capture_waits_for_a_fresh_frame():
    clock = FakeClock(1.0)
    g = grabber(FakeCamera(), clock)
    g.step()
    assert g.wait_for_frame_after(1.0, timeout=0) is None   # the frame on screen is not new
    clock.t = 1.1
    g.step()
    fresh = g.wait_for_frame_after(1.0, timeout=0)
    assert fresh is not None and fresh.t == 1.1 and fresh.seq == 2


def test_manual_reconnect_reopens_the_camera():
    cam = FakeCamera()
    g = grabber(cam, FakeClock())
    g.step()
    g.request_reconnect()
    assert g.step() and cam.release_calls == 1 and cam.open_calls == 2 and g.is_live


# --- configured index, OpenCV calls, diagnostics (cv2.VideoCapture mocked) ---------------

class FakeCapture:
    """Stands in for cv2.VideoCapture. OPENABLE: indices that open; reads: list of (ok, frame) or an exception."""

    OPENABLE: set[int] = {0}
    instances: list["FakeCapture"] = []

    def __init__(self, index, api):
        self.index, self.api = index, api
        self.released = False
        self.props: dict[int, float] = {}
        self.reads: list | None = None
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.index in self.OPENABLE and not self.released

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, -1.0)

    def read(self):
        if self.reads:
            item = self.reads.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        h = int(self.props.get(cv2.CAP_PROP_FRAME_HEIGHT, 480))
        w = int(self.props.get(cv2.CAP_PROP_FRAME_WIDTH, 640))
        img = np.zeros((h, w, 3), np.uint8)
        img[::8, ::8] = 120                       # a picture, not a uniform blank frame
        self.reads_done = getattr(self, "reads_done", 0) + 1
        img[0, 0] = self.reads_done % 256         # a live stream: every frame differs a little (sensor noise)
        return True, img

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeCapture.instances = []
    FakeCapture.OPENABLE = {0}
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    return FakeCapture


def test_capture_config_names_one_camera_index_with_directshow():
    camera = load_config(ROOT / "configs" / "capture" / "droidcam_e1b.yaml")["camera"]
    assert isinstance(camera["index"], int) and camera["index"] >= 0 and camera["backend"] == "dshow"
    assert "preferred_name_contains" not in camera     # no name preference that overrides the index


def test_device_at_never_substitutes_another_camera():
    devices = [VideoDevice(0, "Integrated Camera"), VideoDevice(1, "DroidCam Video")]
    assert device_at(devices, 0).name == "Integrated Camera"
    assert device_at(devices, 1).name == "DroidCam Video"
    assert device_at(devices, 2) is None and device_at([], 0) is None


def test_opencv_camera_uses_directshow_and_requested_size(fake_cv2):
    cam = OpenCVCamera(0, "Integrated Camera", 1280, 720, 30)
    assert cam.open() and cam.is_open
    cap = fake_cv2.instances[-1]
    assert (cap.index, cap.api) == (0, cv2.CAP_DSHOW)
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert cam.read().shape == (720, 1280, 3)
    assert cam.properties() == {"fps_reported": 30.0}
    assert cam.release() is True and cap.released and not cam.is_open
    assert cam.release() is True                       # releasing twice is harmless


def test_opencv_camera_reopen_releases_the_old_handle(fake_cv2):
    cam = OpenCVCamera(0, "Integrated Camera")
    cam.open()
    cam.open()
    first, second = fake_cv2.instances
    assert first.released and not second.released     # never two open handles for one camera


def test_open_failure_is_explained_and_released(fake_cv2):
    cam = OpenCVCamera(1, "DroidCam Video")
    assert cam.open() is False and not cam.is_open
    assert fake_cv2.instances[-1].released
    assert "Cannot open camera 1 ('DroidCam Video') with backend DSHOW" in cam.last_error
    assert "DroidCam client" in cam.last_error
    other = OpenCVCamera(3, "USB Camera")
    other.open()
    assert "check_camera.py --list" in other.last_error


def test_read_checks_the_return_value(fake_cv2):
    cam = OpenCVCamera(0, "Integrated Camera")
    cam.open()
    fake_cv2.instances[-1].reads = [(False, None), (True, None), (True, np.zeros((0, 0, 3), np.uint8)),
                                    cv2.error("driver error")]
    assert [cam.read() for _ in range(4)] == [None, None, None, None]
    assert cam.read() is not None                      # the stream continues afterwards


def test_invalid_index_is_refused():
    for bad in (-1, "0", 1.0, True):
        with pytest.raises(ValueError):
            OpenCVCamera(bad, "x")


def test_check_camera_reports_size_and_releases(fake_cv2):
    result = check_camera(0, "Integrated Camera", width=640, height=480, frames=5)
    assert result.ok and result.opened and result.frames_read == 5
    assert (result.width, result.height, result.channels) == (640, 480, 3)
    assert result.released and fake_cv2.instances[-1].released


def test_check_camera_open_failure(fake_cv2):
    result = check_camera(1, "DroidCam Video", frames=5)
    assert not result.ok and not result.opened and result.frames_read == 0
    assert "Cannot open camera 1" in result.error and result.released


def test_check_camera_read_failure_still_releases(monkeypatch, fake_cv2):
    class NoFrames(FakeCapture):
        def read(self):
            return False, None
    monkeypatch.setattr(cv2, "VideoCapture", NoFrames)
    result = check_camera(0, "Integrated Camera", frames=5)
    assert result.opened and not result.ok and result.frames_read == 0
    assert "cap.read() returned no frame (frame 1 of 5)" in result.error
    assert result.released and NoFrames.instances[-1].released


def test_blank_frame_detection():
    assert is_blank(np.full((48, 64, 3), 1, np.uint8))               # what the covered laptop camera sent
    assert is_blank(np.zeros((480, 640, 3), np.uint8))
    noisy_dark = np.random.default_rng(0).integers(0, 3, (48, 64, 3)).astype(np.uint8)
    assert not is_blank(noisy_dark)                                   # a dark real picture still has noise
    assert not is_blank(frame(5))


def test_blank_picture_is_not_live_and_explained():
    clock = FakeClock()
    blank = np.full((36, 64, 3), 2, np.uint8)
    g = grabber(FakeCamera(frames=[blank] * 4), clock)
    g.step()
    assert g.status == "connecting"                                  # start-up grace period
    clock.t = 1.0
    g.step()
    assert g.status == STATUS_BLANK and not g.is_live and "privacy shutter" in g.message
    clock.t = 1.1
    g.step()
    clock.t = 1.2
    g.step()
    clock.t = 1.3
    g.step()                                                          # a real picture arrives
    assert g.is_live


def test_check_camera_fails_on_blank_frames(monkeypatch, fake_cv2):
    class Covered(FakeCapture):
        def read(self):
            return True, np.full((480, 640, 3), 1, np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", Covered)
    result = check_camera(0, "Integrated Camera", frames=5)
    assert result.opened and result.frames_read == 5 and result.blank_frames == 5
    assert not result.ok and "blank" in result.error and result.released
    assert check_camera(0, "Integrated Camera", frames=1, check_picture=False).ok   # release / re-open check only


def test_check_camera_reports_a_static_picture(monkeypatch, fake_cv2):
    class StartScreen(FakeCapture):         # e.g. DroidCam's static "Start DroidCam" screen: not uniform, never changes
        def read(self):
            img = np.zeros((48, 64, 3), np.uint8)
            img[20:28, 10:50] = 255
            return True, img
    monkeypatch.setattr(cv2, "VideoCapture", StartScreen)
    result = check_camera(0, "DroidCam Video", frames=10)
    assert result.static and not result.ok and "did not change in 10 frames" in result.error
    assert "Start DroidCam" in result.error and result.released


def test_check_camera_tolerates_blank_frames_while_starting(monkeypatch, fake_cv2):
    class WarmingUp(FakeCapture):
        def read(self):
            self.n = getattr(self, "n", 0) + 1
            return (True, np.zeros((48, 64, 3), np.uint8)) if self.n <= 3 else super().read()
    monkeypatch.setattr(cv2, "VideoCapture", WarmingUp)
    result = check_camera(0, "Integrated Camera", frames=10)
    assert result.ok and result.blank_frames == 3

    class GoesBlank(FakeCapture):
        def read(self):
            self.n = getattr(self, "n", 0) + 1
            return super().read() if self.n <= 3 else (True, np.full((48, 64, 3), 1, np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", GoesBlank)
    result = check_camera(0, "Integrated Camera", frames=10)
    assert not result.ok and "the last 7 of 10 frames are blank" in result.error


def test_grabber_shows_the_open_error():
    class Failing(FakeCamera):
        last_error = "Cannot open camera 1 ('DroidCam Video') with backend DSHOW (cap.isOpened() is False)."
    g = grabber(Failing(opens=[False]), FakeClock())
    g.step()
    assert g.status == STATUS_DISCONNECTED and g.message.startswith("Cannot open camera 1")


def test_grabber_reports_opened_but_no_frames():
    g = grabber(FakeCamera(frames=[None, None, None]), FakeClock(), failures=3)
    for _ in range(3):
        g.step()
    assert g.status == STATUS_DISCONNECTED and "opened but cap.read() returned no frame" in g.message


class CountingCamera(FakeCamera):
    """Records which thread releases it; reads are slow like a real device."""

    def __init__(self):
        super().__init__()
        self.release_threads: list[str] = []

    def read(self):
        time.sleep(0.005)
        return super().read()

    def release(self):
        self.release_threads.append(threading.current_thread().name)
        super().release()


def test_grabber_thread_opens_once_and_releases_on_stop():
    cam = CountingCamera()
    g = FrameGrabber(cam)
    g.start()
    g.start()                                          # a second start does not open the camera again
    deadline = time.monotonic() + 2
    while not g.is_live and time.monotonic() < deadline:
        time.sleep(0.01)
    assert g.is_live and cam.open_calls == 1
    g.stop()
    assert cam.release_threads == ["frame-grabber"]    # released once, by the reading thread
    assert not any(t.name == "frame-grabber" for t in threading.enumerate())


def test_grabber_thread_survives_an_unexpected_error():
    class Exploding(CountingCamera):
        def read(self):
            if self.counter == 2:
                self.counter += 1
                raise RuntimeError("driver hiccup")
            return super().read()
    cam = Exploding()
    g = FrameGrabber(cam, reconnect_every_s=0.05)
    g.start()
    deadline = time.monotonic() + 3
    while cam.open_calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    while not g.is_live and time.monotonic() < deadline:
        time.sleep(0.01)
    g.stop()
    assert cam.open_calls == 2 and cam.release_threads.count("frame-grabber") == 2   # closed, re-opened, released


def test_frame_signature_and_png_encoding():
    a, b = frame(1), frame(2)
    assert frame_signature(a) == frame_signature(a.copy()) != frame_signature(b)
    data = encode_png(a)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
