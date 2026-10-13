"""A live OpenCV video device (e.g. the phone through DroidCam) behind the Camera interface.

Wraps the existing acquisition.camera.OpenCVCamera + FrameGrabber (reader thread, blank / frozen / disconnect
detection, reconnect). No fallback: if the device delivers no frame, capture() returns None and the caller
reports it. A camera failure never produces a substitute image.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..acquisition.camera import FrameGrabber, OpenCVCamera
from .base import Camera, CapturedFrame


class LiveCamera(Camera):
    def __init__(self, index: int, name: str = "", width: int | None = None, height: int | None = None,
                 fps: float | None = 30, backend: str = "dshow", fresh_frame_timeout_s: float = 2.0):
        self.device = OpenCVCamera(index, name or f"camera {index}", width, height, fps, backend=backend)
        self.grabber = FrameGrabber(self.device)
        self.camera_id = f"{backend}:{index}:{self.device.name}"
        self.fresh_frame_timeout_s = fresh_frame_timeout_s
        self._started = False

    def open(self, wait_s: float = 8.0) -> bool:
        if not self._started:
            self.grabber.start()
            self._started = True
        deadline = time.monotonic() + wait_s
        while self.grabber.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.grabber.latest() is not None

    @property
    def is_live(self) -> bool:
        return self.grabber.is_live

    @property
    def message(self) -> str:
        return self.grabber.message or self.grabber.status

    def read(self) -> np.ndarray | None:
        frame = self.grabber.latest()
        return None if frame is None else frame.image

    def capture(self) -> CapturedFrame | None:
        shown = self.grabber.latest()
        after = shown.t if shown is not None else float("-inf")
        frame = self.grabber.wait_for_frame_after(after, self.fresh_frame_timeout_s)
        if frame is None:
            return None
        return CapturedFrame(frame.image, frame.wall_time, self.camera_id, self.get_info(), frame.seq)

    def get_info(self) -> dict[str, Any]:
        info = dict(self.grabber.info())
        info.setdefault("camera_id", self.camera_id)
        return info

    def close(self) -> None:
        if self._started:
            self.grabber.stop()
            self._started = False
