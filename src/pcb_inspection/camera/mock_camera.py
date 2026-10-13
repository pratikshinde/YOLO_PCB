"""Test / replay cameras. Never used as a silent substitute for a failed live camera.

MockCamera   serves given BGR frames in order (tests). Every frame it returns is marked
             camera_id "mock:<name>" so a stored capture can never pass for a real camera image.
ReplayCamera serves image files from disk in order (re-inspecting recorded images through the station);
             camera_id "replay:<folder>", and the source file is recorded in get_info().
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..comparison.pipeline import load_bgr
from .base import Camera, CapturedFrame


class MockCamera(Camera):
    def __init__(self, frames: Iterable[np.ndarray], name: str = "test", fail_open: bool = False):
        self.frames = list(frames)
        self.camera_id = f"mock:{name}"
        self.fail_open = fail_open
        self.position = 0
        self.opened = False
        self.closed = False

    def open(self) -> bool:
        self.opened = not self.fail_open
        return self.opened

    def read(self) -> np.ndarray | None:
        if not self.opened or not self.frames:
            return None
        return self.frames[min(self.position, len(self.frames) - 1)]

    def capture(self) -> CapturedFrame | None:
        if not self.opened or self.position >= len(self.frames):
            return None
        image = self.frames[self.position]
        self.position += 1
        return CapturedFrame(image, datetime.now(), self.camera_id, self.get_info(), self.position)

    def get_info(self) -> dict[str, Any]:
        shape = self.frames[0].shape if self.frames else (0, 0)
        return {"camera_id": self.camera_id, "name": self.camera_id, "kind": "mock (test frames)",
                "width": int(shape[1]), "height": int(shape[0])}

    def close(self) -> None:
        self.opened, self.closed = False, True


class ReplayCamera(Camera):
    def __init__(self, files: Iterable[Path]):
        self.files = [Path(f) for f in files]
        self.camera_id = "replay:" + (self.files[0].parent.name if self.files else "-")
        self.position = 0
        self.opened = False
        self._last: np.ndarray | None = None

    def open(self) -> bool:
        self.opened = bool(self.files)
        return self.opened

    def read(self) -> np.ndarray | None:
        return self._last

    def capture(self) -> CapturedFrame | None:
        if not self.opened or self.position >= len(self.files):
            return None
        path = self.files[self.position]
        self.position += 1
        self._last = load_bgr(path)
        return CapturedFrame(self._last, datetime.now(), self.camera_id, {**self.get_info(), "source_file": str(path)},
                             self.position)

    def get_info(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "name": self.camera_id, "kind": "replay of recorded image files",
                "width": None if self._last is None else int(self._last.shape[1]),
                "height": None if self._last is None else int(self._last.shape[0])}

    def close(self) -> None:
        self.opened = False
