"""Camera layer: independent of products and of inspection results.

A camera delivers frames and says what it is; it knows nothing about PCBs, products, GOOD or DEFECT. Any
implementation (phone via DroidCam, industrial camera, a recorded image set) offers the same five methods:

    open() -> bool          read() -> frame | None (newest preview frame)
    capture() -> CapturedFrame | None (one fresh frame taken now, with timestamp and camera identity)
    get_info() -> dict      close()

Replacing the camera never changes the inspection code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray                 # BGR, exactly as delivered
    captured_at: datetime             # local wall-clock time of the frame
    camera_id: str                    # stable identity of the camera (e.g. "dshow:1:DroidCam Source 3")
    camera: dict[str, Any] = field(default_factory=dict)   # get_info() at capture time
    frame_seq: int | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (int(self.image.shape[1]), int(self.image.shape[0]))


class Camera(ABC):
    camera_id: str = ""

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """The newest frame for preview (may be the same frame twice); None if there is none."""

    @abstractmethod
    def capture(self) -> CapturedFrame | None:
        """A frame taken after this call (never one already shown), or None if the camera delivers none."""

    @abstractmethod
    def get_info(self) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
