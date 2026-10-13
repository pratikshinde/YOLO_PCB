"""Automatic capture trigger for the live station: no button, no operator input.

    WAITING  no PCB in view
    SETTLING a PCB is in view; waiting until the picture has been still for settle_s
    CAPTURE  -> one capture is taken (returned once by update())
    DONE     this board was captured; waiting until it is removed (NO PCB for removed_after_s) or replaced
             (the picture changes a lot and settles again)

Presence comes from the caller (board-colour detection on a preview frame); stillness from consecutive
grey previews. Thresholds: configs/inspection.yaml station.*.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WAITING, SETTLING, CAPTURE, DONE = "WAITING", "SETTLING", "CAPTURE", "DONE"


def preview_signature(image: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


@dataclass
class TriggerSettings:
    settle_s: float = 1.5
    still_max_mean_abs_diff: float = 6.0
    removed_after_s: float = 1.0

    @classmethod
    def from_mapping(cls, m: dict) -> TriggerSettings:
        return cls(float(m["settle_s"]), float(m["still_max_mean_abs_diff"]), float(m["removed_after_s"]))


class AutoTrigger:
    def __init__(self, settings: TriggerSettings):
        self.s = settings
        self.state = WAITING
        self._last: np.ndarray | None = None
        self._still_since: float | None = None
        self._absent_since: float | None = None
        self._captured: np.ndarray | None = None
        self._manual = False

    def update(self, t: float, pcb_present: bool, image: np.ndarray | None) -> bool:
        """Feed one preview observation; True exactly when a capture should be taken now."""
        sig = None if image is None else preview_signature(image)
        moved = True
        if sig is not None and self._last is not None and sig.shape == self._last.shape:
            moved = float(np.mean(np.abs(sig - self._last))) > self.s.still_max_mean_abs_diff
        self._last = sig
        if self._manual:
            self._manual, self.state, self._captured = False, DONE, sig
            return True
        if not pcb_present:
            if self.state == DONE:
                self._absent_since = self._absent_since if self._absent_since is not None else t
                if t - self._absent_since >= self.s.removed_after_s:
                    self._reset()
            else:
                self._reset()
            return False
        self._absent_since = None
        if self.state == DONE:
            if sig is not None and self._captured is not None and sig.shape == self._captured.shape and \
                    float(np.mean(np.abs(sig - self._captured))) > 4 * self.s.still_max_mean_abs_diff:
                self.state, self._still_since = SETTLING, None      # a different board / placement: capture again
            else:
                return False
        if moved or self._still_since is None:
            self.state = SETTLING
            self._still_since = t if not moved else None
            if self._still_since is None:
                return False
        if t - self._still_since >= self.s.settle_s:
            self.state = DONE
            self._captured = sig
            return True
        return False

    def manual(self) -> None:
        """Operator asked for a capture now (SPACE): the next update captures, whatever is in view."""
        self._manual = True

    def _reset(self) -> None:
        self.state, self._still_since, self._absent_since, self._captured = WAITING, None, None, None
