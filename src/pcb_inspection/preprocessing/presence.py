"""Is the frame usable, and is there a PCB in it? (Stage before alignment; "camera opened" is not "PCB present".)

Reuses the existing board segmentation (preprocessing.board.board_mask: dominant saturated hue near the image
centre, largest region of that colour) and the blank-frame test of the camera layer. Deliberately coarse: it
only decides whether a board-coloured object of plausible size is in view. Whether it is THE board, whole,
upright and alignable is decided afterwards from the feature alignment (alignment.validation), which is far
more reliable than the colour outline (a second red board in the background merges with the outline).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..acquisition.camera import is_blank
from .board import BoardSegmentationParams, board_mask

VALID, INVALID_FRAME = "VALID", "INVALID_FRAME"
DETECTED, NO_PCB = "DETECTED", "NO_PCB"


@dataclass(frozen=True)
class PresenceParams:
    min_mean_brightness: float = 10.0       # darker = camera covered / no light
    min_board_area_fraction: float = 0.10   # largest board-coloured region / frame area


@dataclass
class FrameCheck:
    status: str                             # VALID | INVALID_FRAME
    reason: str = ""
    mean_brightness: float = 0.0


@dataclass
class Presence:
    status: str                             # DETECTED | NO_PCB
    area_fraction: float
    mask: np.ndarray                        # board-colour region (0/255), full frame size
    reason: str = ""


def check_frame(bgr: np.ndarray | None, params: PresenceParams = PresenceParams()) -> FrameCheck:
    if bgr is None or bgr.size == 0:
        return FrameCheck(INVALID_FRAME, "no image")
    if is_blank(bgr):
        return FrameCheck(INVALID_FRAME, "uniform (blank) frame: camera covered or switched off")
    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) if bgr.ndim == 3 else float(bgr.mean())
    if mean < params.min_mean_brightness:
        return FrameCheck(INVALID_FRAME, f"frame too dark (mean brightness {mean:.1f} < {params.min_mean_brightness:g})", mean)
    return FrameCheck(VALID, "", mean)


def detect_pcb(bgr: np.ndarray, params: PresenceParams = PresenceParams(),
               segmentation: BoardSegmentationParams = BoardSegmentationParams()) -> Presence:
    mask = board_mask(bgr, segmentation)
    fraction = float((mask > 0).mean())
    if fraction < params.min_board_area_fraction:
        return Presence(NO_PCB, fraction, mask,
                        f"no board-coloured object of board size in view (largest region {fraction:.1%} of the frame, "
                        f"needs {params.min_board_area_fraction:.0%})")
    return Presence(DETECTED, fraction, mask)
