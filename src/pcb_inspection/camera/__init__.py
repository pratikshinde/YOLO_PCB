"""Replaceable camera layer: open / read / capture / get_info / close. Knows nothing about products or results."""

from .base import Camera, CapturedFrame
from .mock_camera import MockCamera, ReplayCamera
from .opencv_camera import LiveCamera

__all__ = ["Camera", "CapturedFrame", "LiveCamera", "MockCamera", "ReplayCamera"]
