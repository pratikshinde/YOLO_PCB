"""Camera diagnostic: open -> read frames -> print frame size -> release, exactly as the application does.

    python scripts/check_camera.py               camera.index / backend / resolution from the capture config
    python scripts/check_camera.py --index 1     another camera
    python scripts/check_camera.py --list        list the video devices (DirectShow order = OpenCV index)

Uses cv2.VideoCapture(index, cv2.CAP_DSHOW) on Windows. Checks cap.isOpened(),
the return value of every cap.read(), the frame size, that the frames show a
real picture (a covered / switched-off laptop camera still "reads" uniform
blank frames), and that the camera is released: after release() it must
report closed and be openable again.
Close the acquisition application first; a webcam can usually be opened by
only one program at a time. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.acquisition.camera import BACKENDS, check_camera, device_at, list_video_devices  # noqa: E402
from pcb_inspection.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a camera, read frames and release it.")
    parser.add_argument("--capture-config", type=Path, default=ROOT / "configs" / "capture" / "droidcam_e1b.yaml")
    parser.add_argument("--index", type=int, help="camera index (default: camera.index in the capture config)")
    parser.add_argument("--backend", choices=BACKENDS, help="default: camera.backend in the capture config")
    parser.add_argument("--width", type=int, help="requested width (default: from the capture config)")
    parser.add_argument("--height", type=int, help="requested height (default: from the capture config)")
    parser.add_argument("--frames", type=int, default=30, help="frames to read (default 30)")
    parser.add_argument("--list", action="store_true", help="list the video devices and exit")
    args = parser.parse_args()

    cam = load_config(args.capture_config)["camera"]
    index = cam["index"] if args.index is None else args.index
    backend = args.backend or cam["backend"]
    width = args.width or cam.get("requested_width")
    height = args.height or cam.get("requested_height")
    if index < 0 or args.frames < 1:
        parser.error("--index must be >= 0 and --frames >= 1")

    devices = list_video_devices()
    print("Video devices (DirectShow order = OpenCV index):")
    for d in devices:
        print(f"  {d.index}: {d.name}{'   <- selected' if d.index == index else ''}")
    if not devices:
        print("  none found")
    if args.list:
        return 0
    device = device_at(devices, index)
    if devices and device is None:
        print(f"\nFAILED: camera index {index} does not exist. Set camera.index in {args.capture_config.name} "
              "to one of the indices above, or pass --index.")
        return 1
    name = device.name if device else f"camera {index}"

    print(f"\nOpening camera {index} ('{name}') with cv2.VideoCapture({index}, CAP_{backend.upper()}), "
          f"requested {width} x {height}, reading {args.frames} frames...")
    result = check_camera(index, name, backend, width, height, cam.get("requested_fps"), frames=args.frames)
    print(f"  opened (cap.isOpened()) : {'yes' if result.opened else 'no'}")
    if result.opened:
        print(f"  frames read (cap.read()): {result.frames_read} / {result.frames_requested}")
        if result.width:
            print(f"  frame size              : {result.width} x {result.height}, {result.channels} channel(s)"
                  f"  (shape = ({result.height}, {result.width}, {result.channels}))")
        print(f"  measured FPS            : {result.fps_measured if result.fps_measured else 'n/a'}")
        print(f"  blank (uniform) frames  : {result.blank_frames} / {result.frames_read}"
              f"{'   <- no real picture' if result.frames_read and result.blank_frames == result.frames_read else ''}")
    print(f"  released (cap.release()): {'yes' if result.released else 'NO'}")
    if result.opened and result.released:
        # proves the device was really freed: it can be opened and read again
        again = check_camera(index, name, backend, width, height, cam.get("requested_fps"), frames=1,
                             check_picture=False)
        print(f"  re-opened after release : {'yes' if again.ok else 'NO: ' + again.error}")
        if not again.ok and not result.error:
            result.error = "the camera could not be opened again after release"
    if result.error:
        print(f"\nFAILED: {result.error}")
        if "droidcam" in name.lower():
            print("DroidCam: open the DroidCam app on the phone (camera running), then press Start in the DroidCam PC "
                  "client so its window shows the live picture; then run this check again.")
        return 1
    print(f"\nOK: camera {index} ('{name}') opens, delivers {result.width} x {result.height} frames with a picture "
          "and is released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
