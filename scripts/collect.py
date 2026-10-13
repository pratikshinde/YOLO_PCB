"""Clean real-data collection: live DroidCam camera -> GOOD PCB / FAULT PCB dataset (docs/clean_data_collection.md).

    python scripts/collect.py                    camera.index from configs/capture/droidcam_collection.yaml (1 = DroidCam)
    python scripts/collect.py --camera-index 0   another camera for this run
    python scripts/collect.py --list-cameras     list the video devices

Workflow per image (no typing per image):
    LIVE CAMERA -> 3 s countdown -> CAPTURE -> SAVE ORIGINAL (write-once, SHA-256) -> PCB crop / canonical PCB image
    -> capture check (complete PCB, inside the ROI, watermark outside the PCB, focus, exposure: OK / REVIEW /
    RECAPTURE) -> operator KEEP / REJECT -> data/collection/good|fault/ or data/collection/quarantine/

Only GOOD and FAULT PCB sessions (TEST later). Every image belongs to a physical board ID (GOOD_001, FAULT_001, ...).
Nothing is deleted: originals stay in data/collection/originals/, REJECT goes to quarantine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.acquisition.session import AcquisitionStore, settings_from_product  # noqa: E402
from pcb_inspection.config import load_config  # noqa: E402

DEFAULT_CAPTURE_CONFIG = ROOT / "configs" / "capture" / "droidcam_collection.yaml"


def build_store(capture: dict, product_id: str, project_root: Path = ROOT) -> AcquisitionStore:
    """The collection store of the product, with session IDs unique across the station capture store."""
    from pcb_inspection.products import load_product_profile
    from pcb_inspection.station.capture_store import used_session_ids
    settings = settings_from_product(load_product_profile(product_id, project_root=project_root), capture)
    return AcquisitionStore(settings, used_session_ids(project_root / "data"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean GOOD / FAULT PCB data collection from the live camera.")
    parser.add_argument("--capture-config", type=Path, default=DEFAULT_CAPTURE_CONFIG)
    parser.add_argument("--product", default=None, help="product id (default: 'product' in the capture config)")
    parser.add_argument("--camera-index", type=int, help="camera for this run (default: camera.index in the config)")
    parser.add_argument("--list-cameras", action="store_true", help="print the video devices and exit")
    args = parser.parse_args()
    if args.camera_index is not None and args.camera_index < 0:
        parser.error("--camera-index must be >= 0")

    if args.list_cameras:
        from pcb_inspection.acquisition.camera import list_video_devices
        devices = list_video_devices()
        for d in devices:
            print(f"  {d.index}: {d.name}")
        if not devices:
            print("No video device found. Start the DroidCam client and connect the phone.")
        return 0

    capture = load_config(args.capture_config)
    product_id = args.product or capture.get("product")
    if not product_id:
        parser.error("no product: pass --product or set 'product' in the capture config")
    try:
        store = build_store(capture, product_id)
    except ValueError as e:
        parser.error(str(e))
    s = store.settings
    print(f"Product {product_id}: collection store {s.data_root}")
    print(f"  session types: {', '.join(t.upper() for t in s.dataset_types)}   capture check: "
          f"{'on' if s.pcb_checker else 'OFF'}   KEEP blocked on RECAPTURE: {s.keep_requires_capture_check}")

    from pcb_inspection.acquisition.ui import run
    run(store, capture, camera_index=args.camera_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
