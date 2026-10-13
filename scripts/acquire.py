"""PCB data acquisition: live phone camera (DroidCam) -> guided capture -> quality check -> KEEP / REJECT.

    python scripts/acquire.py                    camera from camera.index in the capture config (default 0)
    python scripts/acquire.py --camera-index 1   another camera for this run (e.g. DroidCam)
    python scripts/acquire.py --list-cameras     list the video devices
    python scripts/acquire.py --product pump_controller   collect data for another product (its own store)

Without --product the store and profile are the LG Indicator ones from the capture config (as before);
--product lg_indicator uses the same store (configs/products/lg_indicator/product.yaml storage.acquisition_root).

Camera-first (Experiment 1B): the configured camera opens at start; choose GOOD / DEFECT /
TEST PCB, the stage and the number of images, press START SESSION. Each image
is captured after a 3-2-1 countdown, saved losslessly to
data/acquisition/<type>/<board>/<session>/images/, checksummed, quality-checked
and queued for review. KEEP copies it to data/curated/<type>/..., REJECT (with
a reason) to data/quarantine/<type>/...; the original always stays.

Guide: docs/experiment_1b_camera_acquisition.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.acquisition.session import AcquisitionStore, settings_from_configs, settings_from_product  # noqa: E402
from pcb_inspection.config import load_config, load_profile  # noqa: E402
from pcb_inspection.dataset.experiment0 import E0_CAPTURES  # noqa: E402
from pcb_inspection.dataset.manifest import read_table  # noqa: E402


def reserved_session_ids(data_root: Path) -> set[str]:
    """Session IDs already used by the Experiment 0 photo ingest, so the two tracks never share an ID."""
    path = data_root / "experiment_0" / "metadata" / E0_CAPTURES.filename
    return {row["session_id"] for row in read_table(path, E0_CAPTURES)[0]} if path.is_file() else set()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-camera PCB data acquisition with human review.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    parser.add_argument("--capture-config", type=Path, default=ROOT / "configs" / "capture" / "droidcam_e1b.yaml")
    parser.add_argument("--camera-index", type=int,
                        help="camera to use for this run (default: camera.index in the capture config)")
    parser.add_argument("--list-cameras", action="store_true", help="print the video devices and exit")
    parser.add_argument("--product", default=None,
                        help="product id from configs/products/ (default: the capture config's profile, LG Indicator)")
    parser.add_argument("--legacy", action="store_true",
                        help="run the Experiment 1B store (data/acquisition, GOOD / DEFECT / TEST) anyway")
    args = parser.parse_args()
    if args.camera_index is not None and args.camera_index < 0:
        parser.error("--camera-index must be >= 0")
    if not args.product and not args.list_cameras and not args.legacy:
        print("Since 2026-10-06 data is collected with the clean GOOD / FAULT collection:\n"
              "    python scripts/collect.py\n"
              "(docs/clean_data_collection.md). This Experiment 1B application writes to the old data/acquisition "
              "layout and offers TEST PCB, which is not collected in this phase. Add --legacy to run it anyway.")
        return 2

    if args.list_cameras:
        from pcb_inspection.acquisition.camera import list_video_devices
        devices = list_video_devices()
        for d in devices:
            print(f"  {d.index}: {d.name}")
        if not devices:
            print("No video device found. Start the DroidCam client and connect the phone.")
        return 0

    project = load_config(args.config)
    capture = load_config(args.capture_config)
    data_root = ROOT / project["data"]["root"]
    if args.product:
        from pcb_inspection.products import ProfileError, load_product_profile
        from pcb_inspection.station.capture_store import used_session_ids
        try:
            settings = settings_from_product(load_product_profile(args.product, project_root=ROOT), capture)
        except (ProfileError, ValueError) as e:
            parser.error(str(e))
        reserved = reserved_session_ids(data_root) | used_session_ids(data_root)   # session IDs unique across stores
        print(f"Product {args.product}: data collection store {settings.data_root}")
    else:
        profile = load_profile(ROOT / project["profiles"]["directory"], capture["profile"])
        settings = settings_from_configs(data_root, profile, capture)
        reserved = reserved_session_ids(data_root)
    store = AcquisitionStore(settings, reserved)

    from pcb_inspection.acquisition.ui import run
    run(store, capture, camera_index=args.camera_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
