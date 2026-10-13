"""Speed test: seconds per epoch for different dataloader worker counts (25 % of fold_1, 1 epoch each)."""
import sys
import time

from common import V3
from experiments import EXPS


def main():
    from ultralytics import YOLO
    cfg = EXPS["E0_base"]
    for w in (0, 4, 8):
        m = YOLO(str(V3.parent / "yolo11n-cls.pt"))
        t0 = time.time()
        m.train(data=str(V3 / "datasets" / "E0_base" / "fold_1"), imgsz=128, epochs=1, batch=64, device=0, project=str(V3 / "bench"), name=f"w{w}", exist_ok=True,
                workers=w, fraction=0.25, plots=False, verbose=False, **cfg["aug"])
        print(f"WORKERS {w}: {time.time() - t0:.1f}s for 25% of an epoch (+ validation)", flush=True)


if __name__ == "__main__":
    main()
