"""Train YOLO11n-cls models for PCB presence detection.

Trains fold_1, fold_2, fold_3, fold_4 and final models according to TRAINING_PROMPT_antigravity.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from ultralytics import YOLO

ML_DIR = Path(r"D:\Projects\YOLO_PCB_Componet\ml\presence")
RUNS_DIR = ML_DIR / "runs"
DATASETS_DIR = ML_DIR / "datasets"
BASE_MODEL_PATH = ML_DIR / "yolo11n-cls.pt"


def train_model(fold_name: str, device: str = "0", epochs: int = 100, batch: int = 64):
    """Train YOLO11n-cls on the specified fold."""
    dataset_path = DATASETS_DIR / fold_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    print(f"\n==================================================")
    print(f"Starting Training: {fold_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}, Epochs: {epochs}, Batch: {batch}")
    print(f"==================================================")

    t0 = time.time()
    model = YOLO(str(BASE_MODEL_PATH))

    # All hyperparameters match Section 7 of TRAINING_PROMPT_antigravity.md
    results = model.train(
        data=str(dataset_path),
        imgsz=128,
        epochs=epochs,
        patience=25,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=fold_name,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5,
        translate=0.04,
        scale=0.15,
        erasing=0.0,
        seed=0,
        workers=0,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - t0
    best_pt = RUNS_DIR / fold_name / "weights" / "best.pt"
    print(f"Completed {fold_name} in {elapsed:.1f}s.")
    print(f"Best weights saved at: {best_pt} (exists: {best_pt.exists()})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-cls for presence detection.")
    parser.add_argument("--fold", type=str, default="all",
                        help="Specific fold (fold_1, fold_2, fold_3, fold_4, final) or 'all'")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fold == "all":
        targets = ["fold_1", "fold_2", "fold_3", "fold_4", "final"]
    else:
        targets = [args.fold]

    for target in targets:
        train_model(target, device=args.device, epochs=args.epochs, batch=args.batch)

    print("\nAll requested training jobs completed successfully!")


if __name__ == "__main__":
    main()
