"""Train YOLO11n-cls models v2 for PCB presence detection.

Follows Section 6 of TRAINING_PROMPT_antigravity_v2.md:
- Pretrained weights: ml/presence/yolo11n-cls.pt (never warm-started from old runs)
- Folds: fold_1, fold_2, fold_3, fold_4, fold_5, and final
- Epochs: 60, patience: 60 (full cosine LR schedule, no early stopping)
- Hyperparameters: batch 64, imgsz 128, cos_lr=True, fliplr 0.5, flipud 0.0,
  hsv_h 0.015, hsv_s 0.5, hsv_v 0.4, degrees 5, translate 0.04, scale 0.15,
  erasing 0, workers 0, seed 0, project=runs_v2, exist_ok=True.
- Keeps last.pt as the model to evaluate and ship.
- Logs to runs_v2/train_log.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ML_DIR = Path(r"D:\Projects\YOLO_PCB_Componet\ml\presence")
RUNS_V2_DIR = ML_DIR / "runs_v2"
DATASETS_V2_DIR = ML_DIR / "datasets_v2"
BASE_MODEL_PATH = ML_DIR / "yolo11n-cls.pt"
LOG_FILE = RUNS_V2_DIR / "train_log.txt"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train_fold(
    fold_name: str,
    device: str = "0",
    epochs: int = 60,
    batch: int = 64,
    patience: int = 60,
):
    dataset_path = DATASETS_V2_DIR / fold_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    log("\n" + "=" * 60)
    log(f"Starting Training: {fold_name}")
    log(f"Dataset: {dataset_path}")
    log(f"Base weights: {BASE_MODEL_PATH}")
    log(f"Device: {device}, Epochs: {epochs}, Patience: {patience}, Batch: {batch}")
    log("=" * 60)

    t0 = time.time()
    # Always load fresh from the original pretrained checkpoint
    model = YOLO(str(BASE_MODEL_PATH))

    results = model.train(
        data=str(dataset_path),
        imgsz=128,
        epochs=epochs,
        patience=patience,
        batch=batch,
        device=device,
        project=str(RUNS_V2_DIR),
        name=fold_name,
        cache=True,
        cos_lr=True,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5,
        translate=0.04,
        scale=0.15,
        erasing=0.0,
        workers=0,
        seed=0,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - t0
    last_pt = RUNS_V2_DIR / fold_name / "weights" / "last.pt"
    log(f"Finished {fold_name} in {elapsed:.1f}s ({elapsed / epochs:.2f}s/epoch).")
    log(f"Model checkpoint saved at: {last_pt} (exists: {last_pt.exists()})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-cls v2 models.")
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Fold to train: fold_1..fold_5, final, or 'all'",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    RUNS_V2_DIR.mkdir(parents=True, exist_ok=True)

    # Environment check
    cuda_avail = torch.cuda.is_available()
    log(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}, CUDA available: {cuda_avail}")
    if not cuda_avail and args.device != "cpu":
        log("WARNING: CUDA is not available! Falling back to CPU for smoke testing.")
        device = "cpu"
    else:
        if cuda_avail:
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = args.device

    if args.fold == "all":
        targets = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "final"]
    else:
        targets = [args.fold]

    t_start = time.time()
    for target in targets:
        train_fold(
            target,
            device=device,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
        )

    total_time = time.time() - t_start
    log(f"\nAll training tasks completed in {total_time:.1f}s ({total_time / 60:.2f} min).")


if __name__ == "__main__":
    main()
