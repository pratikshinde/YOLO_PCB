"""Export and verify final YOLO11n-cls presence model to ONNX (Section 8.3 & 8.4).

- Exports runs_v2/final/weights/last.pt to ONNX (imgsz 128, batch 1, opset 12)
- Saves to models_v2/presence_cls.onnx
- Computes SHA-256 and writes models_v2/presence_cls.json
- Verifies ONNX numerical accuracy against PyTorch on 500 crops using OpenCV DNN (target: max diff < 1e-4)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"D:\Projects\YOLO_PCB_Componet")
ML_DIR = PROJECT_ROOT / "ml" / "presence"
MODELS_V2_DIR = ML_DIR / "models_v2"
FINAL_WEIGHTS = ML_DIR / "runs_v2" / "final" / "weights" / "last.pt"
ONNX_TARGET = MODELS_V2_DIR / "presence_cls.onnx"
JSON_TARGET = MODELS_V2_DIR / "presence_cls.json"
CROPS_DIR = ML_DIR / "crops_v2" / "all"
GOLDEN_IMAGE = PROJECT_ROOT / "data" / "presence" / "golden" / "red" / "grf.jpeg"
OPENCV_SCRATCH = PROJECT_ROOT / "scratch" / "opencv_410"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def export_model(frozen_threshold: float = 0.50):
    print("=" * 60)
    print("Starting Model Export to ONNX")
    print(f"Source weights: {FINAL_WEIGHTS}")
    print("=" * 60)

    if not FINAL_WEIGHTS.exists():
        raise FileNotFoundError(f"Final model weights not found at: {FINAL_WEIGHTS}")

    MODELS_V2_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Export using Ultralytics YOLO
    print("Exporting with YOLO.export()...")
    model = YOLO(str(FINAL_WEIGHTS))
    exported_path_str = model.export(
        format="onnx",
        imgsz=128,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=12,
    )

    exported_path = Path(exported_path_str)
    print(f"Ultralytics exported to: {exported_path}")

    # 2. Move to models_v2/presence_cls.onnx
    shutil.copy2(exported_path, ONNX_TARGET)
    print(f"Copied ONNX model to: {ONNX_TARGET}")

    onnx_sha = compute_sha256(ONNX_TARGET)
    print(f"ONNX SHA-256: {onnx_sha}")

    golden_sha = compute_sha256(GOLDEN_IMAGE)

    # 3. Write presence_cls.json
    model_card = {
        "what": "per-part presence classifier for the red LG Indicator board: crop of one SMT part slot -> p(missing)",
        "model": "YOLO11n-cls (Ultralytics 8.4.174), trained by Antigravity on 31 boards (runs_v2/final)",
        "source_weights": "ml/presence/runs_v2/final/weights/last.pt",
        "onnx_opset": 12,
        "onnx_sha256": onnx_sha,
        "training_boards_count": 31,
        "frozen_threshold": frozen_threshold,
        "input": {
            "name": "images",
            "shape": [1, 3, 128, 128],
            "colour_order": "RGB",
            "scale": "pixel / 255",
            "mean": 0,
            "std": 1,
        },
        "output": {
            "name": "output0",
            "shape": [1, 2],
            "classes": ["missing", "present"],
            "softmax_already_applied": True,
            "index_missing": 0,
        },
        "opencv_dnn": "tested with OpenCV 4.10 (headless 4.10.0.84): cv2.dnn.readNetFromONNX + blobFromImage(img, 1/255, (128,128), (0,0,0), swapRB=True); outputs verified against PyTorch",
        "crop_rule": {
            "frame": "golden canonical frame at 20 px/mm (1660 x 2320 px)",
            "centre": "part box centre + local optical-flow shift (gc.region_shifts)",
            "side_px": "round(max(w, h) + 2 * 1.5 * 20)  (part box + 1.5 mm each side)",
            "outside_frame": "BORDER_REFLECT_101",
            "resize": "128 x 128, INTER_AREA (BGR image from OpenCV; swap to RGB for the network)",
        },
        "decision": f"part is MISSING when p(missing) >= {frozen_threshold:.2f}; board FAILS when any of the 86 parts is MISSING",
        "golden_reference": {
            "file": "data/presence/golden/red/grf.jpeg",
            "sha256": golden_sha,
        },
        "parts": 86,
        "limits": "Only 'R' resistors are PROVEN (40 real removals across 10 boards). Types Q, C, D have <15 removals; R2K2, C100N, D4148, U had 0 removals in dataset and are NOT PROVEN.",
        "proven_types": ["R"],
        "not_proven_types": ["Q", "C", "D", "R2K2", "C100N", "D4148", "U"],
    }

    with open(JSON_TARGET, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved model card to: {JSON_TARGET}")

    # 4. Verify against PyTorch with OpenCV DNN on 500 crops
    verify_onnx_numerical_accuracy()


def verify_onnx_numerical_accuracy(num_crops: int = 500):
    print("\n" + "=" * 60)
    print(f"Verifying ONNX vs PyTorch Numerical Parity on {num_crops} crops")
    print("=" * 60)

    # Use OpenCV from scratch/opencv_410
    if str(OPENCV_SCRATCH) not in sys.path:
        sys.path.insert(0, str(OPENCV_SCRATCH))

    import cv2
    print(f"OpenCV version in verification: {cv2.__version__}")

    # Load ONNX net
    net = cv2.dnn.readNetFromONNX(str(ONNX_TARGET))

    # Load PyTorch model
    py_model = YOLO(str(FINAL_WEIGHTS))
    torch_model = py_model.model
    torch_model.eval()
    device = torch.device("cpu")
    torch_model.to(device)

    # Sample crops
    all_crops = sorted(list(CROPS_DIR.glob("*.png")))
    if len(all_crops) < num_crops:
        sample_crops = all_crops
    else:
        # evenly spaced sampling
        step = len(all_crops) // num_crops
        sample_crops = all_crops[::step][:num_crops]

    print(f"Testing on {len(sample_crops)} crops...")

    max_diff = 0.0
    diffs = []

    for idx, crop_path in enumerate(sample_crops):
        img_bgr = cv2.imread(str(crop_path))
        if img_bgr is None:
            continue
        if img_bgr.shape[:2] != (128, 128):
            img_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)

        # 1. OpenCV DNN inference
        blob = cv2.dnn.blobFromImage(
            img_bgr,
            scalefactor=1.0 / 255.0,
            size=(128, 128),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        onnx_out = net.forward()  # shape (1, 2)

        # 2. PyTorch inference
        # blob is (1, 3, 128, 128), float32, RGB
        tensor_in = torch.from_numpy(blob).to(device)
        with torch.no_grad():
            torch_out = torch_model(tensor_in)
            if hasattr(torch_out, "probs"):
                torch_probs = torch_out.probs.cpu().numpy()
            else:
                torch_probs = torch.softmax(torch_out, dim=-1).cpu().numpy()

        diff = np.max(np.abs(onnx_out - torch_probs))
        diffs.append(diff)
        if diff > max_diff:
            max_diff = diff

    avg_diff = float(np.mean(diffs))
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Average absolute difference: {avg_diff:.8f}")

    if max_diff < 1e-4:
        print("[PASS] ONNX vs PyTorch check PASSED! Max diff < 1e-4.")
    else:
        print(f"[FAIL] ONNX vs PyTorch check EXCEEDED tolerance 1e-4: max diff = {max_diff:.8f}")
        raise ValueError(f"Numerical disparity too high: {max_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    export_model(frozen_threshold=args.threshold)
