"""Blind end-to-end evaluation on the held-out test set (Section 8.2 of prompt v2).

- Copies photos from data/presence/test/red/ to a scratch folder under scrambled names (img_xxxx.jpg)
- Runs ml/presence/predict_photo.py using the frozen final model (runs_v2/final/weights/last.pt)
- Collects predictions blindly
- Evaluates predictions and scores them post-hoc
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Projects\YOLO_PCB_Componet")
TEST_RED_DIR = PROJECT_ROOT / "data" / "presence" / "test" / "red"
ML_DIR = PROJECT_ROOT / "ml" / "presence"
PYTHON_EXE = ML_DIR / ".venv" / "Scripts" / "python.exe"
PREDICT_SCRIPT = ML_DIR / "predict_photo.py"
FINAL_MODEL = ML_DIR / "runs_v2" / "final" / "weights" / "last.pt"
SCRATCH_DIR = PROJECT_ROOT / "scratch" / "blind_test_v2"


def run_blind_test(threshold: float = 0.50):
    print("=" * 60)
    print(f"Starting Blind Test on Test Set: {TEST_RED_DIR}")
    print(f"Model: {FINAL_MODEL}")
    print(f"Frozen Threshold: {threshold}")
    print("=" * 60)

    if not FINAL_MODEL.exists():
        raise FileNotFoundError(f"Final model not found at {FINAL_MODEL}")

    if not TEST_RED_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at {TEST_RED_DIR}")

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"blind_{int(time.time())}"
    blind_folder = SCRATCH_DIR / run_id
    blind_folder.mkdir(parents=True, exist_ok=True)

    test_photos = sorted(list(TEST_RED_DIR.glob("*.jpg")) + list(TEST_RED_DIR.glob("*.jpeg")))
    print(f"Found {len(test_photos)} test photos.")

    rng = random.Random(42)
    mapping = {}
    for p in test_photos:
        scrambled_name = f"img_{rng.randrange(10000, 99999):05d}.jpg"
        while (blind_folder / scrambled_name).exists():
            scrambled_name = f"img_{rng.randrange(10000, 99999):05d}.jpg"
        shutil.copy2(p, blind_folder / scrambled_name)
        mapping[scrambled_name] = p.name

    print(f"Copied {len(mapping)} photos to {blind_folder} with scrambled filenames.")

    out_folder = blind_folder / "out"
    cmd = [
        str(PYTHON_EXE),
        "-I",
        str(PREDICT_SCRIPT),
        str(blind_folder),
        "--model",
        str(FINAL_MODEL),
        "--threshold",
        str(threshold),
        "--out",
        str(out_folder),
    ]

    print("Running predict_photo.py blindly...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("\n--- Predict Output ---")
    print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

    # Parse predictions
    predictions = {}
    for line in res.stdout.splitlines():
        # format: img_XXXXX.jpg: PASS - 86 parts present
        # format: img_XXXXX.jpg: FAIL - 2 missing: R12 (0.95), Q2 (0.88)
        # format: img_XXXXX.jpg: RETAKE - reason
        m = re.match(r"^(img_\d+\.jpg):\s+(PASS|FAIL|RETAKE)\s+-\s+(.*)$", line.strip())
        if m:
            s_name, verdict, details = m.groups()
            missing_parts = []
            if verdict == "FAIL":
                parts_matches = re.findall(r"([A-Z0-9_]+)\s+\((\d+\.\d+)\)", details)
                missing_parts = [(p, float(score)) for p, score in parts_matches]
            predictions[s_name] = {
                "verdict": verdict,
                "details": details,
                "missing": missing_parts,
                "original_filename": mapping.get(s_name, "unknown"),
            }

    # Summary
    print("\n" + "=" * 60)
    print("BLIND TEST RESULTS SUMMARY (Unscrambled)")
    print("=" * 60)
    for s_name, pred in sorted(predictions.items(), key=lambda x: x[1]["original_filename"]):
        orig = pred["original_filename"]
        v = pred["verdict"]
        print(f"{orig} ({s_name}): {v} -> {pred['details']}")

    # Save results to json
    results_path = ML_DIR / "report_v2" / "blind_test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "model": str(FINAL_MODEL),
                "predictions": predictions,
                "mapping": mapping,
            },
            f,
            indent=2,
        )
    print(f"\nSaved blind test results to: {results_path}")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    run_blind_test(threshold=args.threshold)
