"""Export the chosen final model for the app: v3\\final\\presence_cls.onnx + presence_model.json (model card), copied to android_app\\assets\\model\\.
  python export_app_model.py E10_expnorm [--threshold 0.35] [--doubt 0.10]
The card carries everything the phone has to reproduce from training: crop size / margin, exposure normalisation constants, thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from common import CACHE, DATA, ROOT, V3
from experiments import EXPS

ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("--threshold", type=float, default=0.35)
ap.add_argument("--doubt", type=float, default=0.10)
a = ap.parse_args()
cfg = EXPS[a.exp]
assert cfg["margin"] == 1.5 and cfg.get("colornorm") in ("exp", None, False), "the app implements margin 1.5 mm and (optional) exposure normalisation only"

out = V3 / "final"
out.mkdir(exist_ok=True)
weights = V3 / "runs" / a.exp / "final" / "weights" / "last.pt"
shutil.copy2(weights, out / "presence_cls.pt")

from ultralytics import YOLO  # noqa: E402

model = YOLO(str(out / "presence_cls.pt"))
names = model.names
missing_index = [i for i, n in names.items() if n == "missing"][0]
model.export(format="onnx", imgsz=128, batch=1, dynamic=False, simplify=False, opset=12, device="cpu")
onnx = out / "presence_cls.onnx"
assert onnx.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

import summarize  # noqa: E402

m = summarize.metrics(a.exp, a.threshold)

# the thorough mode (median over 9 windows per part): same held-out boards, from decision_rules.py
import pickle  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

thorough = None
wpath = V3 / "results" / f"{a.exp}_windows.pkl"
if wpath.exists():
    win = pickle.load(open(wpath, "rb"))
    by = defaultdict(list)
    for fold, b, st, part, label, row in win:
        by[(b, st)].append((part, label == "missing", float(np.median(row))))
    missed = sum(1 for v in by.values() for _, mis, s_ in v if mis and s_ < a.threshold)
    false_calls = sum(1 for v in by.values() for _, mis, s_ in v if not mis and s_ >= a.threshold)
    good = {k: v for k, v in by.items() if not k[0].startswith("m_")}
    rem = {k: v for k, v in by.items() if k[0].startswith("m_")}
    exact = sum({p for p, _, s_ in v if s_ >= a.threshold} == {p for p, mis, _ in v if mis} for v in rem.values())
    miss_scores = [s_ for v in by.values() for _, mis, s_ in v if mis]
    pres_scores = [s_ for v in by.values() for _, mis, s_ in v if not mis]
    thorough = {"rule": "median of 9 windows", "threshold": a.threshold, "missed": f"{missed}/{len(miss_scores)}", "false_calls": f"{false_calls}/{len(pres_scores)}",
                "good_photos_failing": f"{sum(any(s_ >= a.threshold for _, _, s_ in v) for v in good.values())}/{len(good)}",
                "removed_photos_exact": f"{exact}/{len(rem)}", "lowest_missing_score": round(min(miss_scores), 3), "highest_present_score": round(max(pres_scores), 3)}
exp_params = json.loads((CACHE / "exposure_params.json").read_text()) if cfg.get("colornorm") == "exp" else None
card = {
    "name": f"presence_cls_{a.exp}",
    "architecture": "YOLO11n-cls",
    "trained": "2026-10-12",
    "experiment": a.exp,
    "input_size": 128,
    "classes": [names[i] for i in sorted(names)],
    "missing_index": missing_index,
    "crop_margin_mm": cfg["margin"],
    "px_per_mm": 20,
    "exposure_norm": bool(exp_params),
    "exposure_target_luma": exp_params["target_luma"] if exp_params else 0.0,
    "exposure_gain_min": exp_params["gain_min"] if exp_params else 1.0,
    "exposure_gain_max": exp_params["gain_max"] if exp_params else 1.0,
    "threshold": a.threshold,
    "doubt": a.doubt,
    "max_flagged": 20,
    "windows_default": 9,
    "window_combine": "median",
    "window_offsets_mm": [[0.0, 0.0]] + [[dx * s, dy * s] for s in (0.4, 0.8) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))],
    "golden_sha256": sha(DATA / "golden" / "red" / "grf.jpeg"),
    "onnx_sha256": sha(onnx),
    "cross_validation": {
        "folds": 5, "split": "leave-boards-out", "epochs": cfg["epochs"], "missing_crops": m["n_missing"], "present_crops": m["n_present"],
        "missed_at_threshold": m["FN"], "false_calls_at_threshold": m["FP"], "good_photos_failing": f"{m['good_false_fail']}/{m['good_photos']}",
        "removed_photos_exact": f"{m['removed_exact']}/{m['removed_photos']}", "threshold": a.threshold,
    },
    "cross_validation_thorough": thorough,
    "proven_part_types": ["R"],
    "not_proven_part_types": ["R2K2", "C", "C100N", "D", "D4148", "Q", "U"],
    "note": "Only resistors have enough real removals (45 instances on 6 boards). The other part types have very few or no real missing examples.",
}
(out / "presence_model.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
assets = ROOT / "android_app" / "assets" / "model"
assets.mkdir(parents=True, exist_ok=True)
shutil.copy2(onnx, assets / "presence_cls.onnx")
shutil.copy2(out / "presence_model.json", assets / "presence_model.json")
print(json.dumps({k: card[k] for k in ("name", "exposure_norm", "exposure_target_luma", "threshold", "doubt", "onnx_sha256")}, indent=1))
print("copied to", assets)
