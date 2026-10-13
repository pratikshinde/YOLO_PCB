"""Train + evaluate one experiment with 5-fold leave-boards-out cross-validation.
  python train_eval.py E0_base                  -> train all 5 folds (skips finished ones), evaluate, write results\\E0_base_*.csv / .json
  python train_eval.py E0_base --eval-only      -> evaluation only
  python train_eval.py E0_base --final          -> also train the final model on all boards
  options: --workers 6  --epochs N (override)  --folds fold_1,fold_2
The model of every fold is the LAST epoch (cosine schedule finished); no "best epoch" is picked on validation data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from common import ALIGNED, FOLDS, V3, cut, key, load_aligned, load_meta
from experiments import EXPS

RES = V3 / "results"
SHIFTS = [0.5, 0.9, 1.2, 1.5, 2.0]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


STRESS = ["blur", "dark", "noise", "jpeg"]


def degrade(im, kind, rng):
    """cheap stand-ins for a bad phone photo: out of focus, under-exposed, sensor noise, heavy JPEG"""
    if kind == "blur":
        return cv2.GaussianBlur(im, (0, 0), 1.6)
    if kind == "dark":
        return np.clip(im.astype(np.float32) * 0.6 + 6, 0, 255).astype(np.uint8)
    if kind == "noise":
        return np.clip(im.astype(np.float32) + rng.normal(0, 12, im.shape), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 22])[1], cv2.IMREAD_COLOR)
    raise ValueError(kind)


def load_model(path_or_name):
    from ultralytics import YOLO
    if not os.path.isabs(str(path_or_name)):
        wdir = V3 / "weights"
        wdir.mkdir(exist_ok=True)
        os.chdir(wdir)                       # ultralytics downloads a named base model into the current folder
    return YOLO(str(path_or_name))


def train_one(exp, fold, workers, epochs=None):
    cfg = EXPS[exp]
    run = V3 / "runs" / exp / fold
    if (run / "done.json").exists():
        print(f"[{exp}/{fold}] already trained", flush=True)
        return
    model = load_model(str(V3.parent / "yolo11n-cls.pt") if cfg["model"] == "yolo11n-cls.pt" else cfg["model"])
    n_ep = epochs or cfg["epochs"]
    t0 = time.time()
    model.train(data=str(V3 / "datasets" / exp / fold), imgsz=128, epochs=n_ep, patience=n_ep, batch=64, device=0, project=str(V3 / "runs" / exp), name=fold,
                exist_ok=True, workers=workers, cos_lr=True, seed=0, verbose=False, plots=False, **cfg["aug"])
    (run / "done.json").write_text(json.dumps({"epochs": n_ep, "seconds": round(time.time() - t0), "model": cfg["model"]}))
    print(f"[{exp}/{fold}] trained in {time.time() - t0:.0f}s", flush=True)


def missing_index(model):
    return [i for i, n in model.names.items() if n == "missing"][0]


def predict_p(model, imgs, idx, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        for r in model.predict(imgs[i:i + bs], imgsz=128, device=0, verbose=False):
            out.append(float(r.probs.data[idx]))
    return np.array(out)


def evaluate(exp, folds):
    cfg = EXPS[exp]
    meta = load_meta()
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, shift_rows, stress_rows = [], [], []
    for fold in folds:
        w = V3 / "runs" / exp / fold / "weights" / "last.pt"
        from ultralytics import YOLO
        model = YOLO(str(w))
        idx = missing_index(model)
        val = V3 / "datasets" / exp / fold / "val"
        items = []
        for cls in ("missing", "present"):
            for f in sorted((val / cls).glob("*.png")):
                b, st, part = f.stem.split("__")[:3]
                items.append({"fold": fold, "board": b, "photo": st, "part": part, "label": cls, "path": f})
        p = predict_p(model, [cv2.imread(str(i["path"])) for i in items], idx)
        for it, v in zip(items, p):
            m = meta[key(it["board"], it["photo"])]
            ptype_ = next(x["type"] for x in m["parts"] if x["name"] == it["part"])
            rows.append([fold, it["board"], it["photo"], it["part"], ptype_, it["label"], f"{v:.5f}"])
        # window-shift tolerance on PRESENT parts of the held-out boards (12 random parts per photo, 4 directions)
        by = {}
        for it in items:
            if it["label"] == "present":
                by.setdefault((it["board"], it["photo"]), []).append(it["part"])
        crops, tags = [], []
        for (b, st), parts in sorted(by.items()):
            m = meta[key(b, st)]
            aligned = load_aligned(key(b, st), cfg.get("colornorm", False))
            pdict = {x["name"]: x for x in m["parts"]}
            for name in rng.choice(parts, size=min(12, len(parts)), replace=False):
                for s in SHIFTS:
                    for d in DIRS:
                        crops.append(cut(aligned, pdict[name], cfg["margin"], (d[0] * s, d[1] * s)))
                        tags.append((fold, s))
        ps = predict_p(model, crops, idx)
        shift_rows += [[f, s, f"{v:.5f}"] for (f, s), v in zip(tags, ps)]
        # phone-quality stress: the same held-out crops, degraded
        base_imgs = [cv2.imread(str(i["path"])) for i in items]
        for kind in STRESS:
            pk = predict_p(model, [degrade(im, kind, rng) for im in base_imgs], idx)
            stress_rows += [[fold, kind, it["label"], f"{v:.5f}"] for it, v in zip(items, pk)]
        print(f"[{exp}/{fold}] evaluated {len(items)} val crops, {len(crops)} shifted crops, {len(STRESS)} stress variants", flush=True)
    with open(RES / f"{exp}_preds.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["fold", "board", "photo", "part", "type", "label", "p_missing"])
        w_.writerows(rows)
    with open(RES / f"{exp}_shift.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["fold", "shift_mm", "p_missing"])
        w_.writerows(shift_rows)
    with open(RES / f"{exp}_stress.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["fold", "variant", "label", "p_missing"])
        w_.writerows(stress_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--folds")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--only-final", action="store_true", help="train just the final model (all boards), no cross-validation, no evaluation")
    a = ap.parse_args()
    if a.only_final:
        train_one(a.exp, "final", a.workers, a.epochs)
        return
    folds = a.folds.split(",") if a.folds else list(FOLDS)
    if not a.eval_only:
        for f in folds:
            train_one(a.exp, f, a.workers, a.epochs)
        if a.final:
            train_one(a.exp, "final", a.workers, a.epochs)
    evaluate(a.exp, folds)


if __name__ == "__main__":
    main()
