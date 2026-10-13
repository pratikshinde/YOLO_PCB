"""Build the classification datasets of an experiment from the cache: real crops (hard links), oversampled missing crops, shifted-window
'missing' crops and tolerance-jitter 'present' crops (TRAIN boards only). Validation = real crops only.
Run: ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\v3\\build_datasets.py E0_base [E1_small ...]
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

from degrade import degrade_random
from common import ALIGNED, DATA, FOLDS, V3, corrected_missing, cut, load_aligned, load_meta, norm_tag, valid_parts
from experiments import EXPS


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def usable_photos(meta):
    return {k: m for k, m in meta.items() if m["ok"] and m["kind"] in ("good", "removed") and m["residual_mm"] <= 0.5}


def label_table(meta):
    valid = valid_parts()
    rem = DATA / "raw" / "red" / "removed"
    lab = {}
    for b in sorted({m["board"] for m in meta.values() if m["kind"] == "removed"}):
        lab[b] = corrected_missing(b, rem / b, valid)
    return lab


def real_crops(margin, usable, labels, norm=False):
    """write every real crop once: crops/m<margin*10>[n]/all/<board>__<stem>__<part>.png ; returns index list"""
    tag = f"m{int(round(margin * 10)):02d}" + norm_tag(norm)
    d = V3 / "crops" / tag / "all"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for k, m in usable.items():
        miss = labels.get(m["board"], set())
        aligned = None
        for part in m["parts"]:
            f = d / f"{k}__{part['name']}.png"
            if not f.exists():
                if aligned is None:
                    aligned = load_aligned(k, norm)
                cv2.imwrite(str(f), cut(aligned, part, margin))
            items.append({"k": k, "board": m["board"], "part": part, "label": "missing" if part["name"] in miss else "present", "path": f})
    return items


def build(exp, cfg, fold, items, usable):
    ds = V3 / "datasets" / exp / fold
    if (ds / "READY").exists():
        return json.loads((ds / "READY").read_text())
    if ds.exists():
        shutil.rmtree(ds)
    for s in ("train", "val"):
        for c in ("missing", "present"):
            (ds / s / c).mkdir(parents=True)
    boards = sorted({i["board"] for i in items})
    if fold == "final":
        val_boards = set(FOLDS["fold_1"]["val_removal"] + FOLDS["fold_1"]["val_good"])    # only a sanity-check set: it is inside the training data
        train_boards = set(boards)
    else:
        val_boards = set(FOLDS[fold]["val_removal"] + FOLDS[fold]["val_good"])
        train_boards = set(boards) - val_boards
        assert not (train_boards & val_boards)
    rng = random.Random(1000 + sorted(FOLDS).index(fold) if fold != "final" else 999)

    tr_missing, tr_present = [], []
    excl = set(cfg.get("exclude_missing_types", []))
    for it in items:
        name = f"{it['k']}__{it['part']['name']}.png"
        if it["board"] in train_boards and it["label"] == "missing" and it["part"]["type"] in excl:
            continue                                         # transfer test: no real missing crops of this part type in training
        if it["board"] in val_boards:
            link(it["path"], ds / "val" / it["label"] / name)
        if it["board"] in train_boards:
            link(it["path"], ds / "train" / it["label"] / name)
            (tr_missing if it["label"] == "missing" else tr_present).append(it)
    n_m, n_p = len(tr_missing), len(tr_present)

    # oversample real missing crops (hard-link copies) to about 1 : 4
    target = max(n_m, n_p // 4)
    added, r = 0, 0
    while n_m and added < target - n_m:
        r += 1
        for it in tr_missing:
            if added >= target - n_m:
                break
            link(it["path"], ds / "train" / "missing" / f"{it['k']}__{it['part']['name']}__rep{r}.png")
            added += 1

    # shifted-window crops: need the aligned images -> group by photo
    jobs = defaultdict(list)
    n_syn = min(int(round(cfg["syn_ratio"] * n_m)), n_p)
    for i, it in enumerate(rng.sample(tr_present, n_syn)):
        jobs[it["k"]].append((it, "missing", rng.uniform(cfg["syn_min"], cfg["syn_max"]), f"syn{i:05d}"))
    n_jit = int(round(cfg["jitter_frac"] * n_p))
    for i, it in enumerate(rng.sample(tr_present, n_jit)):
        jobs[it["k"]].append((it, "present", rng.uniform(0.0, cfg["jitter_max"]), f"jit{i:05d}"))
    for k in sorted(jobs):
        aligned = load_aligned(k, cfg.get("colornorm", False))
        for it, cls, dist, tag in jobs[k]:
            ang = rng.uniform(0, 2 * math.pi)
            crop = cut(aligned, it["part"], cfg["margin"], (dist * math.cos(ang), dist * math.sin(ang)))
            cv2.imwrite(str(ds / "train" / cls / f"{k}__{it['part']['name']}__{tag}.png"), crop)

    # degraded copies of real training crops (blur, exposure, saturation, noise, JPEG), per class
    n_deg = 0
    if cfg.get("degrade_frac", 0) > 0:
        nrng = np.random.default_rng(7)
        for cls, pool in (("present", tr_present), ("missing", tr_missing)):
            for i, it in enumerate(rng.sample(pool, int(round(cfg["degrade_frac"] * len(pool))))):
                cv2.imwrite(str(ds / "train" / cls / f"{it['k']}__{it['part']['name']}__deg{i:05d}.png"), degrade_random(cv2.imread(str(it["path"])), nrng))
                n_deg += 1

    stats = {"degraded_copies": n_deg, "train_boards": len(train_boards), "val_boards": sorted(val_boards), "real_missing": n_m, "real_present": n_p, "oversampled": added,
             "synthetic_missing": n_syn, "jitter_present": n_jit,
             "train_missing_files": len(list((ds / "train/missing").glob("*.png"))), "train_present_files": len(list((ds / "train/present").glob("*.png"))),
             "val_missing": len(list((ds / "val/missing").glob("*.png"))), "val_present": len(list((ds / "val/present").glob("*.png")))}
    (ds / "READY").write_text(json.dumps(stats))
    return stats


def build_experiment(exp, with_final=False):
    cfg = EXPS[exp]
    meta = load_meta()
    usable = usable_photos(meta)
    labels = label_table(meta)
    items = real_crops(cfg["margin"], usable, labels, cfg.get("colornorm", False))
    print(f"[{exp}] {len(usable)} usable photos, {len(items)} real crops, missing per removal board: " + ", ".join(f"{b}:{len(v)}" for b, v in labels.items()), flush=True)
    for fold in list(FOLDS) + (["final"] if with_final else []):
        st = build(exp, cfg, fold, items, usable)
        print(f"   {fold}: train missing {st['train_missing_files']} (real {st['real_missing']}, syn {st['synthetic_missing']}) / present {st['train_present_files']} | val missing {st['val_missing']} present {st['val_present']}", flush=True)


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["E0_base"]
    for e in names:
        build_experiment(e, with_final="--final" in sys.argv)
