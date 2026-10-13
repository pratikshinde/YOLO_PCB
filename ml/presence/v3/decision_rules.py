"""Decision rules on top of the fold models (leak-free: each photo is judged by the model that never saw its board).
Window rules: look at the slot through several windows moved by +-0.4 / +-0.8 mm and combine the scores.
Photo rules: combine 2 or 3 photos of the same board.
  python decision_rules.py E7_combo
"""
from __future__ import annotations

import csv
import itertools
import sys
from collections import defaultdict

import cv2
import numpy as np

from common import ALIGNED, FOLDS, V3, cut, key, load_aligned, load_meta
from experiments import EXPS
from train_eval import missing_index, predict_p

RES = V3 / "results"
SH = [(0, 0)] + [(d * s, e * s) for s in (0.4, 0.8) for d, e in ((1, 0), (-1, 0), (0, 1), (0, -1))]   # 9 windows


def window_scores(exp):
    from ultralytics import YOLO
    cfg = EXPS[exp]
    meta = load_meta()
    out = []
    for fold in FOLDS:
        model = YOLO(str(V3 / "runs" / exp / fold / "weights" / "last.pt"))
        idx = missing_index(model)
        val = V3 / "datasets" / exp / fold / "val"
        items = []
        for cls in ("missing", "present"):
            for f in sorted((val / cls).glob("*.png")):
                b, st, part = f.stem.split("__")[:3]
                items.append((b, st, part, cls))
        by = defaultdict(list)
        for it in items:
            by[(it[0], it[1])].append(it)
        for (b, st), lst in sorted(by.items()):
            aligned = load_aligned(key(b, st), cfg.get("colornorm", False))
            pd = {x["name"]: x for x in meta[key(b, st)]["parts"]}
            crops = [cut(aligned, pd[it[2]], cfg["margin"], s) for it in lst for s in SH]
            p = predict_p(model, crops, idx).reshape(len(lst), len(SH))
            for it, row in zip(lst, p):
                out.append((fold, b, st, it[2], it[3], row))
        print(f"[{exp}/{fold}] windows scored", flush=True)
    return out


def evaluate(out, name, rule, taus=(0.3, 0.5, 0.7)):
    lab = np.array([o[4] == "missing" for o in out])
    s = np.array([rule(o[5]) for o in out])
    res = []
    for t in taus:
        fn = int(((s < t) & lab).sum())
        fp = int(((s >= t) & ~lab).sum())
        # board level
        by = defaultdict(list)
        for o, v in zip(out, s):
            by[(o[1], o[2])].append((o[3], v >= t, o[4] == "missing"))
        gf = gn = rx = rn = 0
        for (b, st), v in by.items():
            flagged = {p for p, f, m in v if f}
            truth = {p for p, f, m in v if m}
            if truth or b.startswith("m_"):
                rn += 1
                rx += flagged == truth
            else:
                gn += 1
                gf += bool(flagged)
        res.append(f"@{t}: missed {fn}/{int(lab.sum())}, false calls {fp}, good photos failing {gf}/{gn}, exact lists {rx}/{rn}")
    return f"| {name} | " + " | ".join(res) + " |"


def margins(out, rule):
    lab = np.array([o[4] == "missing" for o in out])
    s = np.array([rule(o[5]) for o in out])
    pm, pp_ = s[lab], s[~lab]
    return (f"missing: min {pm.min():.3f}, 1st pct {np.percentile(pm, 1):.3f}, 5th pct {np.percentile(pm, 5):.3f} | "
            f"present: max {pp_.max():.3f}, 99.9th pct {np.percentile(pp_, 99.9):.4f}, 99th pct {np.percentile(pp_, 99):.4f}")


def main():
    exp = sys.argv[1]
    cache = RES / f"{exp}_windows.pkl"
    if cache.exists():
        import pickle
        out = pickle.load(open(cache, "rb"))
    else:
        out = window_scores(exp)
        import pickle
        pickle.dump(out, open(cache, "wb"))
    rules = {
        "centre only (current)": lambda r: r[0],
        "mean of 5 windows (centre, +-0.4 mm)": lambda r: r[:5].mean(),
        "mean of 9 windows (+-0.4, +-0.8 mm)": lambda r: r.mean(),
        "min of 5 windows (any window sees the part)": lambda r: r[:5].min(),
        "min of 9 windows": lambda r: r.min(),
        "median of 9 windows": lambda r: float(np.median(r)),
        "median of 5 windows (centre, +-0.4 mm)": lambda r: float(np.median(r[:5])),
        "median of 3 windows (centre, +-0.4 mm in x)": lambda r: float(np.median(r[:3])),
        "max of 5 windows (any window says missing)": lambda r: r[:5].max(),
    }
    L = [f"# Decision rules on {exp} (held-out boards)\n", "| rule | 0.3 | 0.5 | 0.7 |", "|---|---|---|---|"]
    for n, r in rules.items():
        L.append(evaluate(out, n, r))
    L += ["", "## Score margins of the rules", ""]
    for n, r in rules.items():
        L.append(f"* {n}: {margins(out, r)}")
    (RES / f"{exp}_rules.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
