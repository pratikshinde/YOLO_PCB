"""Align every photo ONCE (product pipeline, 20 px/mm) and cache: aligned PNG + per-part box / flow shift. Crops of any margin can then be cut in seconds.
Run: ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\v3\\build_cache.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import ALIGNED, META, PX_PER_MM, gc, key, photo_list, ptype, pp, test_photos

_REF = None
_P = None


def _init():
    global _REF, _P
    _P = pp.make_params()
    _REF = gc.prepare_reference(gc.imread(pp.DEFAULT_GOLDEN), _P, gc.load_regions(pp.REGIONS, _P))


def _work(task):
    board, path, kind = task
    k = key(board, path.stem)
    meta = {"board": board, "stem": path.stem, "path": str(path), "kind": kind}
    try:
        img = gc.imread(path)
        aligned, valid, al, margin_added = pp.align(img, _REF, _P)
        rg, ng, _ = gc.comparison_grays(_REF.clahe, gc.clahe_gray(aligned), np.bitwise_and(_REF.area, valid), _P)
        shifts = gc.region_shifts(gc.local_flow(rg, ng), _REF, _REF.regions, _P)
        parts = []
        for r in _REF.regions:
            if r.kind != "part":
                continue
            x, y, w, h = r.bbox_px
            dx, dy = shifts.get(r.id, (0.0, 0.0))
            parts.append({"name": r.name, "type": ptype(r.name), "x": float(x), "y": float(y), "w": float(w), "h": float(h), "dx": float(dx), "dy": float(dy)})
        cv2.imwrite(str(ALIGNED / f"{k}.png"), aligned)
        meta.update(ok=True, residual_mm=float(al.residual_mm), margin_added=bool(margin_added), parts=parts)
    except Exception as e:   # not alignable
        meta.update(ok=False, error=str(e))
    return k, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ALIGNED.mkdir(parents=True, exist_ok=True)
    tasks = photo_list() + [("TEST", p, "test") for p in test_photos()]
    done = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    todo = [t for t in tasks if key(t[0], t[1].stem) not in done or not (ALIGNED / f"{key(t[0], t[1].stem)}.png").exists() and done[key(t[0], t[1].stem)].get("ok")]
    print(f"{len(tasks)} photos, {len(todo)} to align", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for i, (k, meta) in enumerate(ex.map(_work, todo), 1):
            done[k] = meta
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    META.write_text(json.dumps(done, indent=1), encoding="utf-8")
    ok = [m for m in done.values() if m["ok"]]
    gate = [m for m in ok if m["residual_mm"] <= 0.5]
    print(f"aligned {len(ok)} / {len(done)}; residual <= 0.5 mm: {len(gate)}")
    for m in done.values():
        if not m["ok"]:
            print("  FAILED", m["board"], m["stem"], m["error"][:90])
        elif m["residual_mm"] > 0.5:
            print("  RESIDUAL", m["board"], m["stem"], round(m["residual_mm"], 3))


if __name__ == "__main__":
    main()
