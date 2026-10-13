"""Which photos show the SAME physical board?  (solves "are the good photos independent boards?")

Solder fillets / flux marks are unique per board, so high-pass crops of the same slot correlate strongly between two photos of one board
and weakly between two different boards. Calibrated with the removed boards m_01..m_04 (photos of one m_0N = one board, m_0N vs m_0M = two
boards). The photos the training script skipped are cut with the product pipeline (predict_photo.py).

Run:  ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\checks\\identity_all.py
Prints the physical boards found among the good photos, which of them straddle the fold blocks of make_dataset.py, and which good photos
have a same-board photo in the training set of their own fold. Writes out\\identity_all.pkl (photo-by-photo similarity matrix).
"""
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ML = Path(__file__).resolve().parents[1]
ROOT = ML.parents[1]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ML))
import predict_photo as pp          # noqa: E402  (same alignment + crop rule as the product)

gc = pp.gc
P = pp.make_params()
ref = gc.prepare_reference(gc.imread(pp.DEFAULT_GOLDEN), P, gc.load_regions(pp.REGIONS, P))
parts = [r for r in ref.regions if r.kind == "part"]
pnames = [r.name for r in parts]


def vec(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 5)          # remove lighting, keep solder / flux texture
    hp -= hp.mean()
    n = float(np.linalg.norm(hp))
    return (hp / n).ravel() if n > 0 else np.zeros(hp.size, np.float32)


# 1. crops that exist (photos the training script used)
rows = [r for r in csv.DictReader(open(ML / "crops/index.csv", encoding="utf-8")) if r["synthetic_shift"] == "0"]
names = sorted({(r["board"], r["photo"]) for r in rows})
vecs = {n: np.zeros((len(pnames), 128 * 128), np.float32) for n in names}
ok = {n: np.zeros(len(pnames), bool) for n in names}
pidx = {n: i for i, n in enumerate(pnames)}
for r in rows:
    im = cv2.imread(str(ROOT / r["crop_path"]))
    vecs[(r["board"], r["photo"])][pidx[r["part"]]] = vec(im)
    ok[(r["board"], r["photo"])][pidx[r["part"]]] = r["label"] == "present"

# 2. the skipped GOOD photos, cut by the product pipeline
skipped = [s for s in json.load(open(ML / "report/skipped_photos.json", encoding="utf-8")) if s["kind"] == "good"]
for s in skipped:
    img = gc.imread(Path(s["path"]))
    aligned, valid, al, _ = pp.align(img, ref, P)
    rg, ng, _ = gc.comparison_grays(ref.clahe, gc.clahe_gray(aligned), np.bitwise_and(ref.area, valid), P)
    shifts = gc.region_shifts(gc.local_flow(rg, ng), ref, ref.regions, P)
    key = (s["board"], Path(s["photo"]).stem)
    vecs[key] = np.zeros((len(pnames), 128 * 128), np.float32)
    ok[key] = np.ones(len(pnames), bool)
    for i, r in enumerate(parts):
        x, y, w, h = r.bbox_px
        dx, dy = shifts.get(r.id, (0.0, 0.0))
        side = int(round(max(w, h) + 2 * 1.5 * P.px_per_mm))
        vecs[key][i] = vec(pp.square_crop(aligned, x + w / 2.0 + dx, y + h / 2.0 + dy, side))
    print("skipped photo cut:", key, flush=True)

keys = sorted(vecs, key=lambda k: (k[0].startswith("m_"), k[0], k[1]))
N = len(keys)
S = np.zeros((N, N), np.float32)
for i in range(N):
    for j in range(i, N):
        both = ok[keys[i]] & ok[keys[j]]
        c = np.einsum("pk,pk->p", vecs[keys[i]][both], vecs[keys[j]][both])
        S[i, j] = S[j, i] = float(np.median(c)) if c.size else np.nan
pickle.dump({"keys": keys, "S": S}, open(OUT / "identity_all.pkl", "wb"))

within = [S[i, j] for i in range(N) for j in range(i + 1, N) if keys[i][0].startswith("m_") and keys[i][0] == keys[j][0]]
cross = [S[i, j] for i in range(N) for j in range(i + 1, N) if keys[i][0].startswith("m_") and keys[j][0].startswith("m_") and keys[i][0] != keys[j][0]]
THR = (min(within) + max(cross)) / 2                       # midpoint of the gap between "same board" and "different boards"
print(f"\ncalibration: same board min {min(within):.3f} | different boards max {max(cross):.3f} -> threshold {THR:.2f}")
if min(within) <= max(cross):
    print("WARNING: the two groups overlap - the grouping below is not reliable")

parent = list(range(N))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for i in range(N):
    for j in range(i + 1, N):
        if S[i, j] >= THR:
            parent[find(i)] = find(j)
groups = defaultdict(list)
for i in range(N):
    groups[find(i)].append(i)


def num(p):
    return int(re.search(r"R_(\d+)", p).group(1))


def block(n):                      # the fold blocks of make_dataset.py
    return 1 if n <= 11 else 2 if n <= 22 else 3 if n <= 33 else 4


skipped_names = {(s["board"], Path(s["photo"]).stem) for s in skipped}
used = {num(p) for b, p in keys if b.startswith("good_") and (b, p) not in skipped_names}
good_groups, mixed = [], []
for g in groups.values():
    ks = [keys[i] for i in g]
    if all(b.startswith("good_") for b, _ in ks):
        good_groups.append(sorted(num(p) for _, p in ks))
    elif any(b.startswith("good_") for b, _ in ks):
        mixed.append(ks)
good_groups.sort()
print(f"\ngood photos: {sum(len(g) for g in good_groups)} -> {len(good_groups)} physical boards")
leaky = []
for g in good_groups:
    blocks = sorted({block(n) for n in g})
    for n in g:
        if any(m != n and block(m) != block(n) and m in used for m in g):
            leaky.append(n)
    print("  ", ", ".join(f"R_{n:02d}{'*' if n not in used else ''}" for n in g), "  <-- spans fold blocks %s" % blocks if len(blocks) > 1 else "")
print("(* = photo skipped by the training script)")
print("good photos that match a removed-board photo (same physical board):", mixed or "none")
print("good photos with a same-board photo in the training set of their own fold:", sorted(leaky))
