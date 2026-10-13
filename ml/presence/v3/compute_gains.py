"""Per-photo colour gains: median colour of the red solder-mask pixels of every aligned photo -> gain that moves it to the median of all photos.
Applying the gain before cutting crops removes most of the exposure / white-balance difference between sessions.
Run: python compute_gains.py   (writes cache\\gains.json)"""
import json

import cv2
import numpy as np

from common import ALIGNED, CACHE, load_meta

meta = load_meta()
meds = {}
for k, m in meta.items():
    if not m["ok"]:
        continue
    im = cv2.imread(str(ALIGNED / f"{k}.png"))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 120) & (v >= 50)
    if int(mask.sum()) < 20000:
        print("too few red pixels:", k, int(mask.sum()))
        continue
    meds[k] = np.median(im[mask], axis=0)
arr = np.stack(list(meds.values()))
target = np.median(arr, axis=0)
print("photos:", len(meds), "target BGR", np.round(target, 1))
print("per-channel median colour: min", np.round(arr.min(0), 0), "max", np.round(arr.max(0), 0))
gains = {k: np.clip(target / m, 0.5, 2.0).tolist() for k, m in meds.items()}
(CACHE / "gains.json").write_text(json.dumps(gains))
# exposure-only: ONE scalar per photo (same gain on B, G, R) from the luminance of the red-mask colour -> no colour cast on solder / silkscreen
luma = lambda c: 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
tl = luma(target)
gains_exp = {k: [float(np.clip(tl / luma(m), 0.6, 1.7))] * 3 for k, m in meds.items()}
(CACHE / "gains_exp.json").write_text(json.dumps(gains_exp))
print("exposure gains: min %.2f max %.2f" % (min(g[0] for g in gains_exp.values()), max(g[0] for g in gains_exp.values())))
(CACHE / "exposure_params.json").write_text(json.dumps({
    "target_bgr": [float(x) for x in target], "target_luma": float(tl), "gain_min": 0.6, "gain_max": 1.7,
    "mask": "HSV (OpenCV 0-180): (H <= 10 or H >= 170) and S >= 120 and V >= 50 on the aligned 20 px/mm photo; median B, G, R; luma = 0.114 B + 0.587 G + 0.299 R",
    "gain": "clip(target_luma / luma, gain_min, gain_max), the same gain on B, G, R"}, indent=1))
print("target luma %.6f" % tl)
worst = sorted(gains.items(), key=lambda kv: -abs(np.log(np.array(kv[1])).sum()))[:5]
print("largest corrections:", [(k, np.round(g, 2).tolist()) for k, g in worst])
