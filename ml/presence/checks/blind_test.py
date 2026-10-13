"""Blind end-to-end test of predict_photo.py (the real tool, as a subprocess):
- photos are COPIED to a scratch folder under scrambled names (img_xxxx.jpg): no text files, no folder or name hints
- every photo is judged by the fold model that never saw its board (removed board m_0k -> fold_k; good boards -> fold of their block,
  using only good boards that are NOT also in the training set of that fold - see identity_all.py)
- only AFTER all predictions are collected are the text files read, to score them.

Run:  ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\checks\\blind_test.py      (about 5 minutes)
"""
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
ROOT = ML.parents[1]
PY = ML / ".venv" / "Scripts" / "python.exe"
RAW = ROOT / "data" / "presence" / "raw" / "red"
scratch = Path(__file__).parent / "_scratch"
scratch.mkdir(exist_ok=True)
blind = scratch / f"blind_{int(time.time())}"
blind.mkdir()
random.seed(7)

# good photos whose board is not in the training set of the fold that judges them (from identity_all.py, 2026-10-09)
clean_good = {1: range(1, 8), 2: range(15, 22), 3: range(23, 29), 4: range(38, 44)}
plan = {}
for k in (1, 2, 3, 4):
    folder = blind / f"fold_{k}"
    folder.mkdir()
    truth = {}
    srcs = [(p, "removed") for p in sorted((RAW / "removed" / f"m_{k:02d}").glob("*.jpg"))]
    srcs += [(RAW / "good" / f"R_{n:02d}.jpg", "good") for n in clean_good[k]]
    random.shuffle(srcs)
    for p, kind in srcs:
        name = f"img_{random.randrange(10000):04d}.jpg"
        while (folder / name).exists():
            name = f"img_{random.randrange(10000):04d}.jpg"
        shutil.copy2(p, folder / name)
        truth[name] = (kind, str(p))
    assert all(f.suffix == ".jpg" for f in folder.iterdir()), "only jpg files may be in the blind folder"
    plan[k] = truth
print("blind folders:", {k: len(v) for k, v in plan.items()}, "| contents are only scrambled .jpg files")

pred = {}
for k in (1, 2, 3, 4):
    out = blind / f"out_{k}"
    r = subprocess.run([str(PY), "-I", str(ML / "predict_photo.py"), str(blind / f"fold_{k}"), "--model", str(ML / "runs" / f"fold_{k}" / "weights" / "best.pt"), "--no-norm", "--windows", "1", "--threshold", "0.5", "--doubt", "0.2", "--out", str(out)],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = re.match(r"^(img_\d+\.jpg): (PASS|FAIL|RETAKE) - (.*)$", line)
        if m:
            names = set(re.findall(r"([A-Z][A-Z0-9_]*?) \(\d\.\d\d\)", m.group(3))) if m.group(2) == "FAIL" else set()
            pred[(k, m.group(1))] = (m.group(2), names)
    print(f"fold {k}: {sum(1 for (kk, _) in pred if kk == k)} photos judged", flush=True)


def labels(folder):             # read only now, after the predictions exist
    out = set()
    for t in folder.glob("*.txt*"):
        out |= {x.strip().upper() for x in re.split(r"[,;\s]+", t.read_text(encoding="utf-8-sig")) if x.strip()}
    return out


tot = {"removed": [0, 0, 0], "good": [0, 0, 0]}      # photos, decision right, exact part list right
bad = []
for k, truth in plan.items():
    miss = labels(RAW / "removed" / f"m_{k:02d}")
    for name, (kind, src) in truth.items():
        verdict, names = pred.get((k, name), ("NONE", set()))
        want = miss if kind == "removed" else set()
        want_v = "FAIL" if kind == "removed" else "PASS"
        t = tot[kind]
        t[0] += 1
        t[1] += verdict == want_v
        ok = names == {w.upper() for w in want}
        t[2] += ok and verdict == want_v
        if verdict != want_v or not ok:
            bad.append((k, kind, Path(src).name, verdict, sorted(names), sorted(want)))
print()
for kind, (n, dec, exact) in tot.items():
    print(f"{kind:8s}: {n} photos | right board decision {dec}/{n} | exactly the right part list {exact}/{n}")
print("mismatches:", bad or "none")
json.dump({str(k): v for k, v in plan.items()}, open(blind / "mapping_scrambled_to_original.json", "w"), indent=1)
