"""Read-only audit of ml/presence/datasets: is anything of a validation photo / board / crop also in the training set of the same fold?

Run:  ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\checks\\audit_split.py
Checks per fold: photos in both train and val, synthetic / repeated crops in val, byte-identical crops in both, and whether the slots that are
missing on the held-out board were ever missing on a training board.
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parents[1] / "datasets"


def parse(name):
    parts = name[:-4].split("__")
    return parts[0], parts[1], parts[2], parts[3:]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def block_of(photo):
    n = int(re.search(r"R_(\d+)", photo).group(1))
    return 1 if n <= 11 else 2 if n <= 22 else 3 if n <= 33 else 4


for k in range(1, 5):
    rec = {"train": [], "val": []}
    for split in ("train", "val"):
        for cls in ("missing", "present"):
            for f in (DS / f"fold_{k}" / split / cls).glob("*.png"):
                b, ph, part, flags = parse(f.name)
                rec[split].append({"board": b, "photo": ph, "part": part, "flags": flags, "cls": cls, "file": f})
    tr_ph = {(r["board"], r["photo"]) for r in rec["train"]}
    va_ph = {(r["board"], r["photo"]) for r in rec["val"]}
    print(f"\n=== fold {k}: train crops {len(rec['train'])} from {len(tr_ph)} photos | val crops {len(rec['val'])} from {len(va_ph)} photos")
    print("   photos in BOTH train and val:", len(tr_ph & va_ph))
    print("   boards in train:", sorted({b for b, _ in tr_ph if not b.startswith('good_')}), "| removed board in val:", sorted({b for b, _ in va_ph if not b.startswith('good_')}))
    va_good = sorted(p for b, p in va_ph if b.startswith("good_"))
    tr_good = sorted(p for b, p in tr_ph if b.startswith("good_"))
    print(f"   good photos in val ({len(va_good)}): {va_good[0]}..{va_good[-1]} -> all in block {sorted({block_of(p) for p in va_good})}")
    print(f"   good photos in train ({len(tr_good)}): blocks {sorted(Counter(block_of(p) for p in tr_good).items())}")
    print("   val crops that are synthetic / repeated copies:", sum(1 for r in rec["val"] if r["flags"]))
    h_tr = {md5(r["file"]) for r in rec["train"]}
    h_va = {md5(r["file"]) for r in rec["val"]}
    print("   byte-identical crop files in both train and val:", len(h_tr & h_va))
    slots_tr = sorted({r["part"] for r in rec["train"] if r["cls"] == "missing" and not r["flags"]})
    slots_va = sorted({r["part"] for r in rec["val"] if r["cls"] == "missing"})
    print("   slots really missing on the held-out board:", slots_va)
    print("   ... of which ever missing on a training board:", sorted(set(slots_tr) & set(slots_va)) or "none")
print("\nNOTE: this audit sees photos, not physical boards. Several photos can show ONE board - see identity_all.py.")
