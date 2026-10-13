"""Does taking 2 or 3 photos of the same board and combining them reduce the errors?  (uses the held-out predictions only)
  python votes.py E0_base [tau]
A board of the held-out folds has several photos. For every combination of k photos of the SAME board: a part counts as missing when it is
flagged in ALL k photos (AND), in AT LEAST ONE (OR) or in the MAJORITY.
"""
import csv
import itertools
import sys
from collections import defaultdict

import numpy as np

from common import V3

RES = V3 / "results"
exp = sys.argv[1]
tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
rows = list(csv.DictReader(open(RES / f"{exp}_preds.csv")))
board = defaultdict(lambda: defaultdict(dict))        # board -> photo -> part -> (label, flagged)
for r in rows:
    board[r["board"]][r["photo"]][r["part"]] = (r["label"] == "missing", float(r["p_missing"]) >= tau)

L = [f"# {exp}: combining photos of one board (threshold {tau})\n", "| photos | rule | part-level missed | part-level false calls | good-board combinations failing | removal-board combinations with exactly the right list |", "|---|---|---|---|---|---|"]
for k in (1, 2, 3):
    for rule in (("single photo",) if k == 1 else ("AND (all flag it)", "OR (any flags it)", "majority")):
        fn = fp = n_m = n_p = gf = gn = rx = rn = 0
        for b, photos in board.items():
            names = sorted(photos)
            if len(names) < k:
                continue
            for combo in itertools.combinations(names, k):
                parts = set.intersection(*[set(photos[p]) for p in combo])
                flagged, truth = set(), set()
                for part in parts:
                    votes = [photos[p][part][1] for p in combo]
                    is_m = photos[combo[0]][part][0]
                    f = (votes[0] if k == 1 else all(votes) if rule.startswith("AND") else any(votes) if rule.startswith("OR") else sum(votes) * 2 > k)
                    if is_m:
                        n_m += 1
                        fn += not f
                        truth.add(part)
                    else:
                        n_p += 1
                        fp += f
                    if f:
                        flagged.add(part)
                if truth:
                    rn += 1
                    rx += flagged == truth
                else:
                    gn += 1
                    gf += bool(flagged)
        L.append(f"| {k} | {rule} | {fn} / {n_m} | {fp} / {n_p} | {gf} / {gn} | {rx} / {rn} |")
(RES / f"{exp}_votes.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
