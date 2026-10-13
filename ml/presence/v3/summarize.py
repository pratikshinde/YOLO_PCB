"""Summaries of cross-validation results.
  python summarize.py E0_base            -> results\\E0_base_summary.md (+ flagged-label contact sheet)
  python summarize.py E0_base E1_small   -> also results\\compare.md
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import cv2
import numpy as np

from common import V3

RES = V3 / "results"
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# physical removal boards: m_0x pairs are the same board photographed again after removing more parts (m_05 and m_10 are different boards)
PHYS = {"m_01": "A", "m_07": "A", "m_02": "B", "m_08": "B", "m_03": "C", "m_09": "C", "m_04": "D", "m_06": "D", "m_05": "E", "m_10": "F"}


def load_preds(exp):
    rows = list(csv.DictReader(open(RES / f"{exp}_preds.csv")))
    for r in rows:
        r["p"] = float(r["p_missing"])
    return rows


def load_shift(exp):
    d = defaultdict(list)
    for r in csv.DictReader(open(RES / f"{exp}_shift.csv")):
        d[float(r["shift_mm"])].append(float(r["p_missing"]))
    return {k: np.array(v) for k, v in d.items()}


def choose_tau(pm, pp_):
    """largest-margin threshold with every missing part found (FN = 0) and the fewest false calls; ties -> middle of the feasible range"""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    feas = [(int((pp_ >= t).sum()), t) for t in grid if (pm >= t).all()]
    if not feas:
        return 0.5, None
    best = min(f for f, _ in feas)
    ts = [t for f, t in feas if f == best]
    return float(ts[len(ts) // 2]), best


def metrics(exp, tau=0.5):
    rows = load_preds(exp)
    pm = np.array([r["p"] for r in rows if r["label"] == "missing"])
    pp_ = np.array([r["p"] for r in rows if r["label"] == "present"])
    out = {"exp": exp, "n_missing": len(pm), "n_present": len(pp_), "tau": tau,
           "FN": int((pm < tau).sum()), "FP": int((pp_ >= tau).sum()), "min_missing": float(pm.min()), "max_present": float(pp_.max()),
           "p999_present": float(np.percentile(pp_, 99.9))}
    by = defaultdict(list)
    for r in rows:
        by[(r["board"], r["photo"])].append(r)
    good_fail = good_n = rem_fail = rem_n = rem_exact = 0
    for (b, ph), v in by.items():
        flagged = {x["part"] for x in v if x["p"] >= tau}
        truth = {x["part"] for x in v if x["label"] == "missing"}
        if b in PHYS:
            rem_n += 1
            rem_fail += bool(flagged)
            rem_exact += flagged == truth
        else:
            good_n += 1
            good_fail += bool(flagged)
    out.update(good_photos=good_n, good_false_fail=good_fail, removed_photos=rem_n, removed_fail=rem_fail, removed_exact=rem_exact)
    try:
        sh = load_shift(exp)
        for s, a in sh.items():
            out[f"shift_{s}"] = float((a >= tau).mean())
    except FileNotFoundError:
        pass
    nr = [r for r in rows if r["label"] == "missing" and r["type"] != "R"]
    out["nonR_total"] = len(nr)
    out["nonR_found"] = sum(r["p"] >= tau for r in nr)
    try:
        for v, (sm, sp) in load_stress(exp).items():
            out[f"stress_{v}_FN"] = int((sm < tau).sum())
            out[f"stress_{v}_FP"] = int((sp >= tau).sum())
    except FileNotFoundError:
        pass
    return out


def load_stress(exp):
    d = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(RES / f"{exp}_stress.csv")):
        d[r["variant"]][0 if r["label"] == "missing" else 1].append(float(r["p_missing"]))
    return {k: (np.array(a), np.array(b)) for k, (a, b) in d.items()}


def flagged_sheet(exp, rows, out_path, max_items=24):
    groups = defaultdict(list)
    for r in rows:
        if (r["label"] == "present" and r["p"] >= 0.5) or (r["label"] == "missing" and r["p"] < 0.5):
            groups[(r["board"], r["part"], r["label"])].append(r)
    for r in rows:        # good boards: any slot at 0.3 or more
        if r["board"] not in PHYS and r["label"] == "present" and 0.3 <= r["p"] < 0.5:
            groups[(r["board"], r["part"], "good-doubtful")].append(r)
    tiles = []
    T = 176
    for (b, part, kind), v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_items]:
        r = v[0]
        f = V3 / "datasets" / exp / r["fold"] / "val" / ("missing" if r["label"] == "missing" else "present") / f"{b}__{r['photo']}__{part}.png"
        im = cv2.imread(str(f))
        if im is None:
            continue
        im = cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(im, (0, 0), (T, 34), (255, 255, 255), -1)
        cv2.putText(im, f"{b} {part} {kind}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im, f"p={r['p']:.2f} in {len(v)} photo(s)", (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160), 1, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return 0
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full((T, T, 3), 255, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(groups)


def summary(exp):
    rows = load_preds(exp)
    pm = np.array([r["p"] for r in rows if r["label"] == "missing"])
    pp_ = np.array([r["p"] for r in rows if r["label"] == "present"])
    tau, fp_at = choose_tau(pm, pp_)
    L = [f"# {exp}: cross-validation summary (5 folds, leave-boards-out, last epoch)\n"]
    L.append(f"Held-out crops: {len(pm)} missing, {len(pp_)} present. Lowest score on a real missing part {pm.min():.4f}; highest on a present part {pp_.max():.4f}; 99.9th percentile of present {np.percentile(pp_, 99.9):.4f}.")
    L.append(f"Threshold with every missing part found and the fewest false calls (middle of the feasible range): **{tau}** ({fp_at} false calls)\n")
    L.append("| threshold | missed | found | false calls | false-call rate |\n|---|---|---|---|---|")
    for t in TAUS:
        L.append(f"| {t} | {int((pm < t).sum())} | {int((pm >= t).sum())}/{len(pm)} | {int((pp_ >= t).sum())} / {len(pp_)} | {100 * (pp_ >= t).mean():.3f} % |")
    L.append(f"\n## By part type at threshold {tau}\n")
    L.append("| type | present crops | false calls | missing crops | found | physical boards with a real removal | proven? (>= 15 removed and >= 5 boards) |\n|---|---|---|---|---|---|---|")
    types = sorted({r["type"] for r in rows})
    for ty in types:
        pr = [r for r in rows if r["type"] == ty and r["label"] == "present"]
        ms = [r for r in rows if r["type"] == ty and r["label"] == "missing"]
        inst = {(r["board"], r["part"]) for r in ms}
        boards = {PHYS.get(r["board"], r["board"]) for r in ms}
        proven = len(inst) >= 15 and len(boards) >= 5
        L.append(f"| {ty} | {len(pr)} | {sum(r['p'] >= tau for r in pr)} | {len(ms)} | {sum(r['p'] >= tau for r in ms)} | {len(boards)} ({len(inst)} removed part instances) | {'yes' if proven else '**NOT PROVEN**'} |")
    m = metrics(exp, tau)
    L.append(f"\n## Board level at threshold {tau}\n")
    L.append(f"* good-board photos that FAIL by mistake: **{m['good_false_fail']} of {m['good_photos']}**")
    L.append(f"* photos of boards with removed parts that FAIL: {m['removed_fail']} of {m['removed_photos']}; exactly the right part list: {m['removed_exact']} of {m['removed_photos']}")
    try:
        sh = load_shift(exp)
        L.append(f"\n## Window moved against the part (present parts of held-out boards, 4 directions), share called missing at {tau}\n")
        L.append("| shift | share |\n|---|---|\n| 0.0 mm (real position) | %.2f %% |" % (100 * (pp_ >= tau).mean()))
        for s in sorted(sh):
            L.append(f"| {s} mm | {100 * (sh[s] >= tau).mean():.1f} % |")
    except FileNotFoundError:
        pass
    try:
        L.append(f"\n## Phone-quality stress (held-out crops degraded), threshold {tau}\n")
        L.append("| variant | missed | false calls |\n|---|---|---|")
        for v, (sm, sp) in load_stress(exp).items():
            L.append(f"| {v} | {int((sm < tau).sum())} / {len(sm)} | {int((sp >= tau).sum())} / {len(sp)} |")
    except FileNotFoundError:
        pass
    n = flagged_sheet(exp, rows, RES / f"{exp}_flagged.jpg")
    L.append(f"\n## Disagreements between model and text-file label ({n} slot/board groups)\n")
    L.append(f"Contact sheet: `results/{exp}_flagged.jpg` (each tile: board, part, what the label says, score). A confident disagreement is usually a LABEL error: look at the crop.")
    flagged = defaultdict(list)
    for r in rows:
        if (r["label"] == "present" and r["p"] >= 0.5) or (r["label"] == "missing" and r["p"] < 0.5):
            flagged[(r["board"], r["part"], r["label"])].append(r["p"])
    L.append("\n| board | part | text file says | photos | mean score |\n|---|---|---|---|---|")
    for (b, part, lab), v in sorted(flagged.items()):
        L.append(f"| {b} | {part} | {lab} | {len(v)} | {np.mean(v):.2f} |")
    (RES / f"{exp}_summary.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def compare(exps):
    L = ["# Experiment comparison (5-fold leave-boards-out; threshold 0.5 and the best threshold of each)\n",
         "| experiment | missed @0.5 | false calls @0.5 | non-R missing found | min missing score | max present score | good photos failing | removed photos exact | shift 0.9 mm | 1.2 mm | 1.5 mm | 2.0 mm | stress missed (blur/dark/noise/jpeg) | stress false calls |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in exps:
        m = metrics(e, 0.5)
        m3 = metrics(e, 0.3)
        pc = lambda k: f"{100 * m[k]:.1f} %" if k in m else "-"
        sfn = "/".join(str(m.get(f"stress_{v}_FN", "-")) for v in ("blur", "dark", "noise", "jpeg"))
        sfp = "/".join(str(m.get(f"stress_{v}_FP", "-")) for v in ("blur", "dark", "noise", "jpeg"))
        L.append(f"| {e} | {m['FN']} / {m['n_missing']} (@0.3: {m3['FN']} missed, {m3['FP']} false) | {m['FP']} / {m['n_present']} | {m['nonR_found']} / {m['nonR_total']} | {m['min_missing']:.3f} | {m['max_present']:.3f} | {m['good_false_fail']} / {m['good_photos']} | {m['removed_exact']} / {m['removed_photos']} | {pc('shift_0.9')} | {pc('shift_1.2')} | {pc('shift_1.5')} | {pc('shift_2.0')} | {sfn} | {sfp} |")
    (RES / "compare.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__":
    names = sys.argv[1:]
    for n in names:
        summary(n)
    if len(names) > 1:
        print(compare(names))
    else:
        print((RES / f"{names[0]}_summary.md").read_text(encoding="utf-8"))
