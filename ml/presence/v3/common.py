"""Shared helpers of the v3 pipeline: paths, folds, label files + overrides, photo list, crop cutting."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\Projects\YOLO_PCB_Componet")
ML = ROOT / "ml" / "presence"
V3 = ML / "v3"
DATA = ROOT / "data" / "presence"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ML))
import golden_compare as gc  # noqa: E402
import predict_photo as pp   # noqa: E402

REGIONS = ROOT / "android_app" / "assets" / "golden" / "regions.json"
CACHE = V3 / "cache"
ALIGNED = CACHE / "aligned"
META = CACHE / "meta.json"
OVERRIDES = V3 / "label_overrides.json"
PX_PER_MM = 20.0

# Leave-boards-out folds (as in make_dataset_v2.py: linked removal boards = continued boards stay together)
FOLDS = {
    "fold_1": {"val_removal": ["m_01", "m_07"], "val_good": ["G01", "P1", "P2", "P3"]},
    "fold_2": {"val_removal": ["m_02", "m_08"], "val_good": ["G02", "P4", "P5", "P6"]},
    "fold_3": {"val_removal": ["m_03", "m_09"], "val_good": ["G03", "P7", "P8", "P9"]},
    "fold_4": {"val_removal": ["m_04", "m_06"], "val_good": ["G04", "G05", "P10", "P11"]},
    "fold_5": {"val_removal": ["m_05", "m_10"], "val_good": ["G06", "P12", "P13", "P14", "P15"]},
}


def ptype(name: str) -> str:
    for pre in ("R2K2_", "C100N_", "D4148_"):
        if name.startswith(pre):
            return pre[:-1]
    return re.match(r"[A-Z]+", name).group(0)


def valid_parts() -> list[str]:
    reg = json.loads(REGIONS.read_text(encoding="utf-8"))
    return [r["name"] for r in reg["regions"] if r.get("kind") == "part"]


def read_label_tokens(folder: Path) -> list[str]:
    out = []
    for t in sorted(folder.glob("*.txt*")):
        text = re.sub(r"\([^)]*\)", " ", t.read_text(encoding="utf-8-sig"))      # drop "(comments)"
        out += [x for x in re.split(r"[,;\s]+", text) if x and not x.startswith("#")]
    return out


def corrected_missing(board: str, folder: Path, valid: list[str]) -> set[str]:
    """Parts missing on a removal board = text file (strict) + overrides. Unknown tokens must be declared in the overrides."""
    ov = json.loads(OVERRIDES.read_text(encoding="utf-8"))["boards"].get(board, {})
    vmap = {v.upper(): v for v in valid}
    ignore = set(ov.get("ignore_tokens", []))
    missing = set()
    for tok in read_label_tokens(folder):
        if tok in ignore:
            continue
        if tok.upper() not in vmap:
            raise ValueError(f"{board}: token {tok!r} is not a part name and not declared in label_overrides.json")
        missing.add(vmap[tok.upper()])
    missing |= set(ov.get("add_missing", []))
    missing -= set(ov.get("remove_missing", []))
    bad = [m for m in missing if m not in vmap.values()]
    if bad:
        raise ValueError(f"{board}: unknown part in overrides {bad}")
    return missing


def photo_list() -> list[tuple[str, Path, str]]:
    """(board, path, kind) of every training / validation photo (test photos are separate)."""
    bg = json.loads((ML / "board_groups.json").read_text(encoding="utf-8"))
    old = {p: g for g, pl in bg["good_boards_red"].items() for p in pl}
    out = []
    good = DATA / "raw" / "red" / "good"
    for p in sorted(good.glob("R_*.jpg")):
        out.append((old[p.stem], p, "good"))
    for d in sorted(x for x in good.iterdir() if x.is_dir()):
        out += [(d.name, p, "good") for p in sorted(d.glob("*.jpg"))]
    rem = DATA / "raw" / "red" / "removed"
    for d in sorted(x for x in rem.iterdir() if x.is_dir()):
        out += [(d.name, p, "removed") for p in sorted(d.glob("*.jpg"))]
    return out


def test_photos() -> list[Path]:
    return sorted((DATA / "test" / "red").glob("*.jpg"))


def load_meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8"))


def cut(aligned: np.ndarray, part: dict, margin_mm: float, shift_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """128x128 crop of one part slot: window = max(w,h) + 2*margin, centred on box centre + local flow shift (+ optional window shift)."""
    side = int(round(max(part["w"], part["h"]) + 2 * margin_mm * PX_PER_MM))
    cx = part["x"] + part["w"] / 2.0 + part["dx"] + shift_mm[0] * PX_PER_MM
    cy = part["y"] + part["h"] / 2.0 + part["dy"] + shift_mm[1] * PX_PER_MM
    return pp.square_crop(aligned, cx, cy, side)


def key(board: str, stem: str) -> str:
    return f"{board}__{stem}"


_gains = {}


def norm_tag(norm) -> str:
    return {True: "n", "gain": "n", "exp": "e"}.get(norm, "")


def load_aligned(k: str, norm=False) -> np.ndarray:
    """aligned photo (BGR). norm=True / 'gain': per-channel colour gain (white balance + exposure); norm='exp': one exposure gain per photo.
    Gains come from compute_gains.py."""
    im = cv2.imread(str(ALIGNED / f"{k}.png"))
    tag = norm_tag(norm)
    if tag:
        if tag not in _gains:
            _gains[tag] = json.loads((CACHE / ("gains.json" if tag == "n" else "gains_exp.json")).read_text())
        im = np.clip(im.astype(np.float32) * np.array(_gains[tag][k], np.float32), 0, 255).astype(np.uint8)
    return im
