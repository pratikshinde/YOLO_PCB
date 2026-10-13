"""Build presence classification dataset v2 for PCB inspection.

Follows TRAINING_PROMPT_antigravity_v2.md specifications:
- Alignment via product pipeline (predict_photo.py, scripts/golden_compare.py)
- Quality gate (residual <= 0.5 mm)
- Strict split BY BOARD (no board leakage, no photo leakage)
- Synthetic window shifts (1.3-2.5 mm -> missing, train only)
- Tolerance jitter (0.0-0.6 mm -> present, train only)
- Real missing oversampling (~1:4 missing:present in train)
- Real crops only in validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Project Paths
PROJECT_ROOT = Path(r"D:\Projects\YOLO_PCB_Componet")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ML_DIR = PROJECT_ROOT / "ml" / "presence"
DATA_DIR = PROJECT_ROOT / "data" / "presence"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ML_DIR))

import golden_compare as gc  # noqa: E402
import predict_photo as pp    # noqa: E402

DEFAULT_GOLDEN = DATA_DIR / "golden" / "red" / "grf.jpeg"
REGIONS = PROJECT_ROOT / "android_app" / "assets" / "golden" / "regions.json"
BOARD_GROUPS = ML_DIR / "board_groups.json"

CROPS_V2_DIR = ML_DIR / "crops_v2"
ALL_CROPS_DIR = CROPS_V2_DIR / "all"
INDEX_CSV_PATH = CROPS_V2_DIR / "index.csv"

DATASETS_V2_DIR = ML_DIR / "datasets_v2"
REPORT_V2_DIR = ML_DIR / "report_v2"
SHEETS_DIR = REPORT_V2_DIR / "sheets"


def get_part_type(name: str) -> str:
    """Classify component into one of the 8 canonical types."""
    if name.startswith("R2K2_"):
        return "R2K2"
    elif name.startswith("C100N_"):
        return "C100N"
    elif name.startswith("D4148_"):
        return "D4148"
    elif name.startswith("R"):
        return "R"
    elif name.startswith("C"):
        return "C"
    elif name.startswith("D"):
        return "D"
    elif name.startswith("Q"):
        return "Q"
    elif name.startswith("U"):
        return "U"
    return "UNKNOWN"


def parse_strictly_validated_missing(folder_path: Path, valid_part_names: set[str]) -> set[str]:
    """Parse missing parts from *.txt* strictly against the 86 valid part names."""
    txt_files = list(folder_path.glob("*.txt*"))
    if not txt_files:
        return set()

    missing_parts = set()
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8")
        tokens = [t.strip() for t in re.split(r"[,;\s\n\r]+", content) if t.strip() and not t.strip().startswith("#")]
        for token in tokens:
            token_upper = token.upper()
            match = None
            for valid in valid_part_names:
                if valid.upper() == token_upper:
                    match = valid
                    break
            if match is not None:
                missing_parts.add(match)
    return missing_parts


def create_contact_sheet(
    crops: list[tuple[str, str, np.ndarray]], out_path: Path, title: str
):
    """Save an 86-part contact sheet grid with names and labels, framing missing in red."""
    cols = 10
    rows = 9
    cell_w = 140
    cell_h = 160
    margin = 10
    header_h = 40

    sheet_w = cols * cell_w + 2 * margin
    sheet_h = rows * cell_h + 2 * margin + header_h
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Draw header
    cv2.putText(
        canvas, title, (margin + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    for idx, (part_name, label, crop_img) in enumerate(crops):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * cell_w
        y0 = margin + header_h + r * cell_h

        cx0 = x0 + 6
        cy0 = y0 + 4
        canvas[cy0 : cy0 + 128, cx0 : cx0 + 128] = crop_img

        is_missing = (label == "missing")
        border_color = (0, 0, 255) if is_missing else (60, 180, 60)
        border_thick = 3 if is_missing else 1

        cv2.rectangle(
            canvas, (cx0 - 1, cy0 - 1), (cx0 + 128, cy0 + 128), border_color, border_thick
        )

        tag = f"{part_name} {'[M]' if is_missing else '[OK]'}"
        text_color = (0, 0, 255) if is_missing else (200, 200, 200)
        cv2.putText(
            canvas, tag, (x0 + 8, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


# Folds definition: Leave-boards-out cross-validation
# Every removal board in a fold, each fold validates on 2 removal boards, 4-5 good boards.
FOLDS = {
    "fold_1": {
        "val_removal": ["m_01", "m_07"],
        "val_good": ["G01", "P1", "P2", "P3"],
    },
    "fold_2": {
        "val_removal": ["m_02", "m_08"],
        "val_good": ["G02", "P4", "P5", "P6"],
    },
    "fold_3": {
        "val_removal": ["m_03", "m_09"],
        "val_good": ["G03", "P7", "P8", "P9"],
    },
    "fold_4": {
        "val_removal": ["m_04", "m_06"],
        "val_good": ["G04", "G05", "P10", "P11"],
    },
    "fold_5": {
        "val_removal": ["m_05", "m_10"],
        "val_good": ["G06", "P12", "P13", "P14", "P15"],
    },
}


def process_photo_crops(photo_tuple):
    """Align a single photo, compute optical-flow shifts, cut all 86 crops."""
    board_id, photo_path, missing_parts_list, is_test = photo_tuple
    P = pp.make_params()
    ref = gc.prepare_reference(gc.imread(DEFAULT_GOLDEN), P, gc.load_regions(REGIONS, P))
    part_regions = [r for r in ref.regions if r.kind == "part"]

    img = gc.imread(photo_path)
    try:
        aligned, valid, al, retried = pp.align(img, ref, P)
        residual_mm = float(al.residual_mm)
        rg, ng, _ = gc.comparison_grays(ref.clahe, gc.clahe_gray(aligned), np.bitwise_and(ref.area, valid), P)
        shifts = gc.region_shifts(gc.local_flow(rg, ng), ref, ref.regions, P)

        missing_set = set(missing_parts_list)
        crop_items = []
        for r in part_regions:
            x, y, w, h = r.bbox_px
            dx, dy = shifts.get(r.id, (0.0, 0.0))
            side = int(round(max(w, h) + 2 * 1.5 * P.px_per_mm))
            crop_bgr = pp.square_crop(aligned, x + w / 2.0 + dx, y + h / 2.0 + dy, side)

            label = "missing" if r.name in missing_set else "present"
            part_type = get_part_type(r.name)

            crop_items.append({
                "part": r.name,
                "part_type": part_type,
                "label": label,
                "dx": round(float(dx), 3),
                "dy": round(float(dy), 3),
                "residual_mm": round(residual_mm, 4),
                "crop_bgr": crop_bgr,
                "cx": x + w / 2.0 + dx,
                "cy": y + h / 2.0 + dy,
                "side": side,
            })

        return {
            "success": True,
            "board": board_id,
            "photo_stem": photo_path.stem,
            "photo_name": photo_path.name,
            "photo_path": str(photo_path),
            "residual_mm": residual_mm,
            "crop_items": crop_items,
            "aligned_img": aligned,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "board": board_id,
            "photo_stem": photo_path.stem,
            "photo_name": photo_path.name,
            "photo_path": str(photo_path),
            "residual_mm": None,
            "crop_items": None,
            "aligned_img": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Build presence dataset v2.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=== Step 1: Initialize Paths and Directories ===")
    ALL_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_V2_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_V2_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load 86 part names
    with open(REGIONS, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    valid_parts = {r["name"] for r in reg_data["regions"] if r.get("kind") == "part"}
    print(f"Loaded {len(valid_parts)} valid SMT part names.")

    # Load board groups for old good photos
    with open(BOARD_GROUPS, "r", encoding="utf-8") as f:
        bg = json.load(f)
    old_good_map = {}
    for g_id, plist in bg["good_boards_red"].items():
        for p in plist:
            old_good_map[p] = g_id

    # Parse removal board missing files
    removed_missing = {}
    removed_dir = DATA_DIR / "raw" / "red" / "removed"
    for b_dir in sorted(removed_dir.iterdir()):
        if b_dir.is_dir():
            m_parts = parse_strictly_validated_missing(b_dir, valid_parts)
            removed_missing[b_dir.name] = m_parts
            print(f"Removal board {b_dir.name}: {len(m_parts)} valid missing parts -> {sorted(m_parts)}")

    # Collect non-test photos to process
    photo_tasks = []
    # 1. Old good photos
    good_dir = DATA_DIR / "raw" / "red" / "good"
    for p in sorted(good_dir.glob("R_*.jpg")):
        b_id = old_good_map[p.stem]
        photo_tasks.append((b_id, p, [], False))

    # 2. New good photos in P1..P15
    for s_dir in sorted(good_dir.iterdir()):
        if s_dir.is_dir():
            for p in sorted(s_dir.glob("*.jpg")):
                photo_tasks.append((s_dir.name, p, [], False))

    # 3. Removal boards m_01..m_10
    for b_dir in sorted(removed_dir.iterdir()):
        if b_dir.is_dir():
            m_list = sorted(removed_missing[b_dir.name])
            for p in sorted(b_dir.glob("*.jpg")):
                photo_tasks.append((b_dir.name, p, m_list, False))

    print(f"Collected {len(photo_tasks)} non-test photos for dataset building.")

    print(f"\n=== Step 2: Parallel Alignment and Crop Extraction ({args.workers} workers) ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(process_photo_crops, photo_tasks))

    print(f"Completed extraction in {time.time() - t0:.1f}s.")

    # Quality Gate and Skipped Logging
    skipped = []
    usable_results = []
    for r in results:
        if not r["success"]:
            skipped.append({
                "photo": r["photo_name"],
                "path": r["photo_path"],
                "board": r["board"],
                "reason": r["error"]
            })
        elif r["residual_mm"] > 0.5:
            skipped.append({
                "photo": r["photo_name"],
                "path": r["photo_path"],
                "board": r["board"],
                "reason": f"Residual {r['residual_mm']:.3f} mm exceeded 0.5 mm quality gate"
            })
        else:
            usable_results.append(r)

    print(f"Usable photos passing quality gate: {len(usable_results)} / {len(results)}")
    print(f"Skipped / Failed photos: {len(skipped)}")

    # Update skipped_photos.json with test skipped photo included
    test_skipped = {
        "photo": "IMG_20261010_130118_086_HDR.jpg",
        "path": str(DATA_DIR / "test" / "red" / "IMG_20261010_130118_086_HDR.jpg"),
        "board": "TEST",
        "reason": "Board not completely visible (aspect 1.63, expected 1.40) - retake"
    }
    all_skipped = skipped + [test_skipped]
    with open(REPORT_V2_DIR / "skipped_photos.json", "w", encoding="utf-8") as f:
        json.dump(all_skipped, f, indent=2)

    print("\n=== Step 3: Saving Real Crops and Building Index CSV ===")
    index_rows = []
    # Photo-to-crops dictionary for assembling datasets
    # photo_info: { (board, photo_stem): [ {part, label, crop_path, crop_bgr, cx, cy, side, aligned_img} ] }
    board_photo_crops = defaultdict(list)
    sheet_candidates = {}

    for res in usable_results:
        b_id = res["board"]
        ph_stem = res["photo_stem"]
        aligned_im = res["aligned_img"]

        sheet_crop_list = []
        for item in res["crop_items"]:
            part = item["part"]
            label = item["label"]
            crop_filename = f"{b_id}__{ph_stem}__{part}.png"
            crop_path = ALL_CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), item["crop_bgr"])

            index_rows.append({
                "board": b_id,
                "photo": ph_stem,
                "part": part,
                "part_type": item["part_type"],
                "label": label,
                "step": "none",
                "dx": item["dx"],
                "dy": item["dy"],
                "residual_mm": item["residual_mm"],
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
            })

            board_photo_crops[b_id].append({
                "board": b_id,
                "photo": ph_stem,
                "part": part,
                "part_type": item["part_type"],
                "label": label,
                "crop_path": crop_path,
                "cx": item["cx"],
                "cy": item["cy"],
                "side": item["side"],
                "aligned_img": aligned_im,
            })

            sheet_crop_list.append((part, label, item["crop_bgr"]))

        # Keep one or two contact sheets per board
        if b_id not in sheet_candidates:
            sheet_candidates[b_id] = (sheet_crop_list, f"{b_id} - {ph_stem}")

    # Write index.csv
    with open(INDEX_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["board", "photo", "part", "part_type", "label", "step", "dx", "dy", "residual_mm", "crop_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Saved {len(index_rows)} real crops to {ALL_CROPS_DIR} and indexed in {INDEX_CSV_PATH}")

    # Write contact sheets
    print("\n=== Generating Contact Sheets for Inspection ===")
    for b_id, (crops_list, title) in sheet_candidates.items():
        sheet_path = SHEETS_DIR / f"{b_id}__contact_sheet.png"
        create_contact_sheet(crops_list, sheet_path, title)
    print(f"Saved {len(sheet_candidates)} contact sheets to {SHEETS_DIR}")

    print("\n=== Step 4: Assembling Cross-Validation Folds and Final Dataset ===")
    # Setup all datasets: fold_1 .. fold_5 and final
    all_boards = sorted(board_photo_crops.keys())

    dataset_configs = {}
    for fold_name, fold_info in FOLDS.items():
        val_b = set(fold_info["val_removal"] + fold_info["val_good"])
        train_b = set(all_boards) - val_b
        dataset_configs[fold_name] = {
            "train_boards": sorted(train_b),
            "val_boards": sorted(val_b),
        }

    # Final model dataset: all boards in train, dummy val from fold_1 val
    dataset_configs["final"] = {
        "train_boards": sorted(all_boards),
        "val_boards": sorted(set(FOLDS["fold_1"]["val_removal"] + FOLDS["fold_1"]["val_good"])),
    }

    P = pp.make_params()
    px_per_mm = P.px_per_mm  # 20.0

    for ds_name, cfg in dataset_configs.items():
        print(f"\nBuilding dataset: {ds_name}")
        ds_dir = DATASETS_V2_DIR / ds_name
        if ds_dir.exists():
            shutil.rmtree(ds_dir)

        train_missing_dir = ds_dir / "train" / "missing"
        train_present_dir = ds_dir / "train" / "present"
        val_missing_dir = ds_dir / "val" / "missing"
        val_present_dir = ds_dir / "val" / "present"

        for d in (train_missing_dir, train_present_dir, val_missing_dir, val_present_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. Validation split: REAL crops only, no copies, no synthetic
        val_crops_count = 0
        for b in cfg["val_boards"]:
            for item in board_photo_crops[b]:
                tgt = val_missing_dir if item["label"] == "missing" else val_present_dir
                shutil.copy2(item["crop_path"], tgt / item["crop_path"].name)
                val_crops_count += 1

        # 2. Training split:
        real_missing_items = []
        real_present_items = []

        for b in cfg["train_boards"]:
            for item in board_photo_crops[b]:
                if item["label"] == "missing":
                    real_missing_items.append(item)
                    shutil.copy2(item["crop_path"], train_missing_dir / item["crop_path"].name)
                else:
                    real_present_items.append(item)
                    shutil.copy2(item["crop_path"], train_present_dir / item["crop_path"].name)

        n_real_missing = len(real_missing_items)
        n_real_present = len(real_present_items)

        # 2a. Oversample real missing crops to ~1 : 4 missing : present
        # Target missing = n_real_present // 4
        target_missing = max(n_real_missing, n_real_present // 4)
        rep_needed = target_missing - n_real_missing
        if n_real_missing > 0 and rep_needed > 0:
            rep_cycle = (rep_needed // n_real_missing) + 1
            added = 0
            for r_idx in range(1, rep_cycle + 1):
                for item in real_missing_items:
                    if added >= rep_needed:
                        break
                    rep_name = f"{item['crop_path'].stem}__rep{r_idx}.png"
                    shutil.copy2(item["crop_path"], train_missing_dir / rep_name)
                    added += 1

        # 2b. Synthetic window-shift crops (1.3-2.5 mm -> missing)
        # only from PRESENT parts on TRAINING boards, at most as many as real missing crops
        n_syn_shifts = min(n_real_missing, len(real_present_items))
        if n_syn_shifts > 0:
            sample_present = random.sample(real_present_items, n_syn_shifts)
            for idx, item in enumerate(sample_present):
                dist_mm = random.uniform(1.3, 2.5)
                angle = random.uniform(0, 2 * math.pi)
                shift_x = dist_mm * math.cos(angle) * px_per_mm
                shift_y = dist_mm * math.sin(angle) * px_per_mm

                syn_crop = pp.square_crop(
                    item["aligned_img"],
                    item["cx"] + shift_x,
                    item["cy"] + shift_y,
                    item["side"]
                )
                syn_name = f"{item['board']}__{item['photo']}__{item['part']}__syn_shift{idx:04d}.png"
                cv2.imwrite(str(train_missing_dir / syn_name), syn_crop)

        # 2c. Tolerance jitter for present (0.0-0.6 mm -> present)
        # only from TRAINING boards, ~20% of present training crops (no shift between 0.6 and 1.3 mm)
        n_jitter = int(round(0.20 * n_real_present))
        if n_jitter > 0:
            sample_jitter = random.sample(real_present_items, n_jitter)
            for idx, item in enumerate(sample_jitter):
                dist_mm = random.uniform(0.0, 0.6)
                angle = random.uniform(0, 2 * math.pi)
                shift_x = dist_mm * math.cos(angle) * px_per_mm
                shift_y = dist_mm * math.sin(angle) * px_per_mm

                jitter_crop = pp.square_crop(
                    item["aligned_img"],
                    item["cx"] + shift_x,
                    item["cy"] + shift_y,
                    item["side"]
                )
                jitter_name = f"{item['board']}__{item['photo']}__{item['part']}__jitter{idx:04d}.png"
                cv2.imwrite(str(train_present_dir / jitter_name), jitter_crop)

        total_tr_missing = len(list(train_missing_dir.glob("*.png")))
        total_tr_present = len(list(train_present_dir.glob("*.png")))
        total_va_missing = len(list(val_missing_dir.glob("*.png")))
        total_va_present = len(list(val_present_dir.glob("*.png")))

        print(f"  Train: missing={total_tr_missing} (real {n_real_missing}, syn {n_syn_shifts}, oversampled), present={total_tr_present} (real {n_real_present}, jitter {n_jitter}) -> ratio 1:{total_tr_present / max(1, total_tr_missing):.1f}")
        print(f"  Val:   missing={total_va_missing}, present={total_va_present} (all real)")

    print("\n=== Step 5: Running Split Audit ===")
    run_split_audit()


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_split_audit():
    """Audit datasets_v2 across all folds and generate report_v2/split_audit.md."""
    print("Auditing folds for strict isolation...")
    audit_lines = [
        "# Cross-Validation Split Audit (v2)\n",
        "Date: 2026-10-10  ",
        "Audit Method: Exhaustive file-by-file, board-by-board, and byte-level MD5 collision verification.\n",
        "## Summary of Audit Rules",
        "1. **Split strictly by physical board**: photos of one physical board are never divided between train and val.",
        "2. **0 shared photos** between train and val in any fold.",
        "3. **0 byte-identical crops** between train and val in any fold.",
        "4. **0 synthetic or replicated crops** in any validation set (validation is 100% real crops).",
        "5. **Test boards isolated**: `data/presence/test/red/` is never in any fold or final training set.\n",
        "---\n",
        "## Fold-by-Fold Audit Results\n",
    ]

    all_pass = True
    for k in range(1, 6):
        fold_name = f"fold_1" if k == 1 else f"fold_{k}"
        f_dir = DATASETS_V2_DIR / fold_name

        tr_files = list((f_dir / "train").glob("*/*.png"))
        va_files = list((f_dir / "val").glob("*/*.png"))

        def parse_info(f):
            parts = f.name[:-4].split("__")
            board = parts[0]
            photo = parts[1]
            part = parts[2]
            flags = parts[3:] if len(parts) > 3 else []
            cls = f.parent.name
            return board, photo, part, flags, cls

        tr_records = [parse_info(f) for f in tr_files]
        va_records = [parse_info(f) for f in va_files]

        tr_photos = {(r[0], r[1]) for r in tr_records}
        va_photos = {(r[0], r[1]) for r in va_records}
        shared_photos = tr_photos & va_photos

        tr_boards = {r[0] for r in tr_records}
        va_boards = {r[0] for r in va_records}
        shared_boards = tr_boards & va_boards

        val_flags = sum(1 for r in va_records if r[3])

        # MD5 check
        tr_hashes = {md5_file(f) for f in tr_files}
        va_hashes = {md5_file(f) for f in va_files}
        shared_hashes = tr_hashes & va_hashes

        va_missing_parts = sorted({r[2] for r in va_records if r[4] == "missing"})
        tr_missing_parts = sorted({r[2] for r in tr_records if r[4] == "missing" and not r[3]})

        val_rem_boards = sorted(FOLDS[fold_name]["val_removal"])
        val_good_boards = sorted(FOLDS[fold_name]["val_good"])

        fold_pass = (
            len(shared_photos) == 0
            and len(shared_boards) == 0
            and len(shared_hashes) == 0
            and val_flags == 0
        )
        if not fold_pass:
            all_pass = False

        audit_lines.append(f"### {fold_name}")
        audit_lines.append(f"- **Validation Removal Boards:** `{val_rem_boards}`")
        audit_lines.append(f"- **Validation Good Boards:** `{val_good_boards}`")
        audit_lines.append(f"- **Train Photos:** {len(tr_photos)} | **Val Photos:** {len(va_photos)}")
        audit_lines.append(f"- **Train Crops:** {len(tr_files)} | **Val Crops:** {len(va_files)}")
        audit_lines.append(f"- **Shared Physical Boards:** `{len(shared_boards)}` (Requirement: 0) -> {'PASS' if len(shared_boards) == 0 else 'FAIL'}")
        audit_lines.append(f"- **Shared Photos:** `{len(shared_photos)}` (Requirement: 0) -> {'PASS' if len(shared_photos) == 0 else 'FAIL'}")
        audit_lines.append(f"- **Byte-Identical MD5 Collisions:** `{len(shared_hashes)}` (Requirement: 0) -> {'PASS' if len(shared_hashes) == 0 else 'FAIL'}")
        audit_lines.append(f"- **Val Synthetic/Replicated Crops:** `{val_flags}` (Requirement: 0) -> {'PASS' if val_flags == 0 else 'FAIL'}")
        audit_lines.append(f"- **Held-Out Missing Part Slots:** `{va_missing_parts}`")
        audit_lines.append(f"- **Previously Seen Missing Slots in Train:** `{sorted(set(va_missing_parts) & set(tr_missing_parts))}`")
        audit_lines.append(f"- **Unseen Held-Out Missing Slots (Novel to Val):** `{sorted(set(va_missing_parts) - set(tr_missing_parts))}`")
        audit_lines.append(f"- **Status:** **{'VERIFIED LEAK-FREE' if fold_pass else 'LEAK DETECTED'}**\n")

    audit_lines.append("---\n")
    audit_lines.append("## Overall Isolation Conclusion")
    if all_pass:
        audit_lines.append("**ALL 5 CROSS-VALIDATION FOLDS ARE VERIFIED 100% LEAK-FREE.** No physical boards, photos, or crops are shared between training and validation.")
    else:
        audit_lines.append("**AUDIT FAILED: LEAKAGE FOUND.**")

    report_path = REPORT_V2_DIR / "split_audit.md"
    report_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"Split audit report written to {report_path}. All pass: {all_pass}")


if __name__ == "__main__":
    main()
