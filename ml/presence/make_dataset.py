"""Build presence classification dataset for PCB inspection using golden_compare.

Follows TRAINING_PROMPT_antigravity.md specifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import existing alignment code without modifying anything outside ml/presence
sys.path.insert(0, r"D:\Projects\YOLO_PCB_Componet\scripts")
import golden_compare as gc

# Configuration paths
PROJECT_ROOT = Path(r"D:\Projects\YOLO_PCB_Componet")
GOLDEN_IMAGE_PATH = PROJECT_ROOT / "data" / "presence" / "golden" / "red" / "grf.jpeg"
REGIONS_JSON_PATH = PROJECT_ROOT / "android_app" / "assets" / "golden" / "regions.json"
GOOD_PHOTOS_DIR = PROJECT_ROOT / "data" / "presence" / "raw" / "red" / "good"
REMOVED_PHOTOS_DIR = PROJECT_ROOT / "data" / "presence" / "raw" / "red" / "removed"

ML_DIR = PROJECT_ROOT / "ml" / "presence"
CROPS_DIR = ML_DIR / "crops"
ALL_CROPS_DIR = CROPS_DIR / "all"
SHEETS_DIR = CROPS_DIR / "sheets"
INDEX_CSV_PATH = CROPS_DIR / "index.csv"
DATASETS_DIR = ML_DIR / "datasets"
REPORT_DIR = ML_DIR / "report"


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
    raise ValueError(f"Unknown component type for part: {name}")


def parse_missing_list(folder_path: Path, valid_part_names: set[str]) -> set[str]:
    """Parse missing parts from any *.txt in removed folder, case-insensitively, strictly validated."""
    txt_files = list(folder_path.glob("*.txt*"))
    if not txt_files:
        raise FileNotFoundError(f"No text files found in {folder_path}")

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
            if match is None:
                raise ValueError(f"Unknown part name '{token}' in {txt_file} against 86 valid parts")
            missing_parts.add(match)
    return missing_parts


def extract_square_crop(image: np.ndarray, cx: float, cy: float, side: int, out_size: int = 128) -> np.ndarray:
    """Extract a square crop centered at (cx, cy) with padding if needed, resized to out_size."""
    H, W = image.shape[:2]
    half = side / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + side
    y2 = y1 + side

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)

    src_y1 = max(0, y1)
    src_y2 = min(H, y2)
    src_x1 = max(0, x1)
    src_x2 = min(W, x2)

    patch = image[src_y1:src_y2, src_x1:src_x2]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

    # Ensure patch is exactly (side, side)
    patch = patch[:side, :side]

    # Resize to out_size x out_size
    resized = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized


def create_contact_sheet(
    crops: list[tuple[str, str, np.ndarray]], out_path: Path, title: str
):
    """Save an 86-part contact sheet grid with names and labels, framing missing in red."""
    # 86 items -> 9 rows x 10 columns (90 slots)
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

        # Place 128x128 crop inside 140x160 cell (leaving 6 px margin on left/right, 26 px for text at bottom)
        cx0 = x0 + 6
        cy0 = y0 + 4
        canvas[cy0 : cy0 + 128, cx0 : cx0 + 128] = crop_img

        is_missing = (label == "missing")
        border_color = (0, 0, 255) if is_missing else (60, 180, 60)
        border_thick = 3 if is_missing else 1

        cv2.rectangle(
            canvas, (cx0 - 1, cy0 - 1), (cx0 + 128, cy0 + 128), border_color, border_thick
        )

        # Label text
        tag = f"{part_name} {'[M]' if is_missing else '[OK]'}"
        text_color = (0, 0, 255) if is_missing else (200, 200, 200)
        cv2.putText(
            canvas, tag, (x0 + 8, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Build PCB presence dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=== Step 1: Initialize Alignment and Regions ===")
    P = gc.Params()
    P.px_per_mm = 20.0        # canonical frame 1660 x 2320 px (83 x 116 mm)
    P.max_input_side = 4000   # do not shrink phone photos before aligning

    ref = gc.prepare_reference(gc.imread(GOLDEN_IMAGE_PATH), P, gc.load_regions(REGIONS_JSON_PATH, P))
    part_regions = [r for r in ref.regions if r.kind == "part"]
    part_names = {r.name for r in part_regions}
    print(f"Loaded {len(part_regions)} part regions. Canonical frame: {P.canon_size}")
    if len(part_regions) != 86:
        raise RuntimeError(f"Expected 86 parts, found {len(part_regions)}")

    # Parse removed boards missing lists
    missing_by_board: dict[str, set[str]] = {}
    for b_idx in range(1, 5):
        b_name = f"m_{b_idx:02d}"
        folder = REMOVED_PHOTOS_DIR / b_name
        missing_parts = parse_missing_list(folder, part_names)
        missing_by_board[b_name] = missing_parts
        print(f"Board {b_name}: {len(missing_parts)} missing parts -> {sorted(missing_parts)}")

    # Ensure directories
    ALL_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all photos
    # Good photos: R_01.jpg .. R_43.jpg
    good_photos = sorted(GOOD_PHOTOS_DIR.glob("R_*.jpg"))
    # Removed photos: m_01 .. m_04
    removed_photos = []
    for b_idx in range(1, 5):
        b_name = f"m_{b_idx:02d}"
        photos = sorted((REMOVED_PHOTOS_DIR / b_name).glob("*.jpg"))
        for p in photos:
            removed_photos.append((b_name, p))

    print(f"Found {len(good_photos)} good photos and {len(removed_photos)} removed board photos.")

    all_crop_records = []
    skipped_photos = []
    processed_count = 0

    all_tasks = [(f"good_{p.stem}", p, "good") for p in good_photos] + [
        (b_name, p, "removed") for b_name, p in removed_photos
    ]

    print(f"Total photos to process: {len(all_tasks)}")
    t_start = time.time()

    for item_idx, (board_name, photo_path, photo_kind) in enumerate(all_tasks):
        t0 = time.time()
        photo_stem = photo_path.stem
        raw_bgr = gc.imread(photo_path)

        result, images = gc.inspect(ref, raw_bgr, P)
        if result["decision"] == "RETAKE" or images is None:
            reason = result.get("reason", "Unknown RETAKE")
            print(f"[{item_idx+1}/{len(all_tasks)}] SKIPPED {photo_path.name}: {reason}")
            skipped_photos.append({
                "photo": photo_path.name,
                "path": str(photo_path),
                "board": board_name,
                "kind": photo_kind,
                "reason": reason
            })
            continue

        aligned = images["aligned"]
        residual_mm = result.get("alignment", {}).get("residual_mm", 0.0)

        # Local fine alignment
        rg, ng, _ = gc.comparison_grays(ref.clahe, gc.clahe_gray(aligned), ref.area, P)
        shifts = gc.region_shifts(gc.local_flow(rg, ng), ref, ref.regions, P)

        missing_set = missing_by_board.get(board_name, set()) if photo_kind == "removed" else set()

        contact_sheet_crops = []
        present_parts_in_photo = []

        for region in part_regions:
            x, y, w, h = region.bbox_px
            dx, dy = shifts.get(region.id, (0.0, 0.0))
            cx = x + w / 2.0 + dx
            cy = y + h / 2.0 + dy

            # Crop window side: max(w, h) + 2 * 1.5 mm (1.5 mm = 30 px at 20 px/mm)
            side = int(round(max(w, h) + 2 * (1.5 * P.px_per_mm)))
            crop_128 = extract_square_crop(aligned, cx, cy, side, out_size=128)

            part_name = region.name
            is_missing = part_name in missing_set
            label = "missing" if is_missing else "present"
            part_type = get_part_type(part_name)

            crop_filename = f"{board_name}__{photo_stem}__{part_name}.png"
            crop_path = ALL_CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop_128)

            record = {
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                "board": board_name,
                "photo": photo_stem,
                "part": part_name,
                "part_type": part_type,
                "label": label,
                "synthetic_shift": 0,
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "residual_mm": round(residual_mm, 3)
            }
            all_crop_records.append(record)
            contact_sheet_crops.append((part_name, label, crop_128))

            if label == "present":
                present_parts_in_photo.append((region, cx, cy, side, dx, dy))

        # Synthetic shift crops (Section 4.3):
        # 0.9 mm rule: shift 1.3 - 2.5 mm in random direction -> label missing, synthetic_shift=1
        # Pick 2 present parts per usable photo to have a rich pool of synthetic candidates
        synth_sample_count = min(3, len(present_parts_in_photo))
        sampled_for_shift = random.sample(present_parts_in_photo, synth_sample_count)
        for s_region, s_cx, s_cy, s_side, s_dx, s_dy in sampled_for_shift:
            shift_dist_mm = random.uniform(1.3, 2.5)
            shift_dist_px = shift_dist_mm * P.px_per_mm
            shift_angle = random.uniform(0, 2 * math.pi)
            shifted_cx = s_cx + shift_dist_px * math.cos(shift_angle)
            shifted_cy = s_cy + shift_dist_px * math.sin(shift_angle)

            shift_crop_128 = extract_square_crop(aligned, shifted_cx, shifted_cy, s_side, out_size=128)
            s_name = s_region.name
            s_type = get_part_type(s_name)
            s_filename = f"{board_name}__{photo_stem}__{s_name}__synth.png"
            s_path = ALL_CROPS_DIR / s_filename
            cv2.imwrite(str(s_path), shift_crop_128)

            all_crop_records.append({
                "crop_path": str(s_path.relative_to(PROJECT_ROOT)),
                "board": board_name,
                "photo": photo_stem,
                "part": s_name,
                "part_type": s_type,
                "label": "missing",
                "synthetic_shift": 1,
                "dx": round(s_dx + shift_dist_px * math.cos(shift_angle), 2),
                "dy": round(s_dy + shift_dist_px * math.sin(shift_angle), 2),
                "residual_mm": round(residual_mm, 3)
            })

        # Save contact sheet
        sheet_path = SHEETS_DIR / f"{photo_stem}.jpg"
        create_contact_sheet(contact_sheet_crops, sheet_path, f"Photo: {photo_stem} | Board: {board_name}")

        processed_count += 1
        elapsed = time.time() - t0
        print(f"[{item_idx+1}/{len(all_tasks)}] Processed {photo_stem} ({board_name}) in {elapsed:.2f}s (res: {residual_mm:.2f} mm)")

    total_time = time.time() - t_start
    print(f"\nProcessing complete in {total_time:.1f}s. Processed: {processed_count}, Skipped: {len(skipped_photos)}")

    # Write skipped photos log
    skipped_log_path = REPORT_DIR / "skipped_photos.json"
    skipped_log_path.write_text(json.dumps(skipped_photos, indent=2), encoding="utf-8")

    # Write index.csv
    with open(INDEX_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crop_path", "board", "photo", "part", "part_type", "label",
            "synthetic_shift", "dx", "dy", "residual_mm"
        ])
        writer.writeheader()
        writer.writerows(all_crop_records)
    print(f"Wrote {len(all_crop_records)} crop entries to {INDEX_CSV_PATH}")

    # Build datasets for 4 folds and final model
    print("\n=== Step 2: Build Folds and Datasets ===")
    build_folds_and_final(all_crop_records)


def build_folds_and_final(crop_records: list[dict]):
    """Partition crops into 4 folds and 1 final dataset conforming to Ultralytics classification layout."""
    # Good photo grouping (consecutive photo numbers)
    # R_01 - R_11 -> fold 1
    # R_12 - R_22 -> fold 2
    # R_23 - R_33 -> fold 3
    # R_34 - R_43 -> fold 4
    def get_good_photo_fold(photo_stem: str) -> int:
        match = re.search(r"R_(\d+)", photo_stem)
        if not match:
            return 1
        num = int(match.group(1))
        if num <= 11:
            return 1
        elif num <= 22:
            return 2
        elif num <= 33:
            return 3
        else:
            return 4

    for fold_k in range(1, 5):
        val_removed_board = f"m_{fold_k:02d}"
        fold_dir = DATASETS_DIR / f"fold_{fold_k}"
        train_present_dir = fold_dir / "train" / "present"
        train_missing_dir = fold_dir / "train" / "missing"
        val_present_dir = fold_dir / "val" / "present"
        val_missing_dir = fold_dir / "val" / "missing"

        for d in [train_present_dir, train_missing_dir, val_present_dir, val_missing_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        train_real_present = []
        train_real_missing = []
        train_synth_missing = []
        val_real_present = []
        val_real_missing = []

        for r in crop_records:
            board = r["board"]
            photo = r["photo"]
            is_good = board.startswith("good_")
            is_synth = (r["synthetic_shift"] == 1)

            # Determine if this crop belongs to val or train
            if is_good:
                fold_num = get_good_photo_fold(photo)
                is_val = (fold_num == fold_k)
            else:
                is_val = (board == val_removed_board)

            if is_val:
                # Validation set never gets synthetic crops
                if not is_synth:
                    if r["label"] == "missing":
                        val_real_missing.append(r)
                    else:
                        val_real_present.append(r)
            else:
                # Train set
                if is_synth:
                    train_synth_missing.append(r)
                elif r["label"] == "missing":
                    train_real_missing.append(r)
                else:
                    train_real_present.append(r)

        # 1. Populate val set (never balanced, no synth)
        for r in val_real_present:
            src = PROJECT_ROOT / r["crop_path"]
            dst = val_present_dir / src.name
            shutil.copy2(src, dst)

        for r in val_real_missing:
            src = PROJECT_ROOT / r["crop_path"]
            dst = val_missing_dir / src.name
            shutil.copy2(src, dst)

        # 2. Populate train set
        # A. Present crops
        for r in train_real_present:
            src = PROJECT_ROOT / r["crop_path"]
            dst = train_present_dir / src.name
            shutil.copy2(src, dst)

        # B. Limit synthetic shifts: at most as many as real missing in train
        max_synth = len(train_real_missing)
        selected_synth = train_synth_missing[:max_synth]
        for r in selected_synth:
            src = PROJECT_ROOT / r["crop_path"]
            dst = train_missing_dir / src.name
            shutil.copy2(src, dst)

        # C. Real missing crops and balancing to achieve ~1:4 missing:present ratio
        # Calculate target missing count
        target_missing = len(train_real_present) // 4
        # Existing missing count from 1x real + selected_synth
        current_missing = len(train_real_missing) + len(selected_synth)
        needed_extra = max(0, target_missing - current_missing)
        repeat_factor = max(1, math.ceil(needed_extra / max(1, len(train_real_missing))))

        for r in train_real_missing:
            src = PROJECT_ROOT / r["crop_path"]
            # Copy base
            dst = train_missing_dir / src.name
            shutil.copy2(src, dst)
            # Copy duplicates with suffix
            for rep in range(1, repeat_factor + 1):
                rep_name = f"{src.stem}__rep{rep}{src.suffix}"
                shutil.copy2(src, train_missing_dir / rep_name)

        final_train_present = len(list(train_present_dir.glob("*.png")))
        final_train_missing = len(list(train_missing_dir.glob("*.png")))
        final_val_present = len(list(val_present_dir.glob("*.png")))
        final_val_missing = len(list(val_missing_dir.glob("*.png")))

        print(f"Fold {fold_k}:")
        print(f"  Train: {final_train_present} present, {final_train_missing} missing (ratio 1:{final_train_present/final_train_missing:.1f})")
        print(f"  Val:   {final_val_present} present, {final_val_missing} missing (real missing only)")

    # Build Final dataset (all data in train)
    print("\nBuilding Final Dataset (all data)...")
    final_dir = DATASETS_DIR / "final"
    train_present_dir = final_dir / "train" / "present"
    train_missing_dir = final_dir / "train" / "missing"
    val_present_dir = final_dir / "val" / "present"
    val_missing_dir = final_dir / "val" / "missing"

    for d in [train_present_dir, train_missing_dir, val_present_dir, val_missing_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    all_real_present = [r for r in crop_records if r["synthetic_shift"] == 0 and r["label"] == "present"]
    all_real_missing = [r for r in crop_records if r["synthetic_shift"] == 0 and r["label"] == "missing"]
    all_synth_missing = [r for r in crop_records if r["synthetic_shift"] == 1]

    for r in all_real_present:
        src = PROJECT_ROOT / r["crop_path"]
        shutil.copy2(src, train_present_dir / src.name)

    max_synth = len(all_real_missing)
    for r in all_synth_missing[:max_synth]:
        src = PROJECT_ROOT / r["crop_path"]
        shutil.copy2(src, train_missing_dir / src.name)

    target_missing = len(all_real_present) // 4
    current_missing = len(all_real_missing) + min(len(all_synth_missing), max_synth)
    needed_extra = max(0, target_missing - current_missing)
    repeat_factor = max(1, math.ceil(needed_extra / max(1, len(all_real_missing))))

    for r in all_real_missing:
        src = PROJECT_ROOT / r["crop_path"]
        shutil.copy2(src, train_missing_dir / src.name)
        for rep in range(1, repeat_factor + 1):
            rep_name = f"{src.stem}__rep{rep}{src.suffix}"
            shutil.copy2(src, train_missing_dir / rep_name)

    # For final model val: copy fold 4 val crops so Ultralytics can run validation checks
    f4_val = DATASETS_DIR / "fold_4" / "val"
    for p in (f4_val / "present").glob("*.png"):
        shutil.copy2(p, val_present_dir / p.name)
    for p in (f4_val / "missing").glob("*.png"):
        shutil.copy2(p, val_missing_dir / p.name)

    print(f"Final Dataset:")
    print(f"  Train: {len(list(train_present_dir.glob('*.png')))} present, {len(list(train_missing_dir.glob('*.png')))} missing")
    print(f"  Val:   {len(list(val_present_dir.glob('*.png')))} present, {len(list(val_missing_dir.glob('*.png')))} missing")


if __name__ == "__main__":
    main()
