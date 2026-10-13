"""Make the live inspection (scripts/station.py) use the clean collection: reference, board registry, region map.

    python scripts/setup_reference.py                     dry run with the best verified GOOD image
    python scripts/setup_reference.py --apply             do it
    python scripts/setup_reference.py --reference GOOD_001_TOP_S20261006-03_C007 --apply

Steps (each reported; nothing in data/collection is changed):
1. Reference: the chosen KEPT GOOD capture must pass every automatic clean-reference check
   (scripts/collection_summary.py --verify-reference). LED4 must still be confirmed by eye on the sheet.
2. Board registry (data/collection/registry/boards.csv): every board with kept images is ADDED as declared at capture:
   GOOD -> condition good, split reference (the golden bank); FAULT -> condition fault, split validation, defects from
   the operator's fault report (unverified). Existing rows are never changed.
3. Region map: the 55 DRAFT component boxes (experiments/experiment_002/components.yaml, drawn on the archived
   watermarked reference) are re-attached UNCHANGED (mm) to the new reference: same board design, same canonical
   frame. They stay DRAFT / NOT VERIFIED. A sheet with every box on the new reference is written for checking.
4. golden_bank.yaml geometric_reference and regions.yaml component_regions.file are set; the bank status is shown.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import DEFAULT_CAPTURE_CONFIG, build_store  # noqa: E402
from collection_summary import find_record, reference_candidates, verify_reference  # noqa: E402
from pcb_inspection.acquisition.session import REVIEW_KEPT, AcquisitionStore  # noqa: E402
from pcb_inspection.comparison.components import (  # noqa: E402
    Component, ComponentMap, box_to_px, load_component_map, part_mask, save_component_map)
from pcb_inspection.comparison.pipeline import prepare_reference  # noqa: E402
from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.dataset.experiment0 import sha256_of  # noqa: E402
from pcb_inspection.dataset.registry import BOARD_COLUMNS, load_registry  # noqa: E402
from pcb_inspection.inspection import load_reference_bank  # noqa: E402
from pcb_inspection.inspection.reference_baseline import ReferenceBaselineBackend  # noqa: E402
from pcb_inspection.preprocessing.board import warp_to_canonical  # noqa: E402
from pcb_inspection.preprocessing.watermark import load_template, locate_watermark, remove_watermark  # noqa: E402
from pcb_inspection.products import load_product_profile  # noqa: E402

DRAFT_MAP = ROOT / "experiments" / "experiment_002" / "components.yaml"
DEFECT_REF = re.compile(r"[A-Z]{1,4}\d{1,3}[A-Z]?")


def registry_rows(store: AcquisitionStore) -> list[dict[str, str]]:
    """One row per physical board with kept images, as declared at capture (GOOD / FAULT + fault report)."""
    rows = {}
    with_images = {s.board_id for s in store.sessions() if any(r["review"]["status"] == REVIEW_KEPT for r in s.images)}
    for session in store.sessions():      # the fault report of a board counts from ANY of its sessions
        if not session.board_id or session.board_id not in with_images:
            continue
        row = rows.setdefault(session.board_id, {
            "board_id": session.board_id, "condition": "good" if session.dataset_type == "good" else "fault",
            "split": "reference" if session.dataset_type == "good" else "validation", "defects": "",
            "defect_source": "", "identity_source": "collection_board_id", "verified_by": "",
            "notes": f"declared {session.dataset_type.upper()} at capture (scripts/collect.py)"})
        report = session.data.get("defect_report") or {}
        if report.get("defect_known") == "yes":
            ref = (report.get("component_reference") or "").upper()
            item = f"{ref}:{report.get('taxonomy_defect_type') or report['defect_type']}"
            if DEFECT_REF.fullmatch(ref) and item not in row["defects"].split(";"):
                row["defects"] = ";".join(filter(None, [row["defects"], item]))
                row["defect_source"] = "operator_at_capture_unverified"
            elif not DEFECT_REF.fullmatch(ref):
                row["notes"] += f"; reported {report['defect_type']} without a component reference"
    return list(rows.values())


def add_to_registry(path: Path, rows: list[dict[str, str]], apply: bool) -> list[str]:
    existing = load_registry(path).boards if path.is_file() else {}
    new = [r for r in rows if r["board_id"] not in existing]
    if apply and new:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BOARD_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(new)
    return [f"{r['board_id']}: {r['condition']}, split {r['split']}" + (f", defects {r['defects']}" if r["defects"]
            else (" (fault UNKNOWN)" if r["condition"] == "fault" else "")) for r in new]


def rebind_map(cfg, reference: Path, out_sheet: Path) -> tuple[ComponentMap, list[str]]:
    """The draft boxes re-attached to the canonical frame of the NEW reference (as the engine computes it)."""
    old = load_component_map(DRAFT_MAP)
    ppm = cfg.pipeline.px_per_mm
    info = {"id": reference.stem, "path": str(reference), "board_id": "", "sha256": sha256_of(reference)}
    base = prepare_reference(info, cfg.pipeline, cfg.cache_dir / "reference" / reference.stem, cfg.experiment)
    expected = (round(old.size_mm[0] * ppm), round(old.size_mm[1] * ppm))
    if tuple(base.frame.size) != expected:
        raise ValueError(f"the new reference's canonical frame {base.frame.size} does not match the map {expected}: "
                         "the board lies portrait / another way; place it like the GOOD_001 reference")
    template = load_template(cfg.watermark)
    clean = remove_watermark(base.image, locate_watermark(base.image, template, cfg.watermark))
    canonical, _ = warp_to_canonical(clean, base.frame.homography, base.frame.size, base.frame.source_px_per_mm, ppm)
    # The boxes are kept UNCHANGED (mm): same board design, same canonical frame and orientation. Snapping them to the
    # part pixels was tried and made them worse (they grew over neighbouring parts), so it is not done. Each box is
    # only checked to contain part pixels on the new reference.
    parts = part_mask(canonical, ppm, cfg.check)
    components, notes, empty = [], [], []
    for c in old.components:
        x0, y0, x1, y1 = box_to_px(c.box_mm, ppm)
        filled = float(np.count_nonzero(parts[y0:y1, x0:x1])) / max(1, (x1 - x0) * (y1 - y0))
        note = (f"draft box of the archived reference {old.reference.get('image_id')}, unchanged (mm), re-attached to "
                f"{reference.stem}; NOT VERIFIED")
        if filled < 0.10:
            empty.append(c.refdes)
            note += "; few part pixels in the box on the new reference: check it"
        components.append(Component(c.refdes, c.box_mm, "assistant_draft", False, note))
    notes.append(f"{len(components)} draft boxes re-attached unchanged; boxes with few part pixels on the new reference: "
                 f"{', '.join(empty) or 'none'}")
    cmap = ComponentMap(components, {"image_id": reference.stem, "file": reference.relative_to(ROOT).as_posix(),
                                     "sha256": info["sha256"]}, old.size_mm, old.profile,
                        {**old.header, "component_map_version": int(old.header.get("component_map_version", 1)) + 1})
    sheet = canonical.copy()
    for c in components:
        x0, y0, x1, y1 = box_to_px(c.box_mm, ppm)
        colour = (0, 200, 255) if c.refdes in empty else (60, 220, 60)
        cv2.rectangle(sheet, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(sheet, c.refdes, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    out_sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cmap, notes


def set_yaml_value(path: Path, key: str, value: str, apply: bool) -> str:
    """Replace `key: ...` keeping the comment of that line (configs are hand-commented)."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s*{re.escape(key)}:)\s*[^#\n]*?(\s*#.*)?$", re.M)
    if not pattern.search(text):
        raise ValueError(f"{path.name}: no '{key}:' line")
    new = pattern.sub(lambda m: f"{m.group(1)} {value}{m.group(2) or ''}", text, count=1)
    if apply:
        path.write_text(new, encoding="utf-8")
    return f"{path.relative_to(ROOT).as_posix()}: {key}: {value}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Use a verified clean GOOD capture as the inspection reference.")
    parser.add_argument("--reference", default="best", help="KEPT GOOD image ID, or 'best' (default)")
    parser.add_argument("--apply", action="store_true", help="write the registry / map / configs (default: dry run)")
    parser.add_argument("--capture-config", type=Path, default=DEFAULT_CAPTURE_CONFIG)
    args = parser.parse_args()
    capture = load_config(args.capture_config)
    store = build_store(capture, capture["product"])
    profile = load_product_profile(capture["product"], project_root=ROOT)

    print("1. Reference")
    ok, lines = verify_reference(store, args.reference)
    print("\n".join(f"   {line}" for line in lines))
    if not ok:
        print("   -> not a clean reference; nothing changed.")
        return 1
    image_id = lines[0].split()[3]
    _, record = find_record(store, image_id)
    reference = store.root / record["original_file"]
    boards = sorted(reference_candidates(store))
    print(f"   GOOD boards for the bank: {', '.join(boards)}")

    print("\n2. Board registry")
    registry = profile.registry_dir / "boards.csv"
    added = add_to_registry(registry, registry_rows(store), args.apply)
    print("\n".join(f"   + {a}" for a in added) or "   (every board is already registered)")

    print("\n3. Region map")
    profile.golden["geometric_reference"] = reference.relative_to(ROOT).as_posix()
    cfg = ReferenceBaselineBackend()._config(profile)
    sheet = ROOT / "data" / "processed" / "collection" / "reference_check" / f"{image_id}_components.jpg"
    cmap, notes = rebind_map(replace(cfg, reference_file=reference), reference, sheet)
    print("\n".join(f"   {n}" for n in notes))
    print(f"   Check every box on: {sheet.relative_to(ROOT)} (orange = few part pixels in the box)")
    map_file = profile.folder / "regions" / "components.yaml"
    if args.apply:
        save_component_map(cmap, map_file,
                           "# Component map of LG Indicator / variant_1 on the clean reference (scripts/setup_reference.py).\n"
                           "# STATUS: DRAFT, NOT VERIFIED: the Experiment 2 draft boxes (mm) re-attached unchanged to the new\n"
                           "# clean reference (same board design and frame). No BOM / centroid file exists. Verify / correct:\n"
                           "#     python scripts/define_components.py\n")

    print("\n4. Configuration")
    print("   " + set_yaml_value(profile.folder / "golden" / "golden_bank.yaml", "geometric_reference",
                                  profile.golden["geometric_reference"], args.apply))
    print("   " + set_yaml_value(profile.folder / "regions" / "regions.yaml", "file",
                                  map_file.relative_to(ROOT).as_posix(), args.apply))
    print("   " + set_yaml_value(args.capture_config, "orientation_reference",       # live "turned 180 deg" check
                                  (store.root / record["pcb_file"]).relative_to(ROOT).as_posix(), args.apply))
    if not args.apply:
        print("\nDry run: nothing written. Run again with --apply.")
        return 0
    bank = load_reference_bank(load_product_profile(capture["product"], project_root=ROOT))
    print(f"\nGolden bank: {bank.status} - {bank.reason}")
    if bank.skipped:
        print("   skipped: " + "; ".join(bank.skipped[:5]))
    return 0 if bank.status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
