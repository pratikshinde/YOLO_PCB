"""Summary, audit and reference verification of the clean GOOD / FAULT collection (docs/clean_data_collection.md).

    python scripts/collection_summary.py                               counts per type / board, capture checks, audit
    python scripts/collection_summary.py --verify-reference IMAGE_ID   is this KEPT GOOD capture a clean reference?
    python scripts/collection_summary.py --verify-reference best       the sharpest KEPT GOOD capture with check OK
    python scripts/collection_summary.py --csv data/processed/collection/metadata.csv   flat metadata table

Read-only on the dataset. --verify-reference writes a verification sheet (derived) to
data/processed/collection/reference_check/<image_id>.jpg.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import DEFAULT_CAPTURE_CONFIG, build_store  # noqa: E402
from pcb_inspection.acquisition.pcb_view import OK, RECAPTURE, draw_check  # noqa: E402
from pcb_inspection.acquisition.session import (  # noqa: E402
    BOARD_POOL, REVIEW_KEPT, REVIEW_PENDING, REVIEW_REJECTED, AcquisitionStore, audit, capture_check)
from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.dataset.experiment0 import sha256_of  # noqa: E402

CSV_FIELDS = ("image_id", "product_id", "physical_board_id", "session_id", "dataset_type", "stage", "timestamp",
              "camera", "resolution", "path", "sha256", "pcb_file", "pcb_sha256", "quality_status", "capture_check",
              "capture_check_codes", "review_status", "saved_file", "saved_pcb_file")
SHEET_DIR = ROOT / "data" / "processed" / "collection" / "reference_check"


def archived_counts(project_root: Path) -> list[tuple[str, int, int]]:
    """(archive folder, files, images) of every archive/*/manifest.csv."""
    out = []
    for manifest in sorted((project_root / "archive").glob("*/manifest.csv")):
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out.append((manifest.parent.relative_to(project_root).as_posix(), len(rows),
                    sum(r.get("kind") == "image" for r in rows)))
    return out


def summarise(store: AcquisitionStore, project_root: Path = ROOT) -> dict:
    """Counts of the collection store (all sessions, every review state) + audit issues."""
    per_type = {t: Counter() for t in ("good", "fault")}
    boards: dict[str, dict] = defaultdict(lambda: {"sessions": set(), "status": Counter(), "check": Counter(),
                                                    "type": None, "defect_reports": []})
    checks, test_images = Counter(), 0
    for session in store.sessions():
        t = session.dataset_type
        if t == "test":
            test_images += len(session.images)
            continue
        board = session.board_id or f"(no board ID: {session.id})"
        b = boards[board]
        b["type"] = t
        b["sessions"].add(session.id)
        if session.data.get("defect_report"):
            b["defect_reports"].append((session.id, session.data["defect_report"]))
        for r in session.images:
            status = r["review"]["status"]
            per_type.setdefault(t, Counter())[status] += 1
            per_type[t]["captured"] += 1
            b["status"][status] += 1
            check = capture_check(r)
            if check is not None:
                checks[check["status"]] += 1
                b["check"][check["status"]] += 1
    test_dir = store.curated_root / "test"
    test_images += sum(1 for p in test_dir.rglob("*.png")) if test_dir.is_dir() else 0
    bank = store.good_bank_candidates()
    return {"per_type": per_type, "boards": dict(boards), "checks": checks, "test_images": test_images,
            "bank": bank, "issues": audit(store), "archives": archived_counts(project_root)}


def board_fault_lines(reports: list[tuple[str, dict]]) -> list[str]:
    """One board's fault information over its sessions (operator reports, not verified): the known report, sessions
    that recorded UNKNOWN (e.g. after a restart), and conflicting reports."""
    if not reports:
        return []
    known = [(sid, r) for sid, r in reports if r.get("defect_known") == "yes"]
    unknown = [sid for sid, r in reports if r.get("defect_known") != "yes"]
    describe = lambda r: r["defect_type"] + "".join(f", {k} {r[k]}" for k in ("component_reference", "location", "notes")  # noqa: E731
                                                    if r.get(k))
    if not known:
        return ["fault type UNKNOWN (not labelled)"]
    kinds = {(r["defect_type"], r.get("component_reference")) for _, r in known}
    if len(kinds) > 1:
        return ["CONFLICTING fault reports for one board: " + "; ".join(f"{sid}: {describe(r)}" for sid, r in known)
                + " - check the board and correct the labels"]
    sid, r = known[0]
    line = f"reported {describe(r)} ({', '.join(s for s, _ in known)}) - operator report, not verified"
    if unknown:
        line += f"; session(s) {', '.join(unknown)} recorded UNKNOWN"
    return [line]


def format_summary(store: AcquisitionStore, s: dict, project_root: Path = ROOT) -> str:
    rel = lambda p: Path(p).resolve().relative_to(project_root.resolve()).as_posix()  # noqa: E731
    lines = [f"CLEAN DATA COLLECTION  product {store.settings.context.get('product_id')}  store {rel(store.root)}", ""]
    if s["archives"]:
        for folder, files, images in s["archives"]:
            lines.append(f"Old image data removed/archived: {files} files ({images} images) -> {folder}/ (manifest.csv)")
    else:
        lines.append("Old image data removed/archived: nothing archived yet (python scripts/archive_old_data.py)")
    lines += ["", "New data (this collection phase):"]
    for t in ("good", "fault"):
        c = s["per_type"].get(t, Counter())
        n_boards = sum(1 for b in s["boards"].values() if b["type"] == t and b["status"][REVIEW_KEPT])
        lines.append(f"  {BOARD_POOL[t]} boards: {n_boards:<4} {BOARD_POOL[t]} images: {c[REVIEW_KEPT]:<5} "
                     f"(kept; captured {c['captured']}, pending review {c[REVIEW_PENDING]}, "
                     f"rejected / quarantine {c[REVIEW_REJECTED]})")
    lines.append(f"  TEST images: {s['test_images']} (TEST is not collected in this phase)")
    if s["boards"]:
        lines += ["", f"  {'Board':<26}{'type':<7}{'sessions':>9}{'kept':>6}{'pending':>9}{'rejected':>10}   capture check"]
        for board, b in sorted(s["boards"].items()):
            check = ", ".join(f"{k} {v}" for k, v in sorted(b["check"].items())) or "-"
            lines.append(f"  {board:<26}{BOARD_POOL[b['type']]:<7}{len(b['sessions']):>9}{b['status'][REVIEW_KEPT]:>6}"
                         f"{b['status'][REVIEW_PENDING]:>9}{b['status'][REVIEW_REJECTED]:>10}   {check}")
            lines += [f"      {line}" for line in board_fault_lines(b["defect_reports"])]
    checks = s["checks"]
    lines += ["", f"Capture check (all captures): OK {checks[OK]}, REVIEW {checks['REVIEW']}, RECAPTURE {checks[RECAPTURE]}",
              f"GOOD reference-bank candidates (kept GOOD, PCB image, check not RECAPTURE): {len(s['bank'])} board(s), "
              f"{sum(len(v) for v in s['bank'].values())} image(s); FAULT boards are never candidates",
              f"Live inspection golden bank (scripts/station.py): {s.get('bank_status') or 'not loaded'}",
              "", f"Audit: {'consistent (checksums, folders, records)' if not s['issues'] else str(len(s['issues'])) + ' issue(s)'}"]
    lines += [f"  {i}" for i in s["issues"][:30]]
    return "\n".join(lines)


def write_csv(store: AcquisitionStore, path: Path) -> int:
    rows = []
    for session in store.sessions():
        for r in session.images:
            check = capture_check(r) or {}
            rows.append({**{k: r.get(k) for k in CSV_FIELDS},
                         "resolution": "x".join(str(v) for v in r.get("resolution") or [r.get("width"), r.get("height")]),
                         "path": r.get("path") or r.get("original_file"),
                         "quality_status": r["quality"].get("status"), "capture_check": check.get("status"),
                         "capture_check_codes": ";".join(check.get("problems", []) + check.get("review", [])),
                         "review_status": r["review"]["status"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --- reference verification ------------------------------------------------------------------------

def find_record(store: AcquisitionStore, image_id: str):
    for session in store.sessions():
        for r in session.images:
            if r["image_id"] == image_id:
                return session, r
    return None, None


def reference_candidates(store: AcquisitionStore) -> dict[str, list[dict]]:
    """board -> its KEPT GOOD images whose capture check is OK, sharpest first (stored measurements)."""
    out: dict[str, list[dict]] = {}
    for r in store.records(dataset_types=("good",)):
        check = capture_check(r) or {}
        if r.get("board_id") and check.get("status") == OK and r.get("pcb_file"):
            out.setdefault(r["board_id"], []).append(r)
    for images in out.values():
        images.sort(key=lambda r: -(capture_check(r).get("pcb") or {}).get("sharpness", 0))
    return dict(sorted(out.items()))


def verify_reference(store: AcquisitionStore, image_id: str, sheet_dir: Path = SHEET_DIR) -> tuple[bool, list[str]]:
    """Automatic part of the clean-reference check of one capture. Returns (passed, report lines).
    image_id "best": the sharpest KEPT GOOD image whose capture check is OK."""
    candidates = reference_candidates(store)
    if image_id.lower() == "best":
        ranked = sorted((r for images in candidates.values() for r in images),
                        key=lambda r: -(capture_check(r).get("pcb") or {}).get("sharpness", 0))
        if not ranked:
            return False, ["No KEPT GOOD image with capture check OK yet: collect GOOD boards first "
                           "(python scripts/collect.py)."]
        image_id = ranked[0]["image_id"]
    session, r = find_record(store, image_id)
    if r is None:
        lines = [f"{image_id}: not in the collection store. Use a real image ID (no <...> placeholders), or 'best'."]
        if candidates:
            lines.append("KEPT GOOD images with capture check OK, sharpest first:")
            for board, images in candidates.items():
                lines.append(f"  {board} ({len(images)}): " + ", ".join(i["image_id"] for i in images[:3])
                             + (" ..." if len(images) > 3 else ""))
        return False, lines
    lines, ok = [f"Reference check of {image_id} (board {r['board_id']}, session {r['session_id']})"], True

    def item(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    item("GOOD PCB, kept", r["dataset_type"] == "good" and r["review"]["status"] == REVIEW_KEPT,
         f"type {BOARD_POOL[r['dataset_type']]}, review {r['review']['status']}")
    original = store.root / r["original_file"]
    item("original intact", original.is_file() and sha256_of(original) == r["sha256"], r["original_file"])
    pcb_file = store.root / r["pcb_file"] if r.get("pcb_file") else None
    item("PCB image intact", bool(pcb_file and pcb_file.is_file() and sha256_of(pcb_file) == r["pcb_sha256"]),
         r.get("pcb_file") or "no PCB image")
    checker = store.settings.pcb_checker
    if checker is None or not original.is_file():
        item("capture check", False, "no capture checker configured / original missing")
        return False, lines
    bgr = cv2.imread(str(original))
    c = checker.check(bgr)                           # re-run now, full resolution, current thresholds
    codes = set(c.problems + c.review)
    item("full PCB visible", not codes & {"NO_PCB", "PCB_INCOMPLETE"},
         f"board {c.board_area_fraction:.0%} of the frame, {c.edge_margin_px} px from the image edge")
    item("PCB inside the inspection ROI", "PCB_OUTSIDE_ROI" not in codes, f"ROI {c.roi_px}")
    wm = c.watermark
    clear = False                                    # watermark proven to be off the PCB
    if not wm.get("checked"):
        item("no watermark over the PCB", False, "watermark check disabled: cannot be confirmed")
    elif not wm.get("found"):
        clear = bool(wm.get("not_on_pcb")) and "WATERMARK_NOT_LOCATED" not in codes
        item("no watermark over the PCB / components", clear,
             "watermark not in view (outside the picture / cut off at the image edge); the PCB area itself has no "
             f"watermark-like match (best {wm.get('on_pcb_best_score')}, threshold "
             f"{checker.params.watermark.min_score})" if clear else
             "watermark-like text on the PCB area that is not clearly located: confirm on the sheet")
    else:
        clear = not wm.get("over_pcb") and "WATERMARK_OVER_PCB" not in codes
        item("no watermark over the PCB / components", clear,
             f"watermark box {wm['box_px']} is {wm.get('distance_mm')} mm from the board edge "
             f"(min {checker.params.watermark_clearance_mm} mm)")
    item("correct crop", "BAD_CROP" not in codes and bool(c.pcb),
         f"outline aspect error {c.aspect_error:.3f}, fill {c.fill_fraction:.3f}, PCB image "
         f"{c.pcb.get('width')} x {c.pcb.get('height')} px at {c.pcb.get('px_per_mm')} px/mm")
    item("acceptable quality", not codes & {"BLURRY", "OVEREXPOSED", "UNDEREXPOSED"},
         f"PCB sharpness {c.pcb.get('sharpness')} (min {checker.params.min_pcb_sharpness}), clipped white "
         f"{c.pcb.get('near_white_fraction')}, black {c.pcb.get('near_black_fraction')}")
    item("LED4 not covered by the watermark", bool(c.status != RECAPTURE and clear),
         "follows from: whole board in view and the watermark outside the board outline. LED4 PRESENCE is not "
         "detected automatically (no verified region map yet): confirm it on the sheet")
    sheet = write_sheet(bgr, c, image_id, sheet_dir)
    lines.append(f"  Verification sheet (confirm LED4 and the crop by eye): {sheet}")
    lines.append(f"  RESULT: {'CLEAN REFERENCE CANDIDATE (automatic checks passed; visual confirmation pending)' if ok else 'NOT a clean reference'}")
    return ok, lines


def write_sheet(bgr: np.ndarray, check, image_id: str, folder: Path) -> Path:
    """Original with ROI / board outline / watermark box, next to the PCB image with a 10 mm grid. Derived."""
    scale = 900 / bgr.shape[1]
    left = draw_check(cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), check, scale)
    right = np.full((left.shape[0], 900, 3), 40, np.uint8)
    if check.canonical is not None:
        pcb = check.canonical.copy()
        ppm = check.pcb["px_per_mm"]
        for x in range(0, pcb.shape[1], int(10 * ppm)):
            cv2.line(pcb, (x, 0), (x, pcb.shape[0]), (255, 255, 255), 1)
        for y in range(0, pcb.shape[0], int(10 * ppm)):
            cv2.line(pcb, (0, y), (pcb.shape[1], y), (255, 255, 255), 1)
        s = min(right.shape[1] / pcb.shape[1], right.shape[0] / pcb.shape[0])
        small = cv2.resize(pcb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        right[:small.shape[0], :small.shape[1]] = small
    sheet = np.hstack([left, right])
    header = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{image_id}  capture check {check.status}  {' '.join(check.problems + check.review)}  "
                "| right: PCB image, 10 mm grid", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image_id}.jpg"
    cv2.imwrite(str(path), np.vstack([header, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summary / audit / reference check of the clean collection.")
    parser.add_argument("--capture-config", type=Path, default=DEFAULT_CAPTURE_CONFIG)
    parser.add_argument("--product", default=None)
    parser.add_argument("--verify-reference", metavar="IMAGE_ID")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="write the flat metadata table")
    args = parser.parse_args()
    capture = load_config(args.capture_config)
    store = build_store(capture, args.product or capture["product"])
    if args.verify_reference:
        ok, lines = verify_reference(store, args.verify_reference)
        print("\n".join(lines))
        return 0 if ok else 1
    s = summarise(store)
    from pcb_inspection.inspection import load_reference_bank
    from pcb_inspection.products import load_product_profile
    bank = load_reference_bank(load_product_profile(args.product or capture["product"], project_root=ROOT))
    s["bank_status"] = f"{bank.status} - {bank.reason}" + (f" ({len(bank.skipped)} GOOD image(s) skipped as examples, "
                                                           "e.g. placed the other way round)" if bank.skipped else "")
    print(format_summary(store, s))
    if args.csv:
        print(f"\nMetadata table: {write_csv(store, args.csv)} rows -> {args.csv}")
    return 1 if s["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
