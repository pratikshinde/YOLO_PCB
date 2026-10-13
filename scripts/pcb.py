"""PCB inspection system: products, boards, dataset validation, inspection of stored images, human review.

    python scripts/pcb.py products                         every product profile, its golden bank and readiness
    python scripts/pcb.py boards --product lg_indicator    physical boards of a product (pool, condition, split, images)
    python scripts/pcb.py validate [--product ID] [--no-unlabelled] [--no-bank] [--save]
                                                           every image of every product: resolved / unresolved /
                                                           pending / synthetic / invalid, duplicates, orphans, ...
    python scripts/pcb.py inspect <image> [<image> ...] [--product AUTO|ID] [--board-id GOOD_003]
                                                           inspect stored image files (recorded, never moved)
    python scripts/pcb.py review <inspection_id> --decision CONFIRM_DEFECT|CONFIRM_GOOD|REJECT_RESULT --reviewer NAME

Live inspection: scripts/station.py. Data collection (GOOD / DEFECT / TEST boards): scripts/acquire.py --product ID.
Architecture: docs/inspection_architecture.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.dataset.product_discovery import format_validation, validate_all  # noqa: E402
from pcb_inspection.inspection import load_reference_bank  # noqa: E402
from pcb_inspection.products import load_all_profiles, load_product_profile  # noqa: E402

SETTINGS = ROOT / "configs" / "inspection.yaml"


def rel(path: Path | str | None) -> str:
    if path is None:
        return "-"
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def cmd_products(args) -> int:
    profiles, problems = load_all_profiles(ROOT)
    for p in profiles:
        bank = load_reference_bank(p)
        print(f"{p.product_id}  ({p.display_name})")
        print(f"  board: {p.board_mm or 'UNKNOWN'} mm, side {p.side or 'UNKNOWN'}, stages {p.stages or 'UNKNOWN'}, "
              f"colours {p.colour_variants or 'UNKNOWN'}")
        print(f"  engine: {p.engine_type or 'UNKNOWN'} ({p.engine.get('status') or '-'}); "
              f"defect types enabled: {', '.join(p.enabled_defect_types) or 'none'}")
        print(f"  golden bank: {bank.status} - {bank.reason}")
        if bank.boards:
            print(f"    boards: {', '.join(bank.boards)}"
                  + (f" (identity unverified: {', '.join(bank.unverified_boards)})" if bank.unverified_boards else ""))
        print(f"  automatic identification: {'possible' if bank.can_identify and p.auto_identification else 'NOT possible'}"
              f"; inspection: {'runs (results may still be REVIEW, see docs/inspection_architecture.md)' if bank.can_inspect else 'every result REVIEW (no usable golden bank)'}")
        print(f"  storage: {rel(p.storage_root)}  data collection: {rel(p.acquisition_root)}  registry: {rel(p.registry_dir)}")
        print(f"  unknown (null) profile values: {len(p.unknowns())}")
    for problem in problems:
        print(f"PROFILE PROBLEM: {problem}")
    return 1 if problems else 0


def _validation(args, product: str | None = None):
    settings = load_config(SETTINGS)
    profiles, problems = load_all_profiles(ROOT)
    if product:
        profiles = [p for p in profiles if p.product_id == product]
        if not profiles:
            raise SystemExit(f"unknown product {product!r}")
    banks = None if args.no_bank else {p.product_id: load_reference_bank(p).summary() for p in profiles}
    v = validate_all(ROOT, profiles, capture_root=ROOT / settings["storage"]["capture_store"],
                     derived=list(settings["storage"].get("derived_paths") or []),
                     include_unlabelled=not args.no_unlabelled, banks=banks,
                     cache_file=ROOT / "data" / "processed" / "checksums.json")
    v.problems = problems + v.problems
    return v


def cmd_boards(args) -> int:
    args.no_unlabelled = True
    v = _validation(args, args.product)
    for pid, r in v.products.items():
        print(f"{pid}: physical boards")
        print(f"  {'board':<11}{'pool':<8}{'condition':<10}{'split':<11}{'sessions':>9}{'images':>8}  bank  defects / identity")
        for b in sorted(r.boards.values(), key=lambda b: b.board_id):
            bank = "-" if b.in_bank is None else ("yes" if b.in_bank else "NO")
            extra = b.defects or (b.identity_source or "") + (f", verified by {b.verified_by}" if b.verified_by else "")
            print(f"  {b.board_id:<11}{b.pool or '-':<8}{b.condition or 'UNREG.':<10}{b.split or '-':<11}"
                  f"{len(b.sessions):>9}{b.images:>8}  {bank:<5} {extra}")
    return 0


def cmd_validate(args) -> int:
    v = _validation(args, args.product)
    text = format_validation(v, limit=args.limit)
    print(text)
    if args.save:
        out = ROOT / "data" / "processed" / "reports" / f"validation_{datetime.now():%Y%m%d_%H%M%S}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nsaved: {rel(out)}")
    return 1 if v.problems or any(r.issues for r in v.products.values()) else 0


def cmd_inspect(args) -> int:
    from pcb_inspection.station import InspectionStation
    station = InspectionStation(ROOT, stage=args.stage)
    for problem in station.problems:
        print(f"PROFILE PROBLEM: {problem}")
    from pcb_inspection.station import STORED_IMAGE
    session = None
    for name in args.images:
        path = Path(name)
        if not path.is_file():
            print(f"{name}: not a file")
            continue
        if session is None:
            session = station.start_session(args.product, {"kind": "stored image files"}, STORED_IMAGE)
        r = station.inspect_file(path, args.product, args.board_id, session)
        o = r.outcome
        print("=" * 60)
        print(f"Image:        {rel(path)}  ({o.image_id}, sha256 {r.capture.record['sha256'][:12]}...)")
        print(f"Product:      {r.product_name or 'NOT DETERMINED'}  [{r.identification.status if r.identification else '-'}]"
              + (f" score {r.identification.score:.2f}" if r.identification and r.identification.score is not None else ""))
        print(f"RESULT:       {o.overall_decision}" + ("" if o.confidence is None else f"  confidence {o.confidence:.2f}"))
        for reason in o.reasons:
            print(f"  {reason.code}: {reason.text}")
        for d in o.defects:
            print(f"  DEFECT {d.defect_type} at {d.region} box {d.location['box_mm']} mm "
                  f"(similarity {d.evidence['similarity']} vs limit {d.evidence['limit']}; confidence: not calibrated)")
        print(f"Alignment:    {o.alignment_status}  quality: {o.quality_status}  engine: {o.engine} ({o.engine_status})")
        print(f"Inspection:   {o.inspection_id}  -> {rel(r.result_path)}")
    if session is not None:
        station.captures.end_session(session)
    return 0


def cmd_review(args) -> int:
    from pcb_inspection.station import find_result, record_review
    settings = load_config(SETTINGS)
    profiles, _ = load_all_profiles(ROOT)
    roots = [p.storage_root for p in profiles] + [ROOT / settings["storage"]["unresolved_product_root"]]
    path = find_result(roots, args.inspection_id)
    if path is None:
        print(f"inspection {args.inspection_id} not found")
        return 1
    review = record_review(path, args.decision, args.reviewer, args.note)
    print(json.dumps(review, indent=2))
    print("Stored in the product's reviews/reviews.jsonl; the AI result was not changed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("products", help="list product profiles and golden banks").set_defaults(func=cmd_products)
    b = sub.add_parser("boards", help="physical boards of a product")
    b.add_argument("--product")
    b.add_argument("--no-bank", action="store_true", help="do not load the golden banks (faster)")
    b.set_defaults(func=cmd_boards)
    v = sub.add_parser("validate", help="validate every image of every product")
    v.add_argument("--product")
    v.add_argument("--no-unlabelled", action="store_true", help="do not checksum unlabelled folders (live captures)")
    v.add_argument("--no-bank", action="store_true", help="do not load the golden banks (faster)")
    v.add_argument("--limit", type=int, default=12, help="list at most this many items per section")
    v.add_argument("--save", action="store_true", help="also save the report under data/processed/reports/")
    v.set_defaults(func=cmd_validate)
    i = sub.add_parser("inspect", help="inspect stored image files")
    i.add_argument("images", nargs="+")
    i.add_argument("--product", default="AUTO", help="AUTO (default) or a product id")
    i.add_argument("--board-id", default=None, help="physical board, if known (leave-one-board-out for bank boards)")
    i.add_argument("--stage", default=None)
    i.set_defaults(func=cmd_inspect)
    r = sub.add_parser("review", help="human review of a stored inspection")
    r.add_argument("inspection_id")
    r.add_argument("--decision", required=True, choices=["CONFIRM_DEFECT", "CONFIRM_GOOD", "REJECT_RESULT"])
    r.add_argument("--reviewer", required=True)
    r.add_argument("--note", default="")
    r.set_defaults(func=cmd_review)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
