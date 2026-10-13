"""Experiment 1B: controlled-capture repeatability, board-to-board variation and verified defects.

Uses exactly the Experiment 1 pipeline and configuration, so the results can be
compared with the handheld baseline (experiments/experiment_001/results.csv).

    python scripts/run_experiment_001b.py --sessions S20261012-01
        [--reference <image_id>] [--reference-board GOOD_001]

Photos are selected from the Experiment 0 capture log by session (the controlled
capture session(s)). Groups come from human records only:
  A/B  other captures of the reference board (re-placed between captures)
  C    other boards recorded as good in boards.csv
  D    boards recorded as defective, with verified defects in defects.csv;
       a defect's location counts as ground truth only if defect_box_mm is filled in.
Nothing is labelled from the images. The raw dataset is never modified.

Outputs (--output, default experiments/experiment_001b/):
  reference/<id>/      canonical reference + reference_mm_grid.jpg (for reading defect boxes in mm)
  runs/<ref>__<test>/  the same images and metrics.json as Experiment 1
  results.csv          one row per pair, with its group
  summary.json, summary.md   per-group statistics, handheld vs controlled, defect findings
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_inspection.comparison.experiment1b import (  # noqa: E402
    GOOD_GROUPS,
    compare_to_baseline,
    defect_findings,
    plan_groups,
    summarise,
)
from pcb_inspection.comparison.pipeline import compare_pair, load_pipeline_config, merge_results, prepare_reference  # noqa: E402
from pcb_inspection.dataset.experiment0 import E0_BOARDS, E0_CAPTURES, E0_DEFECTS  # noqa: E402
from pcb_inspection.dataset.manifest import read_table  # noqa: E402

from pcb_inspection.dataset.registry import load_registry  # noqa: E402

E0_ROOT = ROOT / "data" / "experiment_0"


def read(schema, e0_root: Path = E0_ROOT) -> list[dict[str, str]]:
    path = e0_root / "metadata" / schema.filename
    return read_table(path, schema)[0] if path.is_file() else []


def load_all_captures(root: Path) -> list[dict[str, Any]]:
    captures = []
    e0_root = root / "data" / "experiment_0"
    # 1. Experiment 0 captures
    for c in read(E0_CAPTURES, e0_root):
        c_copy = dict(c)
        c_copy["full_path"] = e0_root / c["image_file"].replace("\\", "/")
        captures.append(c_copy)

    # 2. Acquisition sessions (from scripts/acquire.py and curated)
    dataset_dir = root / "data" / "dataset"
    registry = load_registry(dataset_dir / "boards.csv", dataset_dir / "image_boards.csv", dataset_dir / "session_boards.csv")

    for path in sorted((root / "data" / "acquisition").glob("*/*/S*/session.json")):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        session_id = s.get("session_id", "")
        sess_board_id = s.get("board_id")
        for r in s.get("images", []):
            rev = r.get("review", {})
            # Only include kept captures (or those with non-rejected review)
            if rev.get("status") not in ("kept", None, ""):
                continue
            image_id = r.get("image_id", "")
            board = sess_board_id or ""
            if session_id in registry.session_boards:
                board = registry.session_boards[session_id][0]
            if image_id in registry.image_boards:
                board = registry.image_boards[image_id][0]

            rel_file = r.get("saved_file") or r.get("original_file") or ""
            full_path = root / "data" / rel_file if rel_file else None
            if full_path and full_path.is_file():
                captures.append({
                    "image_id": image_id,
                    "session_id": session_id,
                    "board_id": board,
                    "captured_at": r.get("captured_at", ""),
                    "sha256": r.get("sha256", ""),
                    "image_file": rel_file,
                    "full_path": full_path,
                })
    return captures


def load_all_boards(root: Path) -> list[dict[str, str]]:
    boards_by_id = {}
    for b in read(E0_BOARDS):
        boards_by_id[b["board_id"]] = b
    dataset_boards = root / "data" / "dataset" / "boards.csv"
    if dataset_boards.is_file():
        with dataset_boards.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                bid = r.get("board_id", "").strip()
                if bid and bid not in boards_by_id:
                    boards_by_id[bid] = {
                        "board_id": bid,
                        "condition": r.get("condition", "good"),
                        "notes": r.get("notes", ""),
                    }
    return list(boards_by_id.values())


def fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_markdown(path: Path, reference_id: str, sessions: list[str], summary: dict, baseline: dict,
                   findings: list[dict], skipped: list[str]) -> None:
    lines = ["# Experiment 1B summary", "", f"Reference: `{reference_id}`; sessions: {', '.join(sessions)}.",
             "Candidate regions on good boards are false candidates. Nothing here is a defect verdict.", "",
             "## Per group (medians, registered pairs)", "",
             "| Group | Pairs | Registered | Residual mm | SSIM | dE mean | Edge mismatch | Candidates | % board flagged | Largest mm² | s/pair |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group, s in summary.items():
        m = lambda k, d=3: fmt(s[k]["median"], d) if s[k] else "n/a"  # noqa: E731
        pct = f"{s['candidate_area_fraction']['median'] * 100:.2f}" if s["candidate_area_fraction"] else "n/a"
        lines.append(f"| {group} | {s['pairs']} | {s['registered']} | {m('residual_median_mm', 4)} | {m('ssim_mean')} | "
                     f"{m('delta_e_mean', 2)} | {m('edge_mismatch_fraction')} | {m('candidate_regions', 0)} | {pct} | "
                     f"{m('largest_candidate_mm2', 1)} | {m('total_seconds', 1)} |")
    lines += ["", "## Handheld (Experiment 1) vs controlled (medians)", "",
              "| Comparison | Metric | Handheld | Controlled | Change | Change % |", "|---|---|---|---|---|---|"]
    for label, entry in baseline.items():
        for key, v in entry.items():
            if key.endswith("_pairs"):
                continue
            if v is None:
                lines.append(f"| {label} | {key} | n/a | n/a | n/a | n/a |")
            else:
                lines.append(f"| {label} | {key} | {fmt(v['handheld'])} | {fmt(v['controlled'])} | {fmt(v['change'])} | "
                             f"{fmt(v['change_percent'], 1)} |")
        lines.append(f"| {label} | pairs | {entry['handheld_pairs']} | {entry['controlled_pairs']} | | |")
    lines += ["", "## Verified defects", ""]
    if not findings:
        lines.append("No verified defective board in the selected sessions: defect separation NOT evaluated.")
    for f in findings:
        if not f["ground_truth_location"]:
            lines.append(f"- {f['test_id']} ({f['board']}): {f['note']}")
        else:
            lines.append(f"- {f['test_id']} ({f['board']}), box {f['box_mm']} mm: overlapped = {f['overlapped']}; "
                         f"largest overlapping region {fmt(f['largest_overlapping_area_mm2'], 1)} mm² "
                         f"(rank {f['rank_among_candidates']} of {f['candidates_on_this_board']} candidates); "
                         f"largest false candidate on good boards {fmt(f['largest_false_candidate_on_good_boards_mm2'], 1)} mm²; "
                         f"larger than every false candidate = {f['larger_than_every_false_candidate']}")
    if skipped:
        lines += ["", "## Skipped photos", ""] + [f"- {s}" for s in skipped]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 1B: controlled capture vs handheld baseline.")
    parser.add_argument("--sessions", nargs="+", required=True, help="controlled capture session IDs, e.g. S20261012-01")
    parser.add_argument("--reference", help="reference image ID (default: first capture of --reference-board)")
    parser.add_argument("--reference-board", default=None, help="reference board ID (e.g. GOOD_001, GOOD_002, GOOD_003)")
    parser.add_argument("--baseline", type=Path, default=ROOT / "experiments" / "experiment_001" / "results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "experiment_001b")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_001" / "config.yaml",
                        help="defaults to the Experiment 1 config, so both experiments use identical parameters")
    args = parser.parse_args()

    cfg = load_pipeline_config(ROOT, args.config)
    all_captures = load_all_captures(ROOT)
    captures = [c for c in all_captures if c["session_id"] in args.sessions]
    if not captures:
        raise SystemExit(f"no photos found in sessions {args.sessions}; check acquisition sessions or experiment_0")
    captures.sort(key=lambda c: c["captured_at"])

    reference_id = args.reference
    if not reference_id:
        if args.reference_board:
            reference_id = next((c["image_id"] for c in captures if c["board_id"] == args.reference_board), None)
        if not reference_id:
            # Prefer first capture of any known good board
            reference_id = next((c["image_id"] for c in captures if (c.get("board_id") or "").startswith("GOOD")), None)
        if not reference_id and captures:
            reference_id = captures[0]["image_id"]

    if reference_id is None:
        raise SystemExit(f"no valid reference photo found in sessions {args.sessions}; use --reference")

    planned, skipped = plan_groups(captures, load_all_boards(ROOT), read(E0_DEFECTS), reference_id)

    ref_row = next(c for c in captures if c["image_id"] == reference_id)
    ref_info = {"id": reference_id, "path": str(ref_row["full_path"]), "board_id": ref_row["board_id"],
                "sha256": ref_row["sha256"]}
    print(f"Reference: {reference_id} (board {ref_info['board_id']}); {len(planned)} test photo(s); {len(skipped)} skipped")
    reference = prepare_reference(ref_info, cfg, args.output / "reference" / reference_id, "Experiment 1B")

    rows = []
    for p in planned:
        c = p.capture
        test = {"id": c["image_id"], "path": str(c["full_path"]), "board_id": c["board_id"], "sha256": c["sha256"]}
        row = compare_pair(reference, test, cfg, args.output / "runs" / f"{reference_id}__{c['image_id']}",
                           group=p.group, verified_boxes_mm=p.boxes_mm, verified_defects=p.defects)
        rows.append(row)
        status = "registration FAILED" if not row["registration_ok"] else (
            f"SSIM {row['ssim_mean']}, candidates {row['candidate_regions']} ({row['candidate_area_fraction']:.2%} of board)"
            + (f", verified boxes overlapped {row['defect_boxes_overlapped']}/{row['verified_defect_boxes']}"
               if p.group not in GOOD_GROUPS else ""))
        print(f"  [{p.group}] {c['image_id']} ({c['board_id']}): {status}")
    for s in skipped:
        print(f"  SKIPPED {s}")

    merge_results(args.output / "results.csv", rows)
    summary = summarise(rows)
    baseline_rows = list(csv.DictReader(args.baseline.open(encoding="utf-8"))) if args.baseline.is_file() else []
    baseline = compare_to_baseline(baseline_rows, rows)
    findings = defect_findings(rows)
    (args.output / "summary.json").write_text(json.dumps(
        {"reference": reference_id, "sessions": args.sessions, "groups": summary, "handheld_vs_controlled": baseline,
         "defect_findings": findings, "skipped": skipped}, indent=2, default=str), encoding="utf-8")
    write_markdown(args.output / "summary.md", reference_id, args.sessions, summary, baseline, findings, skipped)
    print(f"\nSummary: {args.output / 'summary.md'}  ;  results: {args.output / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
