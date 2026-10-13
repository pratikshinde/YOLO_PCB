"""Experiment 1B: grouping, summaries and the handheld-vs-controlled comparison.

Groups (all from human records, never from the images):
  A/B  same physical board as the reference, re-placed between captures
       (A and B coincide when every capture is taken after re-placing the board)
  C    other boards recorded as good in boards.csv
  D    boards recorded as defective, with their verified defects from defects.csv

On good boards (A/B, C) every candidate region is a false candidate.
For D, only verified defect boxes (defect_box_mm) count as location ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..dataset.experiment0 import parse_defect_box

GROUP_SAME = "A/B same board re-placed"
GROUP_OTHER_GOOD = "C other good board"
GROUP_DEFECTIVE = "D verified defective"
GOOD_GROUPS = (GROUP_SAME, GROUP_OTHER_GOOD)
SUMMARY_METRICS = (
    "inlier_rmse_mm", "residual_median_mm", "ssim_mean", "delta_e_mean", "edge_mismatch_fraction",
    "candidate_regions", "candidate_area_fraction", "largest_candidate_mm2", "total_seconds",
)


@dataclass
class PlannedTest:
    group: str
    capture: Mapping[str, str]
    defects: list[Mapping[str, str]] = field(default_factory=list)
    boxes_mm: list[tuple[float, float, float, float]] = field(default_factory=list)


def plan_groups(captures: Iterable[Mapping[str, str]], boards: Iterable[Mapping[str, str]],
                defects: Iterable[Mapping[str, str]], reference_id: str) -> tuple[list[PlannedTest], list[str]]:
    """Assign each capture to a group, or explain why it is skipped."""
    captures = list(captures)
    board_rows = {b["board_id"]: b for b in boards}
    defects_by_board: dict[str, list[Mapping[str, str]]] = {}
    for d in defects:
        defects_by_board.setdefault(d["board_id"], []).append(d)
    reference = next((c for c in captures if c["image_id"] == reference_id), None)
    if reference is None:
        raise ValueError(f"reference {reference_id} is not among the selected captures")
    reference_board = reference["board_id"]

    planned, skipped = [], []
    for c in captures:
        image_id, board_id = c["image_id"], c["board_id"]
        if image_id == reference_id:
            continue
        if not board_id:
            skipped.append(f"{image_id}: no board_id, so the physical board is unknown")
            continue
        record = board_rows.get(board_id)
        if record is None or record.get("condition") not in ("good", "defective"):
            skipped.append(f"{image_id}: board {board_id} has no verified condition in boards.csv")
            continue
        if record["condition"] == "good":
            planned.append(PlannedTest(GROUP_SAME if board_id == reference_board else GROUP_OTHER_GOOD, c))
            continue
        board_defects = defects_by_board.get(board_id, [])
        if not board_defects:
            skipped.append(f"{image_id}: defective board {board_id} has no verified defect in defects.csv")
            continue
        boxes = [parse_defect_box(d["defect_box_mm"]) for d in board_defects if d.get("defect_box_mm")]
        planned.append(PlannedTest(GROUP_DEFECTIVE, c, list(board_defects), boxes))
    return planned, skipped


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for r in rows:
        try:
            values.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def summarise(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per group: count, registration successes, median and max of the key metrics."""
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(str(r["group"]), []).append(r)
    summary = {}
    for group, group_rows in sorted(by_group.items()):
        ok = [r for r in group_rows if str(r["registration_ok"]) in ("True", "true", "1")]
        entry: dict[str, Any] = {"pairs": len(group_rows), "registered": len(ok)}
        for key in SUMMARY_METRICS:
            values = _numbers(ok, key)
            entry[key] = {"median": median(values), "max": max(values)} if values else None
        summary[group] = entry
    return summary


def compare_to_baseline(baseline_rows: Iterable[Mapping[str, Any]], controlled_rows: Iterable[Mapping[str, Any]],
                        keys: tuple[str, ...] = ("candidate_regions", "candidate_area_fraction", "ssim_mean",
                                                 "delta_e_mean", "residual_median_mm")) -> dict[str, Any]:
    """Medians of the handheld baseline (Experiment 1) vs the controlled run, for same-board and other-board pairs."""
    baseline_rows, controlled_rows = list(baseline_rows), list(controlled_rows)
    is_true = lambda v: str(v) in ("True", "true", "1")  # noqa: E731
    pairs = {
        "same board": ([r for r in baseline_rows if is_true(r["same_board"])],
                       [r for r in controlled_rows if r["group"] == GROUP_SAME]),
        "other good board": ([r for r in baseline_rows if not is_true(r["same_board"])],
                             [r for r in controlled_rows if r["group"] == GROUP_OTHER_GOOD]),
    }
    out: dict[str, Any] = {}
    for label, (before, after) in pairs.items():
        entry: dict[str, Any] = {"handheld_pairs": len(before), "controlled_pairs": len(after)}
        for key in keys:
            b, a = _numbers(before, key), _numbers(after, key)
            if not b or not a:
                entry[key] = None
                continue
            mb, ma = median(b), median(a)
            entry[key] = {"handheld": mb, "controlled": ma, "change": ma - mb,
                          "change_percent": (ma - mb) / mb * 100 if mb else None}
        out[label] = entry
    return out


def defect_findings(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For each verified defect box: was it overlapped, by what, and how does that compare with
    the largest false candidate seen on good boards in the same run? Descriptive only."""
    rows = list(rows)
    false_areas = [float(r["largest_candidate_mm2"]) for r in rows
                   if r["group"] in GOOD_GROUPS and r.get("largest_candidate_mm2") not in ("", None)]
    largest_false = max(false_areas) if false_areas else None
    findings = []
    for r in rows:
        if r["group"] != GROUP_DEFECTIVE:
            continue
        boxes = r.get("_verified_boxes") or []
        if not boxes:
            findings.append({"test_id": r["test_id"], "board": r["test_board"], "ground_truth_location": False,
                             "note": "no defect_box_mm recorded: overlap cannot be evaluated"})
            continue
        for b in boxes:
            regions = b.get("overlapping_regions", [])
            biggest = max((reg["area_mm2"] for reg in regions), default=0.0)
            findings.append({
                "test_id": r["test_id"], "board": r["test_board"], "ground_truth_location": True,
                "box_mm": b["box_mm"], "overlapped": bool(regions), "overlapping_regions": len(regions),
                "largest_overlapping_area_mm2": biggest, "rank_among_candidates": b.get("largest_overlapping_rank"),
                "candidates_on_this_board": r.get("candidate_regions"),
                "largest_false_candidate_on_good_boards_mm2": largest_false,
                "larger_than_every_false_candidate": (biggest > largest_false) if (regions and largest_false is not None) else None,
            })
    return findings
