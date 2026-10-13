"""Inspection results and human reviews, stored per product.

    <product root>/inspections/<session_id>/<inspection_id>/result.json     the full InspectionOutcome + image record ref
    <product root>/inspections/<session_id>/<inspection_id>/*.jpg / report.txt   evidence (overview, aligned,
                                                                                 annotated, evidence crops)
    <product root>/inspections/results.jsonl                               one line per inspection (append-only)
    <product root>/reviews/reviews.jsonl                                   human reviews (append-only)

<product root> is the product's storage.root (data/products/<product_id>); inspections whose product could not
be determined go to data/products/_unresolved/. A human review NEVER changes result.json: it is a separate record
with ai_decision, human_decision, reviewer and review_timestamp. The latest review of an inspection wins.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..inspection.types import DEFECT, GOOD, InspectionOutcome

CONFIRM_DEFECT, CONFIRM_GOOD, REJECT_RESULT = "CONFIRM_DEFECT", "CONFIRM_GOOD", "REJECT_RESULT"
REVIEW_DECISIONS = (CONFIRM_DEFECT, CONFIRM_GOOD, REJECT_RESULT)
RESULTS_INDEX = "results.jsonl"
REVIEWS_INDEX = "reviews.jsonl"
_lock = threading.Lock()


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_unreadable_line": line[:200]})
    return rows


def write_result(product_root: Path, session_id: str, outcome: InspectionOutcome, image_record: dict[str, Any],
                 rel: Any = str) -> Path:
    """result.json in the inspection folder (never overwritten) + one line in results.jsonl."""
    folder = product_root / "inspections" / session_id / outcome.inspection_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "result.json"
    if path.exists():
        raise FileExistsError(f"{rel(path)} exists: inspection results are never overwritten")
    payload = {"result_schema": 1, "written_at": datetime.now().isoformat(timespec="milliseconds"),
               **outcome.to_dict(),
               "image": {k: image_record.get(k) for k in ("image_id", "session_id", "file_path", "sha256",
                                                         "capture_timestamp", "camera_id", "resolution", "stage",
                                                         "mode", "synthetic")}}
    with _lock, path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, default=str))
    _append(product_root / "inspections" / RESULTS_INDEX, {
        "inspection_id": outcome.inspection_id, "image_id": outcome.image_id, "session_id": session_id,
        "product_id": outcome.product_id, "variant_id": outcome.variant_id, "board_id": outcome.board_id,
        "decision": outcome.overall_decision, "confidence": outcome.confidence,
        "reasons": [r.code for r in outcome.reasons],
        "defects": [f"{d.defect_type}@{d.region}" for d in outcome.defects],
        "engine": outcome.engine, "file_path": image_record.get("file_path"), "sha256": image_record.get("sha256"),
        "result": rel(path), "written_at": payload["written_at"]})
    return path


def find_result(search_roots: list[Path], inspection_id: str) -> Path | None:
    for root in search_roots:
        for path in (root / "inspections").glob(f"*/{inspection_id}/result.json"):
            return path
    return None


def record_review(result_path: Path, decision: str, reviewer: str, note: str = "",
                  at: datetime | None = None) -> dict[str, Any]:
    """Human review of one stored inspection. The AI result file is read, never changed."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
    if not reviewer or not reviewer.strip():
        raise ValueError("a review needs the reviewer's name")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    product_root = result_path.parents[3]
    review = {
        "inspection_id": result["inspection_id"], "image_id": result["image_id"],
        "product_id": result.get("product_id"), "variant_id": result.get("variant_id"),
        "board_id": result.get("board_id"),
        "ai_decision": result["overall_decision"],
        "ai_defects": [f"{d['defect_type']}@{d['region']}" for d in result.get("defects", [])],
        "human_decision": decision,
        "human_condition": {CONFIRM_DEFECT: "defective", CONFIRM_GOOD: "good", REJECT_RESULT: None}[decision],
        "agrees_with_ai": {CONFIRM_DEFECT: result["overall_decision"] == DEFECT,
                           CONFIRM_GOOD: result["overall_decision"] == GOOD, REJECT_RESULT: False}[decision],
        "reviewer": reviewer.strip(), "note": note.strip() or None,
        "review_timestamp": (at or datetime.now()).isoformat(timespec="seconds"),
        "result_file": str(result_path),
    }
    _append(product_root / "reviews" / REVIEWS_INDEX, review)
    return review


def latest_reviews(product_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(product_root / "reviews" / REVIEWS_INDEX):
        if "inspection_id" in row:
            latest[row["inspection_id"]] = row
    return latest
