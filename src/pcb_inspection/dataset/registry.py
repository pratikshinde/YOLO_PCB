"""Board registry: PHYSICAL boards, their ground truth and their evaluation split (Experiment 2 data).

Image count is not board count: many photos of one PCB are one sample. Every curated image is mapped to a
physical board, and every board belongs to exactly one split:

    reference   GOOD boards whose images form the GOOD reference set (appearance + per-component calibration)
    validation  boards used to check / develop the method (e.g. known defect boards)
    test        held-out boards, used only for the final evaluation, never to set anything
    unassigned  not used yet

Files (data/dataset/, written by people, read by the software):
    boards.csv        board_id, condition, split, defects, defect_source, identity_source, verified_by, notes
                      condition: good | fault | unknown (unknown = TEST board not yet resolved)
                      defects:   ground truth recorded by a person, "R37:missing;LED4:missing" (never inferred
                                 from images)
    image_boards.csv  image_id, board_id, assignment_source, notes
                      only for images captured WITHOUT a board ID (the acquisition app records board IDs per
                      session); an entry here overrides the session's board ID

Board IDs follow the project convention (dataset/naming.py, acquisition app): GOOD_###, DEFECT_###, TEST_###;
the prefix records the intake pool, the registry records the verified condition and the split.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

SESSION_COLUMNS = ("session_id", "board_id", "assignment_source", "notes")
BOARD_COLUMNS = ("board_id", "condition", "split", "defects", "defect_source", "identity_source", "verified_by", "notes")
IMAGE_COLUMNS = ("image_id", "board_id", "assignment_source", "notes")
INDEX_COLUMNS = ("image_id", "file", "dataset_type", "board_id", "board_source", "condition", "split", "defects",
                 "session_id", "capture_id", "stage", "lighting", "camera", "resolution", "captured_at")
CONDITIONS = ("good", "fault", "unknown")
SPLITS = ("reference", "validation", "test", "unassigned")
BOARD_ID = re.compile(r"(GOOD|FAULT|DEFECT|TEST)_\d{3}")
DEFECT = re.compile(r"([A-Z]{1,4}\d{1,3}[A-Z]?):([a-z_]+)")


@dataclass(frozen=True)
class Board:
    board_id: str
    condition: str
    split: str
    defects: tuple[tuple[str, str], ...]     # (component, defect type)
    defect_source: str = ""
    identity_source: str = ""
    verified_by: str = ""
    notes: str = ""

    @property
    def provisional(self) -> bool:
        return self.identity_source.startswith("inferred")


@dataclass
class IndexRow:
    image_id: str
    file: str                     # relative to the project root
    dataset_type: str             # acquisition session type: good | defect | test
    board_id: str
    board_source: str             # session | session_boards | image_boards | ""
    condition: str = ""
    split: str = ""
    defects: str = ""
    session_id: str = ""
    capture_id: str = ""
    stage: str = ""
    lighting: str = ""
    camera: str = ""
    resolution: str = ""
    captured_at: str = ""


@dataclass
class Registry:
    boards: dict[str, Board]
    image_boards: dict[str, tuple[str, str]]        # image_id -> (board_id, assignment_source)
    session_boards: dict[str, tuple[str, str]] = field(default_factory=dict)  # session_id -> (board_id, assignment_source)
    issues: list[str] = field(default_factory=list)


def _rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return [{c: (r.get(c) or "").strip() for c in columns} for r in reader if any((v or "").strip() for v in r.values())]


def load_registry(boards_csv: Path, image_boards_csv: Path | None = None,
                  session_boards_csv: Path | None = None) -> Registry:
    reg = Registry({}, {})
    for line, r in enumerate(_rows(boards_csv, BOARD_COLUMNS), start=2):
        where = f"{boards_csv.name}:{line}"
        bid = r["board_id"]
        defects = []
        for item in filter(None, (d.strip() for d in r["defects"].split(";"))):
            m = DEFECT.fullmatch(item)
            if m is None:
                reg.issues.append(f"{where}: defect {item!r} must look like R37:missing")
            else:
                defects.append((m.group(1), m.group(2)))
        if not BOARD_ID.fullmatch(bid):
            reg.issues.append(f"{where}: board_id {bid!r} must look like GOOD_011, DEFECT_021 or TEST_016")
        elif bid in reg.boards:
            reg.issues.append(f"{where}: {bid} listed twice (a physical board has exactly one row and one split)")
        if r["condition"] not in CONDITIONS:
            reg.issues.append(f"{where}: condition {r['condition']!r} must be one of {CONDITIONS}")
        if r["split"] not in SPLITS:
            reg.issues.append(f"{where}: split {r['split']!r} must be one of {SPLITS}")
        if r["condition"] == "fault" and not defects:
            reg.issues.append(f"{where}: fault board {bid} needs its actual defects (e.g. R37:missing)")
        if r["condition"] == "good" and defects:
            reg.issues.append(f"{where}: good board {bid} lists defects")
        if r["split"] == "reference" and r["condition"] != "good":
            reg.issues.append(f"{where}: only good boards can be reference boards ({bid} is {r['condition']})")
        reg.boards.setdefault(bid, Board(bid, r["condition"], r["split"], tuple(defects), r["defect_source"],
                                         r["identity_source"], r["verified_by"], r["notes"]))
    for line, r in enumerate(_rows(image_boards_csv, IMAGE_COLUMNS) if image_boards_csv else [], start=2):
        if r["image_id"] in reg.image_boards:
            reg.issues.append(f"{image_boards_csv.name}:{line}: {r['image_id']} assigned twice")
        reg.image_boards[r["image_id"]] = (r["board_id"], r["assignment_source"])
    if session_boards_csv is None and boards_csv.parent.is_dir():
        cand = boards_csv.parent / "session_boards.csv"
        if cand.is_file():
            session_boards_csv = cand
    if session_boards_csv:
        for line, r in enumerate(_rows(session_boards_csv, SESSION_COLUMNS), start=2):
            sid = r["session_id"]
            if sid in reg.session_boards:
                reg.issues.append(f"{session_boards_csv.name}:{line}: {sid} assigned twice")
            reg.session_boards[sid] = (r["board_id"], r["assignment_source"])
    return reg


def curated_images(data_root: Path, acquisition_dir: str = "acquisition") -> list[dict]:
    """Every KEPT image of every acquisition session, plus Experiment 0 captures, with metadata.

    The image file is the KEEP copy of the full frame (saved_file) or, where a KEEP copied only the PCB image (clean
    collection, data/collection: originals/ layout), the full-frame original (original_file)."""
    images = []
    # 1. Acquisition sessions
    for path in sorted((data_root / acquisition_dir).glob("*/*/S*/session.json")):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in s.get("images", []):
            if r.get("review", {}).get("status") != "kept":
                continue
            images.append({**r, "saved_file": r.get("saved_file") or r.get("original_file"),
                           "session_board_id": s.get("board_id"), "stage_session": s.get("stage"),
                           "lighting": s.get("lighting") or "", "resolution": s.get("resolution"),
                           "notes": s.get("notes") or ""})

    # 2. Experiment 0 captures
    e0_captures = data_root / "experiment_0" / "metadata" / "captures.csv"
    if e0_captures.is_file():
        with e0_captures.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                img_rel = r.get("image_file", "").replace("\\", "/")
                full_path = data_root / "experiment_0" / img_rel
                if full_path.is_file():
                    images.append({
                        "image_id": r["image_id"],
                        "session_id": r["session_id"],
                        "dataset_type": "good",
                        "capture_id": r.get("capture_id") or r["image_id"].split("_")[-1],
                        "stage": r.get("inspection_stage", "after_tht"),
                        "camera": r.get("camera_device") or "",
                        "lighting": r.get("lighting_condition") or "",
                        "width": int(r["width"]) if r.get("width") and r["width"].isdigit() else 0,
                        "height": int(r["height"]) if r.get("height") and r["height"].isdigit() else 0,
                        "captured_at": r.get("captured_at", ""),
                        "saved_file": f"experiment_0/{img_rel}",
                        "session_board_id": r.get("board_id"),
                        "review": {"status": "kept"},
                    })
    return images


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return Path(os.path.relpath(path, project_root)).as_posix()
    except ValueError:            # another drive
        return path.as_posix()


def build_index(data_root: Path, registry: Registry, project_root: Path | None = None,
                extra_roots: Iterable[Path] = (), acquisition_dir: str = "acquisition"
                ) -> tuple[list[IndexRow], list[str], list[str]]:
    """One row per curated image with its physical board, condition and split; plus issues and warnings.

    data_root and extra_roots are acquisition stores (each with <acquisition_dir>/, curated/; data_root may also hold
    experiment_0/). IndexRow.file is relative to project_root (default: data_root's parent)."""
    issues, warnings = list(registry.issues), []
    rows = []
    project_root = data_root.parent if project_root is None else project_root
    found = [(root, r) for root in (data_root, *extra_roots) for r in curated_images(root, acquisition_dir)]
    for store, r in found:
        board, source = r.get("session_board_id") or "", "session" if r.get("session_board_id") else ""
        session_id = r.get("session_id", "")
        if session_id in registry.session_boards:
            board, source = registry.session_boards[session_id][0], "session_boards"
        if r["image_id"] in registry.image_boards:
            board, source = registry.image_boards[r["image_id"]][0], "image_boards"
        b = registry.boards.get(board)
        res = r.get("resolution") or [None, None]
        rows.append(IndexRow(
            image_id=r["image_id"], file=_project_relative(store / r["saved_file"], project_root) if r.get("saved_file") else "",
            dataset_type=r.get("dataset_type", ""), board_id=board, board_source=source,
            condition=b.condition if b else "", split=b.split if b else "",
            defects=";".join(f"{c}:{t}" for c, t in b.defects) if b else "",
            session_id=session_id, capture_id=r.get("capture_id", ""), stage=r.get("stage", ""),
            lighting=r.get("lighting", ""), camera=str(r.get("camera") or ""),
            resolution=f"{r.get('width')}x{r.get('height')}", captured_at=r.get("captured_at", "")))
        row = rows[-1]
        if not board:
            issues.append(f"{row.image_id}: no physical board (enter a board ID at capture, or add it to image_boards.csv or session_boards.csv)")
        elif b is None:
            issues.append(f"{row.image_id}: board {board} is not in the registry (boards.csv)")
        else:
            if row.dataset_type == "good" and b.condition == "fault":
                issues.append(f"{row.image_id}: captured as GOOD but board {board} is a fault board")
            if row.dataset_type in ("defect", "fault") and b.condition == "good":
                issues.append(f"{row.image_id}: captured as DEFECT but board {board} is recorded as good")
            if row.dataset_type == "test" and b.split != "test":
                warnings.append(f"{row.image_id}: TEST image of board {board}, which is in split '{b.split}': "
                                "this TEST image is NOT an unseen board")
        if not (project_root / row.file).is_file():
            issues.append(f"{row.image_id}: curated file missing ({row.file})")
    for bid, b in registry.boards.items():
        if b.provisional:
            warnings.append(f"{bid}: identity is {b.identity_source} (not verified as one physical board)")
        if not any(r.board_id == bid for r in rows):
            warnings.append(f"{bid}: in the registry but no curated image")
    return rows, issues, warnings


def counts(rows: list[IndexRow], registry: Registry) -> list[dict]:
    """Images vs physical boards per condition and split."""
    table = defaultdict(lambda: {"images": 0, "boards": set()})
    for r in rows:
        key = (r.condition or "(no board)", r.split or "-")
        table[key]["images"] += 1
        if r.board_id:
            table[key]["boards"].add(r.board_id)
    return [{"condition": k[0], "split": k[1], "images": v["images"], "physical_boards": len(v["boards"]),
             "boards": ", ".join(sorted(v["boards"]))} for k, v in sorted(table.items())]


def reference_images(rows: list[IndexRow], split: str = "reference") -> list[IndexRow]:
    """GOOD images of the boards in the given split: the GOOD reference set."""
    return [r for r in rows if r.split == split and r.condition == "good" and r.dataset_type == "good"]


def write_index(rows: list[IndexRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def board_counts_text(rows: list[IndexRow]) -> str:
    c = Counter(r.board_id or "(none)" for r in rows)
    return ", ".join(f"{b}: {n} images" for b, n in sorted(c.items()))
