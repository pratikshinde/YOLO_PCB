"""Unified dataset discovery and deduplication pipeline for PCB inspection.

Discovers all valid GOOD PCB data across supported data sources:
    - data/experiment_0/     (historical phone photos, captures.csv)
    - data/acquisition/      (camera acquisition originals and session.json)
    - data/curated/          (human-reviewed and kept image copies)
    - data/raw/              (production track captures)
    - data/experiment_1b/    (controlled capture experiment data if present)

Key principles:
1. Physical Board Identity:
   A physical PCB is independent from an image. One board (e.g. GOOD_001) can have
   multiple sessions and multiple images. Different physical boards receive distinct
   board IDs (GOOD_001, GOOD_002, GOOD_003).
2. Deduplication by SHA-256:
   Images appearing across pipeline stages (e.g. raw acquisition vs curated copy,
   or experiment_0 images vs inbox/_ingested) share identical SHA-256 checksums.
   Deduplication recognizes them as the same underlying sample, retains traceability,
   and prevents double-counting.
3. Explicit Board Assignment:
   If physical board identity was not recorded at capture (UNASSIGNED), it can be
   assigned explicitly at the SESSION level (session_boards.csv) or image level
   (image_boards.csv). Never guessed from image similarity.
4. Test and Defect Isolation:
   TEST and DEFECT images are strictly excluded from the GOOD dataset.
5. Quality Checks:
   Images are evaluated against project quality checks; flags are reported without
   destructive deletion.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import load_config, load_profile
from ..image_quality.checks import ImageQuality, QualityParams, check_image, flag_relative_sharpness
from .experiment0 import E0_CAPTURES, sha256_of
from .manifest import read_table
from .naming import BOARD_ID_PATTERN, parse_capture_filename

SUPPORTED_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
VALID_GOOD_BOARD_PATTERN = re.compile(r"GOOD_\d{3}")
VALID_BOARD_PATTERN = re.compile(r"(GOOD|DEFECT|TEST)_\d{3}")
SESSION_ID_PATTERN = re.compile(r"S\d{8}-\d{2}")


@dataclass
class DiscoveredImage:
    image_id: str
    file_path: Path
    relative_path: str
    sha256: str
    board_id: str | None
    board_source: str
    session_id: str
    dataset_type: str            # good | defect | test
    condition: str               # good | fault | unknown
    split: str                   # reference | validation | test | unassigned
    stage: str
    source_stage: str            # curated | acquisition | experiment_0 | raw
    width: int | None = None
    height: int | None = None
    captured_at: str = ""
    camera: str = ""
    lighting: str = ""
    quality_status: str = "pass"
    quality_flags: list[str] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    is_duplicate: bool = False
    duplicate_paths: list[str] = field(default_factory=list)


@dataclass
class DiscoveredSession:
    session_id: str
    dataset_type: str
    board_id: str | None
    source_location: str
    stage: str = ""
    images: list[DiscoveredImage] = field(default_factory=list)


@dataclass
class DiscoveredBoard:
    board_id: str
    condition: str
    split: str
    sessions: dict[str, DiscoveredSession] = field(default_factory=dict)
    images: list[DiscoveredImage] = field(default_factory=list)


@dataclass
class GoodDatasetReport:
    boards: dict[str, DiscoveredBoard] = field(default_factory=dict)
    unique_images: list[DiscoveredImage] = field(default_factory=list)
    duplicate_images: list[tuple[str, str, str]] = field(default_factory=list)  # (path, sha256, primary_path)
    invalid_images: list[tuple[str, str]] = field(default_factory=list)        # (path, reason)
    unassigned_sessions: list[str] = field(default_factory=list)
    unassigned_images: list[str] = field(default_factory=list)
    unknown_board_ids: list[str] = field(default_factory=list)
    quality_flagged: list[tuple[str, list[str]]] = field(default_factory=list)
    not_reviewed: list[tuple[str, str]] = field(default_factory=list)   # (image path, review status): GOOD captures
                                                                        # not KEPT yet (pending) or rejected; reported,
                                                                        # not counted as GOOD data
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_good_boards(self) -> int:
        return len(self.boards)

    @property
    def total_good_sessions(self) -> int:
        seen = set()
        for b in self.boards.values():
            seen.update(b.sessions.keys())
        return len(seen)

    @property
    def total_good_images(self) -> int:
        return len(self.unique_images)

    @property
    def images_per_board(self) -> dict[str, int]:
        return {b_id: len(b.images) for b_id, b in sorted(self.boards.items())}

    @property
    def sessions_per_board(self) -> dict[str, int]:
        return {b_id: len(b.sessions) for b_id, b in sorted(self.boards.items())}

    @property
    def duplicate_image_count(self) -> int:
        return len(self.duplicate_images)

    @property
    def invalid_image_count(self) -> int:
        return len(self.invalid_images)

    @property
    def missing_metadata_count(self) -> int:
        return len(self.unassigned_images)

    @property
    def source_locations(self) -> dict[str, list[str]]:
        locs: dict[str, set[str]] = defaultdict(set)
        for b_id, b in self.boards.items():
            for s in b.sessions.values():
                locs[b_id].add(s.source_location)
        return {k: sorted(v) for k, v in sorted(locs.items())}


def load_session_boards(session_boards_csv: Path | None) -> dict[str, tuple[str, str]]:
    """Loads session_id -> (board_id, assignment_source) from session_boards.csv."""
    if session_boards_csv is None or not session_boards_csv.is_file():
        return {}
    results = {}
    with session_boards_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for line, row in enumerate(reader, start=2):
            sid = (row.get("session_id") or "").strip()
            bid = (row.get("board_id") or "").strip()
            src = (row.get("assignment_source") or "session_boards").strip()
            if sid and bid:
                results[sid] = (bid, src)
    return results


def discover_good_dataset(
    data_root: Path,
    *,
    boards_csv: Path | None = None,
    image_boards_csv: Path | None = None,
    session_boards_csv: Path | None = None,
    run_quality_checks: bool = False,
    quality_params: QualityParams | None = None,
    board_dims_mm: tuple[float, float] | None = None,     # product board size (quality shape check); None = skip
) -> GoodDatasetReport:
    """Discovers all valid GOOD PCB data across all supported sources with SHA-256 deduplication."""
    data_root = Path(data_root).resolve()
    project_root = data_root.parent

    # 1. Load registries & board mappings
    dataset_dir = data_root / "dataset"
    boards_path = boards_csv or (dataset_dir / "boards.csv")
    image_boards_path = image_boards_csv or (dataset_dir / "image_boards.csv")
    session_boards_path = session_boards_csv or (dataset_dir / "session_boards.csv")

    known_boards: dict[str, dict[str, str]] = {}
    if boards_path.is_file():
        with boards_path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                bid = (r.get("board_id") or "").strip()
                if bid:
                    known_boards[bid] = {k: (v or "").strip() for k, v in r.items()}

    # Also incorporate metadata/boards.csv and experiment_0/metadata/boards.csv if available
    for fallback in [data_root / "metadata" / "boards.csv", data_root / "experiment_0" / "metadata" / "boards.csv"]:
        if fallback.is_file():
            with fallback.open(newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    bid = (r.get("board_id") or "").strip()
                    if bid and bid not in known_boards:
                        known_boards[bid] = {
                            "board_id": bid,
                            "condition": r.get("condition", "good"),
                            "split": "reference",
                            "defects": "",
                            "defect_source": "",
                            "identity_source": r.get("source", "metadata_table"),
                            "verified_by": r.get("verified_by", ""),
                            "notes": r.get("notes", ""),
                        }

    # Image-level assignments
    image_boards: dict[str, tuple[str, str]] = {}
    if image_boards_path.is_file():
        with image_boards_path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                iid = (r.get("image_id") or "").strip()
                bid = (r.get("board_id") or "").strip()
                src = (r.get("assignment_source") or "image_boards").strip()
                if iid and bid:
                    image_boards[iid] = (bid, src)

    # Session-level assignments
    session_boards = load_session_boards(session_boards_path)

    report = GoodDatasetReport()

    # Track unique samples by SHA-256
    unique_by_sha: dict[str, DiscoveredImage] = {}

    # Helper: resolve board ID for an image/session
    def resolve_board(image_id: str, session_id: str, session_board: str | None, path_board: str | None) -> tuple[str | None, str]:
        if image_id in image_boards:
            return image_boards[image_id]
        if session_id in session_boards:
            return session_boards[session_id]
        if session_board and session_board != "UNASSIGNED":
            return session_board, "session"
        if path_board and path_board != "UNASSIGNED" and VALID_BOARD_PATTERN.fullmatch(path_board):
            return path_board, "path"
        return None, ""

    # =========================================================================
    # Step A: Scan Curated GOOD images (Preferred high-quality canonical copies)
    # =========================================================================
    curated_good_dir = data_root / "curated" / "good"
    if curated_good_dir.is_dir():
        for img_path in sorted(curated_good_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue

            rel_data = img_path.relative_to(data_root).as_posix()
            parts = img_path.relative_to(curated_good_dir).parts
            path_board = parts[0] if len(parts) >= 2 else None
            session_id = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else "")
            if not SESSION_ID_PATTERN.fullmatch(session_id):
                # Search parts for session pattern
                session_id = next((p for p in parts if SESSION_ID_PATTERN.fullmatch(p)), "")

            digest = sha256_of(img_path)
            image_id = img_path.stem

            board_id, board_source = resolve_board(image_id, session_id, None, path_board)

            # Check if this exact image was already registered
            if digest in unique_by_sha:
                existing = unique_by_sha[digest]
                existing.duplicate_paths.append(rel_data)
                report.duplicate_images.append((rel_data, digest, existing.relative_path))
                continue

            # Quality metrics
            q_status, q_flags, q_metrics = "pass", [], {}
            if run_quality_checks and quality_params:
                q = check_image(img_path, quality_params, board_dims_mm)
                q_status, q_flags = q.status, list(q.flags)
                q_metrics = {"sharpness": q.sharpness, "mean_brightness": q.board_mean_brightness}
                if q_flags:
                    report.quality_flagged.append((image_id, q_flags))

            disc_img = DiscoveredImage(
                image_id=image_id,
                file_path=img_path,
                relative_path=rel_data,
                sha256=digest,
                board_id=board_id,
                board_source=board_source,
                session_id=session_id,
                dataset_type="good",
                condition="good",
                split="reference",
                stage="curated",
                source_stage="curated",
                quality_status=q_status,
                quality_flags=q_flags,
                quality_metrics=q_metrics,
            )
            unique_by_sha[digest] = disc_img

    # =========================================================================
    # Step B: Scan Acquisition GOOD sessions (session.json + images)
    # =========================================================================
    acq_good_dir = data_root / "acquisition" / "good"
    if acq_good_dir.is_dir():
        for sess_path in sorted(acq_good_dir.rglob("session.json")):
            try:
                s_data = json.loads(sess_path.read_text(encoding="utf-8"))
            except Exception as e:
                report.issues.append(f"{sess_path}: failed to read session.json ({e})")
                continue

            session_id = s_data.get("session_id", "")
            sess_board_id = s_data.get("board_id")
            sess_stage = s_data.get("stage", "")
            sess_camera = str(s_data.get("camera", {}).get("name", ""))
            sess_lighting = s_data.get("lighting") or ""

            # Check folder-level board ID
            parts = sess_path.relative_to(acq_good_dir).parts
            path_board = parts[0] if len(parts) >= 2 else None

            for img_rec in s_data.get("images", []):
                rev = img_rec.get("review", {})
                # Review status must be kept to be part of good dataset; others are reported, never dropped silently
                if rev.get("status") != "kept":
                    report.not_reviewed.append((img_rec.get("original_file", img_rec.get("image_id", "")),
                                                rev.get("status") or "unknown"))
                    continue

                image_id = img_rec.get("image_id", "")
                digest = img_rec.get("sha256") or ""
                orig_file = img_rec.get("original_file", "")
                saved_file = img_rec.get("saved_file") or ""
                full_orig = (data_root / orig_file) if orig_file else None

                if not digest and full_orig and full_orig.is_file():
                    digest = sha256_of(full_orig)

                rel_path = saved_file or orig_file
                best_file = (data_root / rel_path) if rel_path else None
                if best_file is None or not best_file.is_file():
                    if full_orig and full_orig.is_file():
                        best_file = full_orig
                        rel_path = orig_file

                if best_file is None or not best_file.is_file():
                    report.invalid_images.append((orig_file, "file does not exist on disk"))
                    continue

                board_id, board_source = resolve_board(image_id, session_id, sess_board_id, path_board)

                # Quality from session record
                q_rec = img_rec.get("quality", {})
                q_status = q_rec.get("status", "pass")
                q_flags = list(q_rec.get("flags", []))
                q_metrics = {"sharpness": q_rec.get("sharpness"), "brightness": q_rec.get("board_mean_brightness")}
                if q_flags:
                    report.quality_flagged.append((image_id, q_flags))

                if digest in unique_by_sha:
                    existing = unique_by_sha[digest]
                    if orig_file and orig_file not in existing.duplicate_paths and orig_file != existing.relative_path:
                        existing.duplicate_paths.append(orig_file)
                        report.duplicate_images.append((orig_file, digest, existing.relative_path))
                    # Update board ID if existing was unassigned and we now have a resolved board ID
                    if not existing.board_id and board_id:
                        existing.board_id = board_id
                        existing.board_source = board_source
                    continue

                disc_img = DiscoveredImage(
                    image_id=image_id,
                    file_path=best_file,
                    relative_path=rel_path,
                    sha256=digest,
                    board_id=board_id,
                    board_source=board_source,
                    session_id=session_id,
                    dataset_type="good",
                    condition="good",
                    split="reference",
                    stage=sess_stage,
                    source_stage="acquisition",
                    width=img_rec.get("width"),
                    height=img_rec.get("height"),
                    captured_at=img_rec.get("captured_at", ""),
                    camera=sess_camera,
                    lighting=sess_lighting,
                    quality_status=q_status,
                    quality_flags=q_flags,
                    quality_metrics=q_metrics,
                )
                unique_by_sha[digest] = disc_img

    # =========================================================================
    # Step C: Scan Experiment 0 Historical Data (data/experiment_0/)
    # =========================================================================
    e0_captures_path = data_root / "experiment_0" / "metadata" / "captures.csv"
    if e0_captures_path.is_file():
        e0_rows, _ = read_table(e0_captures_path, E0_CAPTURES)
        for r in e0_rows:
            image_id = r["image_id"]
            img_rel = r["image_file"].replace("\\", "/")
            full_path = data_root / "experiment_0" / img_rel
            if not full_path.is_file():
                report.invalid_images.append((f"experiment_0/{img_rel}", "Experiment 0 image file missing"))
                continue

            digest = r["sha256"] or sha256_of(full_path)
            session_id = r["session_id"]
            raw_board = r.get("board_id") or ""
            board_id, board_source = resolve_board(image_id, session_id, raw_board, raw_board)

            if digest in unique_by_sha:
                existing = unique_by_sha[digest]
                rel_str = f"experiment_0/{img_rel}"
                if rel_str != existing.relative_path and rel_str not in existing.duplicate_paths:
                    existing.duplicate_paths.append(rel_str)
                    report.duplicate_images.append((rel_str, digest, existing.relative_path))
                continue

            q_status, q_flags, q_metrics = "pass", [], {}
            if run_quality_checks and quality_params:
                q = check_image(full_path, quality_params, board_dims_mm)
                q_status, q_flags = q.status, list(q.flags)
                q_metrics = {"sharpness": q.sharpness, "mean_brightness": q.board_mean_brightness}
                if q_flags:
                    report.quality_flagged.append((image_id, q_flags))

            disc_img = DiscoveredImage(
                image_id=image_id,
                file_path=full_path,
                relative_path=f"experiment_0/{img_rel}",
                sha256=digest,
                board_id=board_id,
                board_source=board_source or "experiment_0",
                session_id=session_id,
                dataset_type="good",
                condition="good",
                split="reference",
                stage=r.get("inspection_stage", "after_tht"),
                source_stage="experiment_0",
                width=int(r["width"]) if r.get("width") and r["width"].isdigit() else None,
                height=int(r["height"]) if r.get("height") and r["height"].isdigit() else None,
                captured_at=r.get("captured_at", ""),
                camera=r.get("camera_device", ""),
                lighting=r.get("lighting_condition", ""),
                quality_status=q_status,
                quality_flags=q_flags,
                quality_metrics=q_metrics,
            )
            unique_by_sha[digest] = disc_img

    # Also detect duplicates in experiment_0/inbox/_ingested and set_aside
    for aux_dir in [data_root / "experiment_0" / "inbox" / "_ingested", data_root / "experiment_0" / "set_aside"]:
        if aux_dir.is_dir():
            for aux_file in aux_dir.rglob("*"):
                if aux_file.is_file() and aux_file.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    h = sha256_of(aux_file)
                    rel_aux = aux_file.relative_to(data_root).as_posix()
                    if h in unique_by_sha:
                        existing = unique_by_sha[h]
                        if rel_aux != existing.relative_path and rel_aux not in existing.duplicate_paths:
                            existing.duplicate_paths.append(rel_aux)
                            report.duplicate_images.append((rel_aux, h, existing.relative_path))

    # =========================================================================
    # Step D: Scan Production Raw GOOD captures (data/raw/good/) if present
    # =========================================================================
    raw_good_dir = data_root / "raw" / "good"
    if raw_good_dir.is_dir():
        for img_path in sorted(raw_good_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            rel_data = img_path.relative_to(data_root).as_posix()
            try:
                name_info = parse_capture_filename(img_path.name)
                session_id = name_info.session_id
                path_board = name_info.board_id
            except Exception:
                session_id = ""
                path_board = img_path.parent.name

            digest = sha256_of(img_path)
            board_id, board_source = resolve_board(img_path.stem, session_id, path_board, path_board)

            if digest in unique_by_sha:
                existing = unique_by_sha[digest]
                if rel_data != existing.relative_path and rel_data not in existing.duplicate_paths:
                    existing.duplicate_paths.append(rel_data)
                    report.duplicate_images.append((rel_data, digest, existing.relative_path))
                continue

            disc_img = DiscoveredImage(
                image_id=img_path.stem,
                file_path=img_path,
                relative_path=rel_data,
                sha256=digest,
                board_id=board_id,
                board_source=board_source or "raw",
                session_id=session_id,
                dataset_type="good",
                condition="good",
                split="reference",
                stage="raw",
                source_stage="raw",
            )
            unique_by_sha[digest] = disc_img

    # =========================================================================
    # Step E: Aggregate into Boards and Sessions
    # =========================================================================
    report.unique_images = list(unique_by_sha.values())

    for img in report.unique_images:
        board_id = img.board_id
        if not board_id or board_id == "UNASSIGNED":
            report.unassigned_images.append(img.image_id)
            if img.session_id and img.session_id not in report.unassigned_sessions:
                report.unassigned_sessions.append(img.session_id)
            continue

        if not VALID_BOARD_PATTERN.fullmatch(board_id):
            report.unknown_board_ids.append(f"{img.image_id}: invalid board ID format {board_id!r}")
            continue

        if board_id not in known_boards:
            # Report unknown / unverified board ID if not in registry
            if board_id not in report.unknown_board_ids:
                report.unknown_board_ids.append(board_id)

        # Get or create DiscoveredBoard
        if board_id not in report.boards:
            reg_info = known_boards.get(board_id, {})
            cond = reg_info.get("condition", "good")
            split = reg_info.get("split", "reference")
            report.boards[board_id] = DiscoveredBoard(board_id=board_id, condition=cond, split=split)

        board_obj = report.boards[board_id]
        board_obj.images.append(img)

        # Associate with session
        sess_id = img.session_id or "(no_session)"
        if sess_id not in board_obj.sessions:
            board_obj.sessions[sess_id] = DiscoveredSession(
                session_id=sess_id,
                dataset_type=img.dataset_type,
                board_id=board_id,
                source_location=str(img.file_path.parent.relative_to(project_root)).replace("\\", "/"),
                stage=img.stage,
            )
        board_obj.sessions[sess_id].images.append(img)

    return report


def format_good_dataset_summary(report: GoodDatasetReport) -> str:
    """Formats the required GOOD DATASET SUMMARY table."""
    lines = [
        "GOOD DATASET SUMMARY",
        "",
        f"{'Board':<16} {'Sessions':>10} {'Images':>10}",
        "-" * 38,
    ]
    for board_id, board in sorted(report.boards.items()):
        n_sessions = len(board.sessions)
        n_images = len(board.images)
        lines.append(f"{board_id:<16} {n_sessions:>10} {n_images:>10}")

    lines.append("-" * 38)
    lines.append(f"Total boards: {report.total_good_boards}")
    lines.append(f"Total images: {report.total_good_images}")
    lines.append(f"Total unique sessions: {report.total_good_sessions}")
    lines.append(f"Duplicate images: {report.duplicate_image_count}")
    lines.append(f"Invalid images: {report.invalid_image_count}")
    lines.append(f"Missing metadata (unassigned board): {report.missing_metadata_count}")
    if report.not_reviewed:
        lines.append(f"GOOD captures not counted (review not KEEP): {len(report.not_reviewed)}")
        lines += [f"  {status}: {path}" for path, status in report.not_reviewed]

    if report.unknown_board_ids:
        lines.append(f"Unknown board IDs: {', '.join(sorted(set(report.unknown_board_ids)))}")
    if report.unassigned_sessions:
        lines.append(f"Unassigned sessions: {', '.join(sorted(report.unassigned_sessions))}")

    lines.append("\nSource locations:")
    for b_id, locs in report.source_locations.items():
        lines.append(f"  {b_id}: {', '.join(locs)}")

    return "\n".join(lines)
