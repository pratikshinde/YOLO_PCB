"""Product-aware discovery and validation of ALL captured images (nothing silently ignored).

For every product profile (configs/products/<product_id>/) every store it declares is scanned:

    acquisition store   <acquisition_root>/acquisition/<type>/<board>/<session>/session.json + images/ (ALL review
                        states: kept, pending, rejected), and the curated/ and quarantine/ copies (folder names from
                        the profile's storage.layout; the clean collection keeps KEEP copies directly under <type>/
                        and a canonical PCB image per capture, which is derived and not counted as a capture)
    legacy sources      experiment_0 (captures.csv + inbox/_ingested + set_aside), production_raw folders,
                        unlabelled folders (e.g. live captures), operator_inspection_sessions (old station output)
    station captures    data/captures/<session>/<image_id>.json whose product was identified / selected as this product

plus, globally: station captures without a product, and any image file under data/ that belongs to no known source
and is not derived data ("unclaimed").

Every image is classified, never dropped:
    RESOLVED    physical board known (board ID from image_boards.csv > session_boards.csv > session / record)
    UNRESOLVED  no physical board / no ground truth (e.g. live captures, UNASSIGNED sessions)
    PENDING     data-collection capture not reviewed yet (KEEP / REJECT outstanding)
    REJECTED    data-collection capture rejected at review (kept in quarantine/)
    SYNTHETIC   generated image (simulated camera / mock camera): excluded from every dataset count
    INVALID     file missing or unreadable
Duplicates are found by SHA-256 (one sample, all paths kept). Checksums are cached in data/processed/checksums.json
(derived; safe to delete).
"""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..products.profile import ProductProfile
from .experiment0 import sha256_of
from .registry import BOARD_ID, Registry, load_registry

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
RESOLVED, UNRESOLVED, PENDING, REJECTED, SYNTHETIC, INVALID = (
    "RESOLVED", "UNRESOLVED", "PENDING", "REJECTED", "SYNTHETIC", "INVALID")
POOLS = ("GOOD", "FAULT", "DEFECT", "TEST")


# --- checksums -------------------------------------------------------------------------------

class ChecksumCache:
    """sha256 per file, reused while size and mtime are unchanged."""

    def __init__(self, path: Path | None):
        self.path, self.changed = path, False
        self.data: dict[str, list] = {}
        if path is not None and path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def sha(self, file: Path) -> str:
        st = file.stat()
        key = str(file.resolve())
        hit = self.data.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2]
        digest = sha256_of(file)
        self.data[key] = [st.st_size, st.st_mtime_ns, digest]
        self.changed = True
        return digest

    def save(self) -> None:
        if self.path is not None and self.changed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            os.replace(tmp, self.path)
            self.changed = False


# --- report structures -----------------------------------------------------------------------

@dataclass
class ImageEntry:
    path: str                         # project-relative
    sha256: str | None
    source: str                       # acquisition | experiment_0 | production_raw | unlabelled_folder | ...
    status: str                       # RESOLVED | UNRESOLVED | PENDING | REJECTED | SYNTHETIC | INVALID
    image_id: str = ""
    session_id: str = ""
    pool: str | None = None           # GOOD | DEFECT | TEST (data-collection type / board prefix); None = unknown
    board_id: str | None = None
    board_source: str = ""
    stage: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    note: str = ""
    copies: list[str] = field(default_factory=list)    # other paths with the same content


@dataclass
class BoardSummary:
    board_id: str
    pool: str | None
    registered: bool
    condition: str | None = None
    split: str | None = None
    identity_source: str | None = None
    verified_by: str | None = None
    defects: str = ""
    sessions: set[str] = field(default_factory=set)
    images: int = 0
    in_bank: bool | None = None


@dataclass
class ProductReport:
    product_id: str
    name: str | None
    variant_id: str
    images: list[ImageEntry] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)      # (path, primary path)
    orphans: list[tuple[str, str]] = field(default_factory=list)         # (path, reason)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    boards: dict[str, BoardSummary] = field(default_factory=dict)
    bank: dict[str, Any] | None = None
    station: dict[str, Any] = field(default_factory=dict)

    def by_status(self, status: str) -> list[ImageEntry]:
        return [e for e in self.images if e.status == status]

    def counts(self) -> dict[str, Counter]:
        """(pool -> status counts) over unique images."""
        table: dict[str, Counter] = defaultdict(Counter)
        for e in self.images:
            table[e.pool or "UNKNOWN"][e.status] += 1
        return table


@dataclass
class DatasetValidation:
    products: dict[str, ProductReport] = field(default_factory=dict)
    unresolved_captures: list[dict[str, Any]] = field(default_factory=list)
    unclaimed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


# --- helpers --------------------------------------------------------------------------------

class _Collector:
    """Unique images by SHA-256 for one product; later paths with the same content become copies."""

    def __init__(self, report: ProductReport, project_root: Path, cache: ChecksumCache, seen: set[str]):
        self.report, self.root, self.cache, self.seen = report, project_root, cache, seen
        self.by_sha: dict[str, ImageEntry] = {}
        self.recorded: set[Path] = set()            # files referenced by acquisition records (resolved paths)

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def add(self, file: Path, **kw: Any) -> ImageEntry | None:
        rel = self.rel(file)
        self.seen.add(rel)
        if not file.is_file():
            entry = ImageEntry(rel, None, status=INVALID, note=kw.pop("note", "") or "file missing", **kw)
            self.report.images.append(entry)
            return entry
        digest = self.cache.sha(file)
        if digest in self.by_sha:
            primary = self.by_sha[digest]
            if rel != primary.path and rel not in primary.copies:
                primary.copies.append(rel)
                self.report.duplicates.append((rel, primary.path))
            # a later, better-resolved record of the same content improves the entry (never the reverse)
            if primary.board_id is None and kw.get("board_id"):
                primary.board_id, primary.board_source = kw["board_id"], kw.get("board_source", "")
                if primary.status == UNRESOLVED:
                    primary.status = RESOLVED
            return primary
        entry = ImageEntry(rel, digest, **kw)
        self.by_sha[digest] = entry
        self.report.images.append(entry)
        return entry


def _image_files(folder: Path) -> Iterable[Path]:
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def _pool(text: str | None) -> str | None:
    if not text:
        return None
    t = text.upper()
    return t if t in POOLS else (t.split("_")[0] if t.split("_")[0] in POOLS else None)


def _load_registry(profile: ProductProfile, report: ProductReport) -> Registry:
    boards = profile.registry_dir / "boards.csv"
    if not boards.is_file():
        report.warnings.append(f"no board registry yet ({boards.relative_to(profile.project_root).as_posix()})")
        return Registry({}, {})
    try:
        reg = load_registry(boards, profile.registry_dir / "image_boards.csv")
    except ValueError as e:
        report.issues.append(f"board registry unreadable: {e}")
        return Registry({}, {})
    report.issues += [f"registry: {i}" for i in reg.issues]
    return reg


def _resolve_board(image_id: str, session_id: str, recorded: str | None, reg: Registry) -> tuple[str | None, str]:
    if image_id in reg.image_boards:
        return reg.image_boards[image_id][0], "image_boards.csv"
    if session_id in reg.session_boards:
        return reg.session_boards[session_id][0], "session_boards.csv"
    if recorded and recorded.upper() != "UNASSIGNED":
        return recorded, "recorded at capture"
    return None, ""


# --- per-source scanners ----------------------------------------------------------------------

def _scan_acquisition(profile: ProductProfile, reg: Registry, col: _Collector) -> None:
    root = profile.acquisition_root
    layout = profile.collection_layout()
    report = col.report
    for session_file in sorted(profile.collection_dir("acquisition").glob("*/*/S*/session.json")):
        try:
            s = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            report.issues.append(f"{col.rel(session_file)}: unreadable ({e})")
            continue
        owner = s.get("product_id") or (s.get("profile_id") or "").split("/")[0] or None
        if owner and owner != profile.product_id:
            report.issues.append(f"{col.rel(session_file)}: session of product {owner!r} inside the {profile.product_id} "
                                 "store (not counted here)")
            continue
        recorded = set()
        for r in s.get("images", []):
            image_id, session_id = r.get("image_id", ""), s.get("session_id", "")
            board, src = _resolve_board(image_id, session_id, s.get("board_id"), reg)
            review = (r.get("review") or {}).get("status")
            status = {"pending": PENDING, "rejected": REJECTED}.get(review, RESOLVED if board else UNRESOLVED)
            q = r.get("quality") or {}
            original = root / r["original_file"] if r.get("original_file") else None
            if original is None:
                report.orphans.append((col.rel(session_file), f"{image_id}: record without original_file"))
                continue
            recorded.add(original.resolve())
            entry = col.add(original, source="acquisition", status=status, image_id=image_id, session_id=session_id,
                            pool=_pool(s.get("dataset_type")), board_id=board, board_source=src,
                            stage=s.get("stage"), quality_flags=list(q.get("flags") or []),
                            note="" if original.is_file() else "original missing (recorded in session.json)")
            if r.get("saved_file"):
                copy = root / r["saved_file"]
                recorded.add(copy.resolve())
                if copy.is_file():
                    col.add(copy, source="acquisition", status=status, image_id=image_id, session_id=session_id,
                            pool=_pool(s.get("dataset_type")), board_id=board, board_source=src)
                else:
                    report.orphans.append((r["saved_file"], f"{image_id}: review copy recorded but missing"))
            if entry is not None and entry.status == INVALID:
                report.orphans.append((entry.path, f"{image_id}: recorded in session.json, file missing"))
            for key in ("pcb_file", "saved_pcb_file"):      # canonical PCB images: derived, accounted for
                if r.get(key):
                    pcb = root / r[key]
                    recorded.add(pcb.resolve())
                    col.seen.add(col.rel(pcb))
                    if not pcb.is_file():
                        report.orphans.append((r[key], f"{image_id}: PCB image recorded but missing"))
        for folder in ("images", "pcb"):
            for f in _image_files(session_file.parent / folder):
                if f.resolve() not in recorded:
                    report.orphans.append((col.rel(f), "image file in a session folder without a record in session.json"))
                    col.add(f, source="acquisition", status=UNRESOLVED, session_id=s.get("session_id", ""),
                            pool=_pool(s.get("dataset_type")), note="no record in session.json")
        col.recorded |= recorded
    review_folders = [("quarantine", root / layout["quarantine"])]
    if layout["curated"]:
        review_folders.insert(0, ("curated", root / layout["curated"]))
    else:   # KEEP copies directly under <type>/ (clean collection)
        review_folders[:0] = [("curated", root / t.lower()) for t in POOLS]
    for folder, path in review_folders:
        for f in _image_files(path):
            if f.resolve() not in col.recorded:
                report.orphans.append((col.rel(f), f"file in {col.rel(path)}/ without an acquisition record"))
                pool = f.relative_to(path).parts[0] if layout["curated"] or folder == "quarantine" else path.name
                col.add(f, source=folder, status=UNRESOLVED, pool=_pool(pool), note=f"in {folder} without a record")


def _scan_experiment_0(path: Path, reg: Registry, col: _Collector) -> None:
    captures = path / "metadata" / "captures.csv"
    recorded = set()
    if captures.is_file():
        with captures.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                file = path / (r.get("image_file") or "").replace("\\", "/")
                recorded.add(file.resolve())
                board, src = _resolve_board(r.get("image_id", ""), r.get("session_id", ""), r.get("board_id"), reg)
                col.add(file, source="experiment_0", status=RESOLVED if board else UNRESOLVED,
                        image_id=r.get("image_id", ""), session_id=r.get("session_id", ""), pool=_pool(board),
                        board_id=board, board_source=src, stage=r.get("inspection_stage") or None)
    for f in _image_files(path):
        if f.resolve() in recorded:
            continue
        rel_parts = f.relative_to(path).parts
        where = rel_parts[0]
        entry = col.add(f, source="experiment_0", status=UNRESOLVED, pool=_pool(rel_parts[1] if len(rel_parts) > 2 else None),
                        note=f"not in captures.csv ({where}/)")
        if entry is not None and entry.path == col.rel(f) and where in ("images",):
            col.report.orphans.append((entry.path, "Experiment 0 image not in captures.csv"))


def _scan_unlabelled(path: Path, col: _Collector, source: str) -> None:
    for f in _image_files(path):
        col.add(f, source=source, status=UNRESOLVED, note="no board, no ground truth")


def _scan_operator_sessions(path: Path, col: _Collector) -> None:
    recorded = set()
    for record_file in sorted(path.glob("*/records/*.json")):
        try:
            r = json.loads(record_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            col.report.issues.append(f"{col.rel(record_file)}: unreadable record")
            continue
        raw = Path(r.get("raw_path") or "")
        file = raw if raw.is_absolute() else path.parent.parent.parent / raw
        if not file.is_file():
            file = record_file.parent.parent / "raw" / f"{r.get('image_id')}.png"
        recorded.add(file.resolve())
        synthetic = str(r.get("camera", "")).lower() == "simulated"
        col.add(file, source="operator_inspection_sessions", status=SYNTHETIC if synthetic else UNRESOLVED,
                image_id=r.get("image_id", ""), session_id=r.get("session_id", ""),
                pool=_pool(r.get("dataset_type")),
                note=("SYNTHETIC frame generated by the old run_operator_inspection.py simulation (camera 'simulated');"
                      f" labelled board {r.get('board_id')} but it is not a photo of any board" if synthetic else
                      f"old operator station capture, never reviewed (claimed board {r.get('board_id')}, "
                      f"result {r.get('inspection_status')})"))
    for f in _image_files(path):
        if f.resolve() not in recorded:
            col.report.orphans.append((col.rel(f), "operator session image without a record"))
            col.add(f, source="operator_inspection_sessions", status=UNRESOLVED, note="no record")


def _scan_station(profile: ProductProfile, capture_root: Path, col: _Collector) -> dict[str, Any]:
    from ..station.records import RESULTS_INDEX, latest_reviews, read_jsonl
    decisions: Counter = Counter()
    reviews = latest_reviews(profile.storage_root)
    for record_file in sorted(capture_root.glob("S*/*_C*.json")):
        try:
            r = json.loads(record_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("product_id") != profile.product_id:
            continue
        decisions[r.get("decision") or "NOT_INSPECTED"] += 1
        human = next((v for k, v in reviews.items() if k in (r.get("inspection_ids") or [])), None)
        board = r.get("physical_board_id")
        status = SYNTHETIC if r.get("synthetic") else (RESOLVED if board else UNRESOLVED)
        note = f"station capture, AI decision {r.get('decision')}"
        if human:
            note += f", human review {human['human_decision']} by {human['reviewer']}"
        col.add(col.root / r["file_path"], source="station_capture", status=status, image_id=r.get("image_id", ""),
                session_id=r.get("session_id", ""), pool=None, board_id=board,
                board_source="recorded at capture" if board else "", stage=r.get("stage"),
                quality_flags=list(r.get("quality_flags") or []), note=note)
    results = read_jsonl(profile.inspections_dir / RESULTS_INDEX)
    return {"captures_by_decision": dict(decisions), "inspections": len(results),
            "human_reviews": Counter(v["human_decision"] for v in reviews.values())}


# --- main entry points -------------------------------------------------------------------------

def discover_product(profile: ProductProfile, *, capture_root: Path, cache: ChecksumCache,
                     include_unlabelled: bool = True, bank: dict[str, Any] | None = None,
                     seen: set[str] | None = None) -> ProductReport:
    report = ProductReport(profile.product_id, profile.name, profile.variant_id, bank=bank)
    reg = _load_registry(profile, report)
    col = _Collector(report, profile.project_root, cache, seen if seen is not None else set())
    _scan_acquisition(profile, reg, col)
    for src in profile.legacy_sources:
        path = profile.project_root / src["path"]
        kind = src.get("type")
        if not path.exists():
            report.warnings.append(f"legacy source {src['path']} ({kind}) does not exist")
        elif kind == "experiment_0":
            _scan_experiment_0(path, reg, col)
        elif kind in ("production_raw", "unlabelled_folder"):
            if kind == "unlabelled_folder" and not include_unlabelled:
                n = sum(1 for _ in _image_files(path))
                col.seen.update(col.rel(f) for f in _image_files(path))
                report.warnings.append(f"{src['path']}: {n} unlabelled images not checksummed (--no-unlabelled)")
                continue
            _scan_unlabelled(path, col, kind)
        elif kind == "operator_inspection_sessions":
            _scan_operator_sessions(path, col)
        else:
            report.issues.append(f"legacy source {src['path']}: unknown type {kind!r} (not scanned)")
    report.station = _scan_station(profile, capture_root, col)
    _boards(report, reg, bank)
    return report


def _boards(report: ProductReport, reg: Registry, bank: dict[str, Any] | None) -> None:
    for e in report.images:
        if e.status == SYNTHETIC:
            continue
        if e.board_id:
            if not BOARD_ID.fullmatch(e.board_id):
                report.issues.append(f"{e.path}: board ID {e.board_id!r} does not follow GOOD_###/FAULT_###/DEFECT_###/TEST_###")
            b = report.boards.setdefault(e.board_id, BoardSummary(e.board_id, _pool(e.board_id), e.board_id in reg.boards))
            b.images += 1
            if e.session_id:
                b.sessions.add(e.session_id)
            if e.pool and b.pool and e.pool != b.pool and e.source == "acquisition":
                report.warnings.append(f"{e.path}: captured in a {e.pool} session but assigned to board {e.board_id}")
    for bid, rb in reg.boards.items():
        b = report.boards.setdefault(bid, BoardSummary(bid, _pool(bid), True))
        b.condition, b.split, b.identity_source = rb.condition, rb.split, rb.identity_source
        b.verified_by, b.defects = rb.verified_by or None, ";".join(f"{c}:{t}" for c, t in rb.defects)
    bank_boards = set((bank or {}).get("boards") or [])
    for b in report.boards.values():
        if bank is not None:
            b.in_bank = b.board_id in bank_boards
        if not b.registered:
            report.warnings.append(f"board {b.board_id} has images but no row in boards.csv (condition / split unknown)")
        elif b.images == 0:
            report.warnings.append(f"board {b.board_id} is registered but has no images")
        if b.in_bank and b.condition != "good":
            report.issues.append(f"board {b.board_id} ({b.condition}) is in the GOOD reference bank")
        if b.registered and b.condition == "good" and b.split == "reference" and bank is not None and not b.in_bank:
            report.warnings.append(f"GOOD reference board {b.board_id} is NOT in the golden bank (no usable aligned image)")


def _derived(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.rstrip("/") + "/*") for p in patterns)


def validate_all(project_root: Path, profiles: list[ProductProfile], *, capture_root: Path, derived: list[str],
                 include_unlabelled: bool = True, banks: dict[str, dict[str, Any]] | None = None,
                 cache_file: Path | None = None) -> DatasetValidation:
    out = DatasetValidation()
    cache = ChecksumCache(cache_file)
    seen: set[str] = set()
    try:
        for p in profiles:
            out.products[p.product_id] = discover_product(
                p, capture_root=capture_root, cache=cache, include_unlabelled=include_unlabelled,
                bank=(banks or {}).get(p.product_id), seen=seen)
        rel = lambda f: f.resolve().relative_to(project_root.resolve()).as_posix()  # noqa: E731
        for record_file in sorted(capture_root.glob("S*/*_C*.json")):
            try:
                r = json.loads(record_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                out.problems.append(f"{rel(record_file)}: unreadable capture record")
                continue
            seen.add(r.get("file_path", ""))
            if not (project_root / r.get("file_path", "")).is_file():
                out.problems.append(f"{rel(record_file)}: capture file {r.get('file_path')} missing")
            if not r.get("product_id"):
                out.unresolved_captures.append({k: r.get(k) for k in (
                    "image_id", "file_path", "decision", "decision_reasons", "synthetic")})
        for f in _image_files(capture_root):
            if rel(f) not in seen:
                out.problems.append(f"{rel(f)}: capture file without an image record")
                seen.add(rel(f))
        for f in _image_files(project_root / "data"):
            r = rel(f)
            if r not in seen and not _derived(r, derived):
                out.unclaimed.append(r)
    finally:
        cache.save()
    return out


# --- text report ---------------------------------------------------------------------------------

def format_validation(v: DatasetValidation, limit: int = 12) -> str:
    lines: list[str] = []
    for pid, r in v.products.items():
        lines += ["=" * 78, f"PRODUCT {pid}  ({r.name}, {r.variant_id})", "=" * 78]
        if r.bank is not None:
            lines.append(f"Golden reference bank: {r.bank.get('status')} - {r.bank.get('reason')}")
        counts = r.counts()
        statuses = (RESOLVED, UNRESOLVED, PENDING, REJECTED, SYNTHETIC, INVALID)
        lines.append("")
        lines.append(f"{'Unique images':<16}" + "".join(f"{s:>12}" for s in statuses))
        for pool in (*POOLS, "UNKNOWN"):
            if pool in counts:
                lines.append(f"{pool:<16}" + "".join(f"{counts[pool][s]:>12}" for s in statuses))
        by_source = Counter(e.source for e in r.images)
        lines.append("Sources: " + ", ".join(f"{k} {n}" for k, n in sorted(by_source.items())))
        lines.append("")
        lines.append(f"{'Board':<11}{'pool':<8}{'condition':<10}{'split':<11}{'identity':<36}{'sess':>5}{'imgs':>6}  bank")
        for b in sorted(r.boards.values(), key=lambda b: b.board_id):
            identity = (b.identity_source or "-") + (f" ({b.verified_by})" if b.verified_by else "")
            bank = "-" if b.in_bank is None else ("yes" if b.in_bank else "NO")
            lines.append(f"{b.board_id:<11}{b.pool or '-':<8}{b.condition or 'UNREG.':<10}{b.split or '-':<11}"
                         f"{identity[:35]:<36}{len(b.sessions):>5}{b.images:>6}  {bank}")
        for title, items in (
                ("Pending review (KEEP / REJECT outstanding)", [f"{e.path}" for e in r.by_status(PENDING)]),
                ("Missing physical board (UNRESOLVED, data-collection)", sorted({
                    f"session {e.session_id} ({e.pool})" for e in r.by_status(UNRESOLVED) if e.source == "acquisition"})),
                ("Synthetic images (excluded)", [f"{e.path}: {e.note}" for e in r.by_status(SYNTHETIC)]),
                ("Invalid / missing images", [f"{e.path}: {e.note}" for e in r.by_status(INVALID)]),
                ("Quality flags", [f"{e.path}: {', '.join(e.quality_flags)}" for e in r.images if e.quality_flags]),
                ("Orphaned records / files", [f"{p}: {why}" for p, why in r.orphans]),
                ("Issues", r.issues), ("Warnings", r.warnings)):
            lines.append(f"\n{title}: {len(items)}")
            lines += [f"  {x}" for x in items[:limit]] + ([f"  ... {len(items) - limit} more"] if len(items) > limit else [])
        lines.append(f"\nDuplicates (same SHA-256, counted once): {len(r.duplicates)}")
        lines.append(f"Unresolved images by source: " + ", ".join(
            f"{k} {n}" for k, n in sorted(Counter(e.source for e in r.by_status(UNRESOLVED)).items())) or "-")
        st = r.station
        lines.append(f"Station: {st.get('inspections', 0)} inspections, captures by decision {st.get('captures_by_decision') or {}}, "
                     f"human reviews {dict(st.get('human_reviews') or {})}")
        lines.append("")
    lines += ["=" * 78, "ALL PRODUCTS", "=" * 78]
    lines.append(f"Station captures without a product (UNRESOLVED, stored under data/products/_unresolved): "
                 f"{len(v.unresolved_captures)}")
    lines += [f"  {c['image_id']}: {c['decision']} {c['decision_reasons']}" for c in v.unresolved_captures[:limit]]
    lines.append(f"Images under data/ that belong to no known source: {len(v.unclaimed)}")
    lines += [f"  {u}" for u in v.unclaimed[:limit]] + ([f"  ... {len(v.unclaimed) - limit} more"]
                                                       if len(v.unclaimed) > limit else [])
    lines.append(f"Problems: {len(v.problems)}")
    lines += [f"  {p}" for p in v.problems]
    return "\n".join(lines)
