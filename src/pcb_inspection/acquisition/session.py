"""Camera-first acquisition sessions: IDs, files, metadata, quality check and human review.

Experiment 1B (controlled camera capture) with a phone used as a live webcam
(DroidCam). Prototype data, NOT production-representative: it is kept apart
from data/raw/ and data/experiment_0/ (configs/capture/droidcam_e1b.yaml).

Layout, relative to data/:

    acquisition/<type>/<board_id|UNASSIGNED>/<session_id>/session.json
    acquisition/<type>/<board_id|UNASSIGNED>/<session_id>/images/<image_id>.png    original captures
    curated/<type>/<board_id|UNASSIGNED>/<session_id>/<image_id>.png                KEEP: verified copy
    quarantine/<type>/<board_id|UNASSIGNED>/<session_id>/<image_id>.png             REJECT: verified copy

<type> is good, defect or test, chosen once per session and never mixed: an
image is only ever copied into curated/ or quarantine/ under its own session's
type, so test captures cannot reach curated/good/ or curated/defect/.

Clean data collection (2026-10-06, scripts/collect.py, docs/clean_data_collection.md) uses the same store with
its own layout and only the types good and fault (settings.dataset_types), relative to data/collection/:

    originals/<good|fault>/<board_id>/<session_id>/session.json, images/<image_id>.png, pcb/<image_id>_pcb.png
    good/<board_id>/<session_id>/   fault/<board_id>/<session_id>/      KEEP: original + PCB image (verified copies)
    quarantine/<good|fault>/<board_id>/<session_id>/                      REJECT
Every capture there also gets a capture-time PCB check and a canonical PCB image (acquisition.pcb_view); an
image whose check says RECAPTURE cannot be KEPT (settings.keep_requires_capture_check).

session.json is the one metadata record of a session and of every image in it
(capture, quality check, review, future annotation). Originals in acquisition/
are never modified, moved or deleted; review copies them (checksum verified).

IDs (docs/experiment_1b_camera_acquisition.md):
    session_id  S20261005-01                    capture date + number (project format, naming.py)
    board_id    GOOD_001 | DEFECT_001 | TEST_001  one ID per physical board; prefix = session type
    image_id    GOOD_001_TOP_S20261005-01_C001  board (or GOOD_UNASSIGNED), side, session, capture
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from ..dataset.experiment0 import sha256_of
from ..dataset.naming import SESSION_ID_PATTERN, SIDES
from ..image_quality.checks import ImageQuality, QualityParams, check_image, flag_relative_sharpness
from ..image_quality.report import quality_row
from .pcb_view import LESS_SHARP_THAN_SESSION, OK, PCB_IMAGE_ERROR, RECAPTURE, PcbCheck, PcbChecker, PcbCheckParams

SCHEMA_VERSION = 1
SESSION_FILE = "session.json"
IMAGES_FOLDER = "images"
PCB_FOLDER = "pcb"
UNASSIGNED = "UNASSIGNED"

DATASET_TYPES = ("good", "defect", "test", "fault")
BOARD_POOL = {"good": "GOOD", "defect": "DEFECT", "test": "TEST", "fault": "FAULT"}
DECLARED_CONDITION = {"good": "good", "defect": "defective", "test": None,   # test boards are never labelled
                      "fault": "defective"}
LEGACY_DATASET_TYPES = ("good", "defect", "test")      # Experiment 1B store (data/acquisition, curated, quarantine)
ACQUISITION_BOARD_ID_PATTERN = re.compile(r"(GOOD|DEFECT|TEST|FAULT)_(\d{1,3})")
OTHER_STAGE = "other"
REVIEW_MODES = ("after_each", "after_session", "auto_keep_ok")   # auto_keep_ok: OK captures kept automatically

STATUS_RUNNING, STATUS_COMPLETED, STATUS_STOPPED, STATUS_INTERRUPTED = "running", "completed", "stopped", "interrupted"
REVIEW_PENDING, REVIEW_KEPT, REVIEW_REJECTED = "pending", "kept", "rejected"
DECISIONS = {"keep": REVIEW_KEPT, "reject": REVIEW_REJECTED}


class CaptureError(Exception):
    """A capture that did not produce a valid saved image; it is recorded as a failure, never as an image."""


class ReviewError(Exception):
    pass


# --- identifiers -------------------------------------------------------------

def normalise_board_id(dataset_type: str, text: str | None) -> str | None:
    """Operator input -> board ID of this session's pool. '' -> None; '1' -> GOOD_001 in a GOOD session.

    A board ID of another pool is refused: a GOOD board cannot be captured in a
    DEFECT or TEST session, and a TEST board never carries a GOOD / DEFECT label.
    """
    _check_type(dataset_type)
    value = (text or "").strip().upper().replace("-", "_").replace(" ", "")
    if not value:
        return None
    pool = BOARD_POOL[dataset_type]
    if value.isdigit():
        value = f"{pool}_{value}"
    match = ACQUISITION_BOARD_ID_PATTERN.fullmatch(value)
    if match is None or not 1 <= int(match.group(2)) <= 999:
        raise ValueError(f"board ID {text!r} must be a number 1-999 or look like {pool}_001")
    if match.group(1) != pool:
        raise ValueError(f"board {value} cannot be captured in a {pool} PCB session (use a {pool}_### board ID)")
    return f"{pool}_{int(match.group(2)):03d}"


def allocate_session_id(day: date, used: Iterable[str]) -> str:
    """Next free S<YYYYMMDD>-<NN> for the day, after every ID in use (acquisition + reserved)."""
    stamp = day.strftime("%Y%m%d")
    numbers = [int(s[10:12]) for s in used if SESSION_ID_PATTERN.fullmatch(s) and s[1:9] == stamp]
    number = max(numbers, default=0) + 1
    if number > 99:
        raise ValueError(f"no session number left for {day.isoformat()} (99 per day)")
    return f"S{stamp}-{number:02d}"


def make_image_id(dataset_type: str, board_id: str | None, side: str, session_id: str, number: int) -> str:
    """GOOD_001_TOP_S20261005-01_C001; without a board ID: GOOD_UNASSIGNED_TOP_S20261005-01_C001."""
    _check_type(dataset_type)
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(f"invalid session_id {session_id!r}")
    if not 1 <= number <= 999:
        raise ValueError(f"capture number must be 1..999, got {number}")
    board = board_id or f"{BOARD_POOL[dataset_type]}_{UNASSIGNED}"
    return f"{board}_{side}_{session_id}_C{number:03d}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_type(dataset_type: str) -> None:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"dataset type must be one of {DATASET_TYPES}, got {dataset_type!r}")


# --- settings and session configuration -------------------------------------

@dataclass(frozen=True)
class AcquisitionSettings:
    data_root: Path                                 # the project's data/ directory
    stages: tuple[str, ...]                         # confirmed stages from the profile
    board_mm: tuple[float, float] | None            # (long side, short side); None = unknown
    quality: QualityParams
    reject_reasons: tuple[str, ...]
    defect_types: Mapping[str, str | None]          # operator choice -> project taxonomy (None: no equivalent)
    side: str = "TOP"
    image_format: str = "png"
    acquisition_dir: str = "acquisition"
    curated_dir: str = "curated"                    # "" = KEEP copies directly under <type>/ (clean collection)
    quarantine_dir: str = "quarantine"
    context: Mapping[str, Any] = field(default_factory=dict)   # pcb model, variant, profile, capture config
    dataset_types: tuple[str, ...] = LEGACY_DATASET_TYPES      # the session types this store accepts
    pcb_checker: PcbChecker | None = None           # set: capture-time PCB check + canonical PCB image per capture
    keep_requires_capture_check: bool = False       # True: a capture whose check is RECAPTURE cannot be KEPT
    copy_original_on_keep: bool = True              # False: KEEP puts only the PCB image into the dataset folder


def _dataset_types(capture: Mapping[str, Any], allowed: Iterable[str] | None = None) -> tuple[str, ...]:
    types = tuple(capture.get("dataset_types") or LEGACY_DATASET_TYPES)
    for t in types:
        _check_type(t)
    if allowed is not None:
        types = tuple(t for t in types if t in set(allowed))
    if not types:
        raise ValueError("no dataset type left: the capture config and the product's storage.dataset_types do not "
                         "share a type")
    return types


def _pcb_checker(capture: Mapping[str, Any], board_mm: tuple[float, float] | None,
                 project_root: Path) -> PcbChecker | None:
    section = capture.get("pcb_check")
    return None if not section else PcbChecker(PcbCheckParams.from_mapping(section, board_mm, project_root))


def settings_from_configs(data_root: Path, profile: Mapping[str, Any], capture: Mapping[str, Any]) -> AcquisitionSettings:
    pcb = profile["pcb"]
    paths = capture["paths"]
    board_mm = (pcb["dimensions_mm"]["long_side"], pcb["dimensions_mm"]["short_side"])
    return AcquisitionSettings(
        data_root=Path(data_root),
        stages=tuple(profile["inspection_stages"]["confirmed"]),
        board_mm=board_mm,
        quality=QualityParams.from_mapping(capture["quality_check"]),
        reject_reasons=tuple(capture["review"]["reject_reasons"]),
        defect_types=dict(capture["defect_types"]),
        side={"top": "TOP", "bottom": "BOT"}[pcb["inspection_sides"][0]],
        image_format=capture["image_format"],
        acquisition_dir=paths["acquisition"], curated_dir=paths["curated"], quarantine_dir=paths["quarantine"],
        context={
            "pcb_model": pcb["model"], "variant": pcb["variant_label"], "profile_id": profile["profile_id"],
            "capture_config_id": capture["capture_config_id"], "purpose": capture["purpose"],
            "device_model": capture["device"].get("model"), "device_connection": capture["device"].get("connection"),
        },
        dataset_types=_dataset_types(capture),
        pcb_checker=_pcb_checker(capture, board_mm, Path(data_root).parent),
        keep_requires_capture_check=bool(capture.get("review", {}).get("keep_requires_capture_check")),
        copy_original_on_keep=bool(capture.get("review", {}).get("copy_original_on_keep", True)),
    )


def settings_from_product(profile: Any, capture: Mapping[str, Any]) -> AcquisitionSettings:
    """Settings for data collection of one product (pcb_inspection.products.ProductProfile): its own store
    (storage.acquisition_root, folder names from storage.layout or the capture config's paths), stages, board size
    and side; the session types are the capture config's dataset_types that the product's storage.dataset_types
    allows. Sessions record product_id / variant_id."""
    side = {"top": "TOP", "bottom": "BOT"}.get(str(profile.side).lower())
    if side is None:
        raise ValueError(f"{profile.product_id}: board.inspection_side is UNKNOWN in product.yaml; set it (top / "
                         "bottom) before collecting data (it is part of every image ID)")
    layout = profile.collection_layout(capture["paths"])
    if profile.storage.get("require_capture_check") and not capture.get("pcb_check"):
        raise ValueError(f"{profile.product_id}: its data collection requires the capture-time PCB check; capture "
                         f"config {capture.get('capture_config_id')!r} has none (use scripts/collect.py)")
    return AcquisitionSettings(
        data_root=profile.acquisition_root,
        stages=tuple(profile.stages),
        board_mm=profile.board_mm,
        quality=QualityParams.from_mapping(capture["quality_check"]),
        reject_reasons=tuple(capture["review"]["reject_reasons"]),
        defect_types=dict(capture["defect_types"]),
        side=side,
        image_format=capture["image_format"],
        acquisition_dir=layout["acquisition"], curated_dir=layout["curated"], quarantine_dir=layout["quarantine"],
        context={
            "product_id": profile.product_id, "variant_id": profile.variant_id,
            "pcb_model": profile.name, "variant": profile.variant_label,
            "profile_id": f"{profile.product_id}/{profile.variant_id}",
            "capture_config_id": capture["capture_config_id"], "purpose": capture["purpose"],
            "device_model": capture["device"].get("model"), "device_connection": capture["device"].get("connection"),
        },
        dataset_types=_dataset_types(capture, profile.storage.get("dataset_types")),
        pcb_checker=_pcb_checker(capture, profile.board_mm, profile.project_root),
        keep_requires_capture_check=bool(capture.get("review", {}).get("keep_requires_capture_check")),
        copy_original_on_keep=bool(capture.get("review", {}).get("copy_original_on_keep", True)),
    )


@dataclass
class SessionConfig:
    dataset_type: str
    stage: str
    requested_image_count: int
    capture_interval_s: float
    board_id: str | None = None
    review_mode: str = "after_each"
    defect_report: dict[str, Any] | None = None
    notes: str = ""
    lighting: str = ""                 # short description of the light for this session, e.g. "desk LED + diffuser"


def make_defect_report(known: bool, defect_types: Mapping[str, str | None], defect_type: str | None = None,
                       component_reference: str = "", location: str = "", notes: str = "") -> dict[str, Any]:
    """Operator-reported defect information for a DEFECT session.

    Annotation information only: it is not verified ground truth and does not
    exempt any image from the quality check or human review.
    """
    if known and defect_type not in defect_types:
        raise ValueError(f"defect type must be one of {sorted(defect_types)} when the defect is known")
    if not known and defect_type:
        raise ValueError("defect type given although the defect is marked UNKNOWN")
    return {
        "defect_known": "yes" if known else "unknown",
        "defect_type": defect_type if known else None,
        "taxonomy_defect_type": defect_types[defect_type] if known else None,
        "component_reference": component_reference.strip().upper() or None,
        "location": location.strip() or None,
        "notes": notes.strip() or None,
        "source": "operator_at_capture",
        "verified": False,
    }


# --- sessions ----------------------------------------------------------------

class Session:
    """One capture run; its session.json is rewritten (atomically) after every change."""

    def __init__(self, folder: Path, data: dict[str, Any]):
        self.folder = folder
        self.data = data
        self.lock = threading.RLock()

    @property
    def id(self) -> str:
        return self.data["session_id"]

    @property
    def dataset_type(self) -> str:
        return self.data["dataset_type"]

    @property
    def board_id(self) -> str | None:
        return self.data["board_id"]

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def images(self) -> list[dict[str, Any]]:
        return self.data["images"]

    @property
    def counts(self) -> dict[str, int]:
        return self.data["counts"]

    @property
    def path(self) -> Path:
        return self.folder / SESSION_FILE

    def image(self, image_id: str) -> dict[str, Any]:
        for record in self.images:
            if record["image_id"] == image_id:
                return record
        raise KeyError(f"{image_id} is not in session {self.id}")

    def pending(self) -> list[dict[str, Any]]:
        return [r for r in self.images if r["review"]["status"] == REVIEW_PENDING]

    def save(self) -> None:
        with self.lock:
            statuses = [r["review"]["status"] for r in self.images]
            self.data["counts"] = {
                "successful_captures": len(self.images),
                "failed_captures": len(self.data["failures"]),
                "kept_count": statuses.count(REVIEW_KEPT),
                "rejected_count": statuses.count(REVIEW_REJECTED),
                "pending_review_count": statuses.count(REVIEW_PENDING),
            }
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)


class AcquisitionStore:
    """Creates sessions, saves captures, applies review decisions. Paths in records are relative to data/."""

    def __init__(self, settings: AcquisitionSettings, reserved_session_ids: Iterable[str] = ()):
        self.settings = settings
        self.root = settings.data_root
        self.acquisition_root = self.root / settings.acquisition_dir
        self.curated_root = self.root / settings.curated_dir
        self.quarantine_root = self.root / settings.quarantine_dir
        self.reserved_session_ids = set(reserved_session_ids)
        self._sessions: dict[str, Session] = {}
        self._loaded_paths: set[Path] = set()
        self._created_here: set[str] = set()
        self._lock = threading.Lock()

    # sessions
    def session_files(self) -> list[Path]:
        return sorted(self.acquisition_root.glob(f"*/*/S*/{SESSION_FILE}"))

    def sessions(self) -> list[Session]:
        for path in self.session_files():
            if path not in self._loaded_paths:
                self.load_session(path)
        return sorted(self._sessions.values(), key=lambda s: s.id)

    def load_session(self, path: Path) -> Session:
        """The session stored at path; read from disk only once, then the in-memory copy is used."""
        cached = self._sessions.get(path.parent.name)
        if cached is not None and cached.path == path:
            return cached
        data = json.loads(path.read_text(encoding="utf-8"))
        session = self._sessions.setdefault(data["session_id"], Session(path.parent, data))
        self._loaded_paths.add(path)
        return session

    def used_session_ids(self) -> set[str]:
        return {p.parent.name for p in self.session_files()} | set(self._sessions) | self.reserved_session_ids

    def create_session(self, config: SessionConfig, camera: Mapping[str, Any], started_at: datetime | None = None) -> Session:
        s = self.settings
        _check_type(config.dataset_type)
        if config.dataset_type not in s.dataset_types:
            raise ValueError(f"this store collects only {', '.join(BOARD_POOL[t] for t in s.dataset_types)} PCB "
                             f"sessions, not {BOARD_POOL[config.dataset_type]}")
        if config.stage not in (*s.stages, OTHER_STAGE):
            raise ValueError(f"stage must be one of {[*s.stages, OTHER_STAGE]}")
        if config.stage == OTHER_STAGE and not config.notes.strip():
            raise ValueError("stage OTHER needs a note describing the stage")
        if not 1 <= int(config.requested_image_count) <= 999:
            raise ValueError("number of images must be 1..999")
        if not config.capture_interval_s > 0:
            raise ValueError("capture interval must be positive")
        if config.review_mode not in REVIEW_MODES:
            raise ValueError(f"review mode must be one of {REVIEW_MODES}")
        if config.board_id is not None and normalise_board_id(config.dataset_type, config.board_id) != config.board_id:
            raise ValueError(f"board ID {config.board_id!r} is not a normalised {BOARD_POOL[config.dataset_type]} board ID")
        if config.defect_report is not None and DECLARED_CONDITION[config.dataset_type] != "defective":
            raise ValueError("defect information belongs to DEFECT / FAULT PCB sessions only")

        started_at = started_at or datetime.now()
        context = dict(s.context)
        connection = context.get("device_connection")
        if connection and str(connection).lower() not in str(camera.get("name") or "").lower():
            # not the configured phone stream (e.g. the laptop's integrated camera): the phone model
            # would be false metadata; only the camera name reported by Windows is known
            context["device_model"] = context["device_connection"] = None
        with self._lock:
            session_id = allocate_session_id(started_at.date(), self.used_session_ids())
            folder = self.acquisition_root / config.dataset_type / (config.board_id or UNASSIGNED) / session_id
            folder.mkdir(parents=True, exist_ok=False)
            (folder / IMAGES_FOLDER).mkdir()
            data = {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "dataset_type": config.dataset_type,
                "declared_condition": DECLARED_CONDITION[config.dataset_type],
                "condition_source": "operator_at_capture" if config.dataset_type != "test" else None,
                "board_id": config.board_id,
                "split_group": config.board_id or f"{UNASSIGNED}:{session_id}",
                "stage": config.stage,
                "side": s.side,
                **context,
                "status": STATUS_RUNNING,
                "start_time": started_at.isoformat(timespec="seconds"),
                "end_time": None,
                "camera": dict(camera),
                "resolution": [camera.get("width"), camera.get("height")],
                "fps": camera.get("fps_measured") or camera.get("fps_reported"),
                "capture_interval_s": config.capture_interval_s,
                "requested_image_count": int(config.requested_image_count),
                "review_mode": config.review_mode,
                "image_format": s.image_format,
                "defect_report": config.defect_report,
                "lighting": config.lighting.strip() or None,
                "notes": config.notes.strip(),
                "counts": {},
                "images": [],
                "failures": [],
            }
            session = Session(folder, data)
            session.save()
            self._sessions[session_id] = session
            self._loaded_paths.add(session.path)
            self._created_here.add(session_id)
        return session

    def finish(self, session: Session, status: str, at: datetime | None = None) -> None:
        if status not in (STATUS_COMPLETED, STATUS_STOPPED):
            raise ValueError(f"status must be {STATUS_COMPLETED} or {STATUS_STOPPED}")
        with session.lock:
            if session.status != STATUS_RUNNING:
                return
            session.data["status"] = status
            session.data["end_time"] = (at or datetime.now()).isoformat(timespec="seconds")
            session.save()

    def mark_interrupted(self) -> list[str]:
        """Sessions left 'running' by an earlier run of the application (e.g. a crash) -> 'interrupted'.
        Their images stay valid and reviewable."""
        marked = []
        for session in self.sessions():
            if session.status == STATUS_RUNNING and session.id not in self._created_here:
                with session.lock:
                    session.data["status"] = STATUS_INTERRUPTED
                    session.save()
                marked.append(session.id)
        return marked

    # captures
    def save_capture(self, session: Session, image_bytes: bytes, captured_at: datetime,
                     camera: Mapping[str, Any], frame_seq: int | None = None) -> dict[str, Any]:
        """Save one captured frame, verify it, run the quality check and append its record.

        Raises CaptureError (nothing recorded as an image) if the frame cannot be
        saved, does not verify, or cannot be read back.
        """
        s = self.settings
        with session.lock:
            if session.status != STATUS_RUNNING:
                raise CaptureError(f"session {session.id} is {session.status}")
            if not image_bytes:
                raise CaptureError("empty frame")
            number = max((r["sequence"] for r in session.images), default=0) + 1
            while True:   # never overwrite: skip any number whose file already exists
                image_id = make_image_id(session.dataset_type, session.board_id, session.data["side"], session.id, number)
                path = session.folder / IMAGES_FOLDER / f"{image_id}.{s.image_format}"
                if not path.exists():
                    break
                number += 1
            digest = sha256_bytes(image_bytes)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as f:
                    f.write(image_bytes)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                raise CaptureError(f"could not write {path.name}: {e}") from e
            if not path.is_file() or path.stat().st_size != len(image_bytes) or sha256_of(path) != digest:
                path.unlink(missing_ok=True)
                raise CaptureError(f"{path.name} on disk does not match the captured frame; not recorded")
            quality = check_image(path, s.quality, s.board_mm)
            if not quality.readable:
                path.unlink(missing_ok=True)
                raise CaptureError(f"{path.name} cannot be read back ({quality.error}); not recorded")
            earlier = [ImageQuality(path="", readable=True, sharpness=float(r["quality"]["sharpness"]))
                       for r in session.images]
            flag_relative_sharpness({session.id: [quality, *earlier]}, s.quality.relative_sharpness_min)

            row = quality_row(image_id, session.id, session.board_id or "", quality)
            summary = {k: v for k, v in row.items() if k not in ("image_id", "session_id", "board_id", "flags")}
            summary.update(status="flagged" if quality.flags else "pass", flags=list(quality.flags))
            pcb_file = pcb_sha = None
            if s.pcb_checker is not None:   # after the original is safely stored, never before
                summary["capture_check"], pcb_file, pcb_sha = self._check_pcb(session, image_id, path)
            timestamp = captured_at.isoformat(timespec="milliseconds")
            record = {
                "image_id": image_id,
                "product_id": s.context.get("product_id"),
                "physical_board_id": session.board_id,
                "session_id": session.id,
                "dataset_type": session.dataset_type,
                "board_id": session.board_id,
                "split_group": session.data["split_group"],
                "stage": session.data["stage"],
                "side": session.data["side"],
                "sequence": number,
                "capture_id": f"C{number:03d}",
                "timestamp": timestamp,
                "captured_at": timestamp,
                "camera": camera.get("name"),
                "resolution": [quality.width, quality.height],
                "width": quality.width,
                "height": quality.height,
                "fps": camera.get("fps_measured") or camera.get("fps_reported"),
                "frame_seq": frame_seq,
                "image_format": s.image_format,
                "path": self.rel(path),
                "original_file": self.rel(path),
                "saved_file": None,
                "file_size_bytes": path.stat().st_size,
                "sha256": digest,
                "pcb_file": pcb_file,
                "pcb_sha256": pcb_sha,
                "saved_pcb_file": None,
                "colour_estimate": quality.colour_estimate or None,
                "colour_source": "automatic_estimate",
                "quality": summary,
                "review_status": REVIEW_PENDING,
                "review": {"status": REVIEW_PENDING, "decided_at": None, "reason": None, "note": None},
                "annotation": {"status": "not_started", "defects": []},
            }
            session.images.append(record)
            session.save()
        return record

    def _check_pcb(self, session: Session, image_id: str, original: Path) -> tuple[dict[str, Any], str | None, str | None]:
        """Capture-time PCB check of the SAVED original, plus its canonical PCB image (pcb/<image_id>_pcb.png,
        write-once, SHA-256 verified). A failure here never loses the capture: it makes the check REVIEW."""
        try:
            bgr = cv2.imread(str(original), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("the saved original cannot be decoded")
            check = self.settings.pcb_checker.check(bgr)
        except Exception as e:  # noqa: BLE001 - recorded, the capture itself is valid
            check = PcbCheck(status=OK)
            check.add(PCB_IMAGE_ERROR)
            return {**check.to_record(), "error": str(e)}, None, None
        pcb_rel = pcb_sha = None
        if check.canonical is not None:
            target = session.folder / PCB_FOLDER / f"{image_id}_pcb.png"
            ok, buffer = cv2.imencode(".png", check.canonical)
            data = buffer.tobytes() if ok else b""
            try:
                if not data:
                    raise OSError("PNG encoding failed")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                if sha256_of(target) != sha256_bytes(data):
                    raise OSError("written file does not match")
                pcb_rel, pcb_sha = self.rel(target), sha256_bytes(data)
            except OSError as e:
                check.add(PCB_IMAGE_ERROR)
                record = check.to_record()
                record["error"] = f"PCB image not saved: {e}"
                return record, None, None
        ratio, mine = self.settings.quality.relative_sharpness_min, check.pcb.get("sharpness")
        earlier = [r["quality"]["capture_check"]["pcb"]["sharpness"] for r in session.images
                   if ((r["quality"].get("capture_check") or {}).get("pcb") or {}).get("sharpness") is not None]
        if ratio is not None and mine is not None and earlier and mine < ratio * median([*earlier, mine]):
            check.add(LESS_SHARP_THAN_SESSION)
        return check.to_record(), pcb_rel, pcb_sha

    def record_failure(self, session: Session, reason: str, at: datetime | None = None) -> dict[str, Any]:
        with session.lock:
            failure = {"attempt": len(session.images) + len(session.data["failures"]) + 1,
                       "time": (at or datetime.now()).isoformat(timespec="milliseconds"), "reason": reason}
            session.data["failures"].append(failure)
            session.save()
        return failure

    # review
    def review(self, session: Session, image_id: str, decision: str, reason: str | None = None,
               note: str = "", at: datetime | None = None, by: str = "operator") -> dict[str, Any]:
        """KEEP -> verified copy in curated/<type>/...; REJECT (with a reason) -> verified copy in quarantine/<type>/...
        The original in acquisition/ stays where it is. With copy_original_on_keep False (clean collection) a KEEP
        copies only the PCB image (no background, no watermark) into the dataset folder; the original stays in
        originals/ and a REJECT copies both into quarantine/ as evidence.
        `by`: operator, or capture_check for an automatic decision: KEEP only of a capture whose check is OK, REJECT
        only of one whose check is RECAPTURE."""
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {tuple(DECISIONS)}")
        if decision == "reject" and reason not in self.settings.reject_reasons:
            raise ValueError(f"a rejection needs a reason from {self.settings.reject_reasons}")
        with session.lock:
            record = session.image(image_id)
            if record["review"]["status"] != REVIEW_PENDING:
                raise ReviewError(f"{image_id} was already reviewed ({record['review']['status']})")
            check = capture_check(record)
            if by != "operator":
                needed = {"keep": OK, "reject": RECAPTURE}[decision]
                if check is None or check["status"] != needed:
                    raise ValueError(f"an automatic {decision.upper()} needs capture check {needed}; this image is "
                                     f"{check['status'] if check else 'not checked'}: the operator decides")
            if decision == "keep" and self.settings.keep_requires_capture_check and check is not None \
                    and check["status"] == RECAPTURE:
                raise ReviewError(f"{image_id} failed the capture check ({', '.join(check['problems'])}): it cannot be "
                                  "kept. REJECT it (it is kept in quarantine) and capture again.")
            copy_original = decision == "reject" or self.settings.copy_original_on_keep or not record.get("pcb_file")
            files = [(record["original_file"], record["sha256"])] if copy_original else []
            if record.get("pcb_file"):
                files.append((record["pcb_file"], record["pcb_sha256"]))
            for rel, digest in [(record["original_file"], record["sha256"]), *files]:
                source = self.root / rel
                if not source.is_file() or sha256_of(source) != digest:
                    raise ReviewError(f"{rel} is missing or has changed; nothing copied")
            folder = self.destination(session, decision)
            copies = {rel: self._verified_copy(self.root / rel, digest, folder / Path(rel).name) for rel, digest in files}
            record["saved_file"] = copies.get(record["original_file"])
            if record.get("pcb_file"):
                record["saved_pcb_file"] = copies[record["pcb_file"]]
            record["review"] = {
                "status": DECISIONS[decision],
                "decided_at": (at or datetime.now()).isoformat(timespec="seconds"),
                "reason": reason if decision == "reject" else None,
                "note": note.strip() or None,
                "decided_by": by,
            }
            record["review_status"] = DECISIONS[decision]
            session.save()
        return record

    def board_defect_report(self, dataset_type: str, board_id: str) -> tuple[str, dict[str, Any]] | None:
        """(session_id, report) of the latest session of this physical board whose defect / fault was KNOWN."""
        known = [(s.id, s.data["defect_report"]) for s in self.sessions()
                 if s.dataset_type == dataset_type and s.board_id == board_id
                 and (s.data.get("defect_report") or {}).get("defect_known") == "yes"]
        return known[-1] if known else None

    def kept_count(self, dataset_type: str, board_id: str) -> int:
        """KEPT images of one physical board, over all its sessions (the per-board target of the board sequence)."""
        return sum(1 for s in self.sessions() if s.dataset_type == dataset_type and s.board_id == board_id
                   for r in s.images if r["review"]["status"] == REVIEW_KEPT)

    def undo_auto_keep(self, session: Session, image_id: str, reason: str, note: str = "",
                       at: datetime | None = None) -> dict[str, Any]:
        """The operator overrides an AUTOMATIC KEEP: the image is REJECTED into quarantine/ (original + PCB image,
        verified copies) and its dataset copies are removed - only after the copies in quarantine/ are verified and
        the files in originals/ still match, so no content is lost. The earlier decision stays in review_history."""
        if reason not in self.settings.reject_reasons:
            raise ValueError(f"a rejection needs a reason from {self.settings.reject_reasons}")
        with session.lock:
            record = session.image(image_id)
            review = record["review"]
            if review["status"] != REVIEW_KEPT or review.get("decided_by") != "capture_check":
                raise ReviewError(f"{image_id} was not kept automatically; operator decisions are final")
            files = [(record["original_file"], record["sha256"])]
            if record.get("pcb_file"):
                files.append((record["pcb_file"], record["pcb_sha256"]))
            for rel, digest in files:
                if not (self.root / rel).is_file() or sha256_of(self.root / rel) != digest:
                    raise ReviewError(f"{rel} is missing or has changed; nothing changed")
            folder = self.destination(session, "reject")
            copies = {rel: self._verified_copy(self.root / rel, digest, folder / Path(rel).name) for rel, digest in files}
            for key, digest in (("saved_file", "sha256"), ("saved_pcb_file", "pcb_sha256")):
                old = record.get(key)
                if old and (self.root / old).is_file() and sha256_of(self.root / old) == record[digest]:
                    (self.root / old).unlink()         # a verified duplicate of the file kept in originals/
            record.setdefault("review_history", []).append(dict(review, saved_file=record.get("saved_file"),
                                                                saved_pcb_file=record.get("saved_pcb_file")))
            record["saved_file"] = copies[record["original_file"]]
            if record.get("pcb_file"):
                record["saved_pcb_file"] = copies[record["pcb_file"]]
            record["review"] = {"status": REVIEW_REJECTED,
                                "decided_at": (at or datetime.now()).isoformat(timespec="seconds"),
                                "reason": reason, "note": note.strip() or "operator overrode the automatic KEEP",
                                "decided_by": "operator"}
            record["review_status"] = REVIEW_REJECTED
            session.save()
        return record

    def recheck_pending(self) -> list[tuple[str, str, str]]:
        """Re-run the capture check (current limits) on every capture still waiting for review. The earlier result
        is kept in quality.capture_check_history; decided captures and all files stay untouched (the PCB image is
        made by the same unchanged crop). Returns (image_id, old status, new status) for each changed result."""
        if self.settings.pcb_checker is None:
            return []
        changed = []
        for session in self.sessions():
            with session.lock:
                touched = False
                for r in session.pending():
                    old = capture_check(r)
                    original = self.root / r["original_file"]
                    if old is None or not original.is_file() or sha256_of(original) != r["sha256"]:
                        continue
                    bgr = cv2.imread(str(original), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    new = self.settings.pcb_checker.check(bgr)       # deterministic: same crop, same measurements
                    new.canonical = None
                    if LESS_SHARP_THAN_SESSION in old.get("review", []):
                        new.add(LESS_SHARP_THAN_SESSION)             # a session comparison, kept as it was
                    record = new.to_record()
                    if (record["status"], record["problems"], record["review"]) == \
                            (old["status"], old.get("problems"), old.get("review")):
                        continue
                    record["rechecked_at"] = datetime.now().isoformat(timespec="seconds")
                    r["quality"].setdefault("capture_check_history", []).append(old)
                    r["quality"]["capture_check"] = record
                    changed.append((r["image_id"], old["status"], record["status"]))
                    touched = True
                if touched:
                    session.save()
        return changed

    def _verified_copy(self, source: Path, digest: str, target: Path) -> str:
        if target.exists():
            if sha256_of(target) != digest:
                raise ReviewError(f"{self.rel(target)} already exists with different content; nothing overwritten")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(source.read_bytes())
            if sha256_of(target) != digest:
                target.unlink()
                raise ReviewError(f"copy to {self.rel(target)} did not verify; review not recorded")
        return self.rel(target)

    def destination(self, session: Session, decision: str) -> Path:
        """curated/ or quarantine/ folder of this session; always under the session's own dataset type."""
        _check_type(session.dataset_type)
        base = self.curated_root if decision == "keep" else self.quarantine_root
        return base / session.dataset_type / (session.board_id or UNASSIGNED) / session.id

    def pending_reviews(self) -> list[tuple[Session, dict[str, Any]]]:
        return [(s, r) for s in self.sessions() for r in s.pending()]

    # dataset view
    def records(self, dataset_types: Iterable[str] | None = None,
                review_statuses: Iterable[str] = (REVIEW_KEPT,)) -> list[dict[str, Any]]:
        """Image records of the given types (default: every type of this store except TEST) and review states.
        TEST is excluded unless asked for explicitly."""
        if dataset_types is None:
            dataset_types = [t for t in self.settings.dataset_types if t != "test"]
        types, statuses = set(dataset_types), set(review_statuses)
        for t in types:
            _check_type(t)
        return [dict(r) for s in self.sessions() if s.dataset_type in types
                for r in s.images if r["review"]["status"] in statuses]

    def good_bank_candidates(self) -> dict[str, list[dict[str, Any]]]:
        """physical GOOD board -> its KEPT GOOD captures that have a canonical PCB image and passed the capture check
        (OK, or REVIEW confirmed by the operator's KEEP). The candidates for a future golden reference bank: only
        GOOD sessions can contribute, so a FAULT (or DEFECT / TEST) board never enters it. Captures without a board
        ID are left out (they cannot be grouped by physical board)."""
        bank: dict[str, list[dict[str, Any]]] = {}
        for r in self.records(dataset_types=("good",)):
            check = capture_check(r)
            if r.get("board_id") and r.get("saved_pcb_file") and check is not None and check["status"] != RECAPTURE:
                bank.setdefault(r["board_id"], []).append(r)
        return dict(sorted(bank.items()))

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def inherit_defect_report(store: AcquisitionStore, dataset_type: str, board_id: str | None,
                          report: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """A session of a physical board that already has a KNOWN defect / fault report, started with UNKNOWN: the
    board's report is carried over (it is the same board, so the same fault), marked inherited_from_session.
    An explicitly entered report is never replaced. Returns (report, session it came from or None)."""
    if report is None or report.get("defect_known") != "unknown" or not board_id:
        return report, None
    earlier = store.board_defect_report(dataset_type, board_id)
    if earlier is None:
        return report, None
    session_id, known = earlier
    return {**known, "inherited_from_session": known.get("inherited_from_session") or session_id}, session_id


def capture_check(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The capture-time PCB check of an image record (None for stores without the check)."""
    return (record.get("quality") or {}).get("capture_check")


def audit(store: AcquisitionStore) -> list[str]:
    """Consistency check of everything on disk: originals, PCB images and review copies match their checksums,
    every copy lies under its own session's type (TEST isolation), every file in a session folder has a record,
    and no file in the KEEP / REJECT folders is unaccounted for. Returns the issues found (empty = consistent)."""
    s = store.settings
    issues, referenced = [], set()
    prefix = lambda kind: f"{kind}/" if kind else ""  # noqa: E731
    for session in store.sessions():
        recorded = set()
        for r in session.images:
            for key, digest, what in (("original_file", "sha256", "original"), ("pcb_file", "pcb_sha256", "PCB image")):
                if not r.get(key):
                    continue
                recorded.add(r[key])
                file = store.root / r[key]
                if not file.is_file() or sha256_of(file) != r[digest]:
                    issues.append(f"{r[key]}: {what} missing or changed")
            if r["review"]["status"] == REVIEW_PENDING:
                continue
            kind = s.curated_dir if r["review"]["status"] == REVIEW_KEPT else s.quarantine_dir
            expected = f"{prefix(kind)}{session.dataset_type}/"
            for key, digest in (("saved_file", "sha256"), ("saved_pcb_file", "pcb_sha256")):
                if key == "saved_pcb_file" and not r.get("pcb_file"):
                    continue
                saved = r.get(key)
                if key == "saved_file" and saved is None and r["review"]["status"] == REVIEW_KEPT and r.get("pcb_file"):
                    continue                    # KEEP of the PCB image only (the original stays in originals/)
                if not saved or not saved.startswith(expected):
                    issues.append(f"{r['image_id']}: {r['review']['status']} copy {saved!r} is not under {expected}")
                    continue
                referenced.add(saved)
                if not (store.root / saved).is_file() or sha256_of(store.root / saved) != r[digest]:
                    issues.append(f"{saved}: review copy missing or changed")
        for folder in (IMAGES_FOLDER, PCB_FOLDER):
            for path in sorted(p for p in (session.folder / folder).glob("*") if p.is_file()):
                if store.rel(path) not in recorded:
                    issues.append(f"{store.rel(path)}: file in a session folder without a record in {SESSION_FILE}")
    if s.curated_dir:
        bases = [store.curated_root, store.quarantine_root]
    else:   # KEEP copies directly under <type>/: check those folders, and that nothing else sits next to them
        bases = [store.curated_root / t for t in s.dataset_types] + [store.quarantine_root]
        known = {Path(s.acquisition_dir).parts[0], Path(s.quarantine_dir).parts[0], *s.dataset_types}
        if store.root.is_dir():
            for child in sorted(store.root.iterdir()):
                if child.is_dir() and child.name not in known and any(
                        p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                        for p in child.rglob("*")):      # e.g. registry/ (CSV) is fine; images of another type are not
                    issues.append(f"{store.rel(child)}/: folder of a type this store does not collect "
                                  f"({', '.join(s.dataset_types)} only)")
    for base in bases:
        if base.is_dir():
            for path in sorted(p for p in base.rglob("*") if p.is_file()):
                if store.rel(path) not in referenced:
                    issues.append(f"{store.rel(path)}: not referenced by any session record")
    return issues


def split_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """image_id -> physical-board group for leakage-free splitting (dataset.splitting.assign_captures).

    Images without a board ID are grouped by session (one board per session);
    they cannot be matched to other sessions of the same board.
    """
    return {r["image_id"]: r["split_group"] for r in records}
