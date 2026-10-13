"""Capture store: every capture is saved FIRST, before the product is known and before any inspection.

    data/captures/<session_id>/session.json            station session (mode, product selection, camera, failures)
    data/captures/<session_id>/<image_id>.png          original, lossless, written once ('xb'), fsync, SHA-256 verified
    data/captures/<session_id>/<image_id>.json         image record (all fields generated automatically)

Originals are never modified, moved, overwritten or deleted. The image record is created with the original and
updated (atomically) when the identification / inspection finishes: product_id, variant_id, quality_status,
inspection_status, decision, ... An image that ends in ERROR keeps its record with the error.

IDs: session_id SYYYYMMDD-NN (unique across every store of the project), image_id <session_id>_C###,
inspection_id <image_id>_I## (re-inspection of a stored image adds _I02, ...; nothing is overwritten).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..acquisition.camera import encode_png
from ..acquisition.session import allocate_session_id
from ..dataset.experiment0 import sha256_of
from ..camera.base import CapturedFrame

SESSION_FILE = "session.json"
RECORD_SCHEMA = 1
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SESSION_DIR = re.compile(r"S\d{8}-\d{2}")

# inspection_status values of an image record
CAPTURED, INSPECTED, FAILED = "CAPTURED", "INSPECTED", "ERROR"


class CaptureError(Exception):
    """The capture could not be stored (nothing was recorded as an image)."""


def used_session_ids(data_root: Path) -> set[str]:
    """Session IDs already used anywhere under data/ (acquisition stores, capture store, Experiment 0)."""
    used = set()
    if data_root.is_dir():
        for path in data_root.rglob("S*-*"):
            if path.is_dir() and SESSION_DIR.fullmatch(path.name):
                used.add(path.name)
        e0 = data_root / "experiment_0" / "metadata" / "captures.csv"
        if e0.is_file():
            used |= set(re.findall(r"S\d{8}-\d{2}", e0.read_text(encoding="utf-8", errors="replace")))
    return used


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class StoredCapture:
    image_id: str
    session_id: str
    path: Path
    record_path: Path
    record: dict[str, Any]


class CaptureStore:
    def __init__(self, root: Path, data_root: Path, project_root: Path):
        self.root, self.data_root, self.project_root = root, data_root, project_root
        self.lock = threading.Lock()

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return str(path)

    # --- sessions ------------------------------------------------------------------------------
    def new_session(self, mode: str, product_selection: str, camera: dict[str, Any] | None = None,
                    stage: str | None = None, station: str | None = None, at: datetime | None = None) -> str:
        at = at or datetime.now()
        with self.lock:
            own = {p.name for p in self.root.iterdir() if p.is_dir()} if self.root.is_dir() else set()
            session_id = allocate_session_id(at.date(), used_session_ids(self.data_root) | own)
            folder = self.root / session_id
            folder.mkdir(parents=True, exist_ok=False)
            _write_json_atomic(folder / SESSION_FILE, {
                "schema_version": RECORD_SCHEMA, "session_id": session_id, "mode": mode,
                "product_selection": product_selection, "stage": stage, "station": station,
                "camera": camera or {}, "started_at": at.isoformat(timespec="seconds"), "ended_at": None,
                "captures": 0, "failures": []})
        return session_id

    def session(self, session_id: str) -> dict[str, Any]:
        return json.loads((self.root / session_id / SESSION_FILE).read_text(encoding="utf-8"))

    def _update_session(self, session_id: str, **changes: Any) -> None:
        path = self.root / session_id / SESSION_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if key == "failure":
                data["failures"].append(value)
            elif key == "captures_add":
                data["captures"] += value
            else:
                data[key] = value
        _write_json_atomic(path, data)

    def record_failure(self, session_id: str, reason: str, at: datetime | None = None) -> None:
        """A capture attempt that produced no image (e.g. the camera delivered no frame): logged, never silent."""
        with self.lock:
            self._update_session(session_id, failure={"time": (at or datetime.now()).isoformat(timespec="milliseconds"),
                                                      "reason": reason})

    def end_session(self, session_id: str) -> None:
        with self.lock:
            self._update_session(session_id, ended_at=datetime.now().isoformat(timespec="seconds"))

    # --- captures ------------------------------------------------------------------------------
    def _next_image_id(self, session_id: str) -> tuple[str, Path]:
        folder = self.root / session_id
        n = 1 + max((int(m.group(1)) for p in folder.glob(f"{session_id}_C*.json")
                     if (m := re.fullmatch(rf"{re.escape(session_id)}_C(\d+)", p.stem))), default=0)
        while True:
            image_id = f"{session_id}_C{n:03d}"
            if not any((folder / f"{image_id}{ext}").exists() for ext in (*IMAGE_EXTS, ".json")):
                return image_id, folder
            n += 1

    def save(self, session_id: str, frame: CapturedFrame, *, board_id: str | None = None, stage: str | None = None,
             mode: str = "live_inspection", extra: dict[str, Any] | None = None) -> StoredCapture:
        """Write the original (lossless PNG) and its image record. Raises CaptureError if it cannot be verified."""
        if frame.image is None or not frame.image.size:
            raise CaptureError("empty frame")
        data = encode_png(frame.image)
        return self._store(session_id, data, ".png", frame.captured_at, frame.camera_id, frame.camera,
                           frame.resolution, board_id, stage, mode, {"frame_seq": frame.frame_seq, **(extra or {})})

    def register_file(self, session_id: str, path: Path, *, copy: bool, board_id: str | None = None,
                      stage: str | None = None, mode: str = "stored_image") -> StoredCapture:
        """An existing image file: copied into the store (copy=True) or recorded where it is (already stored)."""
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
        meta = {"source_file": self.rel(path), "source_sha256": sha256_of(path)}
        when = datetime.fromtimestamp(path.stat().st_mtime)
        if copy:
            return self._store(session_id, path.read_bytes(), path.suffix.lower(), when, "file", {"kind": "imported file"},
                               (width, height), board_id, stage, mode, meta)
        with self.lock:
            image_id, folder = self._next_image_id(session_id)
            record = self._record(image_id, session_id, path, meta["source_sha256"], path.stat().st_size, when, "file",
                                  {"kind": "file stored elsewhere (recorded in place, not copied)"}, (width, height),
                                  board_id, stage, mode, {**meta, "stored_as": "in_place"})
            record_path = folder / f"{image_id}.json"
            _write_json_atomic(record_path, record)
            self._update_session(session_id, captures_add=1)
        return StoredCapture(image_id, session_id, path, record_path, record)

    def _store(self, session_id, data: bytes, ext: str, when: datetime, camera_id: str, camera: dict[str, Any],
               resolution: tuple[int, int], board_id, stage, mode, extra: dict[str, Any]) -> StoredCapture:
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            image_id, folder = self._next_image_id(session_id)
            path = folder / f"{image_id}{ext}"
            try:
                with path.open("xb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                raise CaptureError(f"could not write {path.name}: {e}") from e
            if path.stat().st_size != len(data) or sha256_of(path) != digest:
                raise CaptureError(f"{path.name} on disk does not match the captured frame (kept for inspection, "
                                   "not recorded as a capture)")
            record = self._record(image_id, session_id, path, digest, len(data), when, camera_id, camera, resolution,
                                  board_id, stage, mode, {**extra, "stored_as": "original"})
            record_path = folder / f"{image_id}.json"
            _write_json_atomic(record_path, record)
            self._update_session(session_id, captures_add=1)
        return StoredCapture(image_id, session_id, path, record_path, record)

    def _record(self, image_id, session_id, path: Path, digest, size, when: datetime, camera_id, camera, resolution,
                board_id, stage, mode, extra) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA,
            "image_id": image_id, "inspection_id": None, "inspection_ids": [],
            "product_id": None, "variant_id": None, "physical_board_id": board_id, "session_id": session_id,
            "capture_timestamp": when.isoformat(timespec="milliseconds"), "camera_id": camera_id,
            "camera": {k: v for k, v in (camera or {}).items() if isinstance(v, (str, int, float, bool, type(None)))},
            "resolution": f"{resolution[0]}x{resolution[1]}", "stage": stage, "mode": mode,
            "file_path": self.rel(path), "file_size_bytes": size, "sha256": digest,
            "quality_status": None, "quality_flags": [], "identification": None,
            "inspection_status": CAPTURED, "decision": None, "decision_confidence": None, "result_path": None,
            "review_status": None, "synthetic": camera_id.startswith("mock:"),
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"), **extra}

    def update(self, capture: StoredCapture, **fields: Any) -> dict[str, Any]:
        """Update the image record (never the image). inspection_id is appended to inspection_ids."""
        with self.lock:
            record = json.loads(capture.record_path.read_text(encoding="utf-8"))
            if fields.get("inspection_id"):
                record["inspection_ids"] = [*record.get("inspection_ids", []), fields["inspection_id"]]
            record.update(fields)
            record["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
            _write_json_atomic(capture.record_path, record)
            capture.record = record
        return record

    # --- lookup --------------------------------------------------------------------------------
    def records(self) -> Iterable[tuple[Path, dict[str, Any]]]:
        if not self.root.is_dir():
            return
        for session in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for path in sorted(session.glob("*_C*.json")):
                try:
                    yield path, json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    yield path, {}

    def find(self, *, path: Path | None = None, sha256: str | None = None) -> StoredCapture | None:
        rel = self.rel(path) if path is not None else None
        for record_path, r in self.records():
            if r and ((rel and r.get("file_path") == rel) or (sha256 and r.get("sha256") == sha256)):
                return StoredCapture(r["image_id"], r["session_id"], self.project_root / r["file_path"], record_path, r)
        return None

    def next_inspection_id(self, capture: StoredCapture) -> str:
        n = len(capture.record.get("inspection_ids") or []) + 1
        return f"{capture.image_id}_I{n:02d}"

