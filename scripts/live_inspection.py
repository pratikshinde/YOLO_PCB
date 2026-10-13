"""Live PCB inspection station: camera -> one capture every N seconds -> staged Experiment 2 inspection.

    python scripts/live_inspection.py --camera-index 1 --interval 3 [--board-id GOOD_016]

A wrapper only; the inspection is the existing Experiment 2 code (same config, same thresholds):

  camera      pcb_inspection.acquisition.camera: OpenCVCamera (cv2.VideoCapture(index, cv2.CAP_DSHOW), isOpened
              check), FrameGrabber (reader thread, cap.read() checks, live / blank / frozen / disconnected,
              reconnect, fresh-frame wait, releases the camera on stop), encode_png
  inspection  pcb_inspection.comparison.board_inspection.inspect_image: frame check -> watermark located and
              inpainted -> PCB present? -> SIFT/RANSAC alignment -> alignment validation -> component comparison
              with the GOOD reference set -> plausibility; write_outputs for the result files
  preview     pcb_inspection.preprocessing.presence.detect_pcb on preview frames (every 0.5 s, background thread)
              to show "PCB: DETECTED / NO PCB" before the capture

States: NO_CAMERA, NO_PCB, READY_TO_INSPECT, INSPECTING, then per inspection INVALID_FRAME, NO_PCB,
PCB_POSITION_INVALID, ALIGNMENT_FAILED, PASS, FAIL, REVIEW, INSPECTION_ERROR. PASS only after every stage.

Outputs: a capture with a board in view -> data/testing/live/LIVE_<date>_<time>.png (lossless, never overwritten)
and data/testing/live_output/<name>/ (report.txt, result.json, overview.jpg, annotated.jpg, evidence.jpg,
capture.json); no board / invalid frame -> only a small thumbnail in live_output/no_pcb/. Every inspection is one
row of live_output/summary.csv: timestamp, board_id, image, pcb_detected, alignment_status, inspection_status,
defect_components, scores, camera, resolution, message.

Timing: one capture every --interval seconds (time.monotonic, capture start to capture start), never two at once;
the preview stays live (inspections run on a worker thread). Keys: SPACE inspect now, R reset, Q / ESC quit.
The board ID field: the physical board under the camera (optional). A board of the GOOD reference set is
inspected against the OTHER boards only (leave-one-board-out).

Known limitation (experiments/experiment_002/README.md): unvalidated prototype on an unverified 55-part draft map,
GOOD reference set of 2 physical boards; it missed the R37 defect in the existing TEST image.
Window: Tkinter (the installed OpenCV is the headless build, no cv2.imshow).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pcb_inspection.acquisition.camera import (  # noqa: E402
    STATUS_LIVE,
    FrameGrabber,
    OpenCVCamera,
    device_at,
    encode_png,
    list_video_devices,
)
from pcb_inspection.comparison.board_inspection import (  # noqa: E402
    ALIGNMENT_FAILED,
    INSPECTION_ERROR,
    INVALID_FRAME,
    NO_PCB,
    POSITION_INVALID,
    GoodGallery,
    InspectionConfig,
    InspectionReference,
    inspect_image,
    load_calibration,
    load_gallery,
    load_inspection_config,
    load_reference_and_map,
    write_outputs,
)
from pcb_inspection.comparison.components import FAIL, PASS, REVIEW, ComponentMap  # noqa: E402
from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.preprocessing.presence import DETECTED, PresenceParams, detect_pcb  # noqa: E402

NO_CAMERA, READY, INSPECTING, WAITING = "NO_CAMERA", "READY_TO_INSPECT", "INSPECTING", "WAITING"
SUMMARY_COLUMNS = ("timestamp", "board_id", "image", "status", "defect_count", "defect_components",
                   "score", "pcb_detected", "alignment_status", "inspection_status",
                   "scores", "camera", "resolution", "message")
BOARD_ID = re.compile(r"(GOOD|DEFECT|TEST)_\d{3}")
LINE = "=" * 32


def shown(state: str) -> str:
    return state.replace("_", " ")


# --- timing ------------------------------------------------------------------------

class Cadence:
    """One inspection every `interval` seconds (capture start to capture start), never two at once."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval, self.clock = interval, clock
        self.busy = False
        self.next_due = clock() + interval      # first capture after one interval: time to place the board

    def due(self) -> bool:
        return not self.busy and self.clock() >= self.next_due

    def seconds_left(self) -> float:
        return max(0.0, self.next_due - self.clock())

    def started(self) -> None:
        self.busy = True
        self.next_due = self.clock() + self.interval

    def finished(self) -> None:
        self.busy = False                       # if the inspection took longer than the interval: due at once

    def now(self) -> None:
        """SPACE: inspect as soon as possible (after a running inspection, if any)."""
        self.next_due = self.clock()

    def reset(self) -> None:
        self.next_due = self.clock() + self.interval


# --- capture -> staged inspection -> outputs -------------------------------------------------

@dataclass
class LiveResult:
    timestamp: datetime
    status: str                                  # final state of the staged inspection
    defects: list[tuple[str, float]] = field(default_factory=list)   # (component, score)
    board_id: str = ""
    image: Path | None = None
    output_dir: Path | None = None
    message: str = ""
    pcb: str = ""                                # YES / NO / "" (frame invalid)
    alignment: str = ""                          # OK / FAILED / POSITION_INVALID / ""
    score: float | None = None
    seconds: float = 0.0
    resolution: str = ""

    def __post_init__(self):
        if isinstance(self.defects, str) and isinstance(self.board_id, list):
            self.defects, self.board_id = self.board_id, self.defects


class LiveInspector:
    """Runs the staged Experiment 2 inspection on a captured frame, saves the capture and outputs, logs summary.csv."""

    def __init__(self, cfg: InspectionConfig, reference: InspectionReference, cmap: ComponentMap, gallery: GoodGallery,
                 capture_dir: Path, output_dir: Path, camera_info: Callable[[], dict[str, Any]] = dict,
                 limits: dict[str, float] | None = None, limits_note: str = ""):
        self.cfg, self.reference, self.cmap, self.gallery = cfg, reference, cmap, gallery
        self.capture_dir, self.output_dir = capture_dir, output_dir
        self.camera_info, self.limits, self.limits_note = camera_info, limits, limits_note
        self.lock = threading.Lock()
        self._prepare_summary()

    def _prepare_summary(self) -> None:
        """Keep an older summary.csv with different columns: renamed, never overwritten or deleted."""
        path = self.output_dir / "summary.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if tuple(header) != SUMMARY_COLUMNS:
                old = path.with_name(f"summary_previous_format_{datetime.now():%Y%m%d_%H%M%S}.csv")
                path.rename(old)
                print(f"note: {path.name} had other columns; kept as {old.name}")

    def capture_path(self, when: datetime) -> Path:
        """data/testing/live/LIVE_20261005_134501.png; _2, _3 ... if that name exists (never overwritten)."""
        base = f"LIVE_{when:%Y%m%d_%H%M%S}"
        path, n = self.capture_dir / f"{base}.png", 2
        while path.exists() or (self.output_dir / "no_pcb" / f"{path.stem}.jpg").exists():
            path, n = self.capture_dir / f"{base}_{n}.png", n + 1
        return path

    def inspect(self, image: np.ndarray | None, when: datetime, board_id: str = "") -> LiveResult:
        start = time.perf_counter()
        with self.lock:
            result = self._inspect(image, when, board_id)
            result.seconds = round(time.perf_counter() - start, 2)
            self._append_summary(result)
        return result

    def _inspect(self, image: np.ndarray | None, when: datetime, board_id: str) -> LiveResult:
        if image is None:
            return LiveResult(when, "INSPECTION ERROR", board_id, message="no new frame from the camera (cap.read() failed or timed out)")
        exclude = board_id if board_id in self.gallery.board_ids else None
        try:
            r = inspect_image(image, self.reference, self.cmap, self.cfg, self.gallery, exclude, self.limits, self.limits_note)
        except Exception as e:
            return LiveResult(when, "INSPECTION ERROR", board_id, message=f"inspection failed: {e}",
                              resolution=f"{image.shape[1]}x{image.shape[0]}")
        check = r.alignment_check
        score = max((c.score for c in r.missing), default=None)
        if r.status == INVALID_FRAME:
            return LiveResult(when, "INSPECTION ERROR", board_id, message=f"blank picture: {r.reason}",
                              resolution=f"{image.shape[1]}x{image.shape[0]}", pcb="NO", alignment="")
        if r.status in (NO_PCB, ALIGNMENT_FAILED, POSITION_INVALID):
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            path = self.capture_path(when)
            with path.open("xb") as f:
                f.write(encode_png(image))
            return LiveResult(when, "INSPECTION ERROR", board_id, message=f"not aligned: {r.reason}",
                              image=path, resolution=f"{image.shape[1]}x{image.shape[0]}",
                              pcb="NO" if r.status == NO_PCB else "YES",
                              alignment="FAILED" if check is None else check.status)
        result = LiveResult(
            when, r.status, board_id, [(c.refdes, c.score) for c in r.missing], message=r.reason,
            pcb="YES", alignment="" if check is None else check.status,
            resolution=f"{image.shape[1]}x{image.shape[0]}", score=score)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_path(when)
        with path.open("xb") as f:
            f.write(encode_png(image))
        r.image = path
        out = self.output_dir / path.stem
        write_outputs(r, self.reference, self.cmap, self.cfg, out, frame=image)
        result.image, result.output_dir = path, out
        (out / "capture.json").write_text(json.dumps({
            "timestamp": when.isoformat(timespec="milliseconds"), "board_id": board_id or None,
            "captured_image": str(path), "image_size": [int(image.shape[1]), int(image.shape[0])],
            "camera": self.camera_info(), "status": r.status, "stages": r.stages,
            "defects": [{"component": c, "score": s} for c, s in result.defects], "message": r.reason,
            "pipeline": "Experiment 2 staged inspection (board_inspection.inspect_image)",
        }, indent=2, default=str), encoding="utf-8")
        return result

    def _append_summary(self, r: LiveResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "summary.csv"
        new = not path.is_file()
        camera = self.camera_info() or {}
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({
                "timestamp": r.timestamp.isoformat(timespec="milliseconds"), "board_id": r.board_id or "-",
                "image": "" if r.image is None else _rel(r.image),
                "status": r.status,
                "defect_count": str(len(r.defects)),
                "defect_components": ";".join(c for c, _ in r.defects),
                "score": f"{r.score:.3f}" if r.score is not None else "",
                "pcb_detected": r.pcb or "-",
                "alignment_status": r.alignment or "-", "inspection_status": r.status,
                "scores": ";".join(f"{c}={s:.3f}" for c, s in r.defects),
                "camera": f"{camera.get('index', '')} {camera.get('name', '')}".strip(), "resolution": r.resolution,
                "message": r.message,
            })


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


# --- texts ---------------------------------------------------------------------------

def console_text(r: LiveResult) -> str:
    head = (f"[{r.timestamp:%H:%M:%S}] {r.board_id or '-'}  {r.image.name if r.image else '(no image)'}  ({r.seconds:.1f} s)"
            f"  PCB {r.pcb or '-'} | alignment {r.alignment or '-'}")
    if r.status == PASS:
        body = ["PCB STATUS: PASS", "All expected components detected."]
    elif r.status == FAIL:
        body = ["PCB STATUS: FAIL", "Defects detected:", ""] + [f"{i}. {c}" for i, (c, _) in enumerate(r.defects, 1)]
        body += ["", "Score: " + (f"{r.defects[0][1]:.2f}" if len(r.defects) == 1
                                  else ", ".join(f"{c} {s:.2f}" for c, s in r.defects))]
    else:
        body = [f"PCB STATUS: {shown(r.status)}", r.message, "No PASS / FAIL decision was made."]
    return "\n".join([head, LINE, *body, LINE])


def overlay_lines(camera: str, interval: float, state: str, result: LiveResult | None, seconds_left: float,
                  camera_message: str = "", pcb_preview: str = "", board_id: str = ""
                  ) -> list[tuple[str, tuple[int, int, int]]]:
    """Text of the live overlay (BGR colours)."""
    white, green, red, amber = (255, 255, 255), (80, 220, 80), (60, 60, 255), (0, 190, 255)
    lines = [("PCB INSPECTION", white), (f"Camera: {camera}", white)]
    if camera_message:
        lines.append((f"CAMERA: {camera_message[:46]}", red))
    lines += [(f"Interval: {interval:g} sec" + (f"    Board: {board_id}" if board_id else ""), white)]
    lines.append((f"PCB: {shown(pcb_preview)}" if pcb_preview else "", green if pcb_preview == DETECTED else amber))
    if result is not None:
        lines.append((f"Status: {shown(result.status)}", {PASS: green, FAIL: red, REVIEW: amber}.get(result.status, red)))
        if result.status == FAIL:
            lines.append(("DEFECT DETECTED", red))
            if len(result.defects) == 1:
                lines.append((f"Component: {result.defects[0][0]}", red))
            else:
                lines += [("Components:", red)] + [(f"- {c}", red) for c, _ in result.defects]
        elif result.status == PASS:
            lines.append(("All expected components detected", green))
        elif result.message:
            lines += [(part, amber) for part in _wrap(result.message, 46)[:3]]
    if camera_message and not result:
        lines.append((f"Next inspection in {seconds_left:.1f} s", white))
    else:
        lines.append((f"State: {shown(state)}" + ("" if state == INSPECTING else f"   Next inspection in {seconds_left:.1f} s"), white))
    return lines


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    return lines + ([current] if current else [])


def draw_overlay(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    scale = max(0.5, image.shape[1] / 1600)
    step = round(28 * scale)
    width = round(520 * scale)
    height = step * len(lines) + round(16 * scale)
    out = image.copy()
    box = out[:height, :width]
    box[:] = (box * 0.35).astype(np.uint8)
    for i, (text, colour) in enumerate(lines):
        bold = i == 0 or text.startswith(("Inspection", "DEFECT", "Component", "PCB:"))
        cv2.putText(out, text, (round(12 * scale), step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, colour,
                    2 if bold else 1, cv2.LINE_AA)
    return out


# --- live window (Tkinter) ----------------------------------------------------------------

class LiveApp:
    PREVIEW_WIDTH = 1100
    PRESENCE_EVERY_S = 0.5

    def __init__(self, root, grabber: FrameGrabber, inspector: LiveInspector, camera_label: str, interval: float,
                 fresh_frame_timeout_s: float = 2.0, on_result: Callable[[LiveResult], None] = lambda r: print(console_text(r)),
                 board_id: str = "", presence: PresenceParams | None = None, segmentation=None):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.root, self.grabber, self.inspector = root, grabber, inspector
        self.camera_label, self.cadence = camera_label, Cadence(interval)
        self.fresh_frame_timeout_s, self.on_result = fresh_frame_timeout_s, on_result
        self.presence_params = presence or inspector.cfg.presence
        self.segmentation = segmentation or inspector.cfg.pipeline.segmentation
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")
        self.presence_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="presence")
        self.future: Future | None = None
        self.presence_future: Future | None = None
        self.presence_at = 0.0
        self.pcb_preview = ""
        self.result: LiveResult | None = None
        self.results: list[LiveResult] = []
        self.closing = False
        root.title("Live PCB inspection (Experiment 2 prototype)")
        root.configure(bg="black")
        blank = Image.new("RGB", (self.PREVIEW_WIDTH, round(self.PREVIEW_WIDTH * 9 / 16)), "black")
        self._preview_photo = ImageTk.PhotoImage(blank)
        self.preview = tk.Label(root, image=self._preview_photo, bg="black")
        self.preview.grid(row=0, column=0, rowspan=3)
        side = tk.Frame(root, bg="black")
        side.grid(row=0, column=1, sticky="nw", padx=6, pady=6)
        tk.Label(side, text="Board ID (physical board under the camera):", fg="white", bg="black").pack(anchor="w")
        self.board_var = tk.StringVar(value=board_id)
        self.board_entry = tk.Entry(side, textvariable=self.board_var, width=14, font=("Consolas", 12))
        self.board_entry.pack(anchor="w", pady=(0, 6))
        self._result_photo = ImageTk.PhotoImage(Image.new("RGB", (420, 300), "#222222"))
        self.result_image = tk.Label(side, image=self._result_photo, bg="black")
        self.result_image.pack(anchor="w")
        self.result_text = tk.StringVar(value="No inspection yet.")
        tk.Label(side, textvariable=self.result_text, justify="left", anchor="nw", fg="white", bg="black",
                 font=("Consolas", 10), width=50, wraplength=430).pack(anchor="w")
        tk.Label(root, text="SPACE inspect now    R reset    Q / ESC quit    (prototype: unverified draft component map, "
                            "2 GOOD reference boards; see experiments/experiment_002/README.md)",
                 fg="#bbbbbb", bg="black").grid(row=3, column=0, columnspan=2, sticky="w", padx=6)
        for key in ("<Escape>", "<Control-q>"):
            root.bind(key, lambda e: self.close())
        root.bind("<KeyPress>", self._key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.tick)

    # keys (letters are ignored while typing in the board ID field)
    def _key(self, event) -> None:
        if event.widget is self.board_entry:
            return
        key = event.keysym.lower()
        if key in ("q",):
            self.close()
        elif key == "space":
            self.inspect_now()
        elif key == "r":
            self.reset()

    def board_id(self) -> str:
        value = self.board_var.get().strip().upper()
        return value if BOARD_ID.fullmatch(value) else ""

    def inspect_now(self) -> None:
        self.cadence.now()

    def reset(self) -> None:
        self.cadence.reset()
        if not self.cadence.busy:
            self.result = None
            self.result_text.set("Reset. Waiting for the next inspection.")

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.executor.shutdown(wait=True)      # let a running inspection finish and be saved
        self.presence_executor.shutdown(wait=True)
        self._collect()
        self.grabber.stop()                    # the reader thread releases the camera
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    # state
    def state(self) -> str:
        if self.cadence.busy:
            return INSPECTING
        if not self.grabber.is_live:
            return NO_CAMERA
        if self.pcb_preview == NO_PCB:
            return NO_PCB
        return READY if self.pcb_preview == DETECTED else WAITING

    # loop
    def tick(self) -> None:
        if self.closing:
            return
        try:
            self._collect()
            self._presence()
            if self.cadence.due() and self.grabber.is_live:
                self._start_inspection()
            camera_message = "" if self.grabber.is_live else (self.grabber.message or self.grabber.status.upper())
            self._show(camera_message)
        except Exception:
            traceback.print_exc()
        self.root.after(30, self.tick)

    def _presence(self) -> None:
        if self.presence_future is not None and self.presence_future.done():
            try:
                self.pcb_preview = self.presence_future.result()
            except Exception:
                self.pcb_preview = ""
            self.presence_future = None
        frame = self.grabber.latest()
        now = time.monotonic()
        if self.presence_future is None and frame is not None and self.grabber.is_live and now - self.presence_at >= self.PRESENCE_EVERY_S:
            self.presence_at = now
            small = cv2.resize(frame.image, (1280, round(1280 * frame.image.shape[0] / frame.image.shape[1])),
                               interpolation=cv2.INTER_AREA)
            self.presence_future = self.presence_executor.submit(
                lambda: detect_pcb(small, self.presence_params, self.segmentation).status)
        if not self.grabber.is_live:
            self.pcb_preview = ""

    def _start_inspection(self) -> None:
        self.cadence.started()
        shown_frame = self.grabber.latest()
        after = shown_frame.t if shown_frame is not None else float("-inf")
        board = self.board_id()

        def job() -> LiveResult:
            # a frame that arrives after the capture moment, never the one already on screen
            frame = self.grabber.wait_for_frame_after(after, self.fresh_frame_timeout_s)
            return self.inspector.inspect(None if frame is None else frame.image, datetime.now(), board)

        self.future = self.executor.submit(job)

    def _collect(self) -> None:
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
        except Exception as e:
            result = LiveResult(datetime.now(), INSPECTION_ERROR, message=f"inspection crashed: {e}")
        self.cadence.finished()
        self.result = result
        self.results.append(result)
        self.on_result(result)
        if self.closing:
            return
        self.result_text.set(console_text(result))
        picture = None
        if result.output_dir and (result.output_dir / "annotated.jpg").is_file():
            picture = result.output_dir / "annotated.jpg"
        elif result.output_dir and (result.output_dir / "overview.jpg").is_file():
            picture = result.output_dir / "overview.jpg"
        elif result.image is not None and result.image.suffix == ".jpg":
            picture = result.image
        if picture is not None:
            img = self.Image.open(picture)
            img.thumbnail((420, 300))
            self._result_photo = self.ImageTk.PhotoImage(img)
            self.result_image.configure(image=self._result_photo)

    def _show(self, camera_message: str) -> None:
        frame = self.grabber.latest()
        lines = overlay_lines(self.camera_label, self.cadence.interval, self.state(), self.result,
                              self.cadence.seconds_left(), camera_message, self.pcb_preview, self.board_id())
        if frame is None:
            image = np.zeros((round(self.PREVIEW_WIDTH * 9 / 16), self.PREVIEW_WIDTH, 3), np.uint8)
        else:
            h, w = frame.image.shape[:2]
            image = cv2.resize(frame.image, (self.PREVIEW_WIDTH, round(h * self.PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        shown_image = draw_overlay(image, lines)
        self._preview_photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(shown_image, cv2.COLOR_BGR2RGB)))
        self.preview.configure(image=self._preview_photo)


# --- startup -------------------------------------------------------------------------

def main() -> int:
    capture_cfg = load_config(ROOT / "configs" / "capture" / "droidcam_e1b.yaml")["camera"]
    parser = argparse.ArgumentParser(description="Live PCB inspection with the staged Experiment 2 pipeline.")
    parser.add_argument("--camera-index", type=int, default=capture_cfg["index"],
                        help=f"camera index (default: camera.index in configs/capture/droidcam_e1b.yaml = {capture_cfg['index']})")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between inspections (default 3)")
    parser.add_argument("--board-id", default="", help="physical board under the camera, e.g. GOOD_016 (editable in the window)")
    parser.add_argument("--width", type=int, default=1920, help="requested frame width (default 1920)")
    parser.add_argument("--height", type=int, default=1080, help="requested frame height (default 1080)")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "experiment_002" / "config.yaml")
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "data" / "testing" / "live")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "testing" / "live_output")
    parser.add_argument("--run-seconds", type=float, default=0,
                        help="stop by itself after this many seconds, as if Q were pressed (default 0: until Q / ESC)")
    args = parser.parse_args()
    if args.camera_index < 0 or args.interval <= 0:
        parser.error("--camera-index must be >= 0 and --interval > 0")
    if args.board_id and not BOARD_ID.fullmatch(args.board_id.upper()):
        parser.error("--board-id must look like GOOD_016, DEFECT_021 or TEST_030")

    cfg = load_inspection_config(ROOT, args.config)
    reference, cmap = load_reference_and_map(cfg, ROOT / "experiments" / cfg.experiment / "reference" / cfg.reference_file.stem)
    print("=" * 40, "LIVE PCB INSPECTION", "=" * 40, sep="\n")
    print(f"Camera: {args.camera_index}")
    print(f"Resolution: {args.width}x{args.height} (requested)")
    print(f"Inspection interval: {args.interval:g} seconds")
    print(f"Component map: {len(cmap.components)} components"
          + (f" ({len(cmap.unverified)} unverified draft boxes: NOT production-grade)" if cmap.unverified else ""))
    print("Press SPACE for immediate inspection")
    print("Press Q/ESC to exit")
    print("=" * 40)
    gallery = load_gallery(cfg, reference)
    limits, limits_note = load_calibration(cfg, gallery)
    print(f"GOOD reference set: {len(gallery.names)} images of {len(gallery.board_ids)} physical boards "
          f"({', '.join(gallery.board_ids)}); {limits_note}")
    print("Known limitation: unvalidated prototype; it missed the R37 defect in the existing TEST image "
          "(experiments/experiment_002/README.md). PASS means 'no difference found', not a guarantee.")
    device = device_at(list_video_devices(), args.camera_index)
    name = device.name if device else f"camera {args.camera_index}"
    grabber = FrameGrabber(OpenCVCamera(args.camera_index, name, args.width, args.height, 30, backend="dshow"))
    grabber.start()
    deadline = time.monotonic() + 8
    while grabber.latest() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    frame = grabber.latest()
    if frame is None:
        print(f"Camera {args.camera_index} ({name}): NO CAMERA picture yet. {grabber.message} The window opens anyway and keeps retrying.")
    else:
        print(f"Camera {args.camera_index} ({name}) CONNECTED: actual frame resolution "
              f"{frame.image.shape[1]}x{frame.image.shape[0]}, stream {grabber.status}"
              + ("" if grabber.status == STATUS_LIVE else f" ({grabber.message})"))
    inspector = LiveInspector(cfg, reference, cmap, gallery, args.capture_dir, args.output_dir, grabber.info,
                              limits, limits_note)

    import tkinter as tk
    root = tk.Tk()
    app = LiveApp(root, grabber, inspector, f"{args.camera_index} ({name})", args.interval,
                  capture_cfg["fresh_frame_timeout_s"], board_id=args.board_id.upper())
    if args.run_seconds > 0:
        root.after(round(args.run_seconds * 1000), app.close)
    try:
        root.mainloop()
    finally:
        if not app.closing:
            app.close()
    counts: dict[str, int] = {}
    for r in app.results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"Stopped. Inspections: {len(app.results)} ({', '.join(f'{shown(k)} {v}' for k, v in sorted(counts.items())) or '-'}). "
          f"Camera released. Summary: {_rel(args.output_dir / 'summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
