"""Live inspection station: camera -> automatic capture -> stored -> product -> inspection -> GOOD / DEFECT / REVIEW.

    python scripts/station.py                          window; camera from configs/inspection.yaml station.camera_config
    python scripts/station.py --camera-index 1         another camera for this run
    python scripts/station.py --product lg_indicator   operator-selected product instead of AUTO identification
    python scripts/station.py --headless --captures 5  no window: print each result, stop after 5 captures

No operator input is needed: when a PCB is in view and the picture is still, ONE capture is taken (configs/
inspection.yaml station.*); the next capture needs the board to be removed or replaced. SPACE captures at once.
Every capture is saved FIRST (data/captures/<session>/, SHA-256 verified), then the product is identified
(AUTO) or taken from the selection, and the product's own inspection runs (pcb_inspection.station).

Window: PRODUCT (AUTO DETECTING... / name), RESULT (GOOD / DEFECT with type, location, confidence / REVIEW with
the reason / ERROR), evidence picture, and human review buttons (CONFIRM DEFECT / CONFIRM GOOD / REJECT RESULT,
stored separately; the AI result is never changed). Q / ESC quits (a running inspection finishes and is saved).

No simulation: if the camera delivers no frame, nothing is invented; the failure is logged in the session.
Inspection engine: REFERENCE_BASELINE, EXPERIMENTAL (docs/inspection_architecture.md).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pcb_inspection.camera import Camera, LiveCamera  # noqa: E402
from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.inspection.identification import AUTO  # noqa: E402
from pcb_inspection.inspection.types import DEFECT, ERROR, GOOD, REVIEW  # noqa: E402
from pcb_inspection.preprocessing.board import BoardSegmentationParams  # noqa: E402
from pcb_inspection.preprocessing.presence import DETECTED, PresenceParams, detect_pcb  # noqa: E402
from pcb_inspection.station import (  # noqa: E402
    AutoTrigger,
    CaptureError,
    InspectionStation,
    StationResult,
    TriggerSettings,
    record_review,
)

COLOURS = {GOOD: "#2e9e44", DEFECT: "#d32f2f", REVIEW: "#e0a000", ERROR: "#7b1fa2"}


# --- texts -------------------------------------------------------------------------------------

def result_lines(r: StationResult) -> list[str]:
    """What the operator reads for one result (also printed in headless mode)."""
    o = r.outcome
    ident = r.identification
    product = (r.product_name or (o.product_id or "NOT DETERMINED")).upper()

    # Alignment line
    if o.alignment_status == "OK":
        align_str = "PASS"
        if o.alignment_error:
            ae = o.alignment_error
            parts = []
            if ae.get("inliers") is not None:
                parts.append(f"{ae['inliers']} inliers")
            if ae.get("fit_error_mm") is not None:
                parts.append(f"fit error {ae['fit_error_mm']:.2f} mm")
            if parts:
                align_str += f" ({', '.join(parts)})"
    elif o.alignment_status == "FAILED":
        align_str = "FAIL"
        if o.alignment_error and o.alignment_error.get("reason"):
            align_str += f" ({o.alignment_error['reason']})"
    else:
        align_str = "NOT RUN"

    # Inspection line
    if o.overall_decision in (GOOD, DEFECT):
        checked = f" ({o.regions_checked} regions checked)" if o.regions_checked is not None else ""
        insp_str = f"COMPLETE{checked}"
    elif o.alignment_status == "FAILED":
        insp_str = "STOPPED AT ALIGNMENT"
    elif o.overall_decision == ERROR:
        insp_str = "ERROR"
    else:
        insp_str = f"COMPLETE ({o.regions_checked} regions checked)" if o.regions_checked else "REVIEW"

    # Reference used string
    rb = o.reference_bank or {}
    boards = rb.get("boards", [])
    geom = rb.get("geometric_reference")
    geom_name = Path(geom).name if geom else ""
    if boards:
        ref_text = ", ".join(boards)
        if len(boards) == 1:
            ref_text += " [EXPERIMENTAL / SINGLE_REFERENCE]"
    elif geom_name:
        ref_text = geom_name
    else:
        ref_text = "none"
    if geom_name and boards:
        ref_text += f" ({geom_name})"

    line_sep = "=" * 32
    lines = [
        f"PRODUCT: {product}" + (f"  (match ratio {ident.score:.2f})" if ident and ident.score is not None else ""),
        f"ALIGNMENT: {align_str}",
        f"INSPECTION: {insp_str}",
        "",
        line_sep,
        f"FINAL RESULT: {o.overall_decision}",
        line_sep,
    ]

    is_single_ref = (
        o.details.get("reference_mode") == "EXPERIMENTAL / SINGLE_REFERENCE"
        or (o.reference_bank and len(o.reference_bank.get("boards", [])) == 1)
    )
    if is_single_ref:
        lines.append("MODE: EXPERIMENTAL / SINGLE_REFERENCE (not production-valid)")
    elif o.engine_status:
        lines.append(f"MODE: {o.engine_status}")

    lines.append("")

    if o.overall_decision == DEFECT:
        for idx, d in enumerate(o.defects, 1):
            if len(o.defects) > 1:
                lines.append(f"--- DEFECT {idx} OF {len(o.defects)} ---")
            lines.append(f"DEFECT TYPE:\n  {d.defect_type.replace('_', ' ').title()}")
            lines.append(f"COMPONENT:\n  {d.region or 'UNKNOWN'}")
            loc = d.location or {}
            center = loc.get("center_mm", [0, 0])
            box = loc.get("box_mm", [0, 0, 0, 0])
            lines.append(
                f"LOCATION:\n  x: {center[0]:.1f} mm, y: {center[1]:.1f} mm "
                f"(canonical board coordinates; box [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] mm)"
            )
            lines.append(f"REGION:\n  {d.region or 'UNKNOWN'}")

            ev = d.evidence or {}
            sim = ev.get("similarity")
            lim = ev.get("limit")
            below = ev.get("below_limit_by")
            diff = ev.get("difference")
            if sim is not None and diff is not None:
                lines.append(f"MEASURED SCORE:\n  similarity {sim:.3f} (difference {diff:.3f})")
            elif sim is not None:
                lines.append(f"MEASURED SCORE:\n  similarity {sim:.3f}")
            else:
                lines.append("MEASURED SCORE:\n  N/A")

            if lim is not None and below is not None:
                lines.append(f"THRESHOLD:\n  limit {lim:.3f} (below threshold by {below:.3f})")
            elif lim is not None:
                lines.append(f"THRESHOLD:\n  limit {lim:.3f}")
            else:
                lines.append("THRESHOLD:\n  N/A")

            reason_str = (
                f"Component '{d.region}' appearance differs significantly from the GOOD reference bank "
                f"at this location (similarity {sim:.3f} < threshold {lim:.3f}). Component is absent, damaged, or obscured."
                if (sim is not None and lim is not None)
                else f"Component '{d.region}' failed the reference baseline check."
            )
            lines.append(f"REASON:\n  {reason_str}")
            lines.append(f"REFERENCE:\n  {ref_text}")
            conf_str = f"{d.confidence:.2f}" if d.confidence is not None else "NOT CALIBRATED"
            lines.append(f"CONFIDENCE:\n  {conf_str}")
            lines.append("")

        if o.reasons:
            other_reasons = [r.text for r in o.reasons if r.code != "DEFECT_FOUND"]
            if other_reasons:
                lines.append("ADDITIONAL NOTES:\n  " + "; ".join(other_reasons))

    elif o.overall_decision == GOOD:
        lines.append("SUMMARY:\n  No region exceeded the configured defect threshold.")
        lines.append(f"REFERENCE USED:\n  {ref_text}")
        lines.append(f"ALIGNMENT STATUS:\n  {align_str}")
        regions_str = f"All {o.regions_checked} inspected regions" if o.regions_checked else "All regions"
        lines.append(f"INSPECTION SCORE:\n  {regions_str} matched reference within allowed tolerances.")
        conf_str = f"{o.confidence:.2f}" if o.confidence is not None else "NOT CALIBRATED"
        lines.append(f"CONFIDENCE:\n  {conf_str}")
        if o.reasons:
            lines.append(f"DETAILS:\n  {o.reasons[0].text}")

    elif o.overall_decision == REVIEW:
        lines.append("EXACT REVIEW REASON:")
        for x in o.reasons:
            code_desc = {
                "ALIGNMENT": "alignment uncertainty",
                "IMAGE_QUALITY": "image quality",
                "UNKNOWN_PRODUCT": "unknown product",
                "AMBIGUOUS_PRODUCT": "ambiguous product",
                "INSUFFICIENT_GOLDEN_REFERENCES": "reference uncertainty (insufficient references)",
                "NO_GOLDEN_REFERENCES": "reference uncertainty (no references configured)",
                "UNCERTAIN": "insufficient evidence / measurement uncertainty",
                "NOT_CHECKABLE": "unverified or partly hidden region",
                "IMPLAUSIBLE": "implausible defect count (motion or obstruction)",
                "NO_REGIONS": "no regions in map",
                "NO_PCB": "no PCB detected",
            }.get(x.code, x.code)
            lines.append(f"  * {code_desc}: {x.text}")
        lines.append(f"REFERENCE USED:\n  {ref_text}")
        lines.append("CONFIDENCE:\n  NOT CALIBRATED")

    else:  # ERROR
        lines.append("ERROR DETAILS:")
        for x in o.reasons:
            lines.append(f"  * {x.code}: {x.text}")
        lines.append("CONFIDENCE:\n  NOT CALIBRATED")

    lines.append("")
    sec_str = f"  ({o.seconds:.2f} s)" if o.seconds is not None else ""
    lines.append(f"INSPECTION ID: {o.inspection_id}{sec_str}")
    return lines


# --- controller (no window) ---------------------------------------------------------------------

@dataclass
class Observation:
    t: float
    pcb: str                  # DETECTED / NO_PCB / ""


class LiveStation:
    """Preview -> presence -> automatic trigger / interval -> capture -> station.process (worker thread).

    Threads: the camera's reader thread; the observe loop (start_loop: presence + trigger / interval, never a GUI thread);
    one inspection worker. on_result is called from the observe loop: a GUI must hand it to its own thread."""

    PRESENCE_WIDTH = 640                        # presence / stillness need no more (112 ms at 960 px)

    def __init__(self, station: InspectionStation, camera: Camera, selection: str, trigger: AutoTrigger,
                 presence: PresenceParams, segmentation: BoardSegmentationParams, presence_every_s: float,
                 on_result: Callable[[StationResult | None, str], None] = lambda r, msg: None,
                 auto_inspection: bool = False, interval_s: float = 3.0):
        self.station, self.camera, self.selection = station, camera, selection
        self.trigger, self.presence, self.segmentation = trigger, presence, segmentation
        self.presence_every_s, self.on_result = presence_every_s, on_result
        self.auto_inspection = auto_inspection
        self.interval_s = float(interval_s)
        self.next_capture_at = 0.0
        self.session_id = station.start_session(selection, camera.get_info())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")
        self.future: Future | None = None
        self.results: list[StationResult] = []
        self.failures: list[str] = []
        self.pcb = ""
        self._presence_at = 0.0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._loop: threading.Thread | None = None

    def start_loop(self, every_s: float = 0.05) -> None:
        """Run observe() on a background thread until close()."""
        if self._loop is not None:
            return

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.observe(time.monotonic())
                except Exception:  # noqa: BLE001 - the loop must survive a bad frame
                    traceback.print_exc()
                self._stop.wait(every_s)

        self._loop = threading.Thread(target=run, name="live-observe", daemon=True)
        self._loop.start()

    @property
    def busy(self) -> bool:
        return self.future is not None and not self.future.done()

    @property
    def seconds_until_next_capture(self) -> float:
        if not self.auto_inspection or self.busy:
            return 0.0
        if self.next_capture_at == 0.0 or self.next_capture_at == float("inf"):
            return self.interval_s
        return max(0.0, self.next_capture_at - time.monotonic())

    def start_auto(self) -> None:
        with self.lock:
            self.auto_inspection = True
            self.next_capture_at = time.monotonic() + self.interval_s

    def stop_auto(self) -> None:
        with self.lock:
            self.auto_inspection = False

    def set_interval(self, seconds: float) -> None:
        with self.lock:
            self.interval_s = max(0.1, float(seconds))
            if self.auto_inspection:
                self.next_capture_at = time.monotonic() + self.interval_s

    def observe(self, now: float) -> None:
        """One preview step: handles presence, auto-interval trigger, or manual trigger; starts capture when due."""
        self.collect()
        if now - self._presence_at >= self.presence_every_s:
            self._presence_at = now
            frame = self.camera.read()
            if frame is None:
                self.pcb = ""
                # If camera is disconnected, attempt safe reconnect
                if hasattr(self.camera, "open") and not getattr(self.camera, "is_open", lambda: True)():
                    try:
                        self.camera.open()
                    except Exception:
                        pass
            else:
                w = self.PRESENCE_WIDTH
                small = cv2.resize(frame, (w, round(w * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
                self.pcb = detect_pcb(small, self.presence, self.segmentation).status

        if self.busy:
            return

        if self.auto_inspection:
            if self.next_capture_at == 0.0:
                self.next_capture_at = now + self.interval_s
            elif now >= self.next_capture_at:
                # Do not capture multiple frames simultaneously; schedule capture
                self.next_capture_at = float("inf")
                self.future = self.executor.submit(self._capture_and_inspect)
        else:
            frame = self.camera.read()
            if frame is not None:
                w = self.PRESENCE_WIDTH
                small = cv2.resize(frame, (w, round(w * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
                if self.trigger.update(now, self.pcb == DETECTED, small) and not self.busy:
                    self.future = self.executor.submit(self._capture_and_inspect)

    def capture_now(self) -> None:
        if not self.busy:
            self.next_capture_at = float("inf")
            self.future = self.executor.submit(self._capture_and_inspect)

    def _capture_and_inspect(self) -> tuple[StationResult | None, str]:
        frame = self.camera.capture()
        if frame is None:
            msg = "camera delivered no frame: nothing captured (logged in the session)"
            self.station.captures.record_failure(self.session_id, msg)
            return None, msg
        try:
            return self.station.process(frame, self.session_id, self.selection), ""
        except CaptureError as e:
            self.station.captures.record_failure(self.session_id, f"capture not stored: {e}")
            return None, f"ERROR: capture could not be stored ({e})"

    def collect(self) -> None:
        with self.lock:                          # called by the observe loop and by close(): handle a result once
            if self.future is None or not self.future.done():
                return
            future, self.future = self.future, None
        try:
            result, message = future.result()
        except Exception as e:  # noqa: BLE001
            result, message = None, f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()
        if result is not None:
            self.results.append(result)
        else:
            self.failures.append(message)
        # Automatic 3-second cycle: wait interval -> capture -> inspect -> result shown -> wait interval -> next capture
        if self.auto_inspection:
            self.next_capture_at = time.monotonic() + self.interval_s
        self.on_result(result, message)

    def close(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.join(timeout=10)
        self.executor.shutdown(wait=True)          # a running inspection finishes and is saved
        self.collect()
        self.camera.close()
        self.station.captures.end_session(self.session_id)


# --- window ---------------------------------------------------------------------------------------

class StationWindow:
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    def __init__(self, root, live: LiveStation, camera_label: str):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.root, self.live, self.closing = root, live, False
        self.last: StationResult | None = None
        root.title("LIVE PCB INSPECTION STATION (REFERENCE_BASELINE)")
        root.configure(bg="#121212")

        # Left panel: Live Camera (top) + Inspected PCB with Defect Overlay (bottom)
        left_panel = tk.Frame(root, bg="#121212", padx=8, pady=8)
        left_panel.grid(row=0, column=0, sticky="nsew")

        # Live Camera Header & Image
        self.live_title_var = tk.StringVar(value="LIVE CAMERA FEED")
        tk.Label(left_panel, textvariable=self.live_title_var, fg="#64b5f6", bg="#121212",
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", pady=(0, 2))
        self._photo = ImageTk.PhotoImage(Image.new("RGB", (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), "black"))
        self.preview = tk.Label(left_panel, image=self._photo, bg="black", bd=1, relief="solid")
        self.preview.pack(anchor="nw", pady=(0, 8))

        # Inspected PCB Header & Image
        self.defect_title_var = tk.StringVar(value="INSPECTED PCB & DEFECT LOCATION (Awaiting inspection...)")
        self.defect_header_label = tk.Label(left_panel, textvariable=self.defect_title_var, fg="#ffd54f", bg="#121212",
                                            font=("Segoe UI", 11, "bold"), anchor="w")
        self.defect_header_label.pack(fill="x", pady=(0, 2))
        self._defect_photo = ImageTk.PhotoImage(Image.new("RGB", (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), "#1a1a1a"))
        self.defect_display = tk.Label(left_panel, image=self._defect_photo, bg="black", bd=1, relief="solid")
        self.defect_display.pack(anchor="nw")

        # Right panel: Controls, Result Banner, Detailed Findings, Zoomed Evidence, Human Review
        side = tk.Frame(root, bg="#1e1e1e", padx=12, pady=8, bd=1, relief="ridge")
        side.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)

        # 1. Automatic Inspection Controls Box
        ctrl_frame = tk.LabelFrame(side, text=" AUTOMATIC INSPECTION CONTROLS ", fg="#e0e0e0", bg="#252525",
                                   font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        ctrl_frame.pack(fill="x", pady=(0, 8))

        # Status row
        status_row = tk.Frame(ctrl_frame, bg="#252525")
        status_row.pack(fill="x", pady=(0, 4))
        self.auto_status_var = tk.StringVar(
            value="AUTO INSPECTION: ON" if live.auto_inspection else "AUTO INSPECTION: OFF"
        )
        self.auto_status_label = tk.Label(status_row, textvariable=self.auto_status_var,
                                          fg="#00e676" if live.auto_inspection else "#ff5252",
                                          bg="#252525", font=("Segoe UI", 11, "bold"))
        self.auto_status_label.pack(side="left")

        self.countdown_var = tk.StringVar(value="(3.0s interval)")
        self.countdown_label = tk.Label(status_row, textvariable=self.countdown_var, fg="#bdbdbd",
                                        bg="#252525", font=("Segoe UI", 10))
        self.countdown_label.pack(side="right")

        # Buttons row: START / STOP / CAPTURE NOW
        btn_row = tk.Frame(ctrl_frame, bg="#252525")
        btn_row.pack(fill="x", pady=(0, 6))
        self.start_btn = tk.Button(btn_row, text="START AUTO INSPECTION", bg="#2e7d32", fg="white",
                                   font=("Segoe UI", 9, "bold"), activebackground="#1b5e20", activeforeground="white",
                                   command=self._start_auto, padx=8, pady=2)
        self.start_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = tk.Button(btn_row, text="STOP", bg="#c62828", fg="white",
                                  font=("Segoe UI", 9, "bold"), activebackground="#b71c1c", activeforeground="white",
                                  command=self._stop_auto, padx=12, pady=2)
        self.stop_btn.pack(side="left", padx=(0, 4))
        self.capture_btn = tk.Button(btn_row, text="CAPTURE NOW (SPACE)", bg="#37474f", fg="white",
                                     font=("Segoe UI", 9), command=self.live.capture_now, padx=6, pady=2)
        self.capture_btn.pack(side="right")

        # Interval adjustment row + Product
        cfg_row = tk.Frame(ctrl_frame, bg="#252525")
        cfg_row.pack(fill="x")
        tk.Label(cfg_row, text="Interval:", fg="#e0e0e0", bg="#252525", font=("Segoe UI", 9)).pack(side="left")
        tk.Button(cfg_row, text="-", width=2, bg="#424242", fg="white", command=self._decrease_interval).pack(side="left", padx=2)
        self.interval_text = tk.StringVar(value=f"{live.interval_s:.1f}s")
        tk.Label(cfg_row, textvariable=self.interval_text, fg="#ffffff", bg="#252525",
                 font=("Segoe UI", 9, "bold"), width=5).pack(side="left")
        tk.Button(cfg_row, text="+", width=2, bg="#424242", fg="white", command=self._increase_interval).pack(side="left", padx=2)

        tk.Label(cfg_row, text="  Product:", fg="#e0e0e0", bg="#252525", font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        choices = [AUTO] + [p.product_id for p in live.station.profiles]
        self.product_var = tk.StringVar(value=live.selection)
        box = ttk.Combobox(cfg_row, textvariable=self.product_var, values=choices, state="readonly", width=14)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._select())

        # 2. Result Header Lines
        self.product_text = tk.StringVar(value="PRODUCT: AUTO DETECTING..." if live.selection == AUTO
                                         else f"PRODUCT: {live.selection.upper()}")
        self.align_text = tk.StringVar(value="ALIGNMENT: -")
        self.insp_text = tk.StringVar(value="INSPECTION: -")
        tk.Label(side, textvariable=self.product_text, fg="white", bg="#1e1e1e", font=("Segoe UI", 12, "bold"),
                 wraplength=480, justify="left").pack(anchor="w", pady=(2, 0))
        tk.Label(side, textvariable=self.align_text, fg="#80d8ff", bg="#1e1e1e", font=("Segoe UI", 10, "bold"),
                 wraplength=480, justify="left").pack(anchor="w")
        tk.Label(side, textvariable=self.insp_text, fg="#80d8ff", bg="#1e1e1e", font=("Segoe UI", 10, "bold"),
                 wraplength=480, justify="left").pack(anchor="w")

        # 3. Final Result Banner
        self.banner_text = tk.StringVar(value="================================\nFINAL RESULT: -\n================================")
        self.result_label = tk.Label(side, textvariable=self.banner_text, fg="white", bg="#333333",
                                     font=("Consolas", 12, "bold"), justify="center", pady=4)
        self.result_label.pack(fill="x", pady=6)

        # 4. Detailed findings (Scrollable Text)
        detail_frame = tk.Frame(side, bg="#1e1e1e")
        detail_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.detail_text = tk.Text(detail_frame, width=54, height=12, bg="#0e0e0e", fg="#e0e0e0",
                                   font=("Consolas", 9), wrap="word", bd=1, relief="solid")
        scrollbar = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=scrollbar.set)
        self.detail_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.detail_text.insert("end", "Place a PCB under the camera. Automatic inspection will begin.")
        self.detail_text.configure(state="disabled")

        # 5. Component Evidence Zoom Thumbnail
        tk.Label(side, text="ZOOMED COMPONENT EVIDENCE (Reference vs Test):", fg="#bdbdbd", bg="#1e1e1e",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x")
        self._thumb = ImageTk.PhotoImage(Image.new("RGB", (480, 140), "#111111"))
        self.thumb = tk.Label(side, image=self._thumb, bg="black", bd=1, relief="solid")
        self.thumb.pack(fill="x", pady=(2, 6))

        # 6. Human review section
        review = tk.Frame(side, bg="#1e1e1e")
        review.pack(fill="x")
        tk.Label(review, text="Reviewer:", fg="white", bg="#1e1e1e").grid(row=0, column=0, sticky="w")
        self.reviewer = tk.Entry(review, width=16)
        self.reviewer.grid(row=0, column=1, columnspan=2, sticky="w", padx=4)
        for col, (text, decision) in enumerate((("CONFIRM DEFECT", "CONFIRM_DEFECT"), ("CONFIRM GOOD", "CONFIRM_GOOD"),
                                                ("REJECT RESULT", "REJECT_RESULT"))):
            tk.Button(review, text=text, font=("Segoe UI", 8),
                      command=lambda d=decision: self._review(d)).grid(row=1, column=col, padx=2, pady=3)
        self.review_text = tk.StringVar(value="")
        tk.Label(side, textvariable=self.review_text, fg="#bbbbbb", bg="#1e1e1e", wraplength=480, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w")

        # Bottom status bar
        self.status = tk.StringVar(value="")
        tk.Label(root, textvariable=self.status, fg="#cccccc", bg="#222222", font=("Consolas", 9), anchor="w",
                 padx=6, pady=2).grid(row=1, column=0, columnspan=2, sticky="we")

        self.camera_label = camera_label
        root.bind("<Escape>", lambda e: self.close())
        root.bind("<KeyPress>", self._key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.results: queue.Queue = queue.Queue()         # Tk may only be used from its own thread
        live.on_result = lambda r, m: self.results.put((r, m))
        self._shown_frame = None
        self._shown_at = 0.0
        live.start_loop()                                  # presence + trigger + capture: never in this thread
        root.after(50, self.tick)

    def _start_auto(self) -> None:
        self.live.start_auto()
        self.auto_status_var.set("AUTO INSPECTION: ON")
        self.auto_status_label.configure(fg="#00e676")

    def _stop_auto(self) -> None:
        self.live.stop_auto()
        self.auto_status_var.set("AUTO INSPECTION: OFF")
        self.auto_status_label.configure(fg="#ff5252")

    def _decrease_interval(self) -> None:
        val = max(1.0, round(self.live.interval_s - 0.5, 1))
        self.live.set_interval(val)
        self.interval_text.set(f"{val:.1f}s")

    def _increase_interval(self) -> None:
        val = min(30.0, round(self.live.interval_s + 0.5, 1))
        self.live.set_interval(val)
        self.interval_text.set(f"{val:.1f}s")

    def _key(self, event) -> None:
        if event.widget is self.reviewer:
            return
        if event.keysym.lower() == "q":
            self.close()
        elif event.keysym == "space":
            self.live.capture_now()

    def _select(self) -> None:
        self.live.selection = self.product_var.get()
        self.product_text.set("PRODUCT: AUTO DETECTING..." if self.live.selection == AUTO
                              else f"PRODUCT: {self.live.selection.upper()} (selected)")

    def _review(self, decision: str) -> None:
        if self.last is None or self.last.result_path is None:
            self.review_text.set("No stored result to review.")
            return
        try:
            row = record_review(self.last.result_path, decision, self.reviewer.get())
            self.review_text.set(f"Review stored: {row['human_decision']} by {row['reviewer']} "
                                 f"(AI: {row['ai_decision']}; AI result unchanged)")
        except ValueError as e:
            self.review_text.set(str(e))

    def show(self, r: StationResult | None, message: str) -> None:
        if self.closing:
            return
        resample = getattr(self.Image, "Resampling", self.Image).LANCZOS
        if r is None:
            self.result_label.configure(bg=COLOURS[ERROR])
            self.banner_text.set("================================\nFINAL RESULT: ERROR\n================================")
            self.align_text.set("ALIGNMENT: ERROR")
            self.insp_text.set("INSPECTION: ERROR")
            self.defect_title_var.set("INSPECTED PCB: ERROR")
            self.defect_header_label.configure(fg="#ff5252")
            self.detail_text.configure(state="normal")
            self.detail_text.delete("1.0", "end")
            self.detail_text.insert("end", f"CAPTURE/INSPECTION FAILED:\n{message}")
            self.detail_text.configure(state="disabled")
            return
        self.last = r
        lines = result_lines(r)
        self.product_text.set(lines[0])
        self.align_text.set(lines[1])
        self.insp_text.set(lines[2])
        self.banner_text.set(f"================================\nFINAL RESULT: {r.decision}\n================================")
        self.result_label.configure(bg=COLOURS.get(r.decision, "#333333"))

        # Put detailed findings in the scrollable text widget
        detail_content = "\n".join(lines[7:])
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", detail_content)
        self.detail_text.configure(state="disabled")
        self.review_text.set("")

        folder = r.result_path.parent if r.result_path else None

        # Update Inspected PCB & Defect location display
        defect_img_path = None
        if folder is not None:
            if r.decision == DEFECT:
                for name in ("defect_location.jpg", "annotated.jpg", "overview.jpg"):
                    if (folder / name).is_file():
                        defect_img_path = folder / name
                        break
            elif r.decision == GOOD:
                for name in ("annotated.jpg", "aligned.jpg", "overview.jpg"):
                    if (folder / name).is_file():
                        defect_img_path = folder / name
                        break
            else:
                for name in ("overview.jpg", "aligned.jpg", "annotated.jpg"):
                    if (folder / name).is_file():
                        defect_img_path = folder / name
                        break

        if defect_img_path and defect_img_path.is_file():
            img = self.Image.open(defect_img_path)
            img.thumbnail((self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), resample)
            canvas = self.Image.new("RGB", (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), "black")
            paste_x = (self.PREVIEW_WIDTH - img.width) // 2
            paste_y = (self.PREVIEW_HEIGHT - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))
            self._defect_photo = self.ImageTk.PhotoImage(canvas)
            self.defect_display.configure(image=self._defect_photo)

        if r.decision == DEFECT:
            def_components = ", ".join(d.region or 'DEFECT' for d in r.outcome.defects)
            self.defect_title_var.set(f"INSPECTED PCB: DEFECT DETECTED at {def_components}")
            self.defect_header_label.configure(fg="#ff5252")
        elif r.decision == GOOD:
            self.defect_title_var.set(f"INSPECTED PCB: GOOD (All {r.outcome.regions_checked or 0} regions match)")
            self.defect_header_label.configure(fg="#00e676")
        else:
            reason_code = r.outcome.reasons[0].code if r.outcome.reasons else "REVIEW"
            self.defect_title_var.set(f"INSPECTED PCB: REVIEW ({reason_code})")
            self.defect_header_label.configure(fg="#ffd54f")

        # Update Zoomed Evidence Thumbnail in Sidebar
        evidence_shown = False
        if folder is not None:
            for name in ("evidence.jpg", "defect_location.jpg", "annotated.jpg", "overview.jpg"):
                if (folder / name).is_file():
                    ev_img = self.Image.open(folder / name)
                    ev_img.thumbnail((480, 140), resample)
                    ev_canvas = self.Image.new("RGB", (480, 140), "#111111")
                    ev_x = (480 - ev_img.width) // 2
                    ev_y = (140 - ev_img.height) // 2
                    ev_canvas.paste(ev_img, (ev_x, ev_y))
                    self._thumb = self.ImageTk.PhotoImage(ev_canvas)
                    self.thumb.configure(image=self._thumb)
                    evidence_shown = True
                    break
        if not evidence_shown:
            blank = self.Image.new("RGB", (480, 140), "#111111")
            self._thumb = self.ImageTk.PhotoImage(blank)
            self.thumb.configure(image=self._thumb)

    def tick(self) -> None:
        if self.closing:
            return
        try:
            while not self.results.empty():
                self.show(*self.results.get_nowait())
            frame = self.live.camera.read()
            now = time.monotonic()
            redraw = frame is not self._shown_frame and now - self._shown_at >= 1 / 15   # new frames only, <= 15 fps

            # Status and Countdown
            if self.live.busy:
                state = "INSPECTING..."
                self.countdown_var.set("(Inspecting capture...)")
            elif self.live.auto_inspection:
                rem = self.live.seconds_until_next_capture
                state = f"AUTO: next capture in {rem:.1f}s"
                self.countdown_var.set(f"(Next in {rem:.1f}s)")
            else:
                state = f"PAUSED ({self.live.trigger.state})"
                self.countdown_var.set("(Auto paused)")

            if self.live.busy and self.live.selection == AUTO:
                self.product_text.set("PRODUCT: AUTO DETECTING...")
            cam = getattr(self.live.camera, "message", "")
            self.status.set(f"Camera {self.camera_label}: {'NO PICTURE ' + cam if frame is None else 'live'}   "
                            f"PCB: {self.live.pcb or '-'}   {state}   captures: {len(self.live.results)}   "
                            f"session {self.live.session_id}   SPACE capture now   Q quit")
            if redraw:
                self._shown_frame, self._shown_at = frame, now
                if frame is None:
                    image = np.zeros((self.PREVIEW_HEIGHT, self.PREVIEW_WIDTH, 3), np.uint8)
                else:
                    h, w = frame.shape[:2]
                    scale = min(self.PREVIEW_WIDTH / w, self.PREVIEW_HEIGHT / h)
                    nw, nh = round(w * scale), round(h * scale)
                    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                    image = np.zeros((self.PREVIEW_HEIGHT, self.PREVIEW_WIDTH, 3), np.uint8)
                    yoff = (self.PREVIEW_HEIGHT - nh) // 2
                    xoff = (self.PREVIEW_WIDTH - nw) // 2
                    image[yoff:yoff + nh, xoff:xoff + nw] = resized
                self._photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                self.preview.configure(image=self._photo)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.root.after(50, self.tick)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.live.close()
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


# --- startup ---------------------------------------------------------------------------------------

def build(args) -> tuple[LiveStation, str]:
    settings = load_config(ROOT / "configs" / "inspection.yaml")
    st = settings["station"]
    camera_cfg = load_config(ROOT / st["camera_config"])
    cam = camera_cfg["camera"]
    index = cam["index"] if args.camera_index is None else args.camera_index
    print("Loading product profiles and golden reference banks ...")
    station = InspectionStation(ROOT, stage=args.stage, station_id=args.station_id)
    for p in station.profiles:
        bank = station.banks[p.product_id]
        print(f"  {p.product_id}: bank {bank.status} - {bank.reason}")
    for problem in station.problems:
        print(f"  PROFILE PROBLEM: {problem}")
    if args.product != AUTO and station.profile(args.product) is None:
        raise SystemExit(f"unknown product {args.product!r}; known: {[p.product_id for p in station.profiles]}")
    from pcb_inspection.acquisition.camera import device_at, list_video_devices
    device = device_at(list_video_devices(), index)
    name = device.name if device else f"camera {index}"
    camera = LiveCamera(index, name, cam.get("requested_width"), cam.get("requested_height"), 30,
                        backend=cam.get("backend", "dshow"), fresh_frame_timeout_s=cam.get("fresh_frame_timeout_s", 2.0))
    if camera.open():
        frame = camera.read()
        print(f"Camera {index} ({name}): {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Camera {index} ({name}): NO PICTURE yet ({camera.message}); retrying in the background.")
    q = camera_cfg["quality_check"].get("board_detection") or {}
    segmentation = BoardSegmentationParams(**{k: q[k] for k in ("saturation_min", "value_min", "hue_tolerance") if k in q})
    auto = not getattr(args, "no_auto", False)
    interval = float(getattr(args, "interval", 3.0))
    live = LiveStation(station, camera, args.product, AutoTrigger(TriggerSettings.from_mapping(st)),
                       PresenceParams(**st["presence"]), segmentation, float(st["presence_every_s"]),
                       auto_inspection=auto, interval_s=interval)
    threading.Thread(target=station.warm_up, name="warm-up", daemon=True).start()   # first capture not slow
    print(f"Session {live.session_id}: captures -> data/captures/{live.session_id}/ (auto_inspection={auto}, interval={interval:.1f}s)")
    return live, f"{index} ({name})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live PCB inspection station (automatic capture).")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--product", default=AUTO, help="AUTO (default) or a product id (operator selection)")
    parser.add_argument("--stage", default=None, help="production stage of this station, if known")
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between automatic captures (default: 3.0)")
    parser.add_argument("--no-auto", action="store_true", help="disable automatic interval capture on start")
    parser.add_argument("--headless", action="store_true", help="no window; print results")
    parser.add_argument("--captures", type=int, default=0, help="headless: stop after this many captures (0 = no limit)")
    parser.add_argument("--run-seconds", type=float, default=0, help="stop after this many seconds (0 = until Q / Ctrl+C)")
    args = parser.parse_args()
    live, camera_label = build(args)
    started = time.monotonic()
    if args.headless:
        live.on_result = lambda r, m: print("\n".join(result_lines(r)) if r else m, flush=True)
        live.start_loop()
        try:
            while True:
                if args.captures and len(live.results) + len(live.failures) >= args.captures and not live.busy:
                    break
                if args.run_seconds and time.monotonic() - started > args.run_seconds:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            live.close()
    else:
        import tkinter as tk
        root = tk.Tk()
        window = StationWindow(root, live, camera_label)
        if args.run_seconds:
            root.after(round(args.run_seconds * 1000), window.close)
        try:
            root.mainloop()
        finally:
            if not window.closing:
                window.close()
    counts: dict[str, int] = {}
    for r in live.results:
        counts[r.decision] = counts.get(r.decision, 0) + 1
    print(f"Stopped. Captures stored: {len(live.results)} {counts}; capture failures: {len(live.failures)}; "
          f"session data/captures/{live.session_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

