"""PCB data acquisition application (Tkinter): live camera, guided batch capture, human review.

Started with `python scripts/acquire.py`. The logic lives in camera.py,
session.py and controller.py (unit-tested without a camera); this module only
shows their state and forwards operator actions.
"""

from __future__ import annotations

import time
import tkinter as tk
import traceback
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .camera import STATUS_LIVE, FrameGrabber, OpenCVCamera, VideoDevice, device_at, list_video_devices
from .controller import (
    SEQ_PLACE,
    SEQ_REMOVE,
    BoardSequence,
    PHASE_CAMERA_LOST,
    PHASE_CAPTURING,
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_WAITING_PCB,
    PHASE_WAITING_REVIEW,
    CaptureController,
    CaptureOutcome,
    capture_once,
)
from .pcb_view import OK, RECAPTURE, PcbCheck, draw_check, inspection_roi
from .session import (
    DECLARED_CONDITION,
    OTHER_STAGE,
    AcquisitionStore,
    ReviewError,
    Session,
    SessionConfig,
    capture_check,
    inherit_defect_report,
    make_defect_report,
    normalise_board_id,
)

TYPE_LABELS = {"good": "GOOD PCB", "fault": "FAULT PCB", "defect": "DEFECT PCB", "test": "TEST PCB"}
TYPE_COLOURS = {"good": "#b9e4c0", "fault": "#f4b8b8", "defect": "#f4b8b8", "test": "#b8cff4"}
STAGE_LABELS = {"after_smt_reflow": "AFTER_SMT", "after_tht": "AFTER_THT", OTHER_STAGE: "OTHER"}
CHECK_COLOURS = {OK: "#2e7d32", "REVIEW": "#e65100", RECAPTURE: "#c62828"}
PREVIEW_SIZE = (800, 450)
REVIEW_SIZE = (440, 248)
REVIEW_SIZE_WITH_PCB = (320, 180)   # original + PCB image both shown: smaller, so the window fits a laptop screen
PREVIEW_CHECK_WIDTH = 2560      # live PCB check: frames up to this width are checked in full resolution, so the
                                # live overlay gives the same answer as the capture check (~0.2 s per 2560 x 1440
                                # frame; at lower resolutions the watermark match was measured to drop out)
FONT = ("Segoe UI", 10)
BOLD = ("Segoe UI", 10, "bold")


def label_of(key: str) -> str:
    return key.replace("_", " ").capitalize()


class AcquisitionApp:
    def __init__(self, root: tk.Tk, store: AcquisitionStore, capture_cfg: Mapping[str, Any],
                 camera_index: int | None = None):
        self.root, self.store, self.cfg = root, store, capture_cfg
        self.cam_cfg, self.session_cfg = capture_cfg["camera"], capture_cfg["session"]
        self.camera_index = self.cam_cfg["index"] if camera_index is None else camera_index
        self.devices: list[VideoDevice] = []
        self.grabber: FrameGrabber | None = None
        self.session: Session | None = None
        self.controller: CaptureController | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture")
        self.capture_future: Future | None = None
        self.checker = store.settings.pcb_checker
        self.sequence_mode = self.checker is not None and bool(self.session_cfg.get("board_sequence"))
        self.sequence: BoardSequence | None = None
        self.seq_cfg: dict[str, Any] = {}
        self.last_auto_kept: tuple[Session, dict[str, Any]] | None = None
        self.check_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-check")
        self.check_future: Future | None = None
        self.live_check: tuple[PcbCheck, float] | None = None      # (check of a preview frame, its scale)
        self.live_check_at = 0.0                                    # monotonic time of that result
        self.next_check_at = 0.0
        self.check_every_s = float((capture_cfg.get("pcb_check") or {}).get("preview_every_s") or 1.0)
        self.last_signature: str | None = None
        self.flash_until = 0.0
        self.preview_seq: tuple[int, bool] | None = None
        self.requested_size: tuple[int, int] | None = None
        self.review_item: tuple[Session, dict[str, Any]] | None = None
        self._photo = self._review_photo = None

        root.title(capture_cfg.get("window_title") or "PCB Data Acquisition")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.option_add("*Font", FONT)
        self._build()
        interrupted = self.store.mark_interrupted()
        notes = []
        if interrupted:
            notes.append(f"Earlier session(s) {', '.join(interrupted)} were not closed properly: marked 'interrupted'; "
                         "their images are kept and can still be reviewed.")
        rechecked = self.store.recheck_pending()     # waiting images: judged with the current check limits
        if rechecked:
            notes.append(f"{len(rechecked)} image(s) waiting for review were re-checked with the current limits: "
                         + ", ".join(f"{i.split('_TOP_')[-1]} {a} -> {b}" for i, a, b in rechecked[:6])
                         + (" ..." if len(rechecked) > 6 else "") + ".")
        if notes:
            self.message(" ".join(notes))
        self.refresh_devices()
        if self.cam_cfg.get("connect_on_start") and self.device_var.get():
            self.connect_camera()
        self.show_next_review()
        self.on_type_change()
        self.root.after(30, self.loop)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        r = self.root
        left = ttk.Frame(r, padding=8)
        centre = ttk.Frame(r, padding=8)
        right = ttk.Frame(r, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        centre.grid(row=0, column=1, sticky="nsew")
        right.grid(row=0, column=2, sticky="ns")
        r.columnconfigure(1, weight=1)
        r.rowconfigure(0, weight=1)

        ttk.Label(left, text="PCB DATA ACQUISITION", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))

        cam = ttk.LabelFrame(centre, text="Camera", padding=4)
        cam.pack(fill="x")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(cam, textvariable=self.device_var, state="readonly", width=30)
        self.device_combo.grid(row=0, column=0, sticky="w")
        self.res_var = tk.StringVar()
        choices = [f"{w} x {h}" for w, h in self.cam_cfg["resolution_choices"]]
        self.res_combo = ttk.Combobox(cam, textvariable=self.res_var, values=choices, state="readonly", width=11)
        self.res_combo.grid(row=0, column=1, padx=4)
        self.res_var.set(f"{self.cam_cfg['requested_width']} x {self.cam_cfg['requested_height']}")
        self.refresh_btn = ttk.Button(cam, text="Refresh", command=self.refresh_devices)
        self.refresh_btn.grid(row=0, column=2, padx=2)
        self.connect_btn = ttk.Button(cam, text="Connect", command=self.connect_camera)
        self.connect_btn.grid(row=0, column=3, padx=2)
        self.reconnect_btn = ttk.Button(cam, text="Reconnect", command=self.reconnect_camera, state="disabled")
        self.reconnect_btn.grid(row=0, column=4, padx=2)
        self.cam_status_var = tk.StringVar(value="Camera: not connected")
        self.cam_info_var = tk.StringVar(value="Resolution: -   FPS: -")
        ttk.Label(cam, textvariable=self.cam_status_var, font=BOLD).grid(row=1, column=0, sticky="w")
        ttk.Label(cam, textvariable=self.cam_info_var, wraplength=520, justify="left").grid(
            row=1, column=1, columnspan=4, sticky="w")

        mode = ttk.LabelFrame(left, text="Dataset Type", padding=6)
        mode.pack(fill="x", pady=(0, 4))
        types = self.store.settings.dataset_types
        self.type_var = tk.StringVar(value=types[0])
        self.type_buttons = []
        for key in types:
            b = tk.Radiobutton(mode, text=TYPE_LABELS[key], value=key, variable=self.type_var, indicatoron=False,
                               width=28, pady=3, selectcolor=TYPE_COLOURS[key], font=BOLD, command=self.on_type_change)
            b.pack(fill="x", pady=1)
            self.type_buttons.append(b)
        self.guide_title = ttk.Label(left, font=("Segoe UI", 11, "bold"))
        self.guide_title.pack(anchor="w")
        self.guide_text = ttk.Label(left, wraplength=310, justify="left")
        self.guide_text.pack(anchor="w", pady=(0, 6))

        form = ttk.LabelFrame(left, text="Session", padding=6)
        form.pack(fill="x")
        self.board_var = tk.StringVar()
        self.stage_var = tk.StringVar()
        self.count_var = tk.StringVar(value=str(self.session_cfg["default_image_count"]))
        self.interval_var = tk.StringVar(value=str(self.session_cfg["default_interval_s"]))
        self.review_each_var = tk.BooleanVar(value=self.session_cfg["default_review_mode"] == "after_each")
        self.notes_var = tk.StringVar()
        self.lighting_var = tk.StringVar()
        stages = [STAGE_LABELS.get(s, s.upper()) for s in (*self.store.settings.stages, OTHER_STAGE)]
        self.stage_keys = dict(zip(stages, (*self.store.settings.stages, OTHER_STAGE)))
        self.stage_var.set(stages[0])
        self.boards_var = tk.StringVar(value=str(self.session_cfg.get("default_boards", 0)))
        self.auto_keep_var = tk.BooleanVar(value=bool(self.cfg["review"].get("auto_keep_ok")))
        seq = self.sequence_mode
        rows = [
            ("First board ID" if seq else "Board ID (required)" if self.session_cfg.get("require_board_id")
             else "Board ID (recommended)", ttk.Entry(form, textvariable=self.board_var, width=16)),
            ("Stage", ttk.Combobox(form, textvariable=self.stage_var, values=stages, state="readonly", width=14)),
            ("Images per board" if seq else "Images", ttk.Spinbox(form, textvariable=self.count_var, from_=1,
                                                                  to=self.session_cfg["max_image_count"], width=6)),
        ]
        if seq:
            rows.append(("Boards (0 = until STOP)", ttk.Spinbox(form, textvariable=self.boards_var, from_=0, to=999,
                                                                width=6)))
        rows += [
            ("Interval (sec)", ttk.Spinbox(form, textvariable=self.interval_var,
                                           from_=self.session_cfg["min_interval_s"], to=60, width=6)),
            ("Lighting", ttk.Entry(form, textvariable=self.lighting_var, width=22)),
            ("Notes", ttk.Entry(form, textvariable=self.notes_var, width=22)),
        ]
        self.form_widgets: list[tk.Widget] = []
        for i, (text, widget) in enumerate(rows):
            ttk.Label(form, text=text).grid(row=i, column=0, sticky="w", pady=1)
            widget.grid(row=i, column=1, sticky="w", pady=1)
            self.form_widgets.append(widget)
        if self.checker is not None:
            check = ttk.Checkbutton(form, text="Keep OK images automatically", variable=self.auto_keep_var)
        else:
            check = ttk.Checkbutton(form, text="Review each image before the next capture",
                                    variable=self.review_each_var)
        check.grid(row=len(rows), column=0, columnspan=2, sticky="w")
        self.form_widgets.append(check)

        self.defect_frame = ttk.LabelFrame(left, text="Defect information (optional, not verified)", padding=6)
        self.defect_known_var = tk.StringVar(value="unknown")
        self.defect_type_var = tk.StringVar()
        self.defect_ref_var = tk.StringVar()
        self.defect_loc_var = tk.StringVar()
        self.defect_notes_var = tk.StringVar()
        self.defect_labels = {label_of(k): k for k in self.store.settings.defect_types}
        self.defect_known_label = ttk.Label(self.defect_frame, text="Defect known?")
        self.defect_known_label.grid(row=0, column=0, sticky="w")
        ttk.Label(self.defect_frame, text="Only what is known: UNKNOWN is fine. Never guess a label.",
                  foreground="#555555").grid(row=6, column=0, columnspan=2, sticky="w")
        known = ttk.Frame(self.defect_frame)
        known.grid(row=0, column=1, sticky="w")
        self.defect_widgets: list[tk.Widget] = []
        for value, text in (("yes", "YES"), ("unknown", "UNKNOWN")):
            b = ttk.Radiobutton(known, text=text, value=value, variable=self.defect_known_var,
                                command=self.on_defect_known)
            b.pack(side="left")
            self.defect_widgets.append(b)
        self.defect_type_combo = ttk.Combobox(self.defect_frame, textvariable=self.defect_type_var,
                                              values=list(self.defect_labels), state="disabled", width=22)
        for i, (text, widget) in enumerate([
            ("Defect type", self.defect_type_combo),
            ("Component / ref", ttk.Entry(self.defect_frame, textvariable=self.defect_ref_var, width=10)),
            ("Location", ttk.Entry(self.defect_frame, textvariable=self.defect_loc_var, width=22)),
            ("Notes", ttk.Entry(self.defect_frame, textvariable=self.defect_notes_var, width=22)),
        ], start=1):
            ttk.Label(self.defect_frame, text=text).grid(row=i, column=0, sticky="w", pady=1)
            widget.grid(row=i, column=1, sticky="w", pady=1)
            self.defect_widgets.append(widget)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8, side="bottom")
        self.start_btn = tk.Button(buttons, text="START AUTO CAPTURE" if self.sequence_mode else "START SESSION",
                                   font=("Segoe UI", 12, "bold"), bg="#2e7d32", fg="white", command=self.start_session,
                                   height=2)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.stop_btn = tk.Button(buttons, text="STOP", font=("Segoe UI", 12, "bold"), bg="#c62828", fg="white",
                                  command=self.stop_session, height=2, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x")
        self.defect_anchor = buttons

        # centre: status, countdown, live preview
        self.big_var = tk.StringVar(value="READY")
        self.sub_var = tk.StringVar(value="Select the camera and press Connect.")
        ttk.Label(centre, textvariable=self.sub_var, font=("Segoe UI", 12), wraplength=780, justify="center").pack()
        self.big_label = tk.Label(centre, textvariable=self.big_var, font=("Segoe UI", 34, "bold"))
        self.big_label.pack()
        self._blank_preview = ImageTk.PhotoImage(Image.new("RGB", PREVIEW_SIZE, "black"))
        self.review_size = REVIEW_SIZE_WITH_PCB if self.checker is not None else REVIEW_SIZE
        self._blank_review = ImageTk.PhotoImage(Image.new("RGB", self.review_size, "#222222"))
        self.preview = tk.Label(centre, bg="black", image=self._blank_preview)
        self.preview.pack(pady=4)
        self.check_var = tk.StringVar(value="" if self.checker is None else "Live PCB check: waiting for the camera")
        self.check_label = tk.Label(centre, textvariable=self.check_var, font=("Segoe UI", 11, "bold"),
                                    wraplength=780, justify="left")
        self.check_label.pack(fill="x")
        stats = ttk.Frame(centre)
        stats.pack(fill="x")
        self.status_var = tk.StringVar(value="Status: READY")
        self.count_text = tk.StringVar(value="Captured: 0 / 0")
        self.failed_text = tk.StringVar(value="Failed: 0")
        self.last_var = tk.StringVar(value="Last image: -")
        self.quality_var = tk.StringVar(value="")
        for col, var in enumerate((self.status_var, self.count_text, self.failed_text)):
            ttk.Label(stats, textvariable=var, font=("Segoe UI", 12, "bold")).grid(row=0, column=col, padx=12, sticky="w")
        ttk.Label(stats, textvariable=self.last_var).grid(row=1, column=0, columnspan=3, padx=12, sticky="w")
        ttk.Label(stats, textvariable=self.quality_var).grid(row=2, column=0, columnspan=3, padx=12, sticky="w")
        self.msg_var = tk.StringVar()
        self.msg_label = ttk.Label(centre, textvariable=self.msg_var, wraplength=780, justify="left")
        self.msg_label.pack(fill="x", pady=4)

        # right: review
        rev = ttk.LabelFrame(right, text="CAPTURE REVIEW", padding=8)
        rev.pack(fill="y", expand=True)
        self.review_img = tk.Label(rev, bg="#222", image=self._blank_review)
        self.review_img.pack()
        self._review_pcb_photo = None
        self.review_pcb_img = tk.Label(rev)
        if self.checker is not None:
            ttk.Label(rev, text="PCB image (what the dataset uses):").pack(anchor="w")
            self.review_pcb_img.pack()
        names = ("Image", "Type", "Board", "Stage", "Capture check", "Quality", "Flags", "Colour estimate", "Queue")
        self.review_vars = {k: tk.StringVar(value="-") for k in names if k != "Capture check" or self.checker}
        info = ttk.Frame(rev)
        info.pack(fill="x", pady=6)
        for i, (k, var) in enumerate(self.review_vars.items()):
            ttk.Label(info, text=f"{k}:", font=BOLD).grid(row=i, column=0, sticky="nw")
            ttk.Label(info, textvariable=var, wraplength=330, justify="left").grid(row=i, column=1, sticky="w")
        decide = ttk.Frame(rev)
        decide.pack(fill="x", pady=4)
        width = 7 if self.checker is not None else 10
        self.keep_btn = tk.Button(decide, text="KEEP", font=("Segoe UI", 13, "bold"), bg="#2e7d32", fg="white",
                                  width=width, command=lambda: self.decide("keep"), state="disabled")
        self.keep_btn.pack(side="left", expand=True)
        self.reject_btn = tk.Button(decide, text="REJECT", font=("Segoe UI", 13, "bold"), bg="#c62828", fg="white",
                                    width=width, command=lambda: self.decide("reject"), state="disabled")
        self.reject_btn.pack(side="left", expand=True)
        self.undo_btn = tk.Button(decide, text="REJECT\nLAST", font=BOLD, bg="#8d6e63", fg="white", width=8,
                                  command=self.undo_last, state="disabled")   # undo the last automatic KEEP
        if self.checker is not None:
            self.undo_btn.pack(side="left", expand=True)
        reasons = ttk.Frame(rev)
        reasons.pack(fill="x")
        self.reason_labels = {label_of(k): k for k in self.store.settings.reject_reasons}
        self.reason_var = tk.StringVar()
        self.review_note_var = tk.StringVar()
        ttk.Label(reasons, text="Reject reason:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(reasons, textvariable=self.reason_var, values=list(self.reason_labels), state="readonly",
                     width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(reasons, text="Note:").grid(row=1, column=0, sticky="w")
        ttk.Entry(reasons, textvariable=self.review_note_var, width=26).grid(row=1, column=1, sticky="w")
        self.review_feedback = tk.StringVar()
        ttk.Label(rev, textvariable=self.review_feedback, font=("Segoe UI", 11, "bold"), wraplength=400,
                  justify="left").pack(pady=6)

    # ------------------------------------------------------------------ camera
    def refresh_devices(self) -> None:
        """List the devices and pre-select the configured index (camera.index); never another camera."""
        self.devices = list_video_devices()
        self.device_combo["values"] = [f"{d.index}: {d.name}" for d in self.devices]
        if self.grabber is not None:
            return                      # keep the selection of the camera that is in use
        pick = device_at(self.devices, self.camera_index)
        self.device_var.set(f"{pick.index}: {pick.name}" if pick else "")
        if pick is None:
            found = ", ".join(f"{d.index}: {d.name}" for d in self.devices) or "none"
            self.message(f"Configured camera index {self.camera_index} not found (devices: {found}). Set camera.index "
                         "in configs/capture/droidcam_e1b.yaml, or pick a camera and press Connect.", error=True)

    def connect_camera(self) -> None:
        choice = self.device_var.get()
        device = next((d for d in self.devices if f"{d.index}: {d.name}" == choice), None)
        if device is None:
            messagebox.showwarning("Camera", "Select a camera first.")
            return
        width, height = (int(v) for v in self.res_var.get().split(" x "))
        if self.grabber is not None:
            self.grabber.stop()         # release the current camera before opening one again
            self.grabber = None
        camera = OpenCVCamera(device.index, device.name, width, height, self.cam_cfg["requested_fps"],
                              backend=self.cam_cfg["backend"])
        self.grabber = FrameGrabber(camera, failures_before_disconnect=self.cam_cfg["read_failures_before_disconnect"],
                                    frozen_after_s=self.cam_cfg["frozen_after_s"],
                                    reconnect_every_s=self.cam_cfg["reconnect_every_s"])
        self.grabber.start()
        self.requested_size = (width, height)
        self.preview_seq = None
        self.reconnect_btn["state"] = "normal"
        self.message(f"Opening camera {device.index} ({device.name}) with backend {self.cam_cfg['backend'].upper()}...")

    def reconnect_camera(self) -> None:
        if self.grabber is not None:
            self.grabber.request_reconnect()
            self.message("Reconnecting to the camera...")

    def camera_live(self) -> bool:
        return self.grabber is not None and self.grabber.status == STATUS_LIVE

    # ------------------------------------------------------------------ mode / form
    def on_type_change(self) -> None:
        key = self.type_var.get()
        try:
            normalise_board_id(key, self.board_var.get())
        except ValueError:
            self.board_var.set("")      # a board ID of another pool never carries over to this mode
        guide = self.cfg["guidance"][key]
        self.guide_title["text"] = guide["title"]
        self.guide_text["text"] = "\n".join(guide["steps"])
        if DECLARED_CONDITION[key] == "defective":
            word = "Fault" if key == "fault" else "Defect"
            self.defect_frame["text"] = f"{word} information (optional, not verified)"
            self.defect_known_label["text"] = f"{word} known?"
            self.defect_frame.pack(fill="x", pady=4, before=self.defect_anchor)
        else:
            self.defect_frame.pack_forget()
        self.on_defect_known()

    def on_defect_known(self) -> None:
        known = self.defect_known_var.get() == "yes"
        self.defect_type_combo["state"] = "readonly" if known and not self.session else "disabled"
        if not known:
            self.defect_type_var.set("")

    def _lock_form(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        for b in self.type_buttons:
            b["state"] = state
        for w in self.form_widgets + self.defect_widgets:
            try:
                w["state"] = "disabled" if locked else ("readonly" if isinstance(w, ttk.Combobox) else "normal")
            except tk.TclError:
                pass
        for w in (self.device_combo, self.res_combo):
            w["state"] = "disabled" if locked else "readonly"
        self.connect_btn["state"] = self.refresh_btn["state"] = state
        self.start_btn["state"] = state
        self.stop_btn["state"] = "normal" if locked else "disabled"
        self.on_defect_known()

    # ------------------------------------------------------------------ session
    def start_session(self) -> None:
        if not self.camera_live():
            messagebox.showwarning("Camera", "The camera is not live. Connect the DroidCam camera first.")
            return
        if self.sequence_mode:
            self._start_sequence()
            return
        dataset_type = self.type_var.get()
        try:
            board_id = normalise_board_id(dataset_type, self.board_var.get())
            count = int(self.count_var.get())
            interval = float(self.interval_var.get())
            if not 1 <= count <= self.session_cfg["max_image_count"]:
                raise ValueError(f"Images must be 1..{self.session_cfg['max_image_count']}")
            if interval < self.session_cfg["min_interval_s"]:
                raise ValueError(f"Interval must be at least {self.session_cfg['min_interval_s']} s")
            defect_report = None
            if DECLARED_CONDITION[dataset_type] == "defective":
                known = self.defect_known_var.get() == "yes"
                defect_report = make_defect_report(
                    known, self.store.settings.defect_types,
                    self.defect_labels.get(self.defect_type_var.get()) if known else None,
                    self.defect_ref_var.get(), self.defect_loc_var.get(), self.defect_notes_var.get())
                defect_report, _ = inherit_defect_report(self.store, dataset_type, board_id, defect_report)
            config = SessionConfig(dataset_type=dataset_type, stage=self.stage_keys[self.stage_var.get()],
                                   requested_image_count=count, capture_interval_s=interval, board_id=board_id,
                                   review_mode="after_each" if self.review_each_var.get() else "after_session",
                                   defect_report=defect_report, notes=self.notes_var.get(),
                                   lighting=self.lighting_var.get())
        except ValueError as e:
            messagebox.showerror("Session", str(e))
            return
        if board_id is None and self.session_cfg.get("require_board_id"):
            pool = TYPE_LABELS[dataset_type].split()[0]
            messagebox.showerror("Board ID", f"Enter the physical board ID ({pool}_001, {pool}_002, ... or just 1, 2, "
                                 "...). Every image must stay grouped by its physical board.")
            return
        if board_id is None and not messagebox.askyesno(
                "No board ID", "No board ID entered. The images cannot be linked to a physical board, so they can "
                "only be split by session later.\n\nStart anyway?"):
            return
        try:
            self.session = self.store.create_session(config, self.grabber.info())
        except (ValueError, OSError) as e:
            messagebox.showerror("Session", str(e))
            return
        self.board_var.set(board_id or "")
        session = self.session
        wait_for_pcb = self.checker is not None and bool(self.session_cfg.get("wait_for_pcb"))
        self.controller = CaptureController(count, interval, self.review_each_var.get(), camera_ok=self.camera_live,
                                            pending_reviews=lambda: len(session.pending()),
                                            scene_ready=self.pcb_ready if wait_for_pcb else None,
                                            max_consecutive_retakes=int(self.session_cfg.get("max_consecutive_retakes") or 3))
        self.last_signature = None
        self.count_text.set(f"Captured: 0 / {count}")
        self.failed_text.set("Failed: 0")
        self._lock_form(True)
        self.controller.start(time.monotonic())
        self.message(f"Session {session.id} started: {self.store.rel(session.folder)}")
        self.sub_var.set("Preparing capture...")

    # ------------------------------------------------------------------ automatic board sequence
    def _start_sequence(self) -> None:
        """START AUTO CAPTURE: board N, N+1, ... each captured automatically until it has `images per board` KEPT
        images; the operator only places / removes boards (and decides REVIEW images)."""
        dataset_type = self.type_var.get()
        pool = TYPE_LABELS[dataset_type].split()[0]
        try:
            first = normalise_board_id(dataset_type, self.board_var.get())
            if first is None:
                raise ValueError(f"Enter the first physical board ID ({pool}_001 or just 1). The boards are then "
                                 f"captured in order: {pool}_001, {pool}_002, ... - label them like that.")
            per_board, boards = int(self.count_var.get()), int(self.boards_var.get() or 0)
            interval = float(self.interval_var.get())
            if not 1 <= per_board <= self.session_cfg["max_image_count"]:
                raise ValueError(f"Images per board must be 1..{self.session_cfg['max_image_count']}")
            if interval < self.session_cfg["min_interval_s"]:
                raise ValueError(f"Interval must be at least {self.session_cfg['min_interval_s']} s")
            stage = self.stage_keys[self.stage_var.get()]
            if stage == OTHER_STAGE and not self.notes_var.get().strip():
                raise ValueError("stage OTHER needs a note describing the stage")
            number = int(first.split("_")[1])
            done = self.store.kept_count(dataset_type, first)
            if done >= per_board:
                nxt = number
                while nxt < 999 and self.store.kept_count(dataset_type, f"{pool}_{nxt:03d}") >= per_board:
                    nxt += 1
                if not messagebox.askyesno("Board already complete", f"{first} already has {done} kept images "
                                           f"(target {per_board}).\n\nStart with {pool}_{nxt:03d} instead? Place that "
                                           "board under the camera."):
                    return
                number = nxt
            self.sequence = BoardSequence(number, boards or None, float(self.session_cfg.get("removal_s") or 1.0))
        except ValueError as e:
            messagebox.showerror("Auto capture", str(e))
            return
        self.seq_cfg = {"dataset_type": dataset_type, "pool": pool, "per_board": per_board, "interval": interval,
                        "stage": stage, "lighting": self.lighting_var.get(), "notes": self.notes_var.get()}
        self._lock_form(True)
        self._set_defect_form(editable=True)          # fault information of the first board can still be entered
        self.message(f"Auto capture started: {pool} boards from {pool}_{number:03d}, {per_board} kept images each"
                     f"{f', {boards} boards' if boards else ', until STOP'}. Place the board in the dashed frame.")

    def _board_id(self) -> str:
        return f"{self.seq_cfg['pool']}_{self.sequence.number:03d}"

    def _start_board_session(self) -> None:
        cfg, board_id = self.seq_cfg, self._board_id()
        try:
            defect_report = None
            if DECLARED_CONDITION[cfg["dataset_type"]] == "defective":
                known = self.defect_known_var.get() == "yes"
                defect_report = make_defect_report(
                    known, self.store.settings.defect_types,
                    self.defect_labels.get(self.defect_type_var.get()) if known else None,
                    self.defect_ref_var.get(), self.defect_loc_var.get(), self.defect_notes_var.get())
            defect_report, inherited = inherit_defect_report(self.store, cfg["dataset_type"], board_id, defect_report)
            config = SessionConfig(dataset_type=cfg["dataset_type"], stage=cfg["stage"],
                                   requested_image_count=cfg["per_board"], capture_interval_s=cfg["interval"],
                                   board_id=board_id, review_mode="auto_keep_ok" if self.auto_keep_var.get()
                                   else "after_each", defect_report=defect_report, notes=cfg["notes"],
                                   lighting=cfg["lighting"])
            self.session = self.store.create_session(config, self.grabber.info())
        except (ValueError, OSError) as e:
            self.sequence.finish(str(e))
            messagebox.showerror("Auto capture", f"{board_id}: {e}")
            return
        session, dataset_type = self.session, cfg["dataset_type"]
        self._set_defect_form(editable=False)
        self.controller = CaptureController(
            cfg["per_board"], cfg["interval"], True, camera_ok=self.camera_live,
            pending_reviews=lambda: len(session.pending()), scene_ready=self.pcb_ready,
            max_consecutive_retakes=int(self.session_cfg.get("max_consecutive_retakes") or 3),
            kept=lambda: self.store.kept_count(dataset_type, board_id))
        self.last_signature = None
        self.board_var.set(board_id)
        self.count_text.set(f"{board_id}: kept {self.controller.kept_count} / {cfg['per_board']}")
        self.failed_text.set("Failed: 0")
        self.controller.start(time.monotonic())
        carried = (f" Fault information carried over from {inherited}: {label_of(defect_report['defect_type'])}"
                   f"{' ' + defect_report['component_reference'] if defect_report.get('component_reference') else ''}."
                   if inherited else "")
        self.message(f"{board_id}: session {session.id} started ({self.store.rel(session.folder)}).{carried}")

    def _drive_sequence(self, now: float) -> None:
        seq = self.sequence
        if self.controller is not None:
            self._drive(now)                           # the board's session; its end calls _board_session_ended
            return
        if not seq.finished and seq.update(self._pcb_present(), now):
            self._start_board_session()
            return
        if seq.finished:
            self._end_sequence()
            return
        board_id, cfg = self._board_id(), self.seq_cfg
        if seq.state == SEQ_REMOVE:
            self.status_var.set(f"Status: {board_id} DONE")
            self.big_var.set(f"REMOVE {board_id}")
            self.big_label["fg"] = "#2e7d32"
            self.sub_var.set(f"{board_id} is complete ({self.store.kept_count(cfg['dataset_type'], board_id)} / "
                             f"{cfg['per_board']} kept). Lift it out of the view; then place the next board "
                             f"{cfg['pool']}_{seq.number + 1:03d}.")
        elif seq.state == SEQ_PLACE:
            self.status_var.set(f"Status: WAITING FOR {board_id}")
            self.big_var.set(f"PLACE {board_id}")
            self.big_label["fg"] = "#1a237e"
            extra = " Enter its fault information first if it is known." if cfg["pool"] == "FAULT" else ""
            self.sub_var.set(f"Place the {cfg['pool']} board labelled {board_id} in the dashed frame: it is captured "
                             f"automatically ({cfg['per_board']} images).{extra}")

    def _pcb_present(self) -> bool | None:
        """From the latest live PCB check: True = a PCB is in view, False = none, None = no recent check."""
        if self.live_check is None or time.monotonic() - self.live_check_at > 3 * self.check_every_s + 1.0:
            return None
        return "NO_PCB" not in self.live_check[0].problems

    def _board_session_ended(self, board_id: str, completed: bool, reason: str | None) -> None:
        self.sequence.board_finished(completed, reason or "stopped by the operator")
        if completed:
            self._set_defect_form(editable=True, reset=True)   # the next board gets its own (or no) fault label
            self.message(f"{board_id} complete. Remove it; the next board is {self.seq_cfg['pool']}_"
                         f"{self.sequence.number + 1:03d}.")

    def _end_sequence(self) -> None:
        seq, cfg = self.sequence, self.seq_cfg
        done = ", ".join(f"{cfg['pool']}_{n:03d}" for n in seq.completed) or "none"
        self.message(f"Auto capture ended ({seq.stop_reason or 'all boards done'}). Complete boards: {done}.",
                     error=bool(seq.stop_reason and seq.stop_reason != "stopped by the operator"))
        self.status_var.set("Status: AUTO CAPTURE ENDED")
        self.sequence, self.seq_cfg = None, {}
        self._set_defect_form(editable=True, reset=True)
        self._lock_form(False)
        self.show_next_review()

    def _set_defect_form(self, editable: bool, reset: bool = False) -> None:
        if reset:
            self.defect_known_var.set("unknown")
            for var in (self.defect_type_var, self.defect_ref_var, self.defect_loc_var, self.defect_notes_var):
                var.set("")
        for w in self.defect_widgets:
            try:
                w["state"] = "normal" if editable else "disabled"
            except tk.TclError:
                pass
        self.on_defect_known()

    def stop_session(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.message("Stopping: the session is saved; captured images are kept.")
        elif self.sequence is not None:
            self.sequence.finish("stopped by the operator")

    def _end_session(self) -> None:
        c_ctl = self.controller
        session, status, reason = self.session, c_ctl.finish_status, c_ctl.finish_reason
        completed = status == "completed" and c_ctl.goal_reached()
        self.store.finish(session, status)
        c = session.counts
        self.message(f"Session {session.id} {status}{': ' + reason if reason else ''}. Requested "
                     f"{session.data['requested_image_count']}, captured {c['successful_captures']} (automatically "
                     f"retaken {c_ctl.retakes}), failed {c['failed_captures']}; kept {c['kept_count']}, "
                     f"rejected {c['rejected_count']}, pending review {c['pending_review_count']}.", error=bool(reason))
        self.status_var.set(f"Status: SESSION {status.upper()}")
        self.session, self.controller = None, None
        if self.sequence is not None:
            self._board_session_ended(session.board_id, completed, reason)
        else:
            self._lock_form(False)
        self.show_next_review()

    # ------------------------------------------------------------------ main loop
    def loop(self) -> None:
        try:
            now = time.monotonic()
            self._update_camera(now)
            if self.sequence is not None:
                self._drive_sequence(now)
            elif self.controller is not None:
                self._drive(now)
            self._update_undo_button()
        except Exception as e:  # keep the station running; show the error
            traceback.print_exc()
            self.message(f"Error: {e}", error=True)
        self.root.after(30, self.loop)

    def _drive(self, now: float) -> None:
        c = self.controller
        if self.capture_future is not None:
            if not self.capture_future.done():
                return
            try:
                outcome = self.capture_future.result()
            except Exception as e:  # unexpected: still recorded as a failed capture, never as an image
                self.store.record_failure(self.session, f"unexpected error: {e}")
                outcome = CaptureOutcome(False, error=str(e))
            self.capture_future = None
            self._auto_decide(outcome)
            self._after_capture(outcome, now)
            c.capture_done(outcome, now)
            retaken = f"   retaken: {c.retakes}" if c.retakes else ""
            if c.kept_count is not None:
                self.count_text.set(f"{self.session.board_id}: kept {c.kept_count} / {c.requested}{retaken}")
            else:
                self.count_text.set(f"Captured: {c.successes} / {c.requested}{retaken}")
        elif c.tick(now):
            self.capture_future = self.executor.submit(
                capture_once, self.grabber, self.store, self.session, now, self.last_signature,
                self.cam_cfg["fresh_frame_timeout_s"])
        if c.finished and self.capture_future is None:
            self._end_session()
            return
        self._show_phase(now)

    def _after_capture(self, outcome: CaptureOutcome, now: float) -> None:
        if outcome.ok:
            r = outcome.record
            self.last_signature = outcome.signature
            self.flash_until = now + 0.8
            self.last_var.set(f"Last image: {Path(r['original_file']).name}")
            q = r["quality"]
            self.quality_var.set(f"Quality: {q['status'].upper()}   flags: {', '.join(q['flags']) or 'none'}   "
                                 f"{r['width']} x {r['height']}   colour estimate: {r['colour_estimate'] or '-'}")
            check = capture_check(r)
            if outcome.retake:
                self.flash_until = 0.0
                self.message(f"{r['image_id']}: capture check RECAPTURE ({', '.join(check['problems'])}). Stored in "
                             "quarantine automatically; capturing again. " + " ".join(check["messages"]), error=True)
            elif r["review"]["status"] == "kept":
                self.message(f"{r['image_id']}: capture check OK, kept automatically (only the PCB image goes into the "
                             "dataset). Wrong? Press REJECT LAST.")
            elif check is not None and check["status"] != OK:
                self.message(f"Capture check {check['status']}: " + " ".join(check["messages"]),
                             error=check["status"] == RECAPTURE)
            else:
                self.message("")
            if self.review_item is None or self.review_item[0] is not self.session:
                self.show_next_review()   # the running session's images come first
            else:
                self._update_queue_count()
        else:
            self.message(f"Capture failed: {outcome.error}. Not recorded as an image; the session continues.",
                         error=True)
        self.failed_text.set(f"Failed: {self.session.counts['failed_captures']}")

    def _auto_decide(self, outcome: CaptureOutcome) -> None:
        """Automatic decisions on a saved capture: check OK -> KEPT (if "Keep OK images automatically"; the operator
        can undo it with REJECT LAST); check RECAPTURE -> quarantine and taken again (review.auto_reject_recapture);
        check REVIEW -> always waits for the operator's KEEP / REJECT."""
        check = capture_check(outcome.record) if outcome.ok else None
        if check is not None and check["status"] == OK and self.auto_keep_var.get():
            try:
                self.store.review(self.session, outcome.record["image_id"], "keep",
                                  note="automatic: capture check OK", by="capture_check")
                self.last_auto_kept = (self.session, outcome.record)
            except (ReviewError, ValueError, OSError) as e:   # stays pending for the operator
                self.message(f"Automatic KEEP failed: {e}", error=True)
            return
        if check is None or check["status"] != RECAPTURE or not self.cfg["review"].get("auto_reject_recapture"):
            return
        reasons = self.cfg["review"].get("reason_for_check") or {}
        reason = next((reasons[c] for c in check["problems"] if reasons.get(c) in self.store.settings.reject_reasons),
                      "other" if "other" in self.store.settings.reject_reasons else self.store.settings.reject_reasons[0])
        try:
            self.store.review(self.session, outcome.record["image_id"], "reject", reason,
                              note=f"automatic: capture check RECAPTURE ({', '.join(check['problems'])})",
                              by="capture_check")
            outcome.retake = True
        except (ReviewError, ValueError, OSError) as e:   # stays pending for the operator
            self.message(f"Automatic quarantine failed: {e}", error=True)

    def pcb_ready(self) -> bool:
        """Countdown gate: a recent live PCB check that is not RECAPTURE (complete PCB in the ROI, watermark outside,
        sharp, exposed)."""
        if self.live_check is None or time.monotonic() - self.live_check_at > 3 * self.check_every_s + 1.0:
            return False
        return self.live_check[0].status != RECAPTURE

    def _show_phase(self, now: float) -> None:
        c = self.controller
        board = f"{self.session.board_id} " if self.sequence is not None and self.session else ""
        self.status_var.set(f"Status: {board}{'CAPTURING' if c.phase in (PHASE_COUNTDOWN, PHASE_CAPTURING) else c.phase}")
        big, sub, colour = {
            PHASE_COUNTDOWN: (str(c.countdown_value(now)), f"{board}NEXT CAPTURE IN" if c.attempts or c.retakes or
                              (c.kept_count or 0) else f"{board}PCB OK - capturing in", "black"),
            PHASE_CAPTURING: ("CAPTURE", "Capturing...", "#c62828"),
            PHASE_WAITING_REVIEW: ("REVIEW", "Review the image on the right: KEEP or REJECT", "#1a237e"),
            PHASE_CAMERA_LOST: ("PAUSED", f"{(self.grabber.message if self.grabber else '') or 'No live picture.'} "
                                          "The countdown restarts automatically when the picture is live.", "#c62828"),
            PHASE_WAITING_PCB: (f"ADJUST {board}".strip() if board else "PLACE PCB",
                                ("Waiting for the live PCB check: " + (" ".join(self.live_check[0].messages)
                                 if self.live_check else "checking...")
                                 + " The 3 s countdown starts as soon as the PCB is OK."), "#e65100"),
            PHASE_FINISHED: ("FINISHING", "Saving the session...", "black"),
        }[c.phase]
        if now < self.flash_until:
            big, colour = "CAPTURED", "#2e7d32"
        self.big_var.set(big)
        self.sub_var.set(sub)
        self.big_label["fg"] = colour

    def _update_camera(self, now: float) -> None:
        g = self.grabber
        if g is None:
            return
        info = g.info()
        status = {"live": "LIVE", "connecting": "CONNECTING", "disconnected": "DISCONNECTED",
                  "frozen": "FROZEN", "blank": "BLANK PICTURE"}.get(g.status, g.status.upper())
        self.cam_status_var.set(f"Camera: {status}  ({g.camera.name})")
        if info["width"]:
            fps = " / ".join(x for x in (
                f"{info['fps_reported']:g} reported" if info["fps_reported"] else "",
                f"{info['fps_measured']:g} measured" if info["fps_measured"] else "") if x) or "-"
            text = f"Resolution: {info['width']} x {info['height']}   FPS: {fps}"
            if self.requested_size and info["width"] * info["height"] < self.requested_size[0] * self.requested_size[1]:
                text += (f"\nRequested {self.requested_size[0]} x {self.requested_size[1]}; the camera delivers only "
                         f"{info['width']} x {info['height']} (raise the resolution in the DroidCam client).")
            self.cam_info_var.set(text)
        if self.controller is None and self.sequence is None:
            self.big_var.set("READY" if g.status == STATUS_LIVE else f"CAMERA {status}")
            self.big_label["fg"] = "black" if g.status == STATUS_LIVE else "#c62828"
            start = "START AUTO CAPTURE" if self.sequence_mode else "START SESSION"
            self.sub_var.set((f"Choose GOOD / FAULT, the first board ID and press {start}." if not self.review_item else
                              "Review the waiting images on the right, or start a new session.")
                             if g.status == STATUS_LIVE else g.message or "Waiting for the camera...")
        frame = g.latest()
        live = g.status == STATUS_LIVE
        self._live_check(frame, live, now)
        if frame is not None and (frame.seq, live) != self.preview_seq:
            self.preview_seq = (frame.seq, live)
            self._photo = ImageTk.PhotoImage(self._preview_image(frame.image, stale=not live))
            self.preview.configure(image=self._photo, width=self._photo.width(), height=self._photo.height())

    # ------------------------------------------------------------------ live PCB check (guidance only)
    def _live_check(self, frame, live: bool, now: float) -> None:
        """Every check_every_s: PCB check of the latest frame in a background thread. It only guides the placement
        (board in the frame, watermark outside the board); every capture is checked again after it is saved."""
        if self.checker is None:
            return
        if self.check_future is not None and self.check_future.done():
            try:
                self.live_check = self.check_future.result()
                self.live_check_at = time.monotonic()
                self._show_live_check()
            except Exception as e:  # guidance only: never stops the application
                self.live_check = None
                self.check_var.set(f"Live PCB check error: {e}")
            self.check_future = None
        if frame is None or not live:
            self.live_check = None
            self.check_var.set("Live PCB check: no live picture")
            self.check_label["fg"] = CHECK_COLOURS[RECAPTURE]
            return
        if self.check_future is None and self.capture_future is None and now >= self.next_check_at:
            self.next_check_at = now + self.check_every_s
            self.check_future = self.check_executor.submit(self._check_preview_frame, frame.image)

    def _check_preview_frame(self, bgr) -> tuple[PcbCheck, float]:
        import cv2
        scale = min(1.0, PREVIEW_CHECK_WIDTH / bgr.shape[1])
        small = bgr if scale >= 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        check = self.checker.check(small)       # incl. the PCB image: focus / exposure as in the capture check
        check.canonical = None
        return check, scale

    def _show_live_check(self) -> None:
        check, _ = self.live_check
        if check.status == OK:
            text = "Live PCB check: OK. Complete PCB inside the frame, watermark outside the PCB. Ready to capture."
        elif check.problems == ["NO_PCB"]:            # an empty table is normal between boards: not an error
            self.check_var.set("Live PCB check: no PCB in view.")
            self.check_label["fg"] = "#1a237e"
            return
        else:
            text = f"Live PCB check: {check.status}. " + " ".join(check.messages)
        self.check_var.set(text)
        self.check_label["fg"] = CHECK_COLOURS.get(check.status, "black")

    def _preview_image(self, bgr, stale: bool = False) -> Image.Image:
        """Downscaled preview with the dashed inspection-area guide (the inspection ROI) and, when the live PCB check
        runs, the detected board outline and watermark box. Never used for saved images.
        A stale picture (stream not live) is darkened and marked, so it is never mistaken for a live view."""
        import cv2
        h, w = bgr.shape[:2]
        scale = min(PREVIEW_SIZE[0] / w, PREVIEW_SIZE[1] / h)
        img = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ph, pw = img.shape[:2]
        if self.live_check is not None and not stale:
            check, check_scale = self.live_check
            img = draw_check(img, check, scale / check_scale)
        else:
            fraction = self.checker.params.roi_fraction if self.checker else self.session_cfg["inspection_area_fraction"]
            guide = PcbCheck(status=OK, roi_px=inspection_roi(pw, ph, self.store.settings.board_mm, float(fraction)))
            img = draw_check(img, guide)
        if stale:
            img = (img * 0.35).astype(img.dtype)
            cv2.putText(img, "NO LIVE PICTURE", (pw // 2 - 170, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (60, 60, 255), 3, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------ review
    def _queue(self) -> list[tuple[Session, dict[str, Any]]]:
        """Pending images: the running session first, then earlier sessions."""
        items = self.store.pending_reviews()
        current = self.session.id if self.session else None
        return sorted(items, key=lambda it: (it[0].id != current, it[0].id, it[1]["sequence"]))

    def _update_queue_count(self) -> None:
        self.review_vars["Queue"].set(f"{len(self._queue())} image(s) waiting for review")

    def _auto_kept_item(self) -> tuple[Session, dict[str, Any]] | None:
        """The last automatically KEPT image while it can still be undone (REJECT LAST)."""
        if self.last_auto_kept is None:
            return None
        review = self.last_auto_kept[1]["review"]
        return self.last_auto_kept if review["status"] == "kept" and review.get("decided_by") == "capture_check" else None

    def _update_undo_button(self) -> None:
        self.undo_btn["state"] = "normal" if self._auto_kept_item() is not None else "disabled"

    def undo_last(self) -> None:
        """Operator override of the last automatic KEEP: the image goes to quarantine. If its board was just completed
        and is still in place, the board is captured again for the missing image."""
        item = self._auto_kept_item()
        if item is None:
            return
        session, record = item
        reason = self.reason_labels.get(self.reason_var.get())
        if reason is None:
            messagebox.showwarning("Reject last", "Choose a reject reason first (right panel, 'Reject reason').")
            return
        try:
            self.store.undo_auto_keep(session, record["image_id"], reason, self.review_note_var.get())
        except (ReviewError, ValueError, OSError) as e:
            messagebox.showerror("Reject last", str(e))
            return
        self.last_auto_kept = None
        board = record["board_id"]
        kept = self.store.kept_count(record["dataset_type"], board)
        text = f"{record['image_id']} moved to quarantine. {board} has {kept} kept images now."
        if self.sequence is not None and board == self._board_id():
            if self.sequence.reopen():
                text += " Leave the board in place: it is captured again."
        elif self.sequence is not None:
            text += " Capture it again later (START with that board ID)."
        if self.controller is not None and self.controller.kept_count is not None:
            self.count_text.set(f"{self.session.board_id}: kept {self.controller.kept_count} / {self.controller.requested}")
        self.message(text)
        self.show_next_review()
        self.review_feedback.set(text)

    def show_next_review(self) -> None:
        queue = self._queue()
        self.review_item = queue[0] if queue else None
        self.reason_var.set("")
        self.review_note_var.set("")
        shown = self.review_item or self._auto_kept_item()
        if shown is None:
            for k, var in self.review_vars.items():
                var.set("-")
            self.review_vars["Queue"].set("no images waiting for review")
            self.review_img.configure(image=self._blank_review)
            self._review_photo = self._review_pcb_photo = None
            self.review_pcb_img.configure(image="")
            self.keep_btn["state"] = self.reject_btn["state"] = "disabled"
            return
        session, r = shown
        q = r["quality"]
        stage = STAGE_LABELS.get(r["stage"], r["stage"])
        values = {
            "Image": r["image_id"], "Type": TYPE_LABELS[r["dataset_type"]], "Board": r["board_id"] or "(no board ID)",
            "Stage": stage, "Quality": q["status"].upper(), "Flags": ", ".join(q["flags"]) or "None",
            "Colour estimate": r["colour_estimate"] or "-",
            "Queue": f"{len(queue)} image(s) waiting for review" if self.review_item else
                     "none waiting; shown: the last image, KEPT automatically",
        }
        check = capture_check(r)
        blocked, advice = False, ""
        if "Capture check" in self.review_vars:
            if check is None:
                values["Capture check"] = "not run"
            else:
                codes = check["problems"] + check["review"]
                values["Capture check"] = check["status"] + (f" ({', '.join(codes)})" if codes else "")
                advice = " ".join(check["messages"])
                blocked = check["status"] == RECAPTURE and self.store.settings.keep_requires_capture_check
                reasons = self.cfg["review"].get("reason_for_check") or {}
                suggested = next((reasons[c] for c in check["problems"] if c in reasons), None)
                if suggested in self.store.settings.reject_reasons and self.review_item is not None:
                    self.reason_var.set(label_of(suggested))   # suggestion only: the operator still decides
        report = session.data.get("defect_report")
        if report and report["defect_known"] == "yes":
            values["Type"] += f"  (reported: {label_of(report['defect_type'])}" \
                              f"{', ' + report['component_reference'] if report['component_reference'] else ''})"
        for k, v in values.items():
            self.review_vars[k].set(v)
        try:
            with Image.open(self.store.root / r["original_file"]) as img:
                img = img.convert("RGB")
                img.thumbnail(self.review_size)
                self._review_photo = ImageTk.PhotoImage(img)
            self.review_img.configure(image=self._review_photo, width=self._review_photo.width(),
                                      height=self._review_photo.height())
        except OSError as e:
            self.review_img.configure(image=self._blank_review)
            self.review_feedback.set(f"Cannot show image: {e}")
        self._review_pcb_photo = None
        self.review_pcb_img.configure(image="")
        if r.get("pcb_file"):
            try:
                with Image.open(self.store.root / r["pcb_file"]) as img:
                    img = img.convert("RGB")
                    img.thumbnail(self.review_size)
                    self._review_pcb_photo = ImageTk.PhotoImage(img)
                self.review_pcb_img.configure(image=self._review_pcb_photo)
            except OSError as e:
                self.review_feedback.set(f"Cannot show the PCB image: {e}")
        if self.review_item is None:                   # a decided image: only REJECT LAST applies
            self.keep_btn["state"] = self.reject_btn["state"] = "disabled"
            return
        self.keep_btn["state"] = "disabled" if blocked else "normal"
        self.reject_btn["state"] = "normal"
        if blocked:
            self.review_feedback.set(f"RECAPTURE: REJECT this image and capture again. {advice}")
        elif advice:
            self.review_feedback.set(f"REVIEW: {advice} KEEP only if you confirmed it.")

    def decide(self, decision: str) -> None:
        if self.review_item is None:
            return
        session, record = self.review_item
        reason = self.reason_labels.get(self.reason_var.get())
        if decision == "reject" and reason is None:
            messagebox.showwarning("Reject", "Choose a reject reason first.")
            return
        try:
            self.store.review(session, record["image_id"], decision, reason, self.review_note_var.get())
        except (ReviewError, ValueError, OSError) as e:
            messagebox.showerror("Review", str(e))
            return
        self.review_feedback.set("Image accepted." if decision == "keep" else "Image quarantined.")
        if self.controller is not None and self.controller.kept_count is not None and session is self.session:
            self.count_text.set(f"{session.board_id}: kept {self.controller.kept_count} / {self.controller.requested}")
        self.show_next_review()

    # ------------------------------------------------------------------ misc
    def message(self, text: str, error: bool = False) -> None:
        self.msg_var.set(text)
        self.msg_label["foreground"] = "#b71c1c" if error else "#1a237e"

    def on_close(self) -> None:
        if self.controller is not None:
            if not messagebox.askyesno("Quit", "A session is running. Stop it (captured images are kept) and quit?"):
                return
            self.controller.stop()
            if self.capture_future is not None:
                try:
                    outcome = self.capture_future.result(timeout=10)
                    self.controller.capture_done(outcome, time.monotonic())
                except Exception:
                    pass
            self.store.finish(self.session, "stopped")
        self.executor.shutdown(wait=True)
        self.check_executor.shutdown(wait=True)
        if self.grabber is not None:
            self.grabber.stop()         # the reader thread releases the camera
            self.grabber = None
        self.root.destroy()


def run(store: AcquisitionStore, capture_cfg: Mapping[str, Any], camera_index: int | None = None) -> None:
    """camera_index overrides camera.index from the capture config for this run."""
    root = tk.Tk()
    AcquisitionApp(root, store, capture_cfg, camera_index)
    root.mainloop()
