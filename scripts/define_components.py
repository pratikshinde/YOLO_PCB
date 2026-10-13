"""Draw, name and verify component boxes on the GOOD reference (component map for Experiment 2).

    python scripts/define_components.py

Without a BOM / centroid file the component positions must be defined once on the reference image.
The window shows the rectified GOOD reference with every box of experiments/experiment_002/components.yaml:
green = verified by a person, orange = draft (not verified).

  click a box      select it (enlarged view on the right)
  V                verify the selected box (name and position checked against the board)
  R                rename the selected box (use the designator printed on the silkscreen)
  Delete           remove the selected box
  N                select the next unverified box
  drag             draw a new box; it snaps to the part inside and asks for the designator
                   (new boxes are saved as source: operator, verified: true)
  Ctrl+S           save

Nothing is changed on disk until you save. Coordinates are mm in the reference's canonical board frame.
"""

from __future__ import annotations

import sys
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, simpledialog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from pcb_inspection.comparison.board_inspection import load_inspection_config, load_reference_and_map  # noqa: E402
from pcb_inspection.comparison.components import (  # noqa: E402
    REFDES_PATTERN,
    Component,
    box_to_px,
    part_mask,
    save_component_map,
    snap_box,
)

HELP = """green = verified, orange = draft (not verified)
click box  select (enlarged view above)
V          verify the selected box
R          rename (designator on the silkscreen)
Delete     remove the selected box
N          next unverified box
drag       draw a new box (snaps to the part)
Ctrl+S     save"""

HEADER = """# Component map for Experiment 2: named component boxes on the GOOD reference image.
# Edited with scripts/define_components.py. source: assistant_draft = drafted automatically, NOT verified;
# source: operator = drawn by a person. verified: true = a person checked name and position on the board.
# Coordinates: mm in the canonical board frame of the reference image (origin top-left, x right, y down).
"""


class Editor:
    def __init__(self, root: tk.Tk, config_path: Path):
        self.root = root
        self.cfg = load_inspection_config(ROOT, config_path)
        reference, self.cmap = load_reference_and_map(self.cfg, ROOT / "experiments" / self.cfg.experiment / "reference"
                                                       / self.cfg.reference_file.stem)
        self.canonical = reference.canonical
        self.ppm = self.cfg.pipeline.px_per_mm
        self.parts = part_mask(self.canonical, self.ppm, self.cfg.check)
        h, w = self.canonical.shape[:2]
        self.scale = min(1.0, (root.winfo_screenheight() - 160) / h, (root.winfo_screenwidth() - 420) / w)
        shown = cv2.resize(self.canonical, (round(w * self.scale), round(h * self.scale)), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        self.selected: str | None = None
        self.dirty = False
        self.drag_start: tuple[int, int] | None = None

        self.canvas = tk.Canvas(root, width=self.photo.width(), height=self.photo.height(), cursor="crosshair")
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        side = tk.Frame(root, padx=8)
        side.grid(row=0, column=1, sticky="n")
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (380, 300), "#222222"))
        self.zoom = tk.Label(side, bg="#222", image=self.blank)
        self.zoom.pack()
        self.info = tk.StringVar()
        tk.Label(side, textvariable=self.info, justify="left", font=("Segoe UI", 11), wraplength=380).pack(anchor="w", pady=6)
        tk.Label(side, justify="left", font=("Consolas", 9), text=HELP).pack(anchor="w")
        for key, action in (("<v>", self.verify), ("<r>", self.rename), ("<Delete>", self.delete), ("<n>", self.next_draft),
                            ("<Control-s>", self.save)):
            root.bind(key, lambda e, a=action: a())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    # drawing
    def view(self, box_px):
        return [round(v * self.scale) for v in box_px]

    def redraw(self) -> None:
        self.canvas.delete("box")
        for c in self.cmap.components:
            x0, y0, x1, y1 = self.view(box_to_px(c.box_mm, self.ppm))
            colour = "#33dd33" if c.verified else "#ff9900"
            width = 3 if c.refdes == self.selected else 1
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ffff" if c.refdes == self.selected else colour,
                                         width=width, tags="box")
            self.canvas.create_text(x0, y0 - 2, text=c.refdes, anchor="sw", fill=colour, font=("Segoe UI", 8, "bold"),
                                    tags="box")
        done = len(self.cmap.components) - len(self.cmap.unverified)
        self.root.title(f"Component map: {done} of {len(self.cmap.components)} verified{' (unsaved)' if self.dirty else ''}")
        self.show_selected()

    def show_selected(self) -> None:
        if self.selected is None:
            self.info.set("Click a box to select it, or drag to draw a new one.")
            self.zoom.configure(image=self.blank)
            return
        c = self.cmap.get(self.selected)
        x0, y0, x1, y1 = box_to_px(c.box_mm, self.ppm)
        m = round(3 * self.ppm)
        crop = self.canonical[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m].copy()
        cv2.rectangle(crop, (x0 - max(0, x0 - m), y0 - max(0, y0 - m)), (x1 - max(0, x0 - m), y1 - max(0, y0 - m)),
                      (255, 255, 0), 2)
        f = min(380 / crop.shape[1], 300 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        self.zoom_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        self.zoom.configure(image=self.zoom_photo, width=self.zoom_photo.width(), height=self.zoom_photo.height())
        self.info.set(f"{c.refdes}   box {[round(v, 1) for v in c.box_mm]} mm\nsource: {c.source}   "
                      f"verified: {'YES' if c.verified else 'NO'}\nCheck that the printed designator next to the part "
                      f"is {c.refdes} and the box covers the part, then press V.")

    # mouse
    def press(self, event) -> None:
        self.drag_start = (event.x, event.y)

    def motion(self, event) -> None:
        if self.drag_start:
            self.canvas.delete("drag")
            self.canvas.create_rectangle(*self.drag_start, event.x, event.y, outline="#00ffff", dash=(4, 2), tags="drag")

    def release(self, event) -> None:
        start, self.drag_start = self.drag_start, None
        self.canvas.delete("drag")
        if start is None:
            return
        x0, x1 = sorted((start[0], event.x))
        y0, y1 = sorted((start[1], event.y))
        if x1 - x0 < 5 and y1 - y0 < 5:          # a click: select the smallest box under the cursor
            px, py = event.x / self.scale / self.ppm, event.y / self.scale / self.ppm
            hits = [c for c in self.cmap.components if c.box_mm[0] <= px <= c.box_mm[2] and c.box_mm[1] <= py <= c.box_mm[3]]
            self.selected = min(hits, key=lambda c: (c.box_mm[2] - c.box_mm[0]) * (c.box_mm[3] - c.box_mm[1])).refdes \
                if hits else None
            self.redraw()
            return
        box = tuple(round(v / self.scale) for v in (x0, y0, x1, y1))
        box = snap_box(self.parts, box) or box
        name = self.ask_name("New component", "Designator printed next to the part (e.g. R37, LED4):")
        if not name:
            return
        existing = [c for c in self.cmap.components if c.refdes == name]
        if existing and not messagebox.askyesno("Replace", f"{name} already exists. Replace its box?"):
            return
        self.cmap.components = [c for c in self.cmap.components if c.refdes != name] + [
            Component(name, tuple(round(v / self.ppm, 2) for v in box), "operator", True)]
        self.selected, self.dirty = name, True
        self.redraw()

    def ask_name(self, title: str, prompt: str, initial: str = "") -> str | None:
        name = simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.root)
        if name is None:
            return None
        name = name.strip().upper()
        if not REFDES_PATTERN.fullmatch(name):
            messagebox.showerror(title, f"{name!r} is not a designator like R37, LED4, U2.")
            return None
        return name

    # keys
    def update_selected(self, **changes) -> None:
        if self.selected is None:
            return
        self.cmap.components = [replace(c, **changes) if c.refdes == self.selected else c for c in self.cmap.components]
        self.dirty = True

    def verify(self) -> None:
        self.update_selected(verified=True)
        self.redraw()

    def rename(self) -> None:
        if self.selected is None:
            return
        name = self.ask_name("Rename", "Designator printed next to the part:", self.selected)
        if name and name != self.selected:
            if any(c.refdes == name for c in self.cmap.components):
                messagebox.showerror("Rename", f"{name} already exists.")
                return
            self.update_selected(refdes=name, source="operator", verified=True)
            self.selected = name
        self.redraw()

    def delete(self) -> None:
        if self.selected and messagebox.askyesno("Delete", f"Remove {self.selected} from the component map?"):
            self.cmap.components = [c for c in self.cmap.components if c.refdes != self.selected]
            self.selected, self.dirty = None, True
            self.redraw()

    def next_draft(self) -> None:
        drafts = self.cmap.unverified
        if drafts:
            later = [d for d in drafts if self.selected is None or d > self.selected]
            self.selected = (later or drafts)[0]
        self.redraw()

    def save(self) -> None:
        save_component_map(self.cmap, self.cfg.component_map_file, HEADER)
        self.dirty = False
        self.redraw()

    def close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Save", "Save changes to the component map?")
            if answer is None:
                return
            if answer:
                self.save()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Editor(root, ROOT / "experiments" / "experiment_002" / "config.yaml")
    root.mainloop()


if __name__ == "__main__":
    main()
