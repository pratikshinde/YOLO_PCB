"""Local web app for the PCB inspection system (http://127.0.0.1:8000).

    python scripts/web.py                 dashboard on http://127.0.0.1:8000
    python scripts/web.py --port 8080

Pages:
    /            products + golden banks, latest inspections of every product (decision, reasons, evidence)
    /inspect     inspect an image: upload a file, or give the path of a stored image (AUTO or selected product)
    /inspection  one inspection: result, defects, evidence images, human review (CONFIRM DEFECT / GOOD / REJECT)
    /live        live camera in the browser: automatic capture + inspection (same pipeline as scripts/station.py)
    /validate    dataset validation of every product (scripts/pcb.py validate)

Same rules as the station: every capture / upload is stored first (data/captures/), results and reviews go to the
product's folder, the AI result is never changed by a review. Listens on 127.0.0.1 only. Python standard library
only (http.server); the camera is opened only while /live is running.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from pcb_inspection.config import load_config  # noqa: E402
from pcb_inspection.inspection.identification import AUTO  # noqa: E402
from pcb_inspection.station import REVIEW_DECISIONS, InspectionStation, find_result, record_review  # noqa: E402
from pcb_inspection.station.records import RESULTS_INDEX, latest_reviews, read_jsonl  # noqa: E402

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
TEXT_TYPES = {".json": "application/json", ".txt": "text/plain; charset=utf-8"}
MAX_UPLOAD = 60 * 1024 * 1024
THUMB_SLOTS = threading.BoundedSemaphore(2)    # at most 2 thumbnails decoded at once: never starve a live inspection


# --- application state -------------------------------------------------------------------------------

class LockedStation:
    """The station, with inspections serialised (web requests and the live camera share one station)."""

    def __init__(self, station: InspectionStation):
        self._station, self._lock = station, threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._station, name)

    def process(self, *args, **kw):
        with self._lock:
            return self._station.process(*args, **kw)

    def inspect_file(self, *args, **kw):
        with self._lock:
            return self._station.inspect_file(*args, **kw)


class LiveController:
    """Runs scripts/station.py's LiveStation (camera -> presence -> automatic trigger -> station) in a thread."""

    def __init__(self, app: App):
        self.app = app
        self.live = None
        self.last: dict[str, Any] | None = None
        self.message = ""
        self.lock = threading.Lock()
        self._jpeg_lock = threading.Lock()
        self._jpeg_frame = None
        self._jpeg: bytes | None = None

    @property
    def running(self) -> bool:
        return self.live is not None

    def start(self, selection: str, camera_index: int | None) -> None:
        with self.lock:
            if self.running:
                return
            from pcb_inspection.camera import LiveCamera
            from pcb_inspection.preprocessing.board import BoardSegmentationParams
            from pcb_inspection.preprocessing.presence import PresenceParams
            from pcb_inspection.station import AutoTrigger, TriggerSettings
            from station import LiveStation
            st = self.app.settings["station"]
            camera_cfg = load_config(self.app.root / st["camera_config"])
            cam = camera_cfg["camera"]
            index = cam["index"] if camera_index is None else camera_index
            camera = LiveCamera(index, f"camera {index}", cam.get("requested_width"), cam.get("requested_height"), 30,
                                backend=cam.get("backend", "dshow"),
                                fresh_frame_timeout_s=cam.get("fresh_frame_timeout_s", 2.0))
            opened = camera.open(wait_s=3)
            self.message = "" if opened else f"camera {index}: no picture yet ({camera.message}); retrying"
            q = camera_cfg["quality_check"].get("board_detection") or {}
            seg = BoardSegmentationParams(**{k: q[k] for k in ("saturation_min", "value_min", "hue_tolerance") if k in q})
            self.live = LiveStation(self.app.station, camera, selection, AutoTrigger(TriggerSettings.from_mapping(st)),
                                    PresenceParams(**st["presence"]), seg, float(st["presence_every_s"]),
                                    on_result=self._on_result)
            self.live.start_loop()

    def _on_result(self, result, message: str) -> None:
        if result is None:
            self.last = {"decision": "ERROR", "message": message}
        else:
            self.last = summary_of(result)

    def stop(self) -> None:
        with self.lock:
            live, self.live = self.live, None
            if live is not None:
                live.close()                     # stops the loop, finishes a running inspection, releases the camera

    def capture_now(self) -> None:
        if self.live is not None:
            self.live.capture_now()

    def status(self) -> dict[str, Any]:
        live = self.live
        if live is None:
            return {"running": False, "message": self.message, "last": self.last}
        camera_live = getattr(live.camera, "is_live", live.camera.read() is not None)
        message = "" if camera_live else getattr(live.camera, "message", "") or self.message
        return {"running": True, "camera_live": camera_live, "pcb": live.pcb,
                "state": "INSPECTING" if live.busy else live.trigger.state, "selection": live.selection,
                "session": live.session_id, "captures": len(live.results), "failures": len(live.failures),
                "message": message, "last": self.last}

    def jpeg(self) -> bytes | None:
        """The newest frame as a small JPEG, encoded once per camera frame however many browsers watch."""
        live = self.live
        frame = None if live is None else live.camera.read()
        if frame is None:
            return None
        with self._jpeg_lock:
            if frame is not self._jpeg_frame:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (800, round(h * 800 / w)), interpolation=cv2.INTER_AREA)
                ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self._jpeg_frame, self._jpeg = frame, (data.tobytes() if ok else None)
            return self._jpeg


class App:
    def __init__(self, root: Path, settings_file: Path | None = None, products_dir: Path | None = None):
        self.root = root
        self.settings_file = settings_file or root / "configs" / "inspection.yaml"
        self.settings = load_config(self.settings_file)
        self.products_dir = products_dir
        self.station: LockedStation | None = None
        self.error = ""
        self.live = LiveController(self)
        self.ready = threading.Event()
        self.thumbs: OrderedDict = OrderedDict()          # (path, mtime, width) -> JPEG bytes (most recent last)
        self.thumbs_lock = threading.Lock()

    def load(self) -> None:
        """Profiles and golden banks (takes a few seconds; pages show 'loading' meanwhile)."""
        try:
            self.station = LockedStation(InspectionStation(self.root, self.settings_file, self.products_dir))
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.ready.set()
        if self.station is not None:
            threading.Thread(target=self.station.warm_up, name="warm-up", daemon=True).start()

    def result_roots(self) -> list[Path]:
        roots = [p.storage_root for p in self.station.profiles]
        return roots + [self._abs(self.settings["storage"]["unresolved_product_root"])]

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    def rel(self, path: Path | str | None) -> str:
        if path is None:
            return ""
        path = Path(path)
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def results(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = []
        for root in self.result_roots():
            reviews = latest_reviews(root)
            for row in read_jsonl(root / "inspections" / RESULTS_INDEX):
                if "inspection_id" in row:
                    row["review"] = reviews.get(row["inspection_id"])
                    rows.append(row)
        rows.sort(key=lambda r: r.get("written_at", ""), reverse=True)
        return rows[:limit]


def summary_of(result) -> dict[str, Any]:
    o = result.outcome
    return {"inspection_id": o.inspection_id, "product": result.product_name, "product_id": o.product_id,
            "identification": result.identification.status if result.identification else None,
            "decision": o.overall_decision, "reasons": [f"{r.code}: {r.text}" for r in o.reasons],
            "defects": [{"type": d.defect_type, "region": d.region, "center_mm": d.location.get("center_mm"),
                         "similarity": d.evidence.get("similarity"), "limit": d.evidence.get("limit")}
                        for d in o.defects], "confidence": o.confidence, "seconds": o.seconds}


# --- HTML -------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#5d6676;--line:#dde1e7;--good:#2e9e44;--defect:#d32f2f;
--review:#c98a00;--error:#7b1fa2;--accent:#1f6feb}
@media (prefers-color-scheme: dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2c3340;--accent:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{display:flex;gap:18px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card)}
header b{font-size:15px}header a{color:var(--fg);text-decoration:none;padding:4px 8px;border-radius:6px}
header a:hover{background:var(--bg)}main{padding:16px;max-width:1300px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-weight:700;font-size:12px}
.GOOD{background:var(--good)}.DEFECT{background:var(--defect)}.REVIEW{background:var(--review)}.ERROR{background:var(--error)}
.big{font-size:34px;padding:6px 22px}.muted{color:var(--muted)}a{color:var(--accent)}
img.thumb{max-width:160px;border-radius:6px;border:1px solid var(--line)}img.ev{max-width:100%;border-radius:8px;border:1px solid var(--line)}
input,select,button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{cursor:pointer;background:var(--accent);color:#fff;border:none}button.secondary{background:var(--muted)}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}
.warn{border-left:4px solid var(--review)}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
"""

e = html.escape


def badge(decision: str | None, big: bool = False) -> str:
    d = decision or "-"
    return f'<span class="badge {e(d)}{" big" if big else ""}">{e(d)}</span>'


def page(title: str, body: str, refresh: int | None = None) -> bytes:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1">{meta}<title>{e(title)} - PCB inspection</title><style>{CSS}</style></head>
<body><header><b>PCB inspection</b><a href="/">Dashboard</a><a href="/inspect">Inspect image</a>
<a href="/live">Live camera</a><a href="/validate">Validate data</a>
<span class="muted" style="margin-left:auto">REFERENCE_BASELINE (experimental) &middot; local only</span></header>
<main>{body}</main></body></html>""".encode("utf-8")


def file_url(rel: str) -> str:
    return "/file?path=" + quote(rel)


def thumb_url(rel: str, width: int = 320) -> str:
    return f"/thumb?w={width}&path=" + quote(rel)


# --- request handler --------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "PCBInspection/1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        if not self.path.startswith(("/live/stream", "/api/live", "/thumb", "/file")):
            sys.stderr.write(f"[web] {self.address_string()} {fmt % args}\n")

    # responses
    def send(self, body: bytes, ctype: str = "text/html; charset=utf-8", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def json(self, data: Any, status: int = 200) -> None:
        self.send(json.dumps(data, default=str).encode(), "application/json", status)

    def not_found(self, what: str = "not found") -> None:
        self.send(page("Not found", f'<div class="card">{e(what)}</div>'), status=404)

    # routing
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/health":
                return self.json({"ready": self.app.ready.is_set(), "error": self.app.error})
            if url.path in ("/file", "/thumb"):
                return self.serve_file(q.get("path", ""), url.path == "/thumb", int(q.get("w", "320")))
            if not self.app.ready.is_set() or self.app.station is None:
                msg = self.app.error or "Loading product profiles and golden reference banks... (a few seconds)"
                return self.send(page("Loading", f'<div class="card warn">{e(msg)}</div>', refresh=2))
            routes = {"/": self.dashboard, "/inspect": self.inspect_form, "/inspection": self.inspection,
                      "/live": self.live_page, "/live/stream": self.live_stream, "/api/live": self.api_live,
                      "/api/results": self.api_results, "/validate": self.validate_page}
            handler = routes.get(url.path)
            return handler(q) if handler else self.not_found()
        except ConnectionError:              # the browser closed the page / stream: normal, nothing to report
            pass
        except Exception:  # noqa: BLE001
            self.fail()

    def fail(self) -> None:
        text = traceback.format_exc()
        traceback.print_exc()
        try:
            self.send(page("Error", f"<pre>{e(text)}</pre>"), status=500)
        except ConnectionError:
            pass

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if not self.app.ready.is_set() or self.app.station is None:
                return self.redirect("/")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD:
                return self.send(page("Too large", '<div class="card">Upload larger than 60 MB.</div>'), status=413)
            body = self.rfile.read(length)
            fields, files = self.parse_body(body)
            routes = {"/inspect": self.inspect_post, "/review": self.review_post, "/live/start": self.live_start,
                      "/live/stop": self.live_stop, "/live/capture": self.live_capture, "/validate": self.validate_post}
            handler = routes.get(url.path)
            return handler(fields, files) if handler else self.not_found()
        except ConnectionError:
            pass
        except Exception:  # noqa: BLE001
            self.fail()

    def parse_body(self, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            msg = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
            fields, files = {}, {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                data = part.get_payload(decode=True) or b""
                if part.get_filename():
                    files[name] = (Path(part.get_filename()).name, data)
                else:
                    fields[name] = data.decode("utf-8", "replace")
            return fields, files
        return {k: v[-1] for k, v in parse_qs(body.decode("utf-8", "replace")).items()}, {}

    # files (only under data/, images and result text)
    def serve_file(self, rel: str, thumb: bool, width: int) -> None:
        data_dir = (self.app.root / "data").resolve()
        path = (self.app.root / rel).resolve()
        if not rel or not path.is_relative_to(data_dir) or not path.is_file():
            return self.not_found("file not available")
        suffix = path.suffix.lower()
        if thumb and suffix in IMAGE_TYPES:
            width = max(64, min(width, 1600))
            key = (str(path), path.stat().st_mtime_ns, width)
            app = self.app
            with app.thumbs_lock:
                data = app.thumbs.get(key)
                if data is not None:
                    app.thumbs.move_to_end(key)
            if data is None:
                with THUMB_SLOTS:
                    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2 if width <= 480 else cv2.IMREAD_COLOR)
                    if img is None:
                        return self.not_found("unreadable image")
                    h, w = img.shape[:2]
                    if w > width:
                        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                    data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                with app.thumbs_lock:
                    app.thumbs[key] = data
                    while len(app.thumbs) > 400:
                        app.thumbs.popitem(last=False)
            return self.send(data, "image/jpeg")
        ctype = IMAGE_TYPES.get(suffix) or TEXT_TYPES.get(suffix)
        if ctype is None:
            return self.not_found("file type not served")
        return self.send(path.read_bytes(), ctype)

    # --- pages -------------------------------------------------------------------------------------
    def dashboard(self, q) -> None:
        st = self.app.station
        cards = []
        for p in st.profiles:
            bank = st.banks[p.product_id]
            boards = ", ".join(bank.boards) or "none"
            cards.append(f"""<div class="card"><div class="row"><b style="font-size:16px">{e(p.name or p.product_id)}</b>
<span class="muted">{e(p.product_id)} / {e(p.variant_id)}</span></div>
<div>Board: {e(str(p.board_mm or 'UNKNOWN'))} mm &middot; side {e(str(p.side or 'UNKNOWN'))} &middot; stages {e(', '.join(p.stages) or 'UNKNOWN')}</div>
<div>Golden bank: {badge('GOOD' if bank.status == 'READY' else 'REVIEW')} <b>{e(bank.status)}</b></div>
<div class="muted">{e(bank.reason)}</div><div>Boards in bank: {e(boards)}</div>
<div class="muted">Defect types enabled: {e(', '.join(p.enabled_defect_types) or 'none')} &middot; unknown profile values: {len(p.unknowns())}</div></div>""")
        problems = "".join(f'<div class="card warn">PROFILE PROBLEM: {e(x)}</div>' for x in st.problems)
        rows = []
        for r in self.app.results():
            folder = Path(r.get("result", "")).parent.as_posix()
            pic = ""
            for name in ("annotated.jpg", "overview.jpg"):
                if (self.app.root / folder / name).is_file():
                    pic = f'<img class="thumb" src="{thumb_url(folder + "/" + name, 200)}" alt="">'
                    break
            if not pic and r.get("file_path"):
                pic = f'<img class="thumb" src="{thumb_url(r["file_path"], 200)}" alt="">'
            review = r.get("review")
            rows.append(f"""<tr><td>{pic}</td><td><a href="/inspection?id={quote(r['inspection_id'])}">{e(r['inspection_id'])}</a>
<div class="muted">{e(r.get('written_at', '')[:19].replace('T', ' '))}</div></td><td>{e(r.get('product_id') or 'not determined')}</td>
<td>{badge(r.get('decision'))}{'<div class="muted">review: ' + e(review['human_decision']) + '</div>' if review else ''}</td>
<td>{e(', '.join(r.get('reasons') or []))}<div>{e(', '.join(r.get('defects') or []))}</div></td></tr>""")
        table = ("<table><tr><th>Evidence</th><th>Inspection</th><th>Product</th><th>Result</th><th>Reasons / defects</th></tr>"
                 + "".join(rows) + "</table>") if rows else '<p class="muted">No inspections yet. Use <a href="/inspect">Inspect image</a> or <a href="/live">Live camera</a>.</p>'
        self.send(page("Dashboard", f"""{problems}<h2>Products</h2><div class="grid">{''.join(cards)}</div>
<h2>Latest inspections</h2><div class="card">{table}</div>"""))

    def product_options(self, selected: str = AUTO) -> str:
        opts = [(AUTO, "AUTO (identify the product)")] + [(p.product_id, f"{p.name} ({p.product_id})")
                                                          for p in self.app.station.profiles]
        return "".join(f'<option value="{e(v)}"{" selected" if v == selected else ""}>{e(t)}</option>' for v, t in opts)

    def inspect_form(self, q, message: str = "") -> None:
        note = f'<div class="card warn">{e(message)}</div>' if message else ""
        self.send(page("Inspect image", f"""{note}<h2>Inspect an image</h2>
<div class="card"><form method="post" action="/inspect" enctype="multipart/form-data">
<p><b>Upload</b> an image file (it is stored in data/captures/ first, then inspected):<br><input type="file" name="image" accept="image/*"></p>
<p>or the <b>path of a stored image</b> (relative to the project, e.g. data/curated/test/...png; recorded in place, not copied):<br>
<input name="path" size="90" placeholder="data/curated/..."></p>
<div class="row"><label>Product <select name="product">{self.product_options()}</select></label>
<label>Physical board (optional) <input name="board_id" size="12" placeholder="GOOD_003"></label>
<button type="submit">Inspect</button></div></form>
<p class="muted">Inspection takes a few seconds. Engine: REFERENCE_BASELINE, experimental; confidence is not calibrated.</p></div>"""))

    def inspect_post(self, fields, files) -> None:
        product = fields.get("product") or AUTO
        board = (fields.get("board_id") or "").strip().upper() or None
        upload = files.get("image")
        if upload and upload[1]:
            tmp = Path(tempfile.mkdtemp(prefix="pcb_upload_"))
            try:
                path = tmp / upload[0]
                path.write_bytes(upload[1])
                result = self.app.station.inspect_file(path, product, board)   # copied into data/captures first
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        elif fields.get("path"):
            path = (self.app.root / fields["path"].strip()).resolve()
            if not path.is_file() or not path.is_relative_to(self.app.root.resolve()):
                return self.inspect_form({}, f"Not a file in this project: {fields['path']}")
            result = self.app.station.inspect_file(path, product, board)
        else:
            return self.inspect_form({}, "Choose a file to upload or give the path of a stored image.")
        self.redirect("/inspection?id=" + quote(result.outcome.inspection_id))

    def inspection(self, q, message: str = "") -> None:
        iid = q.get("id", "")
        path = find_result(self.app.result_roots(), iid) if iid else None
        if path is None:
            return self.not_found(f"inspection {iid} not found")
        r = json.loads(path.read_text(encoding="utf-8"))
        folder = self.app.rel(path.parent)
        reviews = [x for x in read_jsonl(path.parents[3] / "reviews" / "reviews.jsonl") if x.get("inspection_id") == iid]
        defects = "".join(f"""<tr><td>{e(d['defect_type'])}</td><td>{e(str(d['region']))}</td>
<td>{e(str(d['location'].get('center_mm')))} mm<div class="muted">box {e(str(d['location'].get('box_mm')))}</div></td>
<td>similarity {e(str(d['evidence'].get('similarity')))} vs limit {e(str(d['evidence'].get('limit')))}</td>
<td>{e(str(d['confidence'])) if d['confidence'] is not None else 'not calibrated'}</td><td>{e(str(d['severity'] or 'not defined'))}</td></tr>""" for d in r["defects"])
        uncertain = "".join(f"<li>{e(u['region'])}: similarity {u['similarity']} vs limit {u['limit']}</li>" for u in r.get("uncertain_regions") or [])
        not_checkable = "".join(f"<li>{e(u['region'])}: {e(u.get('reason', ''))}</li>" for u in (r.get("details") or {}).get("not_checkable") or [])
        evidence = "".join(f'<div><div class="muted">{e(name)}</div><a href="{file_url(folder + "/" + name)}"><img class="ev" src="{thumb_url(folder + "/" + name, 900)}" alt=""></a></div>'
                           for name in ("annotated.jpg", "evidence.jpg", "overview.jpg", "aligned.jpg")
                           if (path.parent / name).is_file())
        image = r.get("image") or {}
        original = image.get("file_path") or ""
        if original and not evidence:
            evidence = f'<div><div class="muted">original</div><img class="ev" src="{thumb_url(original, 900)}" alt=""></div>'
        ident = r.get("identification") or {}
        review_rows = "".join(f"<li><b>{e(x['human_decision'])}</b> by {e(x['reviewer'])} at {e(x['review_timestamp'])}"
                              f" (AI {e(x['ai_decision'])}){': ' + e(x['note']) if x.get('note') else ''}</li>" for x in reviews)
        buttons = "".join(f'<button name="decision" value="{d}" class="{"secondary" if d == "REJECT_RESULT" else ""}">{d.replace("_", " ")}</button>'
                          for d in REVIEW_DECISIONS)
        note = f'<div class="card warn">{e(message)}</div>' if message else ""
        self.send(page(iid, f"""{note}<h2>{e(iid)}</h2>
<div class="card"><div class="row">{badge(r['overall_decision'], True)}
<div><div><b>Product:</b> {e(r.get('product_id') or 'NOT DETERMINED')} {e(r.get('variant_id') or '')}
<span class="muted">({e(ident.get('status') or '-')}{', match ' + str(ident.get('score')) if ident.get('score') is not None else ''})</span></div>
<div><b>Confidence:</b> {e(str(r['confidence'])) if r['confidence'] is not None else 'not calibrated'} &middot;
<b>Alignment:</b> {e(r['alignment_status'])} &middot; <b>Quality:</b> {e(r['quality_status'])} &middot; <b>Engine:</b> {e(str(r.get('engine')))} ({e(str(r.get('engine_status')))})</div>
<div class="muted">Image {e(image.get('image_id', ''))} &middot; {e(original)} &middot; sha256 {e((image.get('sha256') or '')[:16])}... &middot; board {e(str(r.get('board_id') or 'unknown'))}</div></div></div>
<h3>Reasons</h3><ul>{''.join(f"<li><b>{e(x['code'])}</b>: {e(x['text'])}</li>" for x in r['reasons'])}</ul>
{'<h3>Defects</h3><table><tr><th>Type</th><th>Region</th><th>Location</th><th>Evidence</th><th>Confidence</th><th>Severity</th></tr>' + defects + '</table>' if defects else ''}
{'<h3>Uncertain regions</h3><ul>' + uncertain + '</ul>' if uncertain else ''}
{'<h3>Not checkable</h3><ul>' + not_checkable + '</ul>' if not_checkable else ''}
{'<h3>Limitations</h3><ul>' + ''.join(f'<li class="muted">{e(x)}</li>' for x in r.get('limitations') or []) + '</ul>' if r.get('limitations') else ''}
<p class="muted"><a href="{file_url(folder + '/result.json')}">result.json</a>{' &middot; <a href="' + file_url(folder + '/report.txt') + '">report.txt</a>' if (path.parent / 'report.txt').is_file() else ''}</p></div>
<div class="card"><h3>Human review</h3><p class="muted">Stored separately; the AI result above is never changed.</p>
<form method="post" action="/review" class="row"><input type="hidden" name="id" value="{e(iid)}">
<input name="reviewer" placeholder="Reviewer name" required><input name="note" placeholder="Note (optional)" size="40">{buttons}</form>
<ul>{review_rows or '<li class="muted">No review yet.</li>'}</ul></div>
<h3>Evidence</h3><div class="grid">{evidence or '<p class="muted">No evidence images (not inspected).</p>'}</div>"""))

    def review_post(self, fields, files) -> None:
        iid = fields.get("id", "")
        path = find_result(self.app.result_roots(), iid)
        if path is None:
            return self.not_found(f"inspection {iid} not found")
        try:
            record_review(path, fields.get("decision", ""), fields.get("reviewer", ""), fields.get("note", ""))
        except ValueError as ex:
            return self.inspection({"id": iid}, str(ex))
        self.redirect("/inspection?id=" + quote(iid))

    # --- live ---------------------------------------------------------------------------------------
    def live_page(self, q) -> None:
        status = self.app.live.status()
        cam_index = load_config(self.app.root / self.app.settings["station"]["camera_config"])["camera"]["index"]
        if status["running"]:
            controls = """<form method="post" action="/live/capture" style="display:inline"><button>Capture now</button></form>
<form method="post" action="/live/stop" style="display:inline"><button class="secondary">Stop camera</button></form>"""
        else:
            controls = f"""<form method="post" action="/live/start" class="row"><label>Product <select name="product">{self.product_options()}</select></label>
<label>Camera index <input name="camera" size="3" value="{cam_index}"></label><button>Start camera</button></form>"""
        self.send(page("Live camera", f"""<h2>Live inspection</h2><div class="card">{controls}
<p class="muted">Place a PCB under the camera: when it is in view and still, ONE capture is taken automatically, stored, and inspected.
Remove or replace the board for the next one.</p></div>
<div class="grid"><div class="card"><img id="stream" class="ev" src="{'/live/stream' if status['running'] else ''}" alt="camera"
{'style="display:none"' if not status['running'] else ''}><div id="cam" class="muted"></div></div>
<div class="card"><div id="product" style="font-size:18px;font-weight:700">PRODUCT: -</div>
<div style="margin:10px 0"><span id="result" class="badge big">-</span></div><div id="detail"></div>
<p><a id="link" href="#"></a></p><div id="state" class="muted"></div></div></div>
<script>
async function poll(){{try{{const s=await (await fetch('/api/live')).json();
const st=document.getElementById('state'), last=s.last;
st.textContent=s.running?`camera ${{s.camera_live?'live':'NO PICTURE'}} | PCB ${{s.pcb||'-'}} | ${{s.state}} | captures ${{s.captures}} | session ${{s.session}} ${{s.message||''}}`:('camera stopped '+(s.message||''));
const prod=document.getElementById('product'), res=document.getElementById('result'), det=document.getElementById('detail'), link=document.getElementById('link');
if(s.running && s.state==='INSPECTING' && s.selection==='AUTO') prod.textContent='PRODUCT: AUTO DETECTING...';
if(last){{ if(!(s.running && s.state==='INSPECTING')) prod.textContent='PRODUCT: '+(last.product||'NOT DETERMINED')+(last.identification?' ('+last.identification.toLowerCase()+')':'');
res.textContent=last.decision; res.className='badge big '+last.decision;
let t=''; if(last.decision==='DEFECT'){{for(const d of last.defects) t+=`<div><b>${{d.type.replace('_',' ')}}</b>: ${{d.region}} at x ${{d.center_mm[0]}}, y ${{d.center_mm[1]}} mm (similarity ${{d.similarity}} vs limit ${{d.limit}})</div>`; t+='<div class="muted">confidence: not calibrated</div>';}}
else {{for(const r of (last.reasons||[])) t+=`<div>${{r}}</div>`; if(last.message) t+=`<div>${{last.message}}</div>`;}}
det.innerHTML=t; if(last.inspection_id){{link.href='/inspection?id='+encodeURIComponent(last.inspection_id); link.textContent='Open '+last.inspection_id+' (evidence, review)';}} }}
}}catch(e){{}} setTimeout(poll,700);}} poll();
</script>"""))

    def live_stream(self, q) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        while self.app.live.running:
            data = self.app.live.jpeg()
            if data and data is not last:            # only new frames; ~6 per second is enough to place a board
                last = data
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode()
                                 + b"\r\n\r\n" + data + b"\r\n")
                self.wfile.flush()
            time.sleep(0.16)

    def api_live(self, q) -> None:
        self.json(self.app.live.status())

    def api_results(self, q) -> None:
        self.json(self.app.results(int(q.get("limit", "60"))))

    def live_start(self, fields, files) -> None:
        camera = fields.get("camera", "").strip()
        self.app.live.start(fields.get("product") or AUTO, int(camera) if camera.isdigit() else None)
        self.redirect("/live")

    def live_stop(self, fields, files) -> None:
        self.app.live.stop()
        self.redirect("/live")

    def live_capture(self, fields, files) -> None:
        self.app.live.capture_now()
        self.redirect("/live")

    # --- validation ---------------------------------------------------------------------------------
    def validate_page(self, q, text: str = "") -> None:
        self.send(page("Validate data", f"""<h2>Dataset validation</h2><div class="card">
<form method="post" action="/validate" class="row"><label><input type="checkbox" name="unlabelled" value="1"> include unlabelled
folders (live captures; slower the first time)</label><button>Validate every product</button></form>
<p class="muted">Every image of every product: resolved / unresolved / pending / synthetic / invalid, duplicates, orphans,
boards and golden-bank membership. Same as <code>python scripts/pcb.py validate</code>. Nothing is changed.</p></div>
{'<pre>' + e(text) + '</pre>' if text else ''}"""))

    def validate_post(self, fields, files) -> None:
        from pcb_inspection.dataset.product_discovery import format_validation, validate_all
        st = self.app.station
        storage = self.app.settings["storage"]
        v = validate_all(self.app.root, st.profiles, capture_root=self.app.root / storage["capture_store"],
                         derived=list(storage.get("derived_paths") or []),
                         include_unlabelled=fields.get("unlabelled") == "1",
                         banks={p.product_id: st.banks[p.product_id].summary() for p in st.profiles},
                         cache_file=self.app.root / "data" / "processed" / "checksums.json")
        v.problems = list(st.problems) + v.problems
        self.validate_page({}, format_validation(v, limit=30))


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8000, settings_file: Path | None = None,
                products_dir: Path | None = None, background_load: bool = True) -> ThreadingHTTPServer:
    app = App(root, settings_file, products_dir)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    if background_load:
        threading.Thread(target=app.load, name="load", daemon=True).start()
    else:
        app.load()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Local web app for the PCB inspection system.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this PC only)")
    args = parser.parse_args()
    server = make_server(ROOT, args.host, args.port)
    print(f"PCB inspection web app: http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.app.live.stop()  # type: ignore[attr-defined]
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
