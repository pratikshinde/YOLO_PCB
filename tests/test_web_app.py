"""Local web app (scripts/web.py) on a generated test project: upload -> stored -> inspected -> result page -> review."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import pytest

from test_component_inspection import board, scene
from test_product_architecture import build_project

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import web  # noqa: E402


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kw):
        return None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    project = build_project(tmp_path_factory.mktemp("web_project"))
    srv = web.make_server(project, "127.0.0.1", 0, background_load=False)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, project, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def post(url: str, fields: dict[str, str], file: tuple[str, bytes] | None = None) -> tuple[int, str]:
    opener = urllib.request.build_opener(NoRedirect)
    if file is None:
        data, ctype = urllib.parse.urlencode(fields).encode(), "application/x-www-form-urlencoded"
    else:
        boundary = "----pcbtestboundary"
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{file[0]}"\r\n'
                     "Content-Type: image/png\r\n\r\n".encode() + file[1] + b"\r\n")
        data, ctype = b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype}, method="POST")
    try:
        with opener.open(req, timeout=120) as r:
            return r.status, r.headers.get("Location", "")
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Location", "")


def test_upload_is_stored_inspected_and_reviewable(server):
    srv, project, base = server
    status, body = get(base + "/")
    assert status == 200 and "alpha" in body and "pump_controller" in body and "No inspections yet" in body
    ok, png = cv2.imencode(".png", scene(board(missing=("C3",)), angle=-2.0))
    status, location = post(base + "/inspect", {"product": "AUTO", "board_id": ""}, ("bench_photo.png", png.tobytes()))
    assert status == 303 and location.startswith("/inspection?id=")
    iid = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)["id"][0]
    status, body = get(base + location)
    assert status == 200 and "DEFECT" in body and "C3" in body and "not calibrated" in body
    record = json.loads(next((project / "data" / "captures").rglob("*_C001.json")).read_text(encoding="utf-8"))
    assert record["decision"] == "DEFECT" and record["file_path"].startswith("data/captures/")   # upload stored first
    assert (project / record["file_path"]).is_file()
    status, location = post(base + "/review", {"id": iid, "decision": "CONFIRM_DEFECT", "reviewer": "QA", "note": ""})
    assert status == 303
    assert "CONFIRM_DEFECT" in get(base + "/inspection?id=" + iid)[1]
    rows = json.loads(get(base + "/api/results")[1])
    assert rows[0]["inspection_id"] == iid and rows[0]["review"]["human_decision"] == "CONFIRM_DEFECT"


def test_stored_path_and_files_are_served_only_from_data(server):
    srv, project, base = server
    stored = next((project / "data" / "products" / "alpha" / "curated" / "good" / "GOOD_002").rglob("*.png"))
    status, location = post(base + "/inspect", {"product": "alpha", "path": stored.relative_to(project).as_posix(),
                                                "board_id": "GOOD_002"})
    assert status == 303
    assert "GOOD" in get(base + location)[1]
    assert get(base + "/thumb?w=100&path=" + urllib.parse.quote(stored.relative_to(project).as_posix()))[0] == 200
    assert get(base + "/file?path=configs/inspection.yaml")[0] == 404
    assert get(base + "/file?path=data/../configs/inspection.yaml")[0] == 404
    assert get(base + "/live")[0] == 200 and json.loads(get(base + "/api/live")[1])["running"] is False
