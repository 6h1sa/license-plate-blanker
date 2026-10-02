"""Web UI のサーバー（app.py）のテスト。検出器は差し替えるので、モデルや実写真は不要。"""

import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

import app
import plate
from synth import FRONT, scene


@pytest.fixture()
def server(tmp_path, monkeypatch):
    # 検出は合成画像のナンバーの位置をそのまま返す
    monkeypatch.setattr(plate, "find_plates", lambda img, conf=0.1: [plate.PlateCandidate(np.float32(FRONT), True, 0.9)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(app.Store(str(tmp_path))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(), r.headers


def wait_done(base: str, iid: str, version: int | None = None) -> dict:
    for _ in range(200):
        it = json.loads(call(f"{base}/api/image/{iid}")[0])
        if it["status"] == "error":
            raise AssertionError(it["error"])
        if it["status"] == "done" and (version is None or it["result_version"] == version):
            return it
        time.sleep(0.05)
    raise AssertionError("時間内に処理が終わらなかった")


def upload(base: str, name: str = "car.jpg") -> str:
    img, _ = scene(FRONT)
    exif = Image.Exif()
    exif[0x0110] = "TEST-CAM"
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=95, exif=exif.tobytes())
    return json.loads(call(f"{base}/api/upload", "POST", buf.getvalue(), {"X-Filename": name})[0])["id"]


def test_upload_detect_erase_and_download(server):
    iid = upload(server)
    it = wait_done(server, iid)
    assert it["n_enabled"] == 1 and it["result_version"] == it["version"]

    # 表示用の画像と消去後の画像が取れる
    view = Image.open(io.BytesIO(call(f"{server}/img/{iid}/view")[0]))
    result = Image.open(io.BytesIO(call(f"{server}/img/{iid}/result")[0]))
    assert view.size == result.size == (900, 600)

    # ダウンロードは元の解像度で、EXIF を引き継ぐ
    data, headers = call(f"{server}/api/download/{iid}?format=webp")
    im = Image.open(io.BytesIO(data))
    assert im.format == "WEBP" and im.size == (900, 600)
    assert im.getexif().get(0x0110) == "TEST-CAM"
    assert "car_noplate.webp" in headers["Content-Disposition"]


def test_edit_reruns_erase_and_zip_has_all(server):
    a, b = upload(server, "a.jpg"), upload(server, "b.jpg")
    wait_done(server, a)
    wait_done(server, b)

    # ナンバーを対象外にすると、版が上がって消去がやり直され、元画像と同じになる
    it = json.loads(call(f"{server}/api/image/{a}")[0])
    plates = [{**p, "enabled": False} for p in it["plates"]]
    v = json.loads(call(f"{server}/api/image/{a}/plates", "POST", json.dumps({"plates": plates}).encode())[0])["version"]
    wait_done(server, a, v)
    view = np.asarray(Image.open(io.BytesIO(call(f"{server}/img/{a}/view")[0])), np.float32)
    result = np.asarray(Image.open(io.BytesIO(call(f"{server}/img/{a}/result")[0])), np.float32)
    assert np.abs(view - result).mean() < 1.0

    names = zipfile.ZipFile(io.BytesIO(call(f"{server}/api/download/all?format=jpeg")[0])).namelist()
    assert sorted(names) == ["a_noplate.jpg", "b_noplate.jpg"]


def test_rejects_unsupported_file(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"{server}/api/upload", "POST", b"not an image", {"X-Filename": "notes.txt"})
    assert e.value.code == 400


def test_serves_built_ui_and_hints_when_not_built(server, tmp_path, monkeypatch):
    # 未ビルドならビルド方法を案内する
    monkeypatch.setattr(app, "DIST", str(tmp_path / "no-dist"))
    assert "npm --prefix web run build" in call(f"{server}/")[0].decode("utf-8")

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(app, "DIST", str(dist))
    assert call(f"{server}/")[0] == b"<div id=root></div>"
    body, headers = call(f"{server}/assets/index-abc.js")
    assert body == b"console.log(1)" and headers["Content-Type"].startswith("text/javascript")
    assert "immutable" in headers["Cache-Control"]
    # assets の外のファイルは読めない
    with pytest.raises(urllib.error.HTTPError):
        call(f"{server}/assets/..%2F..%2Fapp.py")
