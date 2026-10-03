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
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(app.Store()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


SID = "0123456789abcdef0123456789abcdef"  # このテストのページ（セッション）の番号


def with_session(url: str, sid: str = SID) -> str:
    """API と画像の URL に、画面と同じくセッションの番号を付ける。"""
    if "/api/" not in url and "/img/" not in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}s={sid}"


def call(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None, sid: str = SID):
    req = urllib.request.Request(with_session(url, sid), data=data, method=method, headers=headers or {})
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


def upload(base: str, name: str = "car.jpg", sid: str = SID) -> str:
    img, _ = scene(FRONT)
    exif = Image.Exif()
    exif[0x0110] = "TEST-CAM"
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=95, exif=exif.tobytes())
    return json.loads(call(f"{base}/api/upload", "POST", buf.getvalue(), {"X-Filename": name}, sid=sid)[0])["id"]


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


def test_events_notify_changes(server):
    # 画面は定期的に問い合わせず、/api/events の通知を合図に一覧を取り直す
    with urllib.request.urlopen(with_session(f"{server}/api/events"), timeout=30) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")

        def next_rev() -> int:
            while True:
                line = r.readline().decode().strip()
                if line.startswith("data: "):
                    return int(line[6:])

        first = next_rev()  # つないだ直後に今の番号が届く
        upload(server)
        assert next_rev() > first  # 写真を追加すると次の通知が届く


def test_never_writes_photos_to_disk(server, monkeypatch):
    # 写真の追加から検出・消去・編集・ダウンロードまで、ファイルを書き込まない（メモリにだけ置く）
    import builtins

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"ファイルに書き込もうとした: {file} ({mode})")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    iid = upload(server)
    it = wait_done(server, iid)
    plates = [{**p, "enabled": False} for p in json.loads(call(f"{server}/api/image/{iid}")[0])["plates"]]
    v = json.loads(call(f"{server}/api/image/{iid}/plates", "POST", json.dumps({"plates": plates}).encode())[0])["version"]
    wait_done(server, iid, v)
    assert call(f"{server}/api/download/{iid}?format=jpeg")[0][:2] == b"\xff\xd8"  # JPEG の先頭
    assert it["status"] == "done"


def test_each_page_sees_only_its_own_photos(server):
    # ページ（セッション）ごとに写真が分かれ、ほかのページの写真は一覧にも出ず、取得も削除もできない
    other = "fedcba9876543210fedcba9876543210"
    mine = upload(server, "mine.jpg")
    theirs = upload(server, "theirs.jpg", sid=other)
    assert [it["id"] for it in json.loads(call(f"{server}/api/images")[0])] == [mine]
    assert [it["id"] for it in json.loads(call(f"{server}/api/images", sid=other)[0])] == [theirs]
    for url, method in ((f"{server}/api/image/{theirs}", "GET"), (f"{server}/img/{theirs}/view", "GET"),
                        (f"{server}/api/image/{theirs}", "DELETE")):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(url, method)
        assert e.value.code == 404  # ほかのページの写真は「無い」扱い
    assert len(json.loads(call(f"{server}/api/images", sid=other)[0])) == 1
    # セッションの番号が無い要求は受け付けない
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{server}/api/images", timeout=30)
    assert e.value.code == 400


def test_photos_expire_when_page_disconnects_or_idle():
    store = app.Store(grace=30, idle=3600)
    t0 = time.monotonic()
    a = store.add("page-a", "a.jpg", b"x")
    b = store.add("page-b", "b.jpg", b"x")
    store.viewer("page-a", True)
    store.viewer("page-b", True)
    store.items[a]["touched"] = store.items[b]["touched"] = t0

    # つながっている間は、操作が無くても 60 分までは消えない。60 分たつと消える
    assert store.expire(t0 + 3599) == []
    store.items[b]["touched"] = t0 + 3000
    assert store.expire(t0 + 3600) == [a]

    # ページの接続が切れても、電波の一瞬の途切れに備えて 30 秒は残り、その後に消える
    store.viewer("page-b", False)
    left = store.left["page-b"]
    assert store.expire(left + 29) == []
    assert store.expire(left + 30) == [b]
    assert store.items == {}

    # 30 秒以内につながり直せば消えない
    c = store.add("page-c", "c.jpg", b"x")
    store.viewer("page-c", True)
    store.viewer("page-c", False)
    store.viewer("page-c", True)
    assert store.expire(time.monotonic() + 60) == []
    assert c in store.items


def test_notices_page_close_quickly(tmp_path, monkeypatch):
    # ページを閉じたら（/api/events の接続が切れたら）、変化が無くてもすぐに気づき、写真を消す猶予を数え始める
    monkeypatch.setattr(plate, "find_plates", lambda img, conf=0.1: [])
    store = app.Store()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        r = urllib.request.urlopen(with_session(f"http://127.0.0.1:{httpd.server_address[1]}/api/events"), timeout=30)
        r.readline()
        assert store.viewers.get(SID) == 1
        r.close()
        for _ in range(50):
            if SID in store.left:
                break
            time.sleep(0.1)
        assert SID not in store.viewers and SID in store.left  # 15 秒ごとの接続維持を待たずに気づく
    finally:
        httpd.shutdown()
