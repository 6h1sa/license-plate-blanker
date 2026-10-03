"""ナンバープレート文字消去の Web UI。

uv run app.py            # http://127.0.0.1:7900 を開く

写真をドラッグ&ドロップすると、裏で検出から消去まで自動で進める。画面で四隅の修正・ナンバーの追加や除外をして、
1 枚ずつ、またはまとめて（ZIP）ダウンロードする。写真はディスクに書かずメモリにだけ置き、ページごとに分けて持つ。
ページを閉じたり再読み込みしたりすると、そのページの写真は消える（開くたびにまっさら）。

画面は web/ の React アプリで、ビルド結果（web/dist）を配信する。初回と画面を変えたあとはビルドが必要:
    npm --prefix web install && npm --prefix web run build
"""

import argparse
import io
import json
import os
import queue
import re
import select
import signal
import socket
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image

import plate

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2"}
NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>ナンバー消去</title>
<body style="font-family:sans-serif;padding:2em;line-height:1.8">
<h1>画面がまだビルドされていません</h1>
<p>次のコマンドを実行してから、このページを再読み込みしてください（Node.js が必要です）。</p>
<pre style="background:#eee;padding:1em">npm --prefix web install
npm --prefix web run build</pre></body>""".encode("utf-8")
DEFAULT_MARGIN = 6.0  # 縁として残す幅（プレート高さに対する %）
PREVIEW_QUALITY = 90
EVENT_KEEPALIVE = 15.0  # 変更が無いときに /api/events が接続を保つため送る間隔（秒）
CLOSE_CHECK_INTERVAL = 1.0  # /api/events がブラウザ側の切断を確かめる間隔（秒）
# 写真を自動で消すまでの時間（秒）。ページの接続が切れてから / 最後に操作してから。
# 切れてすぐ消さないのは、電波が一瞬途切れただけの場合に備えるため（再読み込みは新しいページ扱いでまっさらになる）
DISCONNECT_GRACE = 30
IDLE_TIMEOUT = 60 * 60
JANITOR_INTERVAL = 5
SESSION_RE = re.compile(r"[0-9a-f]{16,64}")


def _round_quad(q) -> list:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in plate.order_quad(np.float32(q))]


class Store:
    """読み込んだ写真と、そのナンバーの状態を持つ。

    写真はディスクに書かず、メモリにだけ置く（元のファイル・表示用の JPEG・消去結果の JPEG）。
    写真は開いたページ（セッション）ごとに分けて持ち、ほかのページからは見えない。
    そのページの接続（/api/events）が切れて grace 秒たつか、idle 秒操作されなかった写真は自動で消す。

    検出と消去は裏の 1 本のスレッドで順番に行う。編集するたびに版（version）を上げ、
    表示中の結果がどの版のものか（result_version）で、最新かどうかを画面に伝える。
    """

    def __init__(self, grace: float = DISCONNECT_GRACE, idle: float = IDLE_TIMEOUT):
        self.items: "OrderedDict[str, dict]" = OrderedDict()
        # 状態を変えたら rev を上げて通知する（画面には /api/events で知らせる）
        self.lock = threading.Condition(threading.RLock())
        self.rev = 0
        self.cache: "OrderedDict[str, tuple[np.ndarray, bytes | None]]" = OrderedDict()  # 展開した元画像（最近使った数枚）
        self.jobs: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.model_lock = threading.Lock()
        self.grace, self.idle = grace, idle
        self.viewers: dict[str, int] = {}  # セッション → つながっている /api/events の数
        self.left: dict[str, float] = {}  # セッション → 接続が無くなった時刻（つながっている間は無い）
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()

    def _changed(self) -> None:
        """状態を変えたことを、待っている /api/events に知らせる（self.lock を持った状態で呼ぶ）。"""
        self.rev += 1
        self.lock.notify_all()

    def wait_change(self, rev: int, timeout: float) -> int:
        """rev より新しい変更があるか timeout 秒たつまで待ち、今の rev を返す。"""
        with self.lock:
            self.lock.wait_for(lambda: self.rev != rev, timeout)
            return self.rev

    def viewer(self, session: str, connected: bool) -> None:
        """ページの接続（/api/events）のつながり・切断を数える。"""
        with self.lock:
            n = self.viewers.get(session, 0) + (1 if connected else -1)
            if n > 0:
                self.viewers[session] = n
                self.left.pop(session, None)
            else:
                self.viewers.pop(session, None)
                self.left[session] = time.monotonic()

    def get(self, iid: str, session: str) -> dict | None:
        """そのページの写真の状態（ほかのページの写真なら None）。使われたものとして、自動で消すまでの時間を延ばす。"""
        with self.lock:
            it = self.items.get(iid)
            if it is None or it["session"] != session:
                return None
            it["touched"] = time.monotonic()
            return it

    def list(self, session: str) -> list[dict]:
        with self.lock:
            return [it for it in self.items.values() if it["session"] == session]

    def load(self, iid: str) -> tuple[np.ndarray, bytes | None]:
        with self.lock:
            if iid in self.cache:
                self.cache.move_to_end(iid)
                return self.cache[iid]
            data = self.items[iid]["data"]
        img, exif = plate.load_image(io.BytesIO(data))
        with self.lock:
            if iid in self.items:  # 読んでいる間に消されていたら覚えない
                self.cache[iid] = (img, exif)
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)
        return img, exif

    def add(self, session: str, name: str, data: bytes) -> str:
        iid = uuid.uuid4().hex[:12]
        with self.lock:
            self.items[iid] = {"id": iid, "session": session, "name": name, "data": data, "status": "queued", "error": "",
                               "plates": [], "version": 0, "result_version": -1, "margin": DEFAULT_MARGIN,
                               "touched": time.monotonic()}
            if session not in self.viewers:
                self.left.setdefault(session, time.monotonic())  # 接続の無いまま追加された（猶予はここから数える）
            self._changed()
        self.jobs.put(("detect", iid))
        return iid

    def remove(self, iid: str) -> None:
        with self.lock:
            self.items.pop(iid, None)
            self.cache.pop(iid, None)
            self._changed()

    def update(self, iid: str, plates: list, margin: float | None) -> int:
        """画面での編集を反映し、消去のやり直しを依頼する。新しい版の番号を返す。"""
        with self.lock:
            it = self.items[iid]
            it["plates"] = [{**p, "quad": _round_quad(p["quad"])} for p in plates]
            if margin is not None:
                it["margin"] = float(margin)
            it["version"] += 1
            it["status"] = "queued"
            it["touched"] = time.monotonic()
            self._changed()
            version = it["version"]
        self.jobs.put(("erase", iid))
        return version

    # --- 自動で消す
    def expire(self, now: float | None = None) -> list[str]:
        """消す条件に当てはまる写真を消し、消した id を返す。"""
        now = time.monotonic() if now is None else now
        with self.lock:
            closed = {sid for sid, t in self.left.items() if now - t >= self.grace}
            gone = [i for i, it in self.items.items() if it["session"] in closed or now - it["touched"] >= self.idle]
            for i in gone:
                self.items.pop(i, None)
                self.cache.pop(i, None)
            for sid in closed:  # 写真が無くなったページは忘れる
                self.left.pop(sid, None)
            if gone:
                self._changed()
        return gone

    def _janitor(self) -> None:
        while True:
            time.sleep(JANITOR_INTERVAL)
            self.expire()

    # --- 裏の処理
    def _worker(self) -> None:
        while True:
            kind, iid = self.jobs.get()
            try:
                if kind == "detect":
                    self._detect(iid)
                    self._erase(iid)
                else:
                    self._erase(iid)
            except Exception as e:  # 1 枚の失敗で全体を止めない
                with self.lock:
                    if iid in self.items:
                        self.items[iid].update(status="error", error=f"{type(e).__name__}: {e}")
                        self._changed()

    def _detect(self, iid: str) -> None:
        with self.lock:
            if iid not in self.items:
                return
            self.items[iid]["status"] = "detecting"
            self._changed()
        img, _ = self.load(iid)
        H, W = img.shape[:2]
        view = plate.encode_image(img, ".jpg", quality=PREVIEW_QUALITY)
        with self.model_lock:
            cands = plate.find_plates(img)
        plates = [{"quad": _round_quad(c.quad), "enabled": bool(c.ok), "ok": bool(c.ok), "score": c.score,
                   "reason": c.reason, "source": "auto"} for c in cands]
        with self.lock:
            if iid in self.items:
                self.items[iid].update(width=W, height=H, view=view, plates=plates, version=1)
                self._changed()

    def _erase(self, iid: str) -> None:
        with self.lock:
            it = self.items.get(iid)
            if it is None or it["result_version"] == it["version"]:
                return  # 削除済み、または最新の結果がもうある
            version, margin = it["version"], it["margin"]
            quads = [np.float32(p["quad"]) for p in it["plates"] if p["enabled"]]
            it["status"] = "erasing"
            self._changed()
        img, _ = self.load(iid)
        out = plate.erase_all(img, quads, margin=margin / 100) if quads else img
        result = plate.encode_image(out, ".jpg", quality=PREVIEW_QUALITY)
        with self.lock:
            it = self.items.get(iid)
            if it is None or it["version"] != version:
                return  # 削除された、または処理中に編集された（新しい版は別に依頼済み）
            it.update(result_version=version, result=result, status="done")
            self._changed()

    def final_bytes(self, iid: str, fmt: str) -> tuple[bytes, str]:
        """ダウンロード用。元の解像度で消去し、EXIF を付けた中身とファイル名を返す（ファイルには書かない）。"""
        with self.lock:
            it = self.items[iid]
            quads = [np.float32(p["quad"]) for p in it["plates"] if p["enabled"]]
            margin, name = it["margin"], it["name"]
        img, exif = self.load(iid)
        out = plate.erase_all(img, quads, margin=margin / 100) if quads else img
        ext = plate.SAVE_FORMATS[fmt]
        return plate.encode_image(out, ext, exif), os.path.splitext(name)[0] + "_noplate" + ext

    def public(self, it: dict, full: bool) -> dict:
        d = {k: it.get(k) for k in ("id", "name", "status", "error", "width", "height", "version", "result_version", "margin")}
        d["n_enabled"] = sum(p["enabled"] for p in it["plates"])
        if full:
            d["plates"] = it["plates"]
        return d


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            headers = {"Cache-Control": "no-store", **(extra or {})}
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: str, ctype: str) -> None:
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)

        def _session(self, q: dict) -> str | None:
            """ページ（セッション）の番号。ページを開くたびに画面が作り、すべての要求に ?s= で付けてくる。"""
            sid = q.get("s", [""])[0]
            return sid if SESSION_RE.fullmatch(sid) else None

        def _parts(self) -> tuple[list[str], dict]:
            u = urlparse(self.path)
            return [unquote(p) for p in u.path.strip("/").split("/")], parse_qs(u.query)

        def do_GET(self):
            parts, q = self._parts()
            if parts == [""]:
                index = os.path.join(DIST, "index.html")
                if not os.path.exists(index):
                    return self._send(200, NOT_BUILT, "text/html; charset=utf-8")
                return self._file(index, "text/html; charset=utf-8")
            if parts == ["favicon.ico"]:
                return self._send(204, b"", "image/x-icon")
            if parts[0] == "assets" and len(parts) == 2:
                # ビルドで生成された JS・CSS など。名前に内容のハッシュが入るので、ずっとキャッシュしてよい
                path = os.path.join(DIST, "assets", os.path.basename(parts[1]))
                if not os.path.isfile(path):
                    return self._send(404, b"not found", "text/plain")
                with open(path, "rb") as f:
                    return self._send(200, f.read(), STATIC_TYPES.get(os.path.splitext(path)[1], "application/octet-stream"),
                                      {"Cache-Control": "public, max-age=31536000, immutable"})
            if parts[0] in ("api", "img"):
                sid = self._session(q)
                if sid is None:
                    return self._json({"error": "セッションがありません。ページを開き直してください"}, 400)
            if parts == ["api", "events"]:
                return self._events(sid)
            if parts == ["api", "images"]:
                with store.lock:
                    return self._json([store.public(it, False) for it in store.list(sid)])
            if len(parts) == 3 and parts[:2] == ["api", "image"]:
                it = store.get(parts[2], sid)
                if it is None:
                    return self._json({"error": "not found"}, 404)
                with store.lock:
                    return self._json(store.public(it, True))
            if len(parts) == 3 and parts[0] == "img":
                # /img/<id>/view（元画像、向きを反映済み）または /img/<id>/result（消去後）
                it = store.get(parts[1], sid)
                data = None if it is None else it.get("view" if parts[2] == "view" else "result")
                if data is None:
                    return self._send(404, b"not ready", "text/plain")
                return self._send(200, data, "image/jpeg")
            if len(parts) == 3 and parts[:2] == ["api", "download"]:
                return self._download(sid, parts[2], q.get("format", ["jpeg"])[0])
            return self._send(404, b"not found", "text/plain")

        def _events(self, sid: str) -> None:
            """状態が変わるたびに、変更の番号を Server-Sent Events で送る。画面はそれを合図に一覧を取り直す。

            つながっている間は届けっぱなしにする。何も起きなくても、接続を保つために定期的にコメント行を送る。
            書き込むまで切断に気づけないので、待つのを短く区切り、その合間にブラウザ側が接続を閉じたかを確かめる
            （ページを閉じたら、すぐにそのページの写真を消す猶予を数え始めるため）。
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            rev = -1  # 最初は必ず送る
            last_sent = 0.0
            store.viewer(sid, True)
            try:
                while not self._peer_closed():
                    new = store.wait_change(rev, CLOSE_CHECK_INTERVAL)
                    if new == rev and time.monotonic() - last_sent < EVENT_KEEPALIVE:
                        continue
                    self.wfile.write(f"data: {new}\n\n".encode() if new != rev else b": keepalive\n\n")
                    self.wfile.flush()
                    rev, last_sent = new, time.monotonic()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass  # 画面が閉じられた
            finally:
                store.viewer(sid, False)

        def _peer_closed(self) -> bool:
            """ブラウザ側が接続を閉じたか。SSE ではブラウザから何も送られてこないので、読めるようになったら閉じられたということ。"""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def _download(self, sid: str, target: str, fmt: str) -> None:
            if fmt not in plate.SAVE_FORMATS:
                return self._json({"error": "対応していない形式です"}, 400)
            with store.lock:
                ids = [it["id"] for it in store.list(sid) if it.get("width") and (target == "all" or it["id"] == target)]
            if not ids:
                return self._json({"error": "ダウンロードできる写真がありません"}, 404)
            try:
                files = [store.final_bytes(i, fmt) for i in ids]
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            if target != "all":
                data, name = files[0]
                ctype = "image/jpeg" if fmt == "jpeg" else "image/webp"
            else:
                buf, used = io.BytesIO(), set()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # JPEG/WebP は圧縮済みなのでそのまま格納
                    for data, name in files:
                        base, ext, n = *os.path.splitext(name), 1
                        while name in used:
                            n += 1
                            name = f"{base}_{n}{ext}"
                        used.add(name)
                        z.writestr(name, data)
                data, name, ctype = buf.getvalue(), time.strftime("noplate_%Y%m%d_%H%M%S.zip"), "application/zip"
            self._send(200, data, ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

        def do_POST(self):
            parts, q = self._parts()
            sid = self._session(q)
            if sid is None:
                return self._json({"error": "セッションがありません。ページを開き直してください"}, 400)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if parts == ["api", "upload"]:
                name = os.path.basename(unquote(self.headers.get("X-Filename", "image.jpg")))
                if not name.lower().endswith(plate.IMAGE_EXTS):
                    return self._json({"error": f"対応していない形式です: {name}"}, 400)
                return self._json({"id": store.add(sid, name, body)})
            if len(parts) != 4 or parts[:2] != ["api", "image"]:
                return self._send(404, b"not found", "text/plain")
            iid, action = parts[2], parts[3]
            it = store.get(iid, sid)
            if it is None or not it.get("width"):
                return self._json({"error": "まだ検出が終わっていません"}, 409)
            data = json.loads(body or b"{}")
            if action == "plates":
                return self._json({"version": store.update(iid, data["plates"], data.get("margin"))})
            if action == "refine":
                # 手動で囲んだ範囲から四隅を推定する。推定できなければ囲んだ四角形をそのまま使う
                xs, ys = sorted(data["box"][0::2]), sorted(data["box"][1::2])
                box = [xs[0], ys[0], xs[1], ys[1]]
                img, _ = store.load(iid)
                with store.model_lock:
                    fit = plate.refine_quad(img, box, strict=False)
                quad = fit.quad if fit.ok else plate.box_to_quad(box)
                return self._json({"quad": _round_quad(quad), "ok": fit.ok, "reason": fit.reason})
            return self._send(404, b"not found", "text/plain")

        def do_DELETE(self):
            parts, q = self._parts()
            sid = self._session(q)
            if len(parts) == 3 and parts[:2] == ["api", "image"] and sid and store.get(parts[2], sid):
                store.remove(parts[2])
                return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="待ち受けるアドレス（Docker では 0.0.0.0）")
    ap.add_argument("--port", type=int, default=7900)
    ap.add_argument("--grace", type=float, default=DISCONNECT_GRACE, help="ページの接続が切れてから写真を消すまでの秒数")
    ap.add_argument("--idle", type=float, default=IDLE_TIMEOUT / 60, help="操作が無い写真を消すまでの分数")
    args = ap.parse_args()
    # docker stop などの停止要求（SIGTERM）も Ctrl+C と同じく受けて終わる。
    # コンテナでは最初のプロセスになるので、受けないと無視されて 10 秒後に強制終了される
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    store = Store(args.grace, args.idle * 60)
    print(f"http://127.0.0.1:{args.port} を開いてください（Ctrl+C で終了）")
    print(f"写真はメモリにだけ置き、ページを閉じて {args.grace:g} 秒、または {args.idle:g} 分操作が無いと消します")
    try:
        ThreadingHTTPServer((args.host, args.port), make_handler(store)).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
