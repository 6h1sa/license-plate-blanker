"""ナンバープレート文字消去の Web UI。

uv run app.py            # http://127.0.0.1:7900 を開く

写真をドラッグ&ドロップすると、裏で検出から消去まで自動で進める。画面で四隅の修正・ナンバーの追加や除外をして、
1 枚ずつ、またはまとめて（ZIP）ダウンロードする。

画面は web/ の React アプリで、ビルド結果（web/dist）を配信する。初回と画面を変えたあとはビルドが必要:
    npm --prefix web install && npm --prefix web run build
"""

import argparse
import io
import json
import os
import queue
import shutil
import tempfile
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


def _round_quad(q) -> list:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in plate.order_quad(np.float32(q))]


class Store:
    """読み込んだ写真と、そのナンバーの状態を持つ。画像は一時フォルダに置き、必要なときに読む。

    検出と消去は裏の 1 本のスレッドで順番に行う。編集するたびに版（version）を上げ、
    表示中の結果がどの版のものか（result_version）で、最新かどうかを画面に伝える。
    """

    def __init__(self, root: str):
        self.root = root
        self.items: "OrderedDict[str, dict]" = OrderedDict()
        self.lock = threading.RLock()
        self.cache: "OrderedDict[str, tuple[np.ndarray, bytes | None]]" = OrderedDict()  # 元画像（最近使った数枚）
        self.jobs: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.model_lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def dir(self, iid: str) -> str:
        return os.path.join(self.root, iid)

    def get(self, iid: str) -> dict | None:
        with self.lock:
            return self.items.get(iid)

    def load(self, iid: str) -> tuple[np.ndarray, bytes | None]:
        with self.lock:
            if iid in self.cache:
                self.cache.move_to_end(iid)
                return self.cache[iid]
            file = self.items[iid]["file"]
        img, exif = plate.load_image(os.path.join(self.dir(iid), file))
        with self.lock:
            self.cache[iid] = (img, exif)
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        return img, exif

    def add(self, name: str, data: bytes) -> str:
        iid = uuid.uuid4().hex[:12]
        os.makedirs(self.dir(iid))
        file = "original" + os.path.splitext(name)[1].lower()
        with open(os.path.join(self.dir(iid), file), "wb") as f:
            f.write(data)
        with self.lock:
            self.items[iid] = {"id": iid, "name": name, "file": file, "status": "queued", "error": "", "plates": [],
                               "version": 0, "result_version": -1, "margin": DEFAULT_MARGIN}
        self.jobs.put(("detect", iid))
        return iid

    def remove(self, iid: str) -> None:
        with self.lock:
            self.items.pop(iid, None)
            self.cache.pop(iid, None)
        shutil.rmtree(self.dir(iid), ignore_errors=True)

    def update(self, iid: str, plates: list, margin: float | None) -> int:
        """画面での編集を反映し、消去のやり直しを依頼する。新しい版の番号を返す。"""
        with self.lock:
            it = self.items[iid]
            it["plates"] = [{**p, "quad": _round_quad(p["quad"])} for p in plates]
            if margin is not None:
                it["margin"] = float(margin)
            it["version"] += 1
            it["status"] = "queued"
            version = it["version"]
        self.jobs.put(("erase", iid))
        return version

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

    def _detect(self, iid: str) -> None:
        with self.lock:
            if iid not in self.items:
                return
            self.items[iid]["status"] = "detecting"
        img, _ = self.load(iid)
        H, W = img.shape[:2]
        Image.fromarray(img).save(os.path.join(self.dir(iid), "view.jpg"), quality=PREVIEW_QUALITY)
        with self.model_lock:
            cands = plate.find_plates(img)
        plates = [{"quad": _round_quad(c.quad), "enabled": bool(c.ok), "ok": bool(c.ok), "score": c.score,
                   "reason": c.reason, "source": "auto"} for c in cands]
        with self.lock:
            if iid in self.items:
                self.items[iid].update(width=W, height=H, plates=plates, version=1)

    def _erase(self, iid: str) -> None:
        with self.lock:
            it = self.items.get(iid)
            if it is None or it["result_version"] == it["version"]:
                return  # 削除済み、または最新の結果がもうある
            version, margin = it["version"], it["margin"]
            quads = [np.float32(p["quad"]) for p in it["plates"] if p["enabled"]]
            it["status"] = "erasing"
        img, _ = self.load(iid)
        out = plate.erase_all(img, quads, margin=margin / 100) if quads else img
        path = os.path.join(self.dir(iid), f"result_{version}.jpg")
        Image.fromarray(out).save(path, quality=PREVIEW_QUALITY)
        with self.lock:
            it = self.items.get(iid)
            if it is None or it["version"] != version:
                os.remove(path)  # 削除された、または処理中に編集された（新しい版は別に依頼済み）
                return
            old = it.get("result_file")
            it.update(result_version=version, result_file=path, status="done")
        if old and os.path.exists(old):
            os.remove(old)

    def final_bytes(self, iid: str, fmt: str) -> tuple[bytes, str]:
        """ダウンロード用。元の解像度で消去し、EXIF を付けて保存した中身とファイル名を返す。"""
        with self.lock:
            it = self.items[iid]
            quads = [np.float32(p["quad"]) for p in it["plates"] if p["enabled"]]
            margin, name = it["margin"], it["name"]
        img, exif = self.load(iid)
        out = plate.erase_all(img, quads, margin=margin / 100) if quads else img
        name = os.path.splitext(name)[0] + "_noplate" + plate.SAVE_FORMATS[fmt]
        path = os.path.join(self.dir(iid), "download" + plate.SAVE_FORMATS[fmt])
        plate.save_image(path, out, exif)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data, name

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
            if parts == ["api", "images"]:
                with store.lock:
                    return self._json([store.public(it, False) for it in store.items.values()])
            if len(parts) == 3 and parts[:2] == ["api", "image"]:
                it = store.get(parts[2])
                if it is None:
                    return self._json({"error": "not found"}, 404)
                with store.lock:
                    return self._json(store.public(it, True))
            if len(parts) == 3 and parts[0] == "img":
                # /img/<id>/view（元画像、向きを反映済み）または /img/<id>/result（消去後）
                it = store.get(parts[1])
                path = None if it is None else (os.path.join(store.dir(it["id"]), "view.jpg") if parts[2] == "view" else it.get("result_file"))
                if not path or not os.path.exists(path):
                    return self._send(404, b"not ready", "text/plain")
                return self._file(path, "image/jpeg")
            if len(parts) == 3 and parts[:2] == ["api", "download"]:
                return self._download(parts[2], q.get("format", ["jpeg"])[0])
            return self._send(404, b"not found", "text/plain")

        def _download(self, target: str, fmt: str) -> None:
            if fmt not in plate.SAVE_FORMATS:
                return self._json({"error": "対応していない形式です"}, 400)
            with store.lock:
                ids = [i for i, it in store.items.items() if it.get("width") and (target == "all" or i == target)]
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
            parts, _ = self._parts()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if parts == ["api", "upload"]:
                name = os.path.basename(unquote(self.headers.get("X-Filename", "image.jpg")))
                if not name.lower().endswith(plate.IMAGE_EXTS):
                    return self._json({"error": f"対応していない形式です: {name}"}, 400)
                return self._json({"id": store.add(name, body)})
            if len(parts) != 4 or parts[:2] != ["api", "image"]:
                return self._send(404, b"not found", "text/plain")
            iid, action = parts[2], parts[3]
            it = store.get(iid)
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
            parts, _ = self._parts()
            if len(parts) == 3 and parts[:2] == ["api", "image"]:
                store.remove(parts[2])
                return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=7900)
    args = ap.parse_args()
    root = tempfile.mkdtemp(prefix="noplate_")
    print(f"http://127.0.0.1:{args.port} を開いてください（Ctrl+C で終了）")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Store(root))).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
