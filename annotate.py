"""サンプル写真に正解の四隅を付ける注釈ツール。

uv run annotate.py            # http://127.0.0.1:7901 を開く
uv run annotate.py --src 写真のフォルダ

自動検出の結果を下書きとして表示し、人が確認・修正して確定する。
結果は .regression/annotations.json に保存する（他人の車が写るのでリポジトリには入れない）。
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np

import plate

ANNOTATIONS = os.path.join(".regression", "annotations.json")
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate.html")
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".tif": "image/tiff", ".tiff": "image/tiff"}

_lock = threading.Lock()  # 検出モデルと注釈ファイルを同時に触らないため


def load_annotations() -> dict:
    if os.path.exists(ANNOTATIONS):
        with open(ANNOTATIONS, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "images": {}}


def save_annotations(data: dict) -> None:
    os.makedirs(os.path.dirname(ANNOTATIONS) or ".", exist_ok=True)
    tmp = ANNOTATIONS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, ANNOTATIONS)


def _round_quad(q) -> list:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in plate.order_quad(np.float32(q))]


def draft(path: str) -> dict:
    """自動検出の結果から下書きを作る。処理対象は plates に、除外した候補は suggestions に入れる。"""
    img, _ = plate.load_image(path)
    H, W = img.shape[:2]
    plates, suggestions = [], []
    for c in plate.find_plates(img):
        q = _round_quad(c.quad)
        if c.ok:
            plates.append({"quad": q, "ignore": False, "kind": "normal", "source": "auto"})
        else:
            suggestions.append({"quad": q, "reason": c.reason, "score": c.score})
    return {"done": False, "width": W, "height": H, "plates": plates, "suggestions": suggestions, "note": ""}


def make_handler(src: str):
    names = sorted(n for n in os.listdir(src) if n.lower().endswith(plate.IMAGE_EXTS))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _name(self, prefix: str) -> str | None:
            name = unquote(self.path[len(prefix):])
            return name if name in names else None

        def do_GET(self):
            if self.path == "/":
                with open(HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/list":
                with _lock:
                    images = load_annotations()["images"]
                self._json([{"name": n, "done": images.get(n, {}).get("done", False)} for n in names])
            elif self.path.startswith("/api/ann/"):
                name = self._name("/api/ann/")
                if name is None:
                    return self._json({"error": "not found"}, 404)
                with _lock:
                    data = load_annotations()
                    if name not in data["images"]:
                        data["images"][name] = draft(os.path.join(src, name))
                        save_annotations(data)
                    self._json(data["images"][name])
            elif self.path.startswith("/img/"):
                name = self._name("/img/")
                if name is None:
                    return self._send(404, b"not found", "text/plain")
                with open(os.path.join(src, name), "rb") as f:
                    self._send(200, f.read(), CONTENT_TYPES.get(os.path.splitext(name)[1].lower(), "application/octet-stream"))
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/api/ann/"):
                return self._send(404, b"not found", "text/plain")
            name = self._name("/api/ann/")
            if name is None:
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            for p in body["plates"]:
                p["quad"] = _round_quad(p["quad"])
            with _lock:
                data = load_annotations()
                data["images"][name] = body
                save_annotations(data)
            self._json({"ok": True})

    return Handler, names


def main():
    global ANNOTATIONS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="sample_picture", help="写真のフォルダ（既定: sample_picture）")
    ap.add_argument("--port", type=int, default=7901)
    ap.add_argument("--out", default=ANNOTATIONS, help=f"注釈の保存先（既定: {ANNOTATIONS}）")
    args = ap.parse_args()
    ANNOTATIONS = args.out
    handler, names = make_handler(args.src)
    print(f"{len(names)} 枚。http://127.0.0.1:{args.port} を開いてください（Ctrl+C で終了）")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
