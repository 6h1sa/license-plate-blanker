"""画像の読み書きと、確認用の描画。"""

from __future__ import annotations

import os

import cv2
import numpy as np


def draw_quads(img: np.ndarray, quads: list[np.ndarray], labels: list[str] | None = None, colors=None) -> np.ndarray:
    vis = img.copy()
    lw = max(2, max(img.shape[:2]) // 1000)
    for i, q in enumerate(quads):
        color = colors[i] if colors else (255, 0, 0)
        cv2.polylines(vis, [np.round(q).astype(np.int32)], True, color, lw, cv2.LINE_AA)
        if labels:
            p = tuple(int(v) for v in q.min(axis=0))
            fs = lw * 0.6
            cv2.putText(vis, labels[i], (p[0], max(p[1] - 4 * lw, 10)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, lw, cv2.LINE_AA)
    return vis


# 読み込める画像の拡張子
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def load_image(path: str) -> tuple[np.ndarray, bytes | None]:
    """画像を RGB で読み込む。EXIF の回転を反映し、保存用に EXIF を返す。"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        exif = im.getexif()
        return np.array(im.convert("RGB")), (exif.tobytes() if exif else None)


# 保存形式 → 拡張子
SAVE_FORMATS = {"jpeg": ".jpg", "webp": ".webp"}
WEBP_MAX_SIDE = 16383  # WebP の仕様上の最大辺長


def save_image(path: str, img: np.ndarray, exif: bytes | None = None, quality: int = 95) -> None:
    """拡張子で形式を決めて保存する。JPEG と WebP は EXIF を引き継ぐ。"""
    from PIL import Image

    kw = {"exif": exif} if exif else {}
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        kw.update(quality=quality, subsampling=0)
    elif ext == ".webp":
        if max(img.shape[:2]) > WEBP_MAX_SIDE:
            raise ValueError(f"WebP で保存できるのは {WEBP_MAX_SIDE}px 以下の画像だけです（{img.shape[1]}x{img.shape[0]}）")
        kw.update(quality=quality, method=4)
    Image.fromarray(img).save(path, **kw)
