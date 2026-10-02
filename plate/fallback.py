"""色の領域で四隅が決まらないときの代替（文字の並び・エッジから外形を推定する）。"""

from __future__ import annotations

import cv2
import numpy as np

from .contour import _refine_edges
from .geometry import box_to_quad, order_quad


# 正面化したナンバーで文字が占める範囲（幅・高さに対する比。doc/License_plate_jp.md 2 章の目安）
TEXT_X = (0.04, 0.96)
TEXT_Y = (0.08, 0.93)


def _quad_from_text(small: np.ndarray, box) -> np.ndarray | None:
    """検出枠の中の文字の並びから、ナンバーの外形の四角形（small の座標）を推定する。

    明るい地に暗い文字を前提に、周囲より暗い細い画素を文字とし、文字の塊が占める回転矩形を求める。
    文字が板に占める範囲の比から外形まで広げ、各辺を近くの輝度エッジへ寄せる（傾き・遠近の歪みはここで吸収する）。
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(5, int(bh * 0.25) | 1)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    roi = np.zeros((sh, sw), np.uint8)
    roi[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 1
    vals = blackhat[roi > 0]
    if vals.size < 100:
        return None
    thr = max(12.0, float(np.percentile(vals, 85)) * 0.6)
    text = ((blackhat > thr) & (roi > 0)).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(text)
    keep = [j for j in range(1, n)
            if 0.10 * bh <= st[j, 3] <= 0.8 * bh and st[j, 2] <= 0.45 * bw and st[j, 4] >= 0.002 * bw * bh]
    if len(keep) < 3:
        return None
    pts = np.column_stack(np.nonzero(np.isin(lbl, keep)))[:, ::-1].astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:  # 長い方を幅にする
        rw, rh, ang = rh, rw, ang + 90
    if abs(((ang + 90) % 180) - 90) > 35 or not (1.2 < rw / max(rh, 1) < 4.0):
        return None
    pw = rw / (TEXT_X[1] - TEXT_X[0])
    ph = rh / (TEXT_Y[1] - TEXT_Y[0])
    # 文字の範囲の中心は板の中心から少しずれている
    off = np.array([((TEXT_X[0] + TEXT_X[1]) / 2 - 0.5) * pw, ((TEXT_Y[0] + TEXT_Y[1]) / 2 - 0.5) * ph])
    t = np.deg2rad(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.array([cx, cy]) - R @ off
    corners = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) @ R.T + c
    quad = order_quad(corners.astype(np.float32))
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    for _ in range(2):
        r = _refine_edges(g, quad, max(3.0, 0.12 * ph))
        if np.allclose(r, quad):
            break
        quad = r
    return quad.astype(np.float32)


def _snap_box_to_edges(img: np.ndarray, box) -> np.ndarray | None:
    """bbox の各辺を、法線方向の輝度勾配が最も強い位置へ寄せた四角形（画像座標）を返す。"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    m = 0.3
    cx1, cy1 = int(max(0, x1 - m * bw)), int(max(0, y1 - m * bh))
    cx2, cy2 = int(min(W, x2 + m * bw)), int(min(H, y2 + m * bh))
    gray = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, bh / 150))
    q = box_to_quad(box) - [cx1, cy1]
    search = max(4.0, 0.12 * bh)
    for _ in range(2):
        r = _refine_edges(gray, q.astype(np.float32), search)
        if np.allclose(r, q):
            break
        q = r
    return (q + [cx1, cy1]).astype(np.float32)
