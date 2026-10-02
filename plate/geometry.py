"""四角形 (quad) の幾何。quad は (4, 2) float32 で、左上・右上・右下・左下の順。"""

from __future__ import annotations

import cv2
import numpy as np

# 日本のナンバープレートは普通・中型・大型いずれも縦横比 1:2
PLATE_ASPECT = 2.0


def order_quad(pts: np.ndarray) -> np.ndarray:
    """4点を 左上・右上・右下・左下 に並べ替える。"""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # 左上から時計回り (画像座標系)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def box_to_quad(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def _line_intersect(l1, l2) -> np.ndarray:
    # l = (vx, vy, x0, y0)
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, [x2 - x1, y2 - y1])
    return np.array([x1 + vx1 * t[0], y1 + vy1 * t[0]], np.float32)


def quad_shape_problem(quad: np.ndarray) -> str:
    """長方形の板を撮影した形としてあり得ない四角形なら理由を返す（問題なければ空文字）。

    プレートの一部だけを拾った三角形に近い当てはめを除く。ナンバーは撮影距離に比べて小さいので、
    遠近で歪んでも向かい合う辺はほぼ平行になる。
    """
    angles = []
    for i in range(4):
        a, b = quad[i - 1] - quad[i], quad[(i + 1) % 4] - quad[i]
        cos = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    if not all(40 < a < 140 for a in angles):
        return f"四隅の角度が不自然 ({min(angles):.0f}°〜{max(angles):.0f}°)"
    for i in (0, 1):
        d1 = quad[i + 1] - quad[i]
        d2 = quad[i + 2] - quad[(i + 3) % 4]
        cos = abs(np.dot(d1, d2)) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-6)
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > 15:
            return "向かい合う辺が平行でない"
    return ""


def _apparent_aspect(quad: np.ndarray) -> float:
    top, bottom, left, right = side_lengths(quad)
    return float((top + bottom) / max(left + right, 1e-6))


def side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    """(上, 下, 左, 右) の辺の長さ。"""
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    return top, bottom, left, right


def rect_corners(rw: int, rh: int) -> np.ndarray:
    """正面化した (rw, rh) の矩形の四隅。"""
    return np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])


def warp_rect(img: np.ndarray, quad: np.ndarray, rw: int, rh: int, **kw) -> np.ndarray:
    """quad の領域を (rw, rh) の矩形に正面化する。kw は cv2.warpPerspective に渡す。"""
    M = cv2.getPerspectiveTransform(order_quad(quad).astype(np.float32), rect_corners(rw, rh))
    return cv2.warpPerspective(img, M, (rw, rh), **kw)
