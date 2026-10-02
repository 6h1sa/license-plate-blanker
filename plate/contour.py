"""輪郭・エッジへの四角形の当てはめ。"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import _line_intersect, order_quad


def _enclosing_quad(hull: np.ndarray) -> np.ndarray | None:
    """凸包を、辺を1本ずつ取り除いて（両隣の辺を延長して）囲む四角形に縮める。

    毎回、増える面積が最小の辺を取り除く。角の欠けや出っ張り（ボルトのキャップなど）に強い。
    """
    pts = [p.astype(np.float64) for p in hull.reshape(-1, 2)]
    while len(pts) > 4:
        n = len(pts)
        best, best_i, best_x = np.inf, -1, None
        for i in range(n):
            a, p, q, b = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            x = _line_intersect((*(p - a), *a), (*(q - b), *b))
            if x is None:
                continue
            # x が辺 p-q の外側（凸包の外）にあるときだけ有効
            if np.dot(x - p, p - a) <= 0 or np.dot(x - q, q - b) <= 0:
                continue
            area = abs(np.cross(p - x, q - x)) / 2
            if area < best:
                best, best_i, best_x = area, i, x
        if best_i < 0:
            return None
        i = best_i
        pts[i] = best_x
        del pts[(i + 1) % n]
    return order_quad(np.array(pts, np.float32))


def _refit_lines(quad: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """各辺の近くにある輪郭点に直線を当てはめ直し、交点を四隅とする。"""
    for _ in range(3):
        lines = []
        for i in range(4):
            p, q = quad[i], quad[(i + 1) % 4]
            d = q - p
            L = float(np.linalg.norm(d))
            if L < 1:
                return None
            u = d / L
            n = np.array([-u[1], u[0]])
            rel = pts - p
            t = rel @ u
            dist = np.abs(rel @ n)
            sel = (t > 0.1 * L) & (t < 0.9 * L) & (dist < max(2.0, 0.06 * L))
            if sel.sum() < 5:
                lines.append((u[0], u[1], p[0], p[1]))
                continue
            vx, vy, x0, y0 = cv2.fitLine(pts[sel], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((vx, vy, x0, y0))
        new = []
        for i in range(4):
            c = _line_intersect(lines[(i - 1) % 4], lines[i])
            if c is None:
                return None
            new.append(c)
        quad = order_quad(np.array(new))
    return quad


def _fit_quad_to_contour(cnt: np.ndarray, region: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """輪郭に四角形を当てはめる。

    (既定, 代替) を返す。既定は、凸包の頂点を4つに減らした四角形を初期値に、各辺を輪郭点に当てはめ直したもの。
    領域 region がその四角形から 3% 以上はみ出すときだけ、凸包を囲む四角形などから、はみ出しが最も少ない
    代替を選ぶ（無ければ None）。
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    starts = []
    lo, hi = 0.005, 0.2
    for _ in range(30):
        eps = (lo + hi) / 2
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            starts.append(order_quad(a))
            break
        if len(a) > 4:
            lo = eps
        else:
            hi = eps
    if not starts:
        starts.append(order_quad(cv2.boxPoints(cv2.minAreaRect(cnt))))
    enc = _enclosing_quad(hull)
    if enc is not None:
        starts.append(enc)

    # 既定は、輪郭点に辺を当てはめた四角形（凸包の頂点を減らした初期値から）
    cpts = cnt.reshape(-1, 2).astype(np.float32)
    default = _refit_lines(starts[0], cpts)
    if region is None:
        return default, None
    hull_region = np.zeros(region.shape, np.uint8)
    cv2.fillConvexPoly(hull_region, hull.reshape(-1, 2).astype(np.int32), 1)
    region_area = max(int(region.sum()), 1)

    def coverage(q):
        m = np.zeros(region.shape, np.uint8)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
        return np.logical_and(m, region).sum() / region_area, np.logical_and(m, hull_region == 0).sum() / max(int(m.sum()), 1)

    if default is None or coverage(default)[0] >= 0.97:
        return default, None
    # 領域が四角形からはみ出している（上段の文字で領域が欠け、辺が内側へ引き込まれた）場合は、
    # 凸包を囲む四角形や、凸包の点への当てはめも候補にし、はみ出しが最も少なく余分が小さいものを選ぶ
    hp = hull.reshape(-1, 2).astype(np.float32)
    hpts = np.concatenate([np.linspace(hp[i], hp[(i + 1) % len(hp)], max(2, int(np.linalg.norm(hp[(i + 1) % len(hp)] - hp[i]))), endpoint=False)
                           for i in range(len(hp))]).astype(np.float32)
    cands = [default] + [f(s0, p) for s0 in starts for f, p in ((_refit_lines, cpts), (_refit_lines, hpts))] + starts[1:]
    best, best_cov = None, coverage(default)[0]
    for q in cands:
        if q is None:
            continue
        cov, extra = coverage(q)
        if extra <= 0.06 and cov > best_cov + 0.01:
            best, best_cov = q, cov
    return default, best


def _refine_edges(gray: np.ndarray, quad: np.ndarray, search: float) -> np.ndarray:
    """各辺の法線方向に輝度勾配のピークを探し、直線を再フィットして四隅をサブピクセル精度で詰める。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    center = quad.mean(axis=0)
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        L = float(np.linalg.norm(d))
        u = d / L
        n = np.array([-u[1], u[0]], np.float32)
        if np.dot(center - (p + q) / 2, n) > 0:
            n = -n  # 外向き法線
        offs = np.linspace(-search, search, int(2 * search) * 2 + 1).astype(np.float32)
        ts = np.linspace(0.08, 0.92, 40)
        found = []
        for t in ts:
            base = p + d * t
            xy = base[None, :] + offs[:, None] * n[None, :]
            mx = xy[:, 0].reshape(1, -1).astype(np.float32)
            my = xy[:, 1].reshape(1, -1).astype(np.float32)
            sx = cv2.remap(gx, mx, my, cv2.INTER_LINEAR).ravel()
            sy = cv2.remap(gy, mx, my, cv2.INTER_LINEAR).ravel()
            g = np.abs(sx * n[0] + sy * n[1])  # 法線方向の勾配
            # 現在の辺に近いピークを優先
            g = g * np.exp(-((offs / search) ** 2))
            k = int(np.argmax(g))
            if g[k] > 1e-3:
                found.append(xy[k])
        if len(found) < 8:
            lines.append((u[0], u[1], p[0], p[1]))
            continue
        found = np.array(found, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(found, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        # 外れ値除去して再フィット
        nn = np.array([-vy, vx])
        res = np.abs((found - [x0, y0]) @ nn)
        good = res < max(1.5, np.median(res) * 3)
        if good.sum() >= 8:
            vx, vy, x0, y0 = cv2.fitLine(found[good], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((vx, vy, x0, y0))
    new = []
    for i in range(4):
        c = _line_intersect(lines[(i - 1) % 4], lines[i])
        if c is None:
            return quad
        new.append(c)
    new = order_quad(np.array(new))
    # 大きく動いた場合は信用しない
    if np.max(np.linalg.norm(new - quad, axis=1)) > search * 1.5:
        return quad
    return new
