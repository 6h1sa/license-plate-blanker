"""ナンバープレートの検出・四隅推定・文字消去のコア処理。

画像はすべて RGB の uint8 ndarray (H, W, 3) で扱う。
四隅 (quad) は (4, 2) float32 で、左上・右上・右下・左下の順。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# 日本のナンバープレートは普通・中型・大型いずれも縦横比 1:2
PLATE_ASPECT = 2.0


# ---------------------------------------------------------------------------
# 検出
# ---------------------------------------------------------------------------

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        # 学習データが多く、小さい・斜めのプレートにも強い YOLOv11 系 (AGPL-3.0)
        path = hf_hub_download("morsetechlab/yolov11-license-plate-detection", "license-plate-finetune-v1s.pt")
        _detector = YOLO(path)
    return _detector


def _grid(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    v = list(range(0, n - tile + 1, step))
    if v[-1] + tile < n:
        v.append(n - tile)
    return v


def detect_plates(img: np.ndarray, conf: float = 0.1, tile: int = 1600, imgsz: int = 800) -> list[tuple[list[int], float]]:
    """タイル分割 + 全体縮小の2系統で YOLO を回し、(bbox, score) のリストを返す。

    高解像度写真ではプレートが小さく、640px に縮小する通常推論では見落とすため
    50% 重なりのタイルで推論する。タイル境界に接する検出は分断の可能性が高いので捨てる。
    """
    model = get_detector()
    H, W = img.shape[:2]
    tiles = [(x, y, min(tile, W), min(tile, H)) for y in _grid(H, tile, tile // 2) for x in _grid(W, tile, tile // 2)]
    crops = [np.ascontiguousarray(img[y:y + h, x:x + w]) for x, y, w, h in tiles]

    boxes: list[np.ndarray] = []
    scores: list[float] = []
    edge = 4
    for (x, y, w, h), r in zip(tiles, model(crops, verbose=False, conf=conf, imgsz=imgsz)):
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if (b[0] < edge and x > 0) or (b[1] < edge and y > 0) or (b[2] > w - edge and x + w < W) or (b[3] > h - edge and y + h < H):
                continue
            boxes.append(b + [x, y, x, y])
            scores.append(float(c))

    # 大きく写ったプレート用に全体推論も行う
    for r in model(img, verbose=False, conf=conf, imgsz=1280):
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            boxes.append(b)
            scores.append(float(c))

    if not boxes:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores, conf, 0.3)
    return [([int(round(v)) for v in boxes[k]], scores[k]) for k in np.array(keep).flatten()]


# ---------------------------------------------------------------------------
# 四隅推定
# ---------------------------------------------------------------------------


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


def _kmeans2(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2クラスタに分け (centers, labels, counts) を返す。"""
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)  # 初期値の乱数を固定し、同じ画像では毎回同じ結果にする
    _, labels, centers = cv2.kmeans(samples.astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    return centers, labels, np.bincount(labels, minlength=2)


def _color_dist(lab: np.ndarray, color: np.ndarray, l_weight: float = 0.5) -> np.ndarray:
    d = lab - color.reshape(1, 1, 3)
    d[..., 0] *= l_weight
    return np.sqrt((d ** 2).sum(axis=-1))


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


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


@dataclass
class PlateFit:
    quad: np.ndarray  # 画像座標
    ok: bool
    reason: str = ""
    bg_color: tuple[int, int, int] | None = None
    from_edges: bool = False  # 色の領域ではなく、検出枠をエッジに寄せて決めた四隅か


def looks_like_plate(img: np.ndarray, quad: np.ndarray) -> bool:
    """正面化した領域に、日本のナンバー特有の「下段の大きな数字」があるか。

    ステッカーや看板などの誤検出を除くための簡易チェック。
    """
    rw, rh = 400, 200
    M = cv2.getPerspectiveTransform(order_quad(quad), np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]))
    lab = _to_lab(cv2.warpPerspective(img, M, (rw, rh)))
    centers, _, counts = _kmeans2(lab[16:-16, 16:-16].reshape(-1, 3)[::2])
    bg, fg = centers[int(np.argmax(counts))], centers[int(np.argmin(counts))]
    ch = (_color_dist(lab, bg) > 0.35 * _color_dist(fg[None, None], bg)[0, 0]).astype(np.uint8)
    ch[:10] = 0
    ch[-10:] = 0
    ch[:, :10] = 0
    ch[:, -10:] = 0
    if _has_big_digit(ch, rw, rh):
        return True
    # 映り込みなどでプレートの色味が片側だけずれると、2色の分け方が「文字/地」でなく「色味の違い」になる。
    # 明るさが周囲（文字を消した背景）からはっきり外れる画素を文字として、もう一度確かめる
    L = lab[..., 0]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dark = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k) - L  # 明るい地に暗い文字
    light = L - cv2.morphologyEx(L, cv2.MORPH_OPEN, k)  # 暗い地に明るい文字（緑・黒ナンバー）
    for diff in (dark, light):
        ch = (diff > max(8.0, 0.4 * float(np.percentile(diff, 99)))).astype(np.uint8)
        ch[:10] = 0
        ch[-10:] = 0
        ch[:, :10] = 0
        ch[:, -10:] = 0
        if _has_big_digit(ch, rw, rh):
            return True
    return False


def _has_big_digit(ch: np.ndarray, rw: int, rh: int) -> bool:
    """文字マスクに、下段の大きな数字（縦長の塊）が1つ以上あるか。"""
    _, _, stats, _ = cv2.connectedComponentsWithStats(ch)
    tall = [st for st in stats[1:] if st[3] > 0.3 * rh and st[1] > 0.3 * rh and st[2] < 0.3 * rw]
    return len(tall) >= 1


def plate_colors_plausible(bg: np.ndarray, inner: np.ndarray) -> bool:
    """地色 bg と四角形内の画素 inner (Lab) が日本のナンバーの配色としてあり得るか。

    白・黄（自家用）: 地色が明るい / 緑（事業用）: 緑地に白文字 / 黒（軽事業用）: 黒地に黄文字。
    黒いグリルやディフューザーなど、暗い地に明るい模様の誤検出を除くため。
    サンプルでは本物の白・黄ナンバーの地色は日陰でも L≥40、誤検出した黒い部品は L≤16 だった。
    """
    L = inner[:, 0]
    # 赤・ピンク・紫系の地色のナンバーは無い（白は照明で少し色づく程度）
    if bg[1] > 12 and bg[1] > bg[2]:
        return False
    if bg[0] >= 30:
        return True
    if bg[1] < -12 and (L > bg[0] + 20).mean() > 0.03:
        return True
    if ((L > bg[0] + 20) & (inner[:, 2] > 25)).mean() > 0.03:
        return True
    return False


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


def _rim_is_plate(lab: np.ndarray, quad: np.ndarray, bg: np.ndarray, fg: np.ndarray) -> bool:
    """四角形の4辺それぞれの内側の細い帯が、ほぼ地色か（辺がプレートの縁に沿っているか）。

    エッジに寄せただけの四角形は、車体の線などに乗って斜めにずれることがあるので、その確認に使う。
    """
    rw, rh = 200, 100
    M = cv2.getPerspectiveTransform(order_quad(quad).astype(np.float32), np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]))
    r = cv2.warpPerspective(lab, M, (rw, rh), flags=cv2.INTER_LINEAR)
    near = _color_dist(r, bg) < max(6.0, 0.5 * float(_color_dist(fg[None, None], bg)[0, 0]))
    strips = (near[4:10, 10:-10], near[-10:-4, 10:-10], near[10:-10, 4:10], near[10:-10, -10:-4])
    ratios = sorted(float(st.mean()) for st in strips)
    # 文字が縁に近い辺や、映り込みで色味がずれた辺が1つあっても許す
    return ratios[0] >= 0.4 and ratios[1] >= 0.6


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


def _text_fits_layout(small: np.ndarray, quad: np.ndarray) -> bool:
    """四角形を正面化したとき、文字が規格どおりの範囲に収まっているか（外形の推定がずれていないか）。

    下段の大きな数字の右端は板の右端近く（0.89〜0.995）、文字全体の左端は板の左端近く（0.12 以下）、
    上端・下端もそれぞれ板の上端・下端の近くにあるはず（範囲は注釈済みの実物 68 件の実測から決めた）。
    ずれていれば、外形が車体側へはみ出している。
    """
    rw, rh = 400, 200
    M = cv2.getPerspectiveTransform(order_quad(quad).astype(np.float32), np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]))
    g = cv2.cvtColor(cv2.warpPerspective(small, M, (rw, rh), flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    m = (bh > max(12.0, 0.6 * float(np.percentile(bh, 85)))).astype(np.uint8)
    m[:4] = 0
    m[-4:] = 0
    m[:, :4] = 0
    m[:, -4:] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    comps = [st[j] for j in range(1, n) if 0.08 * rh <= st[j, 3] <= 0.8 * rh and st[j, 2] <= 0.4 * rw and st[j, 4] >= 30]
    tall = [c for c in comps if c[3] > 0.3 * rh and c[1] > 0.3 * rh]
    if not comps or not tall:
        return False
    xmin = min(c[0] for c in comps) / rw
    ymin = min(c[1] for c in comps) / rh
    ymax = max(c[1] + c[3] for c in comps) / rh
    digit_right = max(c[0] + c[2] for c in tall) / rw
    return bool(0.89 <= digit_right <= 0.995 and xmin <= 0.12 and ymin <= 0.22 and ymax >= 0.85)


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


def refine_quad(img: np.ndarray, box, work_width: int = 360, strict: bool = True, debug: dict | None = None) -> PlateFit:
    """おおまかな bbox からプレートの正確な四隅を推定する。

    1. bbox 中央部の画素を2クラスタに分け、多い方をプレートの地色とする
       （うまくいかなければ少ない方を地色として再試行。プレート中央に物が付いている場合など）
    2. 地色に近い画素のうち中央と連結した領域 (エッジで切断) をプレート本体とする
    3. 文字の穴を埋めた輪郭を4辺にフィット
    4. 原寸で輝度勾配に沿って各辺をサブピクセル補正

    strict=False（手動指定時）は配色と文字配置のチェックを省く。
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 4.0), max(y2 - y1, 4.0)
    # bbox がプレートの一部しか囲っていないことがあるので広めに探索
    cx1, cy1 = int(max(0, x1 - 0.6 * bw)), int(max(0, y1 - 0.8 * bh))
    cx2, cy2 = int(min(W, x2 + 0.6 * bw)), int(min(H, y2 + 0.8 * bh))
    crop = img[cy1:cy2, cx1:cx2]
    s = work_width / bw
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    lab = _to_lab(small)
    sh, sw = small.shape[:2]
    bx1, by1, bx2, by2 = (x1 - cx1) * s, (y1 - cy1) * s, (x2 - cx1) * s, (y2 - cy1) * s
    fail = PlateFit(box_to_quad(box), False)

    # --- 地色推定
    mx, my = 0.2 * (bx2 - bx1), 0.2 * (by2 - by1)
    seed = np.zeros((sh, sw), np.uint8)
    seed[int(by1 + my):int(by2 - my), int(bx1 + mx):int(bx2 - mx)] = 1
    samples = lab[seed > 0]
    if len(samples) < 50:
        fail.reason = "bbox が小さすぎます"
        return fail
    centers, labels, counts = _kmeans2(samples)
    if float(np.linalg.norm(centers[0] - centers[1])) < 15:
        fail.reason = "文字と地色のコントラストが不足"
        return fail

    first_reason = None
    from_edges = False
    for bg_i in np.argsort(-counts):
        bg, fg = centers[bg_i], centers[1 - bg_i]
        quad, reason = _fit_plate_region(lab, seed, samples[labels == bg_i], bg, fg, strict, debug)
        if quad is not None:
            # 白い車体の白ナンバーなどでは、色の領域が車体へ広がり、検出器の枠よりずっと大きい四角形になる。
            # 検出器の枠は多少ずれても大きさはほぼ合っているので、大きすぎるものは採用しない
            qx0, qy0 = quad.min(axis=0)
            qx1, qy1 = quad.max(axis=0)
            if (qx1 - qx0) > 1.5 * (bx2 - bx1) or (qy1 - qy0) > 1.8 * (by2 - by1):
                quad, reason = None, "検出枠より大きすぎる（車体と同色）"
            else:
                break
        first_reason = first_reason or reason
    else:
        # 日陰で車体とほぼ同じ色のプレートや、隣の車の映り込みで色味がずれたプレートは、色の領域が車体へ漏れて決まらない。
        # 検出器の bbox を初期値に、各辺を近くの強いエッジへ寄せた四角形を、同じ妥当性チェックに掛ける
        bg_i = int(np.argmax(counts))
        bg, fg = centers[bg_i], centers[1 - bg_i]
        quad = None
        # まず文字の並びから外形を推定する（白い車体の白ナンバーなど、色では車体と区別できないもの向け）
        q_text = _quad_from_text(small, (bx1, by1, bx2, by2))
        if q_text is not None and not quad_shape_problem(q_text) and _text_fits_layout(small, q_text) and _rim_is_plate(lab, q_text, bg, fg):
            filled = np.zeros((sh, sw), np.uint8)
            cv2.fillConvexPoly(filled, np.round(q_text).astype(np.int32), 1)
            quad, _ = _validate_quad(lab, filled, q_text, bg, fg, strict)
        if quad is None:
            snapped = _snap_box_to_edges(img, box)
            q_small = None if snapped is None else ((snapped - [cx1, cy1]) * s).astype(np.float32)
            if q_small is not None and not quad_shape_problem(q_small) and _rim_is_plate(lab, q_small, bg, fg):
                filled = np.zeros((sh, sw), np.uint8)
                cv2.fillConvexPoly(filled, np.round(q_small).astype(np.int32), 1)
                quad, _ = _validate_quad(lab, filled, q_small, bg, fg, strict)
        if quad is None:
            fail.reason = first_reason
            return fail
        from_edges = True

    # --- 原寸で辺を詰める
    quad_full = quad / s + [cx1, cy1]
    gray = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, 0.5 / s))
    search = max(3.0, 4.0 / s)
    refined = _refine_edges(gray, (quad_full - [cx1, cy1]).astype(np.float32), search) + [cx1, cy1]
    if not quad_shape_problem(refined):
        quad_full = refined

    # 写真の端に接する（端で切れた）プレートは四隅を正しく決められないので、自動では処理しない（手動指定は可）
    over = max(2.0, 0.01 * float(np.linalg.norm(quad_full[1] - quad_full[0])))
    if strict and (quad_full.min() < over or (quad_full[:, 0] > W - 1 - over).any() or (quad_full[:, 1] > H - 1 - over).any()):
        return PlateFit(quad_full.astype(np.float32), False, "写真の端で切れている")
    if strict and not looks_like_plate(img, quad_full):
        return PlateFit(quad_full.astype(np.float32), False, "ナンバーの文字配置に見えない")

    bg_rgb = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).ravel()
    return PlateFit(quad_full.astype(np.float32), True, "", tuple(int(c) for c in np.clip(bg_rgb * 255, 0, 255)), from_edges)


def _apparent_aspect(quad: np.ndarray) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    return float((top + bottom) / max(left + right, 1e-6))


def _extend_quad(lab: np.ndarray, quad: np.ndarray, bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """影などで地色が変わり切り落とされた端を、外側へ伸ばして取り戻す。

    正面化した座標で各辺の外側を1行（列）ずつ見て、地色か文字色に近い画素が大半なら
    プレートの続きとみなす。明るさの差に寛容にするため L の重みを下げて比較する。
    伸ばした結果、縦横比が本来の 1:2 に近づく場合だけ採用する。
    """
    quad = order_quad(quad)
    rw, rh = 200, 100
    ey, ex = 0.5, 0.15  # 探索する外側の幅（縦: 高さ比、横: 幅比）
    oy, ox = int(rh * ey), int(rw * ex)
    dst = np.array([[ox, oy], [ox + rw, oy], [ox + rw, oy + rh], [ox, oy + rh]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    big = cv2.warpPerspective(lab, M, (rw + 2 * ox, rh + 2 * oy), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.ones(lab.shape[:2], np.float32), M, (rw + 2 * ox, rh + 2 * oy), borderValue=0) > 0.5
    thr = max(6.0, 0.4 * float(_color_dist(fg[None, None], bg, 0.25)[0, 0]))
    like = ((_color_dist(big, bg, 0.25) < thr) | (_color_dist(big, fg, 0.25) < thr)) & valid
    bglike = (_color_dist(big, bg, 0.25) < thr) & valid

    def walk(lines_like, lines_bg, start, step, limit):
        last, miss = start, 0
        for i in range(start, limit, step):
            if lines_like[i] >= 0.75 and lines_bg[i] >= 0.3:
                last, miss = i, 0
            else:
                miss += 1
                if miss >= 2:
                    break
        return last

    cx = slice(ox + int(rw * 0.1), ox + int(rw * 0.9))
    cy = slice(oy + int(rh * 0.1), oy + int(rh * 0.9))
    row_like, row_bg = like[:, cx].mean(1), bglike[:, cx].mean(1)
    col_like, col_bg = like[cy, :].mean(0), bglike[cy, :].mean(0)
    top = walk(row_like, row_bg, oy, -1, -1)
    bottom = walk(row_like, row_bg, oy + rh - 1, 1, rh + 2 * oy) + 1
    left = walk(col_like, col_bg, ox, -1, -1)
    right = walk(col_like, col_bg, ox + rw - 1, 1, rw + 2 * ox) + 1
    Minv = np.linalg.inv(M)

    def back(l, t, r, b):
        rect = np.array([[l, t], [r, t], [r, b], [l, b]], np.float32)
        return cv2.perspectiveTransform(rect[None], Minv)[0]

    def err(q):
        return abs(np.log(_apparent_aspect(q) / PLATE_ASPECT))

    if (top, bottom, left, right) != (oy, oy + rh, ox, ox + rw):
        cand = back(left, top, right, bottom)
        if err(cand) < err(quad) - 0.05:
            quad, top, bottom = cand, top, bottom
        else:
            top, bottom, left, right = oy, oy + rh, ox, ox + rw

    # 濃い影で上下の帯が地色と大きく違う場合: 縦横比が横長すぎるなら、上下に「外側が暗くなる」強いエッジを探し、
    # 1:2 に戻る位置まで伸ばす
    if _apparent_aspect(quad) > PLATE_ASPECT * 1.08:
        prof = np.median(big[:, cx, 0], axis=1)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        k = 3

        def edge_strength(r, inside_below):
            if r - k < 0 or r + k > len(prof):
                return 0.0
            a, b = prof[r:r + k].mean(), prof[r - k:r].mean()
            return (a - b) if inside_below else (b - a)

        def peaks(rows, inside_below):
            # エッジ強度が局所最大の行だけを候補にする（強いエッジの手前・奥で止まらないよう）
            st = {r: edge_strength(r, inside_below) for r in range(rows.start - 1, rows.stop + 1)}
            return [(r, st[r]) for r in rows if st[r] > 8 and st[r] >= st[r - 1] and st[r] >= st[r + 1]]

        tops = [(top, np.inf)] + peaks(range(k + 1, top - 2), True)
        bots = [(bottom, np.inf)] + peaks(range(bottom + 2, len(prof) - k - 1), False)
        best, best_score, base_err = None, np.inf, err(quad)
        for t, st in tops:
            for b, sb in bots:
                if (t, b) == (top, bottom):
                    continue
                cand = back(left, t, right, b)
                e = err(cand)
                if not (e < 0.12 and e < base_err - 0.06 and e < best_score):
                    continue
                # 足した帯は、暗くても地色と同じ色味（影の落ちたプレート）でなければならない
                strips = [big[t:top, cx]] if t < top else []
                strips += [big[bottom:b, cx]] if b > bottom else []
                ok = all(s.size and np.hypot(*(np.median(s.reshape(-1, 3)[:, 1:], axis=0) - bg[1:])) < 9
                         and np.median(s[..., 0]) > 0.35 * bg[0] for s in strips)
                if ok:
                    best, best_score = cand, e
        if best is not None:
            quad = best
    return quad


def _fit_plate_region(lab, seed, bg_samples, bg, fg, strict=True, debug=None) -> tuple[np.ndarray | None, str]:
    """地色 bg の連結領域を探して四隅をフィットし、妥当性を確認する。(quad, 失敗理由) を返す。"""
    # 地色クラスタのばらつきから許容距離を決める
    spread = np.percentile(_color_dist(bg_samples[None], bg)[0], 90)
    thr = float(np.clip(max(spread * 1.6, 0.35 * _color_dist(fg[None, None], bg)[0, 0]), 6, 40))

    dist = _color_dist(lab, bg)
    mask = (dist < thr).astype(np.uint8)

    # 隣接する同色領域（白いボディ・枠など）と繋がらないよう強いエッジで切る
    L8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    v = np.median(L8)
    edges = cv2.Canny(L8, int(max(20, 0.5 * v)), int(min(255, max(60, 1.2 * v))))
    mask[edges > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return None, "プレート領域が見つかりません"
    overlap = np.bincount(lbl[seed > 0].ravel(), minlength=n)
    overlap[0] = 0
    k = int(np.argmax(overlap))
    blob = (lbl == k).astype(np.uint8)
    return _fit_blob(lab, blob, bg, fg, strict, debug, mask)


def _fit_blob(lab, blob, bg, fg, strict, debug, mask) -> tuple[np.ndarray | None, str]:
    """プレート本体の領域 blob に四角形を当てはめ、妥当性を確認する。"""
    sh, sw = lab.shape[:2]
    # エッジで切った隙間と文字の穴を埋める
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, [cnt], -1, 1, -1)

    default, alt = _fit_quad_to_contour(cnt, filled)
    if debug is not None:
        debug.update(mask=mask, blob=filled, quad=default, bg=bg, fg=fg)
    quad, reason = _validate_quad(lab, filled, default, bg, fg, strict)
    if quad is None or alt is None:
        return quad, reason
    # 既定の四角形が妥当なときだけ代替を試す（既定で弾かれる誤検出を代替で救わないよう）
    better, _ = _validate_quad(lab, filled, alt, bg, fg, strict)
    return (better, "") if better is not None else (quad, "")


def _validate_quad(lab, filled, quad, bg, fg, strict) -> tuple[np.ndarray | None, str]:
    """当てはめた四角形がナンバーとして妥当か確認する。妥当なら端を伸ばした四角形を返す。"""
    sh, sw = lab.shape[:2]
    if quad is None:
        return None, "四角形フィット失敗"
    qarea = cv2.contourArea(quad)
    if qarea < 100:
        return None, "領域が小さすぎます"
    fill_ratio = filled.sum() / qarea
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    aspect = (top + bottom) / max(left + right, 1e-6)
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return None, "四角形が凸でない"
    shape_reason = quad_shape_problem(quad)
    if shape_reason:
        return None, shape_reason
    if fill_ratio < 0.8:
        return None, f"四角形への当てはまりが悪い ({fill_ratio:.2f})"
    # 当てはまりを確認してから、影で切り落とされた端を伸ばす
    quad = _extend_quad(lab, quad, bg, fg)
    if not (0.5 < aspect < 3.2):
        return None, f"縦横比が不自然 ({aspect:.2f})"
    poly = np.zeros((sh, sw), np.uint8)
    cv2.fillConvexPoly(poly, quad.astype(np.int32), 1)
    poly = cv2.erode(poly, np.ones((5, 5), np.uint8))
    inner = lab[poly > 0]
    if len(inner) < 50:
        return None, "領域が小さすぎます"
    # 四角形の内側だけで改めて2クラスタに分け、少数側（文字）の割合を見る
    c2, _, cnt2 = _kmeans2(inner[:: max(1, len(inner) // 20000)])
    char_ratio = float(cnt2.min() / cnt2.sum())
    if not (0.04 < char_ratio < 0.6):
        return None, f"文字領域の比率が不自然 ({char_ratio:.2f})"
    if strict and not plate_colors_plausible(c2[int(np.argmax(cnt2))], inner):
        return None, "ナンバーの配色ではない"
    return quad, ""


# ---------------------------------------------------------------------------
# 文字消去
# ---------------------------------------------------------------------------


def _normalized_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """マスク付きガウシアン（正規化畳み込み）。マスク外の画素を周囲から補間する。"""
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def _shadow_ratio(rect: np.ndarray, base: np.ndarray, rh: int) -> np.ndarray:
    """正面化したプレート rect (RGB float) に落ちた影を、無地の地色面 base に対する明るさの比 (H, W, 3) で返す。

    文字より太いカーネルでクロージングすると、細い文字は消え、帯状の影は残る。
    残った暗い領域のうち、小さいもの（文字が交わる部分など）と、プレートの辺の大部分に沿っていないもの
    （ぼやけて帯状につながった文字列、端の近くの数字など）は影とみなさない。
    影の中の漢字のように文字が密集した部分は円形カーネルでは消えないので、横長のカーネルでも
    クロージングして明るい方を取る（バンパーの影は横方向の帯になるため、横に長いカーネルでも残る）。
    """
    k = max(5, int(rh * 0.13) | 1)
    closed = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kh = max(9, int(rh * 0.5) | 1)
    closed_h = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, np.ones((1, kh), np.uint8))
    closed = np.maximum(closed, closed_h)
    closed = cv2.GaussianBlur(closed, (0, 0), max(1.0, rh * 0.01))
    ratio = np.clip(closed / np.maximum(base, 1.0), 0, 1)
    dark = (ratio.mean(-1) < 0.92).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(dark)
    H, W = dark.shape
    e = max(2, int(rh * 0.03))
    x, y, w, h, area = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]
    # 外から落ちる影はプレートの端から始まり、その辺の大部分にかかる。
    # ぼやけて帯状につながった文字列（内側にある）や、端の近くにある数字（辺の一部だけ）と区別する
    along_top_bottom = ((y <= e) | (y + h >= H - e)) & (w >= 0.6 * W)
    along_left_right = ((x <= e) | (x + w >= W - e)) & (h >= 0.85 * H)
    keep = (area >= 0.03 * dark.size) & (along_top_bottom | along_left_right)
    keep[0] = False
    shadow = cv2.GaussianBlur(keep[lbl].astype(np.float32), (0, 0), max(1.0, rh * 0.008))[..., None]
    return 1 - (1 - ratio) * shadow


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """各連結成分の内側の穴を埋める（金属キャップの明るい中心など）。"""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


# プレート取付ボルトの標準位置（プレート幅・高さに対する比）。ボルト穴は上段の文字の左右にある
BOLT_X = (0.18, 0.82)
BOLT_Y = 0.15


def _bolt_at(dev: np.ndarray, ex: float, ey: float, tol: tuple[float, float], r_range: tuple[float, float], min_fill: float):
    """(ex, ey) の近く（tol: 幅・高さに対する比の許容）にある、丸くて文字より小さい塊を探す。

    見つかれば (中心 x, 中心 y, 半径, マスク) を、無ければ None を返す（座標は dev の画素）。
    """
    rh, rw = dev.shape
    x0, x1 = max(0, int((ex - 2 * tol[0]) * rw)), min(rw, int((ex + 2 * tol[0]) * rw))
    y0, y1 = max(0, int((ey - 2 * tol[1]) * rh)), min(rh, int((ey + 2 * tol[1]) * rh))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    win = cv2.morphologyEx(dev[y0:y1, x0:x1].astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    win = _fill_holes(win)
    h, w = win.shape
    n, lbl, st, cen = cv2.connectedComponentsWithStats(win)
    best, best_d = None, 1.0
    for j in range(1, n):
        x, y, ww, hh, area = st[j]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue  # 窓の外へ続く塊は文字
        r = max(ww, hh) / 2
        if not (r_range[0] * rh <= r <= r_range[1] * rh):
            continue
        if not (0.55 <= ww / max(hh, 1) <= 1.8) or area / (np.pi * r * r) < min_fill:
            continue
        cx, cy = cen[j][0] + x0, cen[j][1] + y0
        d = ((cx / rw - ex) / tol[0]) ** 2 + ((cy / rh - ey) / tol[1]) ** 2
        if d < best_d:
            mask = np.zeros_like(dev, dtype=np.uint8)
            mask[y0:y1, x0:x1][lbl == j] = 1
            best, best_d = (cx, cy, r, mask), d
    return best


def _small_circle_at(rect: np.ndarray, ex: float, ey: float):
    """(ex, ey)（幅・高さに対する比）のすぐ近くにある小さな円（ボルトの頭）を円の輪郭で探す。

    文字に接した小さなボルトは、地色から外れた画素の塊としては文字とつながってしまうので、輪郭で探す。
    文字の丸い部分を拾わないよう、半径は小さく、位置の許容も狭くする。
    """
    rh, rw = rect.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(rect.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), max(0.7, rh * 0.004))
    x0, x1 = max(0, int((ex - 0.06) * rw)), min(rw, int((ex + 0.06) * rw))
    y0, y1 = max(0, int((ey - 0.10) * rh)), min(rh, int((ey + 0.10) * rh))
    cs = cv2.HoughCircles(g[y0:y1, x0:x1], cv2.HOUGH_GRADIENT, dp=1, minDist=max(4, int(0.03 * rh)), param1=80, param2=10,
                          minRadius=max(2, int(0.015 * rh)), maxRadius=max(3, int(0.06 * rh)))
    if cs is None:
        return None
    best = None
    for cx, cy, r in cs[0]:
        cx, cy = cx + x0, cy + y0
        d = ((cx / rw - ex) / 0.03) ** 2 + ((cy / rh - ey) / 0.05) ** 2
        if d <= 1 and (best is None or d < best[0]):
            best = (d, cx, cy, r)
    if best is None:
        return None
    _, cx, cy, r = best
    mask = np.zeros((rh, rw), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r * 1.15)), 1, -1)
    return float(cx), float(cy), float(r), mask


def _find_bolts(dev: np.ndarray, rect: np.ndarray | None = None, bg: np.ndarray | None = None) -> np.ndarray:
    """正面化したプレートの「地色から外れた画素」dev から、取付ボルト（左は封印のことがある）を探してマスクで返す。

    ボルトは左右対称の位置にあるので、左右そろって見つかったときだけ残す。片方しか見つからなければ、
    見つかった側を左右反転した位置の近くを緩い条件で探し直し、それでも無ければ両方とも塗る
    （片方だけ残すと不自然で、誤認した文字の一部であることも多いため）。
    """
    rh, rw = dev.shape
    strict = dict(tol=(0.06, 0.09), r_range=(0.022, 0.10), min_fill=0.4)
    left = _find_seal(rect) if rect is not None and (bg is None or may_have_seal(bg)) else None
    if left is None:
        left = _bolt_at(dev, BOLT_X[0], BOLT_Y, **strict)
    right = _bolt_at(dev, BOLT_X[1], BOLT_Y, **strict)

    if (left is None) != (right is None):
        cx, cy, r, _ = left or right
        # 封印は右のボルトより大きいので、大きさの条件は広めにする
        relaxed = dict(tol=(0.035, 0.05), r_range=(0.015, 0.12), min_fill=0.3)
        other = _bolt_at(dev, 1 - cx / rw, cy / rh, **relaxed)
        if other is None and rect is not None:
            other = _small_circle_at(rect, 1 - cx / rw, cy / rh)
        if left is None:
            left = other
        else:
            right = other
    found = np.zeros_like(dev, dtype=np.uint8)
    if left is None or right is None or abs(left[1] - right[1]) > 0.1 * rh:
        return found
    return found | left[3] | right[3]


def may_have_seal(bg: np.ndarray) -> bool:
    """地色 bg (Lab) のナンバーに封印が付きうるか。

    封印は登録自動車（白・緑ナンバー）の後部だけに付き、軽自動車（黄・黒ナンバー）には無い（doc/License_plate_jp.md 5 章）。
    前後の区別は画像からはできないので、白・緑なら探す。
    """
    yellow = bg[2] > 25
    dark = bg[0] < 35 and bg[1] > -12  # 黒地（緑地は a が負なので除く）
    return not (yellow or dark)


def _find_seal(rect: np.ndarray):
    """後部の白・緑ナンバーの左ボルトに付く封印（金属キャップ）を円として探す。

    見つかれば (中心 x, 中心 y, 半径, 円板のマスク) を、無ければ None を返す。

    封印は右のボルトより大きく、明るい金属で、上段の文字に接していることが多いので、
    色の塊ではなく円の輪郭で探す。位置は左ボルトの標準位置に限る（文字の丸い部分を拾わないよう）。
    """
    rh, rw = rect.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(rect.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), max(0.8, rh * 0.006))
    x0, x1, y1 = int(0.05 * rw), int(0.33 * rw), int(0.36 * rh)
    cs = cv2.HoughCircles(g[:y1, x0:x1], cv2.HOUGH_GRADIENT, dp=1, minDist=rh, param1=80, param2=14,
                          minRadius=max(2, int(0.045 * rh)), maxRadius=int(0.13 * rh))
    if cs is None:
        return None
    cx, cy, r = cs[0][0]
    cx += x0
    if not (0.11 <= cx / rw <= 0.23 and 0.05 <= cy / rh <= 0.21):
        return None
    mask = np.zeros((rh, rw), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r * 1.1)), 1, -1)
    return float(cx), float(cy), float(r), mask


def _keep_mask(dev: np.ndarray, band: int, bolts: np.ndarray, textlike: np.ndarray) -> np.ndarray:
    """元画像のまま残す部分（縁のエンボス、枠・クリップ、ボルト）のマスクを返す。

    - 縁の帯の中で辺に沿って長く続く線（エンボスの陰影・プレートの縁）
    - 帯の中にほぼ収まり、プレートの外周に接する塊（ナンバー枠のクリップなど）
    - ボルト
    文字は外周に接しないので、帯にはみ出していても残らない。
    縁の陰影の帯に文字が重なることがあるので、文字色の画素 textlike は縁・クリップとしては残さない。
    """
    rh, rw = dev.shape
    d = dev.astype(np.uint8)
    # 縁の線はプレートの辺のほぼ全長にわたる。文字の画（数字の縦線など）より十分長いものだけを線とみなす
    zone = band + 2
    lines = np.zeros_like(d)
    lh = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((1, max(5, int(rw * 0.35))), np.uint8))
    lv = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((max(5, int(rh * 0.6)), 1), np.uint8))
    # エンボスの陰影は細い線。太い帯（縁から落ちる影など）は影として扱い、ここでは残さない
    t = max(3, int(rh * 0.05))
    lh &= 1 - cv2.morphologyEx(lh, cv2.MORPH_OPEN, np.ones((t, 1), np.uint8))
    lv &= 1 - cv2.morphologyEx(lv, cv2.MORPH_OPEN, np.ones((1, t), np.uint8))
    lines[:zone] |= lh[:zone]
    lines[-zone:] |= lh[-zone:]
    lines[:, :zone] |= lv[:, :zone]
    lines[:, -zone:] |= lv[:, -zone:]

    inband = np.ones_like(d)
    inband[band:rh - band, band:rw - band] = 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats(d & (1 - lines))
    edge = 2
    x, y, w, h, area = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    touches = (x <= edge) | (y <= edge) | (x + w >= rw - edge) | (y + h >= rh - edge)
    in_band_area = np.bincount(lbl.ravel(), weights=inband.ravel(), minlength=n)
    hardware = touches & (in_band_area >= 0.7 * np.maximum(area, 1))
    hardware[0] = False
    return (((lines | hardware[lbl].astype(np.uint8)) & (1 - textlike.astype(np.uint8))) | bolts).astype(np.uint8)


def erase_plate(img: np.ndarray, quad: np.ndarray, margin: float = 0.06, seed: int = 0, debug: dict | None = None) -> np.ndarray:
    """四隅で指定されたプレートの文字を消し、無地の板にする。

    プレートを正面化し、文字を除いた地色画素から照明ムラを含む滑らかな地色面を推定、
    元画像相当の粒状ノイズを乗せて内側を置き換え、逆射影で合成する。
    取付ボルト・縁のエンボス・枠のクリップ・バンパーなどの影は残す。
    縁の帯 (margin) は元画像を残し、内側から続く文字だけを置き換える。
    """
    quad = order_quad(quad)
    out = img.copy()
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    # 正面化解像度: 元のプレート画素数と同程度（小さいプレートは拡大して処理）
    rw = int(np.clip(max(top, bottom, 2 * max(left, right)), 240, 1600))
    rh = int(round(rw / PLATE_ASPECT))
    dst = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    rect = cv2.warpPerspective(img, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab = _to_lab(rect)

    # 地色: 内側の画素を2クラスタに分けて多い方
    mm = int(rh * 0.08)
    centers, labels, counts = _kmeans2(lab[mm:-mm, mm:-mm].reshape(-1, 3)[::3])
    bg = centers[int(np.argmax(counts))]
    fg = centers[int(np.argmin(counts))]
    thr = float(np.clip(0.35 * _color_dist(fg[None, None], bg)[0, 0], 5, 25))

    # 地色面の推定（初回は地色との距離、以降は推定面との残差で文字マスクを更新）
    sigma = rh * 0.13
    bgmask = _color_dist(lab, bg, 0.35) < thr * 1.5
    # 縁の帯は地色面の推定に使わない（縁の陰影や、縁に接した文字の輪郭が混ざり、塗った部分に文字の跡が出るため）。
    # 帯の地色は内側から補間する
    eb = max(2, int(round(rh * margin)))
    inner_zone = np.zeros((rh, rw), bool)
    inner_zone[eb:rh - eb, eb:rw - eb] = True
    for _ in range(3):
        k = max(3, int(rh * 0.04) | 1)
        bgmask_e = (cv2.erode(bgmask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & inner_zone
        surface = _normalized_blur(lab, bgmask_e, sigma)
        d = lab - surface
        d[..., 0] *= 0.6
        resid = np.sqrt((d ** 2).sum(-1))
        bgmask = resid < thr
    # 文字だけを含む領域が大きく空いた場合に備え、粗いスケールでも補間
    coarse = _normalized_blur(lab, bgmask_e, rh * 0.3)
    den = cv2.GaussianBlur(bgmask_e.astype(np.float32), (0, 0), sigma)
    w = np.clip(den / 0.15, 0, 1)[..., None]
    surface = surface * w + coarse * (1 - w)

    # 粒状ノイズ: 地色画素の高周波成分の標準偏差に合わせる
    hp = lab - cv2.GaussianBlur(lab, (0, 0), 2.0)
    # 外れ値（文字の縁など）に引っ張られないよう MAD で推定
    noise_std = np.array([1.4826 * np.median(np.abs(hp[..., c][bgmask_e])) if bgmask_e.any() else 0.0 for c in range(3)])
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(lab.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    noise /= max(float(noise.std()), 1e-6)
    synth_lab = surface + noise * noise_std.astype(np.float32)
    synth = cv2.cvtColor(synth_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    # 影は消さずに再現する（文字と一緒に明るく塗ると板が大きく見える）
    base = cv2.cvtColor(surface.astype(np.float32), cv2.COLOR_LAB2RGB) * 255
    synth = np.clip(synth * 255 * _shadow_ratio(rect.astype(np.float32), base, rh), 0, 255)

    # 合成マスク: 内側はすべて置き換え、縁の帯は内側から続く文字だけ。残すもの（ボルト・縁・クリップ）は除く
    dev = ~bgmask
    band = max(2, int(round(rh * margin)))
    bolts = _find_bolts(dev, rect, bg)
    textlike = _color_dist(lab, fg, 0.6) < 0.4 * float(_color_dist(fg[None, None], bg, 0.6)[0, 0])
    keep = _keep_mask(dev, band, bolts, textlike)
    feather = max(1.0, rh * 0.012)
    interior = np.zeros((rh, rw), np.uint8)
    interior[band:rh - band, band:rw - band] = 1
    # 内側に掛かる「地色から外れた塊」= 文字（帯にはみ出した部分も含めて消す）
    text_src = cv2.dilate((dev & (keep == 0)).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lbl, _, _ = cv2.connectedComponentsWithStats(text_src)
    hit = np.zeros(n, bool)
    hit[np.unique(lbl[(interior > 0) & (text_src > 0)])] = True
    hit[0] = False
    text = cv2.dilate(hit[lbl].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(interior.astype(np.float32), (0, 0), feather)
    pad = max(1, int(round(rh * 0.015)))
    keep_soft = cv2.GaussianBlur(cv2.dilate(keep, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(np.float32), (0, 0), feather * 0.7)
    # 残す部分の周りは塗らない。ただし文字は、縁に接していても必ず塗る（残すのは縁そのものだけ）
    text_soft = cv2.GaussianBlur(text.astype(np.float32), (0, 0), feather * 0.7) * (1 - keep)
    alpha = np.maximum(alpha * (1 - keep_soft), text_soft)
    if debug is not None:
        debug.update(rect=rect, dev=dev, bolts=bolts, keep=keep, text=text, alpha=alpha, band=band)

    # 逆射影して合成（プレート周辺の矩形領域だけ処理）
    Minv = np.linalg.inv(M)
    H, W = img.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 2
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    Mback = T @ Minv
    size = (x1 - x0, y1 - y0)
    synth_b = cv2.warpPerspective(synth.astype(np.float32), Mback, size, flags=cv2.INTER_LINEAR)
    alpha_b = cv2.warpPerspective(alpha, Mback, size, flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(region * (1 - alpha_b) + synth_b * alpha_b, 0, 255).astype(np.uint8)
    return out


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


# ---------------------------------------------------------------------------
# 入出力・一括処理
# ---------------------------------------------------------------------------


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


@dataclass
class PlateCandidate:
    quad: np.ndarray
    ok: bool
    score: float | None  # 検出スコア（手動指定は None）
    reason: str = ""


EDGE_FIT_MIN_SCORE = 0.5


def find_plates(img: np.ndarray, conf: float = 0.1) -> list[PlateCandidate]:
    """自動検出 → 四隅推定。四隅推定に失敗したものも ok=False で返す。"""
    out = []
    for box, score in detect_plates(img, conf=conf):
        fit = refine_quad(img, box)
        if fit.ok and fit.from_edges and score < EDGE_FIT_MIN_SCORE:
            # エッジに寄せる方法は色の裏付けが弱いので、検出器が確信しているときだけ使う
            fit = PlateFit(fit.quad, False, f"色で四隅を決められず、検出スコアも低い ({score:.2f})")
        out.append(PlateCandidate(fit.quad, fit.ok, score, fit.reason))
    # 同じプレートに対する重複（四隅推定後に重なったもの）を除く
    out.sort(key=lambda c: (not c.ok, -(c.score or 0)))
    kept: list[PlateCandidate] = []
    for c in out:
        cc = c.quad.mean(axis=0)
        if any(k.ok and cv2.pointPolygonTest(k.quad.reshape(-1, 1, 2), (float(cc[0]), float(cc[1])), False) >= 0 for k in kept):
            continue
        kept.append(c)
    return kept


def erase_all(img: np.ndarray, quads: list[np.ndarray], margin: float = 0.06) -> np.ndarray:
    out = img
    for q in quads:
        out = erase_plate(out, q, margin=margin)
    return out
