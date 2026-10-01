"""ナンバープレートの検出・四隅推定・文字消去のコア処理。

画像はすべて RGB の uint8 ndarray (H, W, 3) で扱う。
四隅 (quad) は (4, 2) float32 で、左上・右上・右下・左下の順。
"""

from __future__ import annotations

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
    _, labels, centers = cv2.kmeans(samples.astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    return centers, labels, np.bincount(labels, minlength=2)


def _color_dist(lab: np.ndarray, color: np.ndarray, l_weight: float = 0.5) -> np.ndarray:
    d = lab - color.reshape(1, 1, 3)
    d[..., 0] *= l_weight
    return np.sqrt((d ** 2).sum(axis=-1))


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _fit_quad_to_contour(cnt: np.ndarray) -> np.ndarray | None:
    """輪郭を4辺に分け、各辺を直線フィットして交点を四隅とする。"""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = None
    lo, hi = 0.005, 0.2
    for _ in range(30):
        eps = (lo + hi) / 2
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a
            break
        if len(a) > 4:
            lo = eps
        else:
            hi = eps
    if approx is None:
        approx = cv2.boxPoints(cv2.minAreaRect(cnt))
    quad = order_quad(approx)

    pts = cnt.reshape(-1, 2).astype(np.float32)
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
    for bg_i in np.argsort(-counts):
        bg, fg = centers[bg_i], centers[1 - bg_i]
        quad, reason = _fit_plate_region(lab, seed, samples[labels == bg_i], bg, fg, strict, debug)
        if quad is not None:
            break
        first_reason = first_reason or reason
    else:
        fail.reason = first_reason
        return fail

    # --- 原寸で辺を詰める
    quad_full = quad / s + [cx1, cy1]
    gray = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, 0.5 / s))
    search = max(3.0, 4.0 / s)
    refined = _refine_edges(gray, (quad_full - [cx1, cy1]).astype(np.float32), search) + [cx1, cy1]
    if not quad_shape_problem(refined):
        quad_full = refined

    if strict and not looks_like_plate(img, quad_full):
        return PlateFit(quad_full.astype(np.float32), False, "ナンバーの文字配置に見えない")

    bg_rgb = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).ravel()
    return PlateFit(quad_full.astype(np.float32), True, "", tuple(int(c) for c in np.clip(bg_rgb * 255, 0, 255)))


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
    if _apparent_aspect(quad) > PLATE_ASPECT * 1.18:
        prof = np.median(big[:, cx, 0], axis=1)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        k = 3

        def edge_strength(r, inside_below):
            if r - k < 0 or r + k > len(prof):
                return 0.0
            a, b = prof[r:r + k].mean(), prof[r - k:r].mean()
            return (a - b) if inside_below else (b - a)

        tops = [(top, np.inf)] + [(r, edge_strength(r, True)) for r in range(k, top - 2)]
        bots = [(bottom, np.inf)] + [(r, edge_strength(r, False)) for r in range(bottom + 2, len(prof) - k)]
        tops = [t for t in tops if t[1] > 8]
        bots = [b for b in bots if b[1] > 8]
        best, best_score = None, err(quad)
        for t, st in tops:
            for b, sb in bots:
                if (t, b) == (top, bottom):
                    continue
                cand = back(left, t, right, b)
                e = err(cand)
                if e < 0.12 and e < best_score - 0.1:
                    best, best_score = cand, e
        if best is not None:
            quad = best
    return quad


def _fit_plate_region(lab, seed, bg_samples, bg, fg, strict=True, debug=None) -> tuple[np.ndarray | None, str]:
    """地色 bg の連結領域を探して四隅をフィットし、妥当性を確認する。(quad, 失敗理由) を返す。"""
    sh, sw = lab.shape[:2]
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
    # エッジで切った隙間と文字の穴を埋める
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, [cnt], -1, 1, -1)

    quad = _fit_quad_to_contour(cnt)
    if debug is not None:
        debug.update(mask=mask, blob=filled, quad=quad, bg=bg, fg=fg, thr=thr)
    if quad is None:
        return None, "四角形フィット失敗"

    # --- 妥当性チェック
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


def erase_plate(img: np.ndarray, quad: np.ndarray, margin: float = 0.035, seed: int = 0) -> np.ndarray:
    """四隅で指定されたプレートの文字・ボルト等を消し、無地の板にする。

    プレートを正面化し、文字を除いた地色画素から照明ムラを含む滑らかな地色面を推定、
    元画像相当の粒状ノイズを乗せて内側を置き換え、逆射影で合成する。
    縁の帯 (margin) は元画像を残し、そこにかかった文字だけを置き換える。
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
    for _ in range(3):
        k = max(3, int(rh * 0.04) | 1)
        bgmask_e = cv2.erode(bgmask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
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
    synth = np.clip(synth * 255, 0, 255)

    # 合成マスク: 内側はすべて置き換え、縁の帯は文字（地色から外れた画素）のみ
    m = int(round(rh * margin))
    alpha = np.zeros((rh, rw), np.float32)
    alpha[m:rh - m, m:rw - m] = 1.0
    feather = max(1.0, rh * 0.012)
    alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    charmask = (~bgmask).astype(np.uint8)
    charmask[: max(1, m // 3)] = 0
    charmask[-max(1, m // 3):] = 0
    charmask[:, : max(1, m // 3)] = 0
    charmask[:, -max(1, m // 3):] = 0
    charmask = cv2.dilate(charmask, np.ones((3, 3), np.uint8), iterations=2).astype(np.float32)
    charmask = cv2.GaussianBlur(charmask, (0, 0), feather * 0.7)
    alpha = np.maximum(alpha, charmask)

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


def load_image(path: str) -> tuple[np.ndarray, bytes | None]:
    """画像を RGB で読み込む。EXIF の回転を反映し、保存用に EXIF を返す。"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        exif = im.getexif()
        return np.array(im.convert("RGB")), (exif.tobytes() if exif else None)


def save_image(path: str, img: np.ndarray, exif: bytes | None = None, quality: int = 95) -> None:
    from PIL import Image

    kw = {"exif": exif} if exif else {}
    if path.lower().endswith((".jpg", ".jpeg")):
        kw.update(quality=quality, subsampling=0)
    Image.fromarray(img).save(path, **kw)


@dataclass
class PlateCandidate:
    quad: np.ndarray
    ok: bool
    score: float | None  # 検出スコア（手動指定は None）
    reason: str = ""


def find_plates(img: np.ndarray, conf: float = 0.1) -> list[PlateCandidate]:
    """自動検出 → 四隅推定。四隅推定に失敗したものも ok=False で返す。"""
    out = []
    for box, score in detect_plates(img, conf=conf):
        fit = refine_quad(img, box)
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


def erase_all(img: np.ndarray, quads: list[np.ndarray], margin: float = 0.035) -> np.ndarray:
    out = img
    for q in quads:
        out = erase_plate(out, q, margin=margin)
    return out
