"""おおまかな bbox からナンバーの正確な四隅を推定する。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .checks import _rim_is_plate, _text_fits_layout, looks_like_plate, plate_colors_plausible
from .color import _color_dist, _kmeans2, _to_lab
from .contour import _fit_quad_to_contour, _refine_edges
from .fallback import _quad_from_text, _snap_box_to_edges
from .geometry import PLATE_ASPECT, _apparent_aspect, box_to_quad, order_quad, quad_shape_problem


@dataclass
class PlateFit:
    quad: np.ndarray  # 画像座標
    ok: bool
    reason: str = ""
    bg_color: tuple[int, int, int] | None = None
    from_edges: bool = False  # 色の領域ではなく、検出枠をエッジに寄せて決めた四隅か


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

    small_box = (bx1, by1, bx2, by2)
    quad, bg, fg, reason = _fit_by_color(lab, seed, samples, centers, labels, counts, small_box, strict, debug)
    from_edges = False
    if quad is None:
        bg_i = int(np.argmax(counts))
        bg, fg = centers[bg_i], centers[1 - bg_i]
        quad = _fit_by_text_or_edges(img, box, small, lab, s, (cx1, cy1), small_box, bg, fg, strict)
        if quad is None:
            fail.reason = reason
            return fail
        from_edges = True

    quad_full = _snap_full_res(img, quad / s + [cx1, cy1], (cx1, cy1, cx2, cy2), s)

    # 写真の端に接する（端で切れた）プレートは四隅を正しく決められないので、自動では処理しない（手動指定は可）
    over = max(2.0, 0.01 * float(np.linalg.norm(quad_full[1] - quad_full[0])))
    if strict and (quad_full.min() < over or (quad_full[:, 0] > W - 1 - over).any() or (quad_full[:, 1] > H - 1 - over).any()):
        return PlateFit(quad_full.astype(np.float32), False, "写真の端で切れている")
    if strict and not looks_like_plate(img, quad_full):
        return PlateFit(quad_full.astype(np.float32), False, "ナンバーの文字配置に見えない")

    bg_rgb = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).ravel()
    return PlateFit(quad_full.astype(np.float32), True, "", tuple(int(c) for c in np.clip(bg_rgb * 255, 0, 255)), from_edges)


def _fit_by_color(lab, seed, samples, centers, labels, counts, small_box, strict, debug):
    """地色の領域から四隅を求める。地色は多い方のクラスタ、だめなら少ない方で試す。

    (quad, bg, fg, 失敗理由) を返す（quad は small の座標。失敗時は None と最初の失敗理由）。
    """
    bx1, by1, bx2, by2 = small_box
    first_reason = None
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
                return quad, bg, fg, ""
        first_reason = first_reason or reason
    return None, None, None, first_reason


def _fit_by_text_or_edges(img, box, small, lab, s, origin, small_box, bg, fg, strict):
    """色の領域で四隅が決まらないときの代替。文字の並びから、それがだめなら検出枠をエッジに寄せて求める。

    日陰で車体とほぼ同じ色のプレートや、隣の車の映り込みで色味がずれたプレートは、色の領域が車体へ漏れて決まらない。
    どちらの方法も、色で求めたときと同じ妥当性チェックに掛ける。quad（small の座標）か None を返す。
    """
    cx1, cy1 = origin
    # まず文字の並びから外形を推定する（白い車体の白ナンバーなど、色では車体と区別できないもの向け）
    q_text = _quad_from_text(small, small_box)
    if q_text is not None and not quad_shape_problem(q_text) and _text_fits_layout(small, q_text) and _rim_is_plate(lab, q_text, bg, fg):
        quad = _validate_filled(lab, q_text, bg, fg, strict)
        if quad is not None:
            return quad
    # 検出器の bbox を初期値に、各辺を近くの強いエッジへ寄せる
    snapped = _snap_box_to_edges(img, box)
    q_small = None if snapped is None else ((snapped - [cx1, cy1]) * s).astype(np.float32)
    if q_small is not None and not quad_shape_problem(q_small) and _rim_is_plate(lab, q_small, bg, fg):
        return _validate_filled(lab, q_small, bg, fg, strict)
    return None


def _validate_filled(lab, quad, bg, fg, strict):
    """quad 自身を領域として妥当性チェックに掛ける（色の領域を持たない代替の四角形用）。"""
    filled = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(filled, np.round(quad).astype(np.int32), 1)
    quad, _ = _validate_quad(lab, filled, quad, bg, fg, strict)
    return quad


def _snap_full_res(img, quad_full, crop, s):
    """原寸で、各辺を輝度勾配に沿って詰める。形が崩れたら詰める前の四角形を返す。"""
    cx1, cy1, cx2, cy2 = crop
    gray = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, 0.5 / s))
    search = max(3.0, 4.0 / s)
    refined = _refine_edges(gray, (quad_full - [cx1, cy1]).astype(np.float32), search) + [cx1, cy1]
    if not quad_shape_problem(refined):
        quad_full = refined
    return quad_full


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
    aspect = _apparent_aspect(quad)
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
