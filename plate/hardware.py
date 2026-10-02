"""正面化したナンバーで、消さずに残すもの（取付ボルト・封印・縁のエンボス・枠のクリップ）を探す。"""

from __future__ import annotations

import cv2
import numpy as np


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
