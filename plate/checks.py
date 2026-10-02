"""四角形がナンバーとして妥当かの判定（配色・文字配置・縁）。"""

from __future__ import annotations

import cv2
import numpy as np

from .color import _color_dist, _kmeans2, _to_lab
from .geometry import warp_rect


def looks_like_plate(img: np.ndarray, quad: np.ndarray) -> bool:
    """正面化した領域に、日本のナンバー特有の「下段の大きな数字」があるか。

    ステッカーや看板などの誤検出を除くための簡易チェック。
    """
    rw, rh = 400, 200
    lab = _to_lab(warp_rect(img, quad, rw, rh))
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


def _rim_is_plate(lab: np.ndarray, quad: np.ndarray, bg: np.ndarray, fg: np.ndarray) -> bool:
    """四角形の4辺それぞれの内側の細い帯が、ほぼ地色か（辺がプレートの縁に沿っているか）。

    エッジに寄せただけの四角形は、車体の線などに乗って斜めにずれることがあるので、その確認に使う。
    """
    rw, rh = 200, 100
    r = warp_rect(lab, quad, rw, rh, flags=cv2.INTER_LINEAR)
    near = _color_dist(r, bg) < max(6.0, 0.5 * float(_color_dist(fg[None, None], bg)[0, 0]))
    strips = (near[4:10, 10:-10], near[-10:-4, 10:-10], near[10:-10, 4:10], near[10:-10, -10:-4])
    ratios = sorted(float(st.mean()) for st in strips)
    # 文字が縁に近い辺や、映り込みで色味がずれた辺が1つあっても許す
    return ratios[0] >= 0.4 and ratios[1] >= 0.6


def _text_fits_layout(small: np.ndarray, quad: np.ndarray) -> bool:
    """四角形を正面化したとき、文字が規格どおりの範囲に収まっているか（外形の推定がずれていないか）。

    下段の大きな数字の右端は板の右端近く（0.89〜0.995）、文字全体の左端は板の左端近く（0.12 以下）、
    上端・下端もそれぞれ板の上端・下端の近くにあるはず（範囲は注釈済みの実物 68 件の実測から決めた）。
    ずれていれば、外形が車体側へはみ出している。
    """
    rw, rh = 400, 200
    g = cv2.cvtColor(warp_rect(small, quad, rw, rh, flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY).astype(np.float32)
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
