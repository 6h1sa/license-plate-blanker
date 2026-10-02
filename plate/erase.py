"""ナンバーの文字を消し、無地の板にする。"""

from __future__ import annotations

import cv2
import numpy as np

from .color import _color_dist, _kmeans2, _to_lab
from .geometry import PLATE_ASPECT, order_quad, rect_corners, side_lengths
from .hardware import _find_bolts, _keep_mask


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


def erase_plate(img: np.ndarray, quad: np.ndarray, margin: float = 0.06, seed: int = 0, debug: dict | None = None) -> np.ndarray:
    """四隅で指定されたプレートの文字を消し、無地の板にする。

    プレートを正面化し、文字を除いた地色画素から照明ムラを含む滑らかな地色面を推定、
    元画像相当の粒状ノイズを乗せて内側を置き換え、逆射影で合成する。
    取付ボルト・縁のエンボス・枠のクリップ・バンパーなどの影は残す。
    縁の帯 (margin) は元画像を残し、内側から続く文字だけを置き換える。
    """
    quad = order_quad(quad)
    rect, M = _rectify(img, quad)
    lab = _to_lab(rect)
    rh = rect.shape[0]

    # 地色: 内側の画素を2クラスタに分けて多い方
    mm = int(rh * 0.08)
    centers, labels, counts = _kmeans2(lab[mm:-mm, mm:-mm].reshape(-1, 3)[::3])
    bg = centers[int(np.argmax(counts))]
    fg = centers[int(np.argmin(counts))]
    thr = float(np.clip(0.35 * _color_dist(fg[None, None], bg)[0, 0], 5, 25))

    surface, bgmask, bgmask_e = _estimate_surface(lab, bg, thr, margin)
    synth = _synthesize(rect, lab, surface, bgmask_e, seed)
    alpha = _composite_alpha(rect, lab, bgmask, bg, fg, margin, debug)
    return _paste_back(img, quad, M, synth, alpha)


def _rectify(img: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """プレートを正面化する。(正面化した画像, 射影行列) を返す。"""
    top, bottom, left, right = side_lengths(quad)
    # 正面化解像度: 元のプレート画素数と同程度（小さいプレートは拡大して処理）
    rw = int(np.clip(max(top, bottom, 2 * max(left, right)), 240, 1600))
    rh = int(round(rw / PLATE_ASPECT))
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), rect_corners(rw, rh))
    rect = cv2.warpPerspective(img, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rect, M


def _estimate_surface(lab: np.ndarray, bg: np.ndarray, thr: float, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """文字を除いた地色画素から、照明ムラを含む滑らかな地色面を推定する。

    (地色面, 地色マスク, 地色面の推定に使った収縮後のマスク) を返す。
    """
    rh, rw = lab.shape[:2]
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
    return surface, bgmask, bgmask_e


def _synthesize(rect: np.ndarray, lab: np.ndarray, surface: np.ndarray, bgmask_e: np.ndarray, seed: int) -> np.ndarray:
    """地色面に粒状ノイズと影を乗せた、無地の板の画像 (RGB float 0–255) を作る。"""
    rh = rect.shape[0]
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
    return np.clip(synth * 255 * _shadow_ratio(rect.astype(np.float32), base, rh), 0, 255)


def _composite_alpha(rect: np.ndarray, lab: np.ndarray, bgmask: np.ndarray, bg: np.ndarray, fg: np.ndarray, margin: float,
                     debug: dict | None) -> np.ndarray:
    """正面化した座標で、無地の板で置き換える割合 (0–1) を返す。

    内側はすべて置き換え、縁の帯は内側から続く文字だけ。残すもの（ボルト・縁・クリップ）は除く。
    """
    rh, rw = lab.shape[:2]
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
    return alpha


def _paste_back(img: np.ndarray, quad: np.ndarray, M: np.ndarray, synth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """正面化した座標の synth を逆射影し、alpha で元画像に合成する（プレート周辺の矩形領域だけ処理）。"""
    out = img.copy()
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
