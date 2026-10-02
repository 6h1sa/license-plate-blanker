"""テスト用の合成画像。実在のナンバーを使わずに、プレートの付いた車体の一部を描く。"""

from __future__ import annotations

import cv2
import numpy as np

PLATE_W, PLATE_H = 330, 165  # 正面化したプレートの大きさ (実物の mm と同じ比)

WHITE = (235, 238, 240)
YELLOW = (240, 200, 40)
GREEN = (20, 110, 60)
DARK_TEXT = (25, 50, 35)
WHITE_TEXT = (240, 240, 240)


# 取付ボルト・封印・縁のエンボスの位置（正面化した座標）
BOLT_R = (int(PLATE_W * 0.82), int(PLATE_H * 0.15))
SEAL_L = (int(PLATE_W * 0.17), int(PLATE_H * 0.14))
RIM = 5  # 縁のエンボスの線の、外周からの距離


BOLT_L = (int(PLATE_W * 0.18), int(PLATE_H * 0.15))


def plate_image(bg=WHITE, text=DARK_TEXT, shadow: float = 0.0, hardware: bool = False, lone_bolt: bool = False) -> np.ndarray:
    """正面から見たプレート (RGB)。

    shadow > 0 なら上部にその濃さの横帯の影を落とす。
    hardware なら、右に黒いボルト、左に金属の封印、外周に沿ってエンボスの細い線を描く。
    lone_bolt なら、左のボルトの位置にだけ黒い丸を描く（右には何も無い）。
    """
    img = np.full((PLATE_H, PLATE_W, 3), bg, np.uint8)
    if lone_bolt:
        cv2.circle(img, BOLT_L, 6, (25, 25, 25), -1, cv2.LINE_AA)
    if hardware:
        rim = tuple(int(c * 0.72) for c in bg)
        cv2.rectangle(img, (RIM, RIM), (PLATE_W - 1 - RIM, PLATE_H - 1 - RIM), rim, 2, cv2.LINE_AA)
        cv2.circle(img, BOLT_R, 6, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(img, SEAL_L, 13, (90, 90, 95), -1, cv2.LINE_AA)
        cv2.circle(img, SEAL_L, 10, (205, 208, 212), -1, cv2.LINE_AA)
    # 上段（地名・分類番号）と、下段の大きな数字
    cv2.putText(img, "ABC 330", (100, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text, 3, cv2.LINE_AA)
    cv2.putText(img, "12-34", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 3.2, text, 10, cv2.LINE_AA)
    if shadow:
        f = np.ones((PLATE_H, 1, 1), np.float32)
        f[: int(PLATE_H * 0.22)] = 1 - shadow
        img = np.clip(img * f, 0, 255).astype(np.uint8)
    return img


def scene(quad, plate: np.ndarray | None = None, size=(900, 600), body=(30, 32, 36), seed: int = 0):
    """plate を quad (左上・右上・右下・左下) の位置に貼った画像と、プレートの領域マスクを返す。"""
    plate = plate_image() if plate is None else plate
    W, H = size
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), body, np.float32)
    # 車体側の質感（グリル風の格子）
    for y in range(0, H, 14):
        img[y:y + 3] *= 0.6
    src = np.float32([[0, 0], [PLATE_W, 0], [PLATE_W, PLATE_H], [0, PLATE_H]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(plate.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((PLATE_H, PLATE_W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    img = img * (1 - mask) + warped * mask
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask[..., 0] > 0.5


def bbox_of(quad, pad: float = 0.05):
    """quad を少しずらした・広げた bbox（検出器の出力のつもり）。"""
    q = np.float32(quad)
    x1, y1 = q.min(axis=0)
    x2, y2 = q.max(axis=0)
    w, h = x2 - x1, y2 - y1
    return [int(x1 - pad * w), int(y1 - pad * h * 0.5), int(x2 + pad * w * 0.6), int(y2 + pad * h)]


def rectify(img: np.ndarray, quad, w: int = PLATE_W, h: int = PLATE_H) -> np.ndarray:
    M = cv2.getPerspectiveTransform(np.float32(quad), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


FRONT = [[300, 200], [630, 200], [630, 365], [300, 365]]
# 斜め（右に振った遠近 + 少し回転）
OBLIQUE = [[310, 215], [600, 185], [612, 345], [318, 380]]
