"""色の扱い。色の比較はすべて Lab で行う。"""

from __future__ import annotations

import cv2
import numpy as np


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
