"""検出 → 四隅推定 → 文字消去 をまとめた一括処理。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import detect_plates
from .erase import erase_plate
from .fit import PlateFit, refine_quad


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
