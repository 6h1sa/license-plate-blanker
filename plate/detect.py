"""YOLO によるナンバーの検出（おおまかな bbox）。"""

from __future__ import annotations

import cv2
import numpy as np


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
