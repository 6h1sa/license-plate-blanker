"""YOLO によるナンバーの検出（おおまかな bbox）。"""

from __future__ import annotations

import cv2
import numpy as np


# 学習データが多く、小さい・斜めのプレートにも強い YOLOv11 系 (AGPL-3.0)。
# 配布元の差し替えで結果が変わらないよう、リポジトリのコミットを固定する
MODEL_REPO = "morsetechlab/yolov11-license-plate-detection"
MODEL_FILE = "license-plate-finetune-v1s.onnx"
MODEL_REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"

STRIDE = 32
NMS_IOU = 0.7  # 1回の推論の中での重複除去（ultralytics の既定値）

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        # cv2.dnn は入力サイズ可変のこのモデルを読めない（Concat の形の推論で失敗する）ので onnxruntime を使う
        path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
        _detector = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _detector


def _letterbox(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """長辺を imgsz に合わせて縮小（拡大）し、32 の倍数になるよう灰色で埋める。倍率と左上の余白を返す。"""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = ((imgsz - nw) % STRIDE) / 2, ((imgsz - nh) % STRIDE) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = round(dh - 0.1), round(dw - 0.1)
    img = cv2.copyMakeBorder(img, top, round(dh + 0.1), left, round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def _infer(imgs: list[np.ndarray], conf: float, imgsz: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """同じ大きさの画像をまとめて推論し、画像ごとに (元画像の座標の xyxy, score) を返す。"""
    boxed = [_letterbox(im, imgsz) for im in imgs]
    # モデルは ultralytics 経由で使っていたときと同じく、チャンネルを逆順にした画像を受け取る
    blob = np.stack([b[0][..., ::-1] for b in boxed]).transpose(0, 3, 1, 2).astype(np.float32) / 255
    out = get_detector().run(None, {"images": blob})[0]  # (batch, 5, N): cx, cy, w, h, score

    results = []
    for pred, (_, r, (left, top)), im in zip(out, boxed, imgs):
        pred = pred.T[pred[4] > conf]
        if len(pred) == 0:
            results.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
            continue
        xywh = np.column_stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2, pred[:, 2], pred[:, 3]])
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), pred[:, 4].tolist(), conf, NMS_IOU)).flatten()
        h, w = im.shape[:2]
        xyxy = np.column_stack([xywh[keep, :2], xywh[keep, :2] + xywh[keep, 2:]])
        xyxy = (xyxy - [left, top, left, top]) / r
        xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(np.float32)
        results.append((xyxy, pred[keep, 4]))
    return results


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
    H, W = img.shape[:2]
    tiles = [(x, y, min(tile, W), min(tile, H)) for y in _grid(H, tile, tile // 2) for x in _grid(W, tile, tile // 2)]
    crops = [np.ascontiguousarray(img[y:y + h, x:x + w]) for x, y, w, h in tiles]

    boxes: list[np.ndarray] = []
    scores: list[float] = []
    edge = 4
    for (x, y, w, h), (bs, cs) in zip(tiles, _infer(crops, conf, imgsz)):
        for b, c in zip(bs, cs):
            if (b[0] < edge and x > 0) or (b[1] < edge and y > 0) or (b[2] > w - edge and x + w < W) or (b[3] > h - edge and y + h < H):
                continue
            boxes.append(b + [x, y, x, y])
            scores.append(float(c))

    # 大きく写ったプレート用に全体推論も行う
    for b, c in zip(*_infer([img], conf, 1280)[0]):
        boxes.append(b)
        scores.append(float(c))

    if not boxes:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores, conf, 0.3)
    return [([int(round(v)) for v in boxes[k]], scores[k]) for k in np.array(keep).flatten()]
