import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

_detector = None


def get_detector() -> YOLO:
    global _detector
    if _detector is None:
        path = hf_hub_download("Koushim/yolov8-license-plate-detection", "best.pt")
        _detector = YOLO(path)
    return _detector


def detect_plates(pil_img: Image.Image) -> list[list[int]]:
    results = get_detector()(pil_img, verbose=False, conf=0.15)
    return [[int(v) for v in box] for r in results for box in r.boxes.xyxy.cpu().numpy()]


def draw_boxes(pil_img: Image.Image, boxes: list) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    lw = max(4, pil_img.width // 600)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=lw)
    return img


def fill_plates(pil_img: Image.Image, boxes: list) -> Image.Image:
    arr = np.array(pil_img.convert("RGB"))

    for x1, y1, x2, y2 in boxes:
        crop = arr[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

        # 白ナンバー: 彩度低・明度高（緩め）
        white_mask = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
        # 黄ナンバー: 黄色相・彩度高・明度高（緩め）
        yellow_mask = cv2.inRange(hsv, (15, 80, 130), (45, 255, 255))
        plate_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # モルフォロジーでノイズ除去・穴埋め
        k = np.ones((5, 5), np.uint8)
        plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, k, iterations=2)
        plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, k, iterations=1)

        # 最大連結成分のみ（プレート本体）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(plate_mask)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            plate_mask = (labels == largest).astype(np.uint8) * 255

        # フォールバック: 色マスクが bbox の 5% 未満なら bbox 全体を明るいピクセルの中央値で塗る
        hit_ratio = plate_mask.sum() / 255 / plate_mask.size
        if hit_ratio < 0.05:
            v = hsv[:, :, 2].astype(np.float32)
            bright = crop[v >= np.percentile(v, 50)]
            bg_color = np.median(bright.reshape(-1, 3), axis=0).astype(np.uint8)
            arr[y1:y2, x1:x2] = bg_color
            continue

        # 輪郭取得
        contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # 輪郭を4頂点に近似（epsilon を段階的に緩めて4点以下に収束）
        peri = cv2.arcLength(cnt, True)
        corners = None
        for eps in [0.02, 0.04, 0.07, 0.10, 0.15]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) <= 4:
                corners = approx.reshape(-1, 2)
                break

        # フォールバック: 最小外接矩形の4頂点
        if corners is None or len(corners) < 3:
            rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(rect).astype(int)

        # 4頂点ポリゴンマスク
        poly_mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(poly_mask, [corners.astype(np.int32)], 255)

        # ポリゴン内の暗ピクセル = 文字マスク（積極的に取得）
        v = hsv[:, :, 2].astype(np.float32)
        plate_v = v[poly_mask > 0]
        # 地色の65パーセンタイル × 0.75 を閾値に → エッジのグレー領域も捕捉
        char_thresh = np.percentile(plate_v, 65) * 0.75
        char_mask = ((v < char_thresh) & (poly_mask > 0)).astype(np.uint8) * 255
        # 膨張を強化してストロークを確実にカバー
        char_mask = cv2.dilate(char_mask, np.ones((7, 7), np.uint8), iterations=2)
        char_mask &= poly_mask  # ポリゴン外には広げない

        if char_mask.sum() == 0:
            continue

        # ポリゴン外を地色で一時置換（車ボディが inpaint の補間ソースに混入しないよう）
        bg_color = np.median(crop[poly_mask > 0].reshape(-1, 3), axis=0).astype(np.uint8)
        temp = crop.copy()
        temp[poly_mask == 0] = bg_color

        # inpaintRadius を大きくして文字ストローク全体をカバー
        inpainted = cv2.inpaint(temp, char_mask, inpaintRadius=25, flags=cv2.INPAINT_TELEA)

        # ポリゴン内のみ元画像に適用（ポリゴン外は元の車ボディを保持）
        result = crop.copy()
        result[poly_mask > 0] = inpainted[poly_mask > 0]
        arr[y1:y2, x1:x2] = result

    return Image.fromarray(arr)


def on_detect(image: Image.Image):
    if image is None:
        return None, [], "画像を選択してください"
    boxes = detect_plates(image)
    preview = draw_boxes(image, boxes)
    msg = f"{len(boxes)} 枚のプレートを検出" if boxes else "プレートが検出されませんでした"
    return preview, boxes, msg


def on_fill(image: Image.Image, boxes: list):
    if image is None:
        gr.Warning("画像を選択してください")
        return None
    if not boxes:
        gr.Warning("プレートが検出されていません。先に検出を実行してください。")
        return None
    return fill_plates(image, boxes)


with gr.Blocks(title="ナンバープレート塗り潰し") as demo:
    gr.Markdown("## ナンバープレート自動塗り潰し")
    boxes_state = gr.State([])

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="入力画像", type="pil")
            detect_btn = gr.Button("① プレート検出")
            status = gr.Textbox(label="検出結果", interactive=False)
            fill_btn = gr.Button("② 文字を塗り潰す", variant="primary")

        with gr.Column():
            preview_img = gr.Image(label="検出プレビュー", type="pil")
            output_img = gr.Image(label="出力結果", type="pil")

    detect_btn.click(
        on_detect,
        inputs=input_img,
        outputs=[preview_img, boxes_state, status],
    )
    fill_btn.click(
        on_fill,
        inputs=[input_img, boxes_state],
        outputs=output_img,
    )

demo.launch(server_port=7900)
