"""ナンバープレート文字消去の Web UI。

uv run app.py で起動し http://127.0.0.1:7900 を開く。
"""

import os
import tempfile

import cv2
import gradio as gr
import numpy as np

import plate

MODE_BOX = "矩形（2点クリック → 四隅を自動推定）"
MODE_QUAD = "四隅（4点クリック → そのまま使用）"

OK_COLOR = (0, 255, 0)
NG_COLOR = (255, 60, 60)
OFF_COLOR = (150, 150, 150)
CLICK_COLOR = (255, 0, 255)


def _label(i: int, c: plate.PlateCandidate) -> str:
    src = f"自動 {c.score:.2f}" if c.score is not None else "手動"
    return f"#{i + 1} {src}" + ("" if c.ok else f"（{c.reason}）")


def _render(img, cands, selected, clicks):
    if img is None:
        return None
    labels = [_label(i, c) for i, c in enumerate(cands)]
    colors = [
        (OK_COLOR if lab in selected else OFF_COLOR) if c.ok or lab in selected else NG_COLOR
        for lab, c in zip(labels, cands)
    ]
    vis = plate.draw_quads(img, [c.quad for c in cands], [f"#{i + 1}" for i in range(len(cands))], colors)
    r = max(4, max(img.shape[:2]) // 400)
    for p in clicks:
        cv2.circle(vis, (int(p[0]), int(p[1])), r, CLICK_COLOR, -1)
    return vis


def _choices(cands):
    labels = [_label(i, c) for i, c in enumerate(cands)]
    return labels, [lab for lab, c in zip(labels, cands) if c.ok]


def on_upload(path):
    if path is None:
        return None, None, [], [], gr.update(choices=[], value=[]), None, None, ""
    img, exif = plate.load_image(path)
    cands = plate.find_plates(img)
    labels, sel = _choices(cands)
    n_ok = len(sel)
    msg = f"{len(cands)} 件の候補を検出（うち {n_ok} 件をプレートと判定）。" if cands else "プレートが検出されませんでした。プレビューをクリックして手動で指定してください。"
    return (
        {"img": img, "exif": exif, "name": os.path.splitext(os.path.basename(path))[0]},
        _render(img, cands, sel, []),
        cands,
        [],
        gr.update(choices=labels, value=sel),
        None,
        None,
        msg,
    )


def on_click(data, cands, clicks, selected, mode, evt: gr.SelectData):
    if data is None:
        return gr.skip(), cands, clicks, gr.skip(), "先に画像を読み込んでください"
    img = data["img"]
    clicks = clicks + [list(evt.index)]
    need = 2 if mode == MODE_BOX else 4
    msg = f"{len(clicks)}/{need} 点"
    if len(clicks) >= need:
        pts = np.array(clicks, np.float32)
        if mode == MODE_BOX:
            box = [*pts.min(axis=0), *pts.max(axis=0)]
            fit = plate.refine_quad(img, box, strict=False)
            if fit.ok:
                cand = plate.PlateCandidate(fit.quad, True, None)
                msg = "四隅を推定して追加しました"
            else:
                # 推定できなければ指定した矩形そのものを使う
                cand = plate.PlateCandidate(plate.box_to_quad(box), True, None)
                msg = f"四隅推定に失敗（{fit.reason}）。指定した矩形をそのまま使います"
        else:
            cand = plate.PlateCandidate(plate.order_quad(pts), True, None)
            msg = "四隅を追加しました"
        cands = cands + [cand]
        clicks = []
        selected = list(selected) + [_label(len(cands) - 1, cand)]
        labels = [_label(i, c) for i, c in enumerate(cands)]
        return _render(img, cands, selected, clicks), cands, clicks, gr.update(choices=labels, value=selected), msg
    return _render(img, cands, selected, clicks), cands, clicks, gr.skip(), msg


def on_select_change(data, cands, selected, clicks):
    if data is None:
        return None
    return _render(data["img"], cands, selected, clicks)


def on_clear_clicks(data, cands, selected):
    if data is None:
        return None, []
    return _render(data["img"], cands, selected, []), []


def on_remove_unselected(data, cands, selected):
    if data is None:
        return None, [], gr.update(choices=[], value=[])
    labels = [_label(i, c) for i, c in enumerate(cands)]
    cands = [c for lab, c in zip(labels, cands) if lab in selected]
    labels = [_label(i, c) for i, c in enumerate(cands)]
    return _render(data["img"], cands, labels, []), cands, gr.update(choices=labels, value=labels)


def on_erase(data, cands, selected, margin):
    if data is None:
        gr.Warning("画像を読み込んでください")
        return None, None
    labels = [_label(i, c) for i, c in enumerate(cands)]
    quads = [c.quad for lab, c in zip(labels, cands) if lab in selected]
    if not quads:
        gr.Warning("処理するプレートが選択されていません")
        return None, None
    out = plate.erase_all(data["img"], quads, margin=margin / 100)
    path = os.path.join(tempfile.mkdtemp(), f"{data['name']}_noplate.jpg")
    plate.save_image(path, out, data["exif"])
    return out, path


with gr.Blocks(title="ナンバープレート消去") as demo:
    gr.Markdown(
        "## ナンバープレート文字消去\n"
        "画像を読み込むと自動でプレートを検出します。見落としや誤りはプレビューをクリックして手動で追加し、"
        "チェックを外したものは処理されません。緑＝処理対象、赤＝プレートと判定されなかった候補、灰＝対象外。"
    )
    data_state = gr.State(None)
    cands_state = gr.State([])
    clicks_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="入力画像", type="filepath", sources=["upload", "clipboard"])
            status = gr.Textbox(label="状態", interactive=False)
            selected = gr.CheckboxGroup(label="処理するプレート", choices=[])
            remove_btn = gr.Button("チェックを外した候補を削除", size="sm")
            mode = gr.Radio([MODE_BOX, MODE_QUAD], value=MODE_BOX, label="手動追加の方法（プレビューをクリック）")
            clear_btn = gr.Button("クリック点をリセット", size="sm")
            margin = gr.Slider(0, 8, value=3.5, step=0.5, label="縁として残す幅（プレート高さに対する %）")
            erase_btn = gr.Button("文字を消去", variant="primary")
        with gr.Column(scale=2):
            preview = gr.Image(label="検出プレビュー（クリックで手動追加）", type="numpy", interactive=False)
            output_img = gr.Image(label="結果", type="numpy", interactive=False)
            output_file = gr.File(label="ダウンロード（元の解像度・EXIF 付き JPEG）")

    input_img.change(
        on_upload,
        inputs=input_img,
        outputs=[data_state, preview, cands_state, clicks_state, selected, output_img, output_file, status],
    )
    preview.select(
        on_click,
        inputs=[data_state, cands_state, clicks_state, selected, mode],
        outputs=[preview, cands_state, clicks_state, selected, status],
    )
    selected.input(on_select_change, inputs=[data_state, cands_state, selected, clicks_state], outputs=preview)
    clear_btn.click(on_clear_clicks, inputs=[data_state, cands_state, selected], outputs=[preview, clicks_state])
    mode.change(on_clear_clicks, inputs=[data_state, cands_state, selected], outputs=[preview, clicks_state])
    remove_btn.click(on_remove_unselected, inputs=[data_state, cands_state, selected], outputs=[preview, cands_state, selected])
    erase_btn.click(on_erase, inputs=[data_state, cands_state, selected, margin], outputs=[output_img, output_file])

if __name__ == "__main__":
    demo.launch(server_port=7900)
