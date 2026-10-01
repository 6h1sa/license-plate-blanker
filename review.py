"""検証用の比較シートを作る。

uv run review.py sample_picture -o review

画像ごとに「全体（検出結果の枠付き）」と「プレートごとの処理前/処理後」を1枚にまとめて保存し、
集計を表示する。パラメータを変えたときに全サンプルの悪化を目視で確認するためのもの。
緑＝処理対象、赤＝候補だが除外。
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

import plate

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
SHEET_W = 1600
CROP_H = 260


def _crop_pair(img, out, quad, ok):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    w, h = x1 - x0, y1 - y0
    H, W = img.shape[:2]
    sl = (slice(int(max(0, y0 - h * 0.5)), int(min(H, y1 + h * 0.5))), slice(int(max(0, x0 - w * 0.3)), int(min(W, x1 + w * 0.3))))
    a = plate.draw_quads(img, [quad], colors=[(0, 255, 0) if ok else (255, 60, 60)])[sl]
    b = out[sl]
    pair = np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], axis=1)
    s = CROP_H / pair.shape[0]
    return cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _pack_rows(tiles, width):
    rows, row, rw = [], [], 0
    for t in tiles:
        if row and rw + t.shape[1] > width:
            rows.append(row)
            row, rw = [], 0
        row.append(t)
        rw += t.shape[1] + 8
    if row:
        rows.append(row)
    out = []
    for r in rows:
        line = np.full((CROP_H, width, 3), 32, np.uint8)
        x = 0
        for t in r:
            t = t[:, : width - x]
            line[:, x:x + t.shape[1]] = t
            x += t.shape[1] + 8
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="サンプル画像のディレクトリ")
    ap.add_argument("-o", "--outdir", default="review")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(args.src, "*")) if p.lower().endswith(EXTS))
    total_ok = total_ng = 0
    for path in paths:
        t = time.time()
        img, _ = plate.load_image(path)
        cands = plate.find_plates(img)
        ok = [c for c in cands if c.ok]
        out = plate.erase_all(img, [c.quad for c in ok])
        dt = time.time() - t
        total_ok += len(ok)
        total_ng += len(cands) - len(ok)

        colors = [(0, 255, 0) if c.ok else (255, 60, 60) for c in cands]
        vis = plate.draw_quads(img, [c.quad for c in cands], [f"#{i + 1}" for i in range(len(cands))], colors)
        s = SHEET_W / vis.shape[1]
        overview = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tiles = [_crop_pair(img, out, c.quad, c.ok) for c in cands]
        sheet = np.concatenate([overview, *_pack_rows(tiles, SHEET_W)], axis=0) if tiles else overview
        name = os.path.splitext(os.path.basename(path))[0]
        plate.save_image(os.path.join(args.outdir, f"{name}.jpg"), sheet, quality=90)

        detail = "".join(f"\n    #{i + 1} 除外 (スコア {c.score:.2f}): {c.reason}" for i, c in enumerate(cands) if not c.ok)
        print(f"{os.path.basename(path)}: 処理 {len(ok)} / 候補 {len(cands)} ({dt:.1f}s){detail}")
    print(f"合計: {len(paths)} 枚, 処理 {total_ok}, 除外 {total_ng}")


if __name__ == "__main__":
    main()
