"""ナンバープレート文字消去の一括処理。

uv run cli.py 画像... [-o 出力先]
"""

import argparse
import os

import plate


def main():
    ap = argparse.ArgumentParser(description="写真内のナンバープレートを自動検出し、文字を消して無地の板にする")
    ap.add_argument("inputs", nargs="+", help="入力画像")
    ap.add_argument("-o", "--outdir", default="output", help="出力ディレクトリ（既定: output）")
    ap.add_argument("--conf", type=float, default=0.1, help="検出スコアの下限")
    ap.add_argument("--margin", type=float, default=6, help="縁として残す幅（プレート高さに対する %%）")
    ap.add_argument("--format", choices=list(plate.SAVE_FORMATS), default="jpeg", help="保存形式（既定: jpeg）")
    ap.add_argument("--preview", action="store_true", help="検出結果を描いたプレビュー画像も保存する")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for path in args.inputs:
        img, exif = plate.load_image(path)
        cands = plate.find_plates(img, conf=args.conf)
        ok = [c for c in cands if c.ok]
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"{path}: {len(ok)} 枚処理" + "".join(f"\n  除外: スコア {c.score:.2f} {c.reason}" for c in cands if not c.ok))
        out = plate.erase_all(img, [c.quad for c in ok], margin=args.margin / 100)
        ext = plate.SAVE_FORMATS[args.format]
        plate.save_image(os.path.join(args.outdir, f"{stem}_noplate{ext}"), out, exif)
        if args.preview:
            colors = [(0, 255, 0) if c.ok else (255, 60, 60) for c in cands]
            vis = plate.draw_quads(img, [c.quad for c in cands], [f"#{i + 1}" for i in range(len(cands))], colors)
            plate.save_image(os.path.join(args.outdir, f"{stem}_preview{ext}"), vis)


if __name__ == "__main__":
    main()
