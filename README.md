車のナンバープレートの文字を消し、無地の板が付いているように加工するツールです。

## 使い方

```bash
npm --prefix web install && npm --prefix web run build   # 初回と、画面（web/）を変えたあと。Node.js が必要
uv run app.py                       # Web UI (http://127.0.0.1:7900)
uv run cli.py 写真/*.jpg -o output  # 一括処理（--preview で検出結果画像も保存）
```

Docker で動かす場合（Node.js も uv も不要。検出モデルはイメージに含まれます）:

```bash
docker build -t plate-blanker .
docker run --rm -p 7900:7900 plate-blanker              # Web UI (http://127.0.0.1:7900)
docker run --rm -v "$PWD/photos:/data" plate-blanker sh -c "python cli.py /data/*.jpg -o /data/output"
```

Windows では Docker Desktop で同じコマンドが使えます。PowerShell では `$PWD` を `${PWD}` に置き換えてください。
検出は CPU（onnxruntime）で動くので、GPU の設定は不要です。

Web UI では、写真をドラッグ&ドロップする（複数可）と、検出から消去まで自動で進みます。

- **確認**: ホイールで拡大し、<kbd>Space</kbd> で処理前と処理後を切り替えます。右側にはナンバーごとの処理前・処理後が並びます
- **修正**: 枠をクリックして選び、角の丸をドラッグして四隅を直します。チェックを外すと、そのナンバーは処理しません
- **追加**: 見落としは <kbd>A</kbd> で範囲をドラッグして囲むと四隅を自動で推定します。ナンバー全体が入るよう、少し外側まで囲むと
  最も正確です（広く囲みすぎると周りの車体に引っ張られます）。<kbd>Q</kbd> なら四隅を直接クリックします。こちらは板の外周の角を狙います。
  点線の枠（自動では除外した候補）はクリックすると対象になります
- **保存**: 「この写真をダウンロード」<kbd>D</kbd> か「すべてダウンロード（ZIP）」

編集すると、消去は裏で自動でやり直されます。読み込んだ写真はサーバーを止めると消えます。

保存形式は JPEG（既定）と WebP から選べます（UI の「保存形式」、CLI は `--format webp`）。
どちらも元の解像度で、EXIF を引き継ぎます。WebP は仕様上、長辺 16383px までです。

## 処理の流れ（plate/）

1. **検出**: YOLOv11 のナンバープレート検出モデル（ONNX 版を onnxruntime で実行）を 50% 重なりのタイルで実行（高解像度写真の小さいプレート対策）
2. **四隅推定**: bbox 内の地色を推定 → 地色領域の輪郭を4辺に直線フィット → 原寸の輝度勾配で各辺をサブピクセル補正。
   縦横比・文字領域の比率・下段の大きな数字の有無で誤検出を除外
3. **文字消去**: プレートを射影変換で正面化し、文字を除いた画素から照明ムラ込みの地色面を推定、
   元画像相当の粒状ノイズを乗せて置き換え、逆射影して合成。次のものは元画像のまま残す
   - 取付ボルトと、後部ナンバー左側の封印（金属キャップ。円として検出）
   - 縁のエンボスの細い線、ナンバー枠のクリップ
   - バンパーなどから落ちる影（濃さを再現して塗る）

## 画面の開発（web/）

画面は React + TypeScript で、Vite でビルドします。ビルド結果（`web/dist`）はリポジトリに入れていません。

```bash
uv run app.py              # API サーバー（別のターミナルで起動しておく）
npm --prefix web run dev   # http://127.0.0.1:5173 。保存すると即座に反映（API は 7900 へ中継）
npm --prefix web test      # 座標計算などの単体テスト
npm --prefix web run build # 型チェックとビルド（app.py が配信するのはこの結果）
```

- `src/App.tsx`: 画面全体と状態のつなぎ込み
- `src/components/`: `Viewer`（拡大表示と四隅の編集をする Canvas）、`PlatePanel`、`FileList` など
- `src/hooks/`: 一覧の定期取得、編集の保存（少し待ってまとめて送る）、キー操作
- `src/geometry.ts`: 座標変換と当たり判定（React に依存しない）

## テスト

```bash
uv run pytest                                # 単体テスト（合成画像のみ・数秒）
uv run pytest -m samples                     # sample_picture/ の実写真で、結果が基準から変わっていないか確認（数分）
uv run pytest -m samples --update-baseline   # 今の結果を基準として保存（意図して結果を変えたとき）
uv run review.py sample_picture -o review    # 全サンプルの処理前後を並べた比較画像を出力（目視確認用）
```

正解の四隅を付けるときは `uv run annotate.py` を起動し、http://127.0.0.1:7901 を開きます。
自動検出の結果が下書きとして表示されるので、ずれた四隅をドラッグで直し、見落としを追加し、
処理しなくてよいナンバー（遠すぎる・読めない・切れている）に「対象外」を付けて確定します。
結果は `.regression/annotations.json` に保存されます。

サンプル写真と基準ファイル（`.regression/`）は他人の車が写るため、リポジトリには入れていません。
どちらかが無い環境では、実写真のチェックはスキップされます。

## ライセンス

[AGPL-3.0](LICENSE)。検出モデル `morsetechlab/yolov11-license-plate-detection`（Ultralytics YOLOv11 で学習）
が AGPL-3.0 のため、それに合わせています。Web UI をネットワーク越しに他者へ公開する場合も、
AGPL に従いソースコードを提供する必要があります。
