# ナンバープレート文字消去の Web UI を動かすイメージ
#
#   docker build -t plate-blanker .
#   docker run --rm -p 7900:7900 plate-blanker               # http://127.0.0.1:7900
#
# 一括処理（CLI）は写真のフォルダをマウントして実行する（Windows の PowerShell では "$PWD" を "${PWD}" に）:
#   docker run --rm -v "$PWD/photos:/data" plate-blanker sh -c "python cli.py /data/*.jpg -o /data/output"
#
# 依存の導入とモデルの取得は builder で uv を使って行い、実行用のイメージには uv を入れない。

# --- 画面（web/）のビルド ---
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- 依存と検出モデルの準備（uv を使う） ---
# 仮想環境は Python の実行ファイルの場所を記録するので、builder と実行用は同じ Python のイメージにそろえる
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    HF_HOME=/opt/hf

WORKDIR /app
# スクリプトとして動かすので、プロジェクト自体はインストールしない
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project

# 検出モデル（ONNX 版、plate/detect.py で固定したリビジョン）を取得しておく。
# detect.py だけを使うので、ほかのコードを変えてもモデルは取り直さない
RUN --mount=type=bind,source=plate/detect.py,target=/tmp/detect.py \
    /app/.venv/bin/python -c "import runpy; runpy.run_path('/tmp/detect.py')['get_detector']()"

# --- 実行用（uv なし） ---
FROM python:3.14-slim-trixie

RUN groupadd --system --gid 999 app \
    && useradd --system --gid 999 --uid 999 --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY --from=builder /opt/hf /opt/hf
COPY plate/ plate/
COPY app.py cli.py LICENSE README.md ./
COPY --from=web /web/dist web/dist

USER app
EXPOSE 7900
CMD ["python", "app.py", "--host", "0.0.0.0"]
