"""手元のサンプル写真での回帰チェック。

    uv run pytest -m samples                     # 基準と比べる
    uv run pytest -m samples --update-baseline   # 今の結果を基準として保存する

写真（sample_picture/）と基準（.regression/baseline.json）は他人の車が写るのでリポジトリに入れない。
どちらかが無ければスキップする。処理したプレートの数と、各プレートの四隅の位置を比べる。
"""

import glob
import json
import os

import numpy as np
import pytest

import plate

SAMPLE_DIR = "sample_picture"
BASELINE = os.path.join(".regression", "baseline.json")
EXTS = plate.IMAGE_EXTS

pytestmark = pytest.mark.samples

SAMPLES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SAMPLE_DIR, "*")) if p.lower().endswith(EXTS))


def _quad_tolerance(q: np.ndarray) -> float:
    # プレート幅の 3%（最低 3px）までのずれは同じ結果とみなす
    return max(3.0, 0.03 * float(np.linalg.norm(q[1] - q[0])))


def _match(expected: list, actual: list) -> tuple[list, list, list]:
    """基準と現在の四隅を対応付け、(ずれたもの, 基準にしか無いもの, 現在にしか無いもの) を返す。"""
    remaining = [np.float32(q) for q in actual]
    unmatched = []
    for q in map(np.float32, expected):
        errs = [float(np.max(np.linalg.norm(q - a, axis=1))) for a in remaining]
        i = int(np.argmin(errs)) if errs else -1
        if i >= 0 and errs[i] <= _quad_tolerance(q):
            remaining.pop(i)
        else:
            unmatched.append(q)
    # 許容範囲は超えたが、同じプレートの位置がずれただけのもの（中心がプレート幅の半分以内）
    moved, lost = [], []
    for q in unmatched:
        width = float(np.linalg.norm(q[1] - q[0]))
        dists = [float(np.linalg.norm(q.mean(axis=0) - a.mean(axis=0))) for a in remaining]
        i = int(np.argmin(dists)) if dists else -1
        if i >= 0 and dists[i] <= 0.5 * width:
            a = remaining.pop(i)
            moved.append((q, float(np.max(np.linalg.norm(q - a, axis=1)))))
        else:
            lost.append(q)
    return moved, lost, remaining


@pytest.fixture(scope="module")
def baseline(request):
    if request.config.getoption("--update-baseline"):
        return {}
    if not os.path.exists(BASELINE):
        pytest.skip(f"{BASELINE} がありません。uv run pytest -m samples --update-baseline で作成してください")
    with open(BASELINE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def updated(request):
    data = {}
    yield data
    if request.config.getoption("--update-baseline") and data:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        old = {}
        if os.path.exists(BASELINE):
            with open(BASELINE, encoding="utf-8") as f:
                old = json.load(f)
        old.update(data)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=1)


@pytest.mark.skipif(not SAMPLES, reason=f"{SAMPLE_DIR}/ にサンプル写真がありません")
@pytest.mark.parametrize("name", SAMPLES or ["-"])
def test_sample_matches_baseline(name, baseline, updated, request):
    img, _ = plate.load_image(os.path.join(SAMPLE_DIR, name))
    quads = [plate.order_quad(c.quad) for c in plate.find_plates(img) if c.ok]

    if request.config.getoption("--update-baseline"):
        updated[name] = [np.round(q, 1).tolist() for q in quads]
        return
    if name not in baseline:
        pytest.skip("基準に含まれていない写真です（--update-baseline で追加できます）")

    moved, lost, new = _match(baseline[name], quads)

    def where(q):
        c = q.mean(axis=0)
        return f"中心 ({c[0]:.0f}, {c[1]:.0f})"

    msgs = [f"プレート {where(q)} の四隅が最大 {err:.1f}px ずれた（許容 {_quad_tolerance(q):.1f}px）" for q, err in moved]
    msgs += [f"基準で処理していたプレート {where(q)} が処理されなくなった" for q in lost]
    msgs += [f"新たに処理されたプレート {where(q)}" for q in new]
    assert not msgs, f"{name}: 処理 {len(baseline[name])} 件 → {len(quads)} 件\n  " + "\n  ".join(msgs)
