"""plate パッケージの単体テスト。合成画像だけを使うので、サンプル写真や検出モデルは不要。"""

import cv2
import numpy as np
import pytest
from PIL import Image

import plate
from synth import BOLT_L, BOLT_R, FRONT, GREEN, OBLIQUE, PLATE_H, PLATE_W, RIM, SEAL_L, WHITE, WHITE_TEXT, YELLOW, bbox_of, plate_image, rectify, scene


def corner_error(a, b) -> float:
    return float(np.max(np.linalg.norm(plate.order_quad(a) - plate.order_quad(b), axis=1)))


# ---------------------------------------------------------------------------
# 幾何・判定の小さな関数
# ---------------------------------------------------------------------------


def test_order_quad_starts_top_left_clockwise():
    pts = np.float32([[10, 50], [100, 0], [0, 0], [110, 60]])
    assert plate.order_quad(pts).tolist() == [[0, 0], [100, 0], [110, 60], [10, 50]]


def test_quad_shape_problem_accepts_perspective_rectangle():
    assert plate.quad_shape_problem(np.float32(OBLIQUE)) == ""


@pytest.mark.parametrize(
    "quad",
    [
        [[0, 0], [100, 0], [100, 50], [95, 50]],  # 三角形に近い
        [[0, 0], [100, 0], [140, 50], [0, 50]],  # 向かい合う辺が平行でない
    ],
)
def test_quad_shape_problem_rejects_non_rectangles(quad):
    assert plate.quad_shape_problem(np.float32(quad)) != ""


def _lab(rgb):
    return plate._to_lab(np.uint8([[rgb]]))[0, 0]


@pytest.mark.parametrize(
    "bg, text, expected",
    [
        (WHITE, (20, 20, 20), True),
        (YELLOW, (20, 20, 20), True),
        (GREEN, WHITE_TEXT, True),  # 事業用（緑地に白文字）
        ((15, 15, 15), (240, 200, 40), True),  # 軽の事業用（黒地に黄文字）
        ((15, 15, 15), (200, 200, 200), False),  # 黒いグリルに明るい模様
    ],
)
def test_plate_colors_plausible(bg, text, expected):
    inner = np.array([_lab(bg)] * 80 + [_lab(text)] * 20)
    assert plate.plate_colors_plausible(_lab(bg), inner) is expected


# ---------------------------------------------------------------------------
# 四隅推定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quad, bg, text",
    [
        (FRONT, WHITE, (25, 50, 35)),
        (OBLIQUE, WHITE, (25, 50, 35)),
        (FRONT, YELLOW, (20, 20, 20)),
        (OBLIQUE, GREEN, WHITE_TEXT),
    ],
    ids=["white-front", "white-oblique", "yellow", "green-oblique"],
)
def test_refine_quad_finds_corners(quad, bg, text):
    img, _ = scene(quad, plate_image(bg, text))
    fit = plate.refine_quad(img, bbox_of(quad))
    assert fit.ok, fit.reason
    assert corner_error(fit.quad, quad) < 3.0


def test_refine_quad_recovers_edge_cut_by_shadow():
    # 上部に濃い影が落ちていても、四角形は影の部分まで含む
    quad = FRONT
    img, _ = scene(quad, plate_image(shadow=0.6))
    fit = plate.refine_quad(img, bbox_of(quad))
    assert fit.ok, fit.reason
    assert corner_error(fit.quad, quad) < 4.0


def test_refine_quad_rejects_area_without_plate():
    img, _ = scene(FRONT, np.full((165, 330, 3), (30, 32, 36), np.uint8))
    assert not plate.refine_quad(img, bbox_of(FRONT)).ok


# ---------------------------------------------------------------------------
# 文字消去
# ---------------------------------------------------------------------------


def _inner(rect, inset=0.08):
    h, w = rect.shape[:2]
    m = int(h * inset)
    return rect[m:h - m, m:w - m]


@pytest.mark.parametrize("quad", [FRONT, OBLIQUE], ids=["front", "oblique"])
@pytest.mark.parametrize("bg, text", [(WHITE, (25, 50, 35)), (YELLOW, (20, 20, 20)), (GREEN, WHITE_TEXT)], ids=["white", "yellow", "green"])
def test_erase_plate_removes_text(quad, bg, text):
    img, _ = scene(quad, plate_image(bg, text))
    out = plate.erase_plate(img, np.float32(quad))
    inner = _inner(rectify(out, quad)).astype(np.float32)
    # 文字が残っていれば、地色から大きく外れた画素が出る
    dev = np.abs(inner - np.array(bg, np.float32)).max(axis=-1)
    assert np.percentile(dev, 99.9) < 30


def test_erase_plate_does_not_touch_outside():
    img, mask = scene(OBLIQUE)
    out = plate.erase_plate(img, np.float32(OBLIQUE))
    near = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert np.array_equal(out[~near], img[~near])


def test_erase_plate_keeps_shadow():
    shadow = 0.6
    img, _ = scene(FRONT, plate_image(shadow=shadow))
    out = rectify(plate.erase_plate(img, np.float32(FRONT)), FRONT).astype(np.float32)
    h, w = out.shape[:2]
    band = out[int(h * 0.06):int(h * 0.16), int(w * 0.1):int(w * 0.9)].mean()
    lit = out[int(h * 0.4):int(h * 0.9), int(w * 0.1):int(w * 0.9)].mean()
    # 影の帯は明るい部分のおよそ (1 - shadow) 倍の明るさで残る
    assert band / lit == pytest.approx(1 - shadow, abs=0.12)


def test_erase_plate_adds_no_shadow_to_evenly_lit_plate():
    img, _ = scene(FRONT)
    out = rectify(plate.erase_plate(img, np.float32(FRONT)), FRONT).astype(np.float32)
    rows = _inner(out).mean(axis=(1, 2))
    assert rows.min() / rows.max() > 0.95


@pytest.mark.parametrize("quad", [FRONT, OBLIQUE], ids=["front", "oblique"])
def test_erase_plate_keeps_bolts_seal_and_rim(quad):
    img, _ = scene(quad, plate_image(hardware=True))
    before = rectify(img, quad).astype(np.float32)
    after = rectify(plate.erase_plate(img, np.float32(quad)), quad).astype(np.float32)

    def diff(x, y, r):
        return float(np.abs(after[y - r:y + r + 1, x - r:x + r + 1] - before[y - r:y + r + 1, x - r:x + r + 1]).mean())

    # 黒いボルト、金属の封印（明るい中心と暗い輪）はそのまま残る
    assert diff(*BOLT_R, 3) < 12
    assert diff(*SEAL_L, 9) < 12
    # 縁のエンボスの線も残る（上辺・下辺の中央部）
    cols = slice(int(PLATE_W * 0.4), int(PLATE_W * 0.6))
    for y in (RIM, PLATE_H - 1 - RIM):
        assert float(np.abs(after[y - 1:y + 2, cols] - before[y - 1:y + 2, cols]).mean()) < 15
    # 文字は消える
    dev = np.abs(after[int(PLATE_H * 0.55):int(PLATE_H * 0.85), int(PLATE_W * 0.12):int(PLATE_W * 0.88)] - np.array(WHITE, np.float32)).max(axis=-1)
    assert np.percentile(dev, 99.5) < 30


def test_erase_plate_paints_over_lone_bolt():
    # 片方しかボルトが無ければ、無理に残さず塗る
    img, _ = scene(FRONT, plate_image(lone_bolt=True))
    after = rectify(plate.erase_plate(img, np.float32(FRONT)), FRONT).astype(np.float32)
    x, y = BOLT_L
    patch = after[y - 3:y + 4, x - 3:x + 4]
    assert float(np.abs(patch - np.array(WHITE, np.float32)).max(axis=-1).mean()) < 20


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


def _exif_with_model(model: str) -> bytes:
    exif = Image.Exif()
    exif[0x0110] = model
    return exif.tobytes()


@pytest.mark.parametrize("ext, fmt", [(".jpg", "JPEG"), (".webp", "WEBP")])
def test_save_image_keeps_size_and_exif(tmp_path, ext, fmt):
    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    path = tmp_path / f"out{ext}"
    plate.save_image(str(path), img, _exif_with_model("TEST-CAM"))
    with Image.open(path) as im:
        assert im.format == fmt
        assert im.size == (200, 120)
        assert im.getexif().get(0x0110) == "TEST-CAM"


def test_save_image_rejects_webp_larger_than_spec(tmp_path):
    img = np.zeros((10, plate.WEBP_MAX_SIDE + 1, 3), np.uint8)
    with pytest.raises(ValueError):
        plate.save_image(str(tmp_path / "big.webp"), img)


@pytest.mark.parametrize(
    "bg, expected",
    [(WHITE, True), (GREEN, True), (YELLOW, False), ((15, 15, 15), False)],
    ids=["white", "green", "yellow", "black"],
)
def test_may_have_seal_only_for_registered_vehicle_colors(bg, expected):
    assert plate.may_have_seal(_lab(bg)) is expected
