"""ナンバープレートの検出・四隅推定・文字消去のコア処理。

画像はすべて RGB の uint8 ndarray (H, W, 3) で扱う。
四隅 (quad) は (4, 2) float32 で、左上・右上・右下・左下の順。

処理の流れ: detect（bbox）→ fit（四隅。contour・checks・fallback を使う）→ erase（文字消去。hardware で残す部分を決める）。
pipeline はそれを1枚の写真にまとめて行う。
"""

from .checks import looks_like_plate, plate_colors_plausible
from .color import _to_lab
from .detect import detect_plates, get_detector
from .erase import erase_plate
from .fit import PlateFit, refine_quad
from .geometry import PLATE_ASPECT, box_to_quad, order_quad, quad_shape_problem
from .hardware import may_have_seal
from .image_io import IMAGE_EXTS, SAVE_FORMATS, WEBP_MAX_SIDE, draw_quads, load_image, save_image
from .pipeline import EDGE_FIT_MIN_SCORE, PlateCandidate, erase_all, find_plates

__all__ = [
    "EDGE_FIT_MIN_SCORE", "IMAGE_EXTS", "PLATE_ASPECT", "SAVE_FORMATS", "WEBP_MAX_SIDE",
    "PlateCandidate", "PlateFit",
    "_to_lab", "box_to_quad", "detect_plates", "draw_quads", "erase_all", "erase_plate", "find_plates", "get_detector",
    "load_image", "looks_like_plate", "may_have_seal", "order_quad", "plate_colors_plausible", "quad_shape_problem",
    "refine_quad", "save_image",
]
