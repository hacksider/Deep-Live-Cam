from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from modules.gpu_processing import gpu_resize


def _fallback_frame(width: int, height: int) -> np.ndarray:
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    return np.zeros((safe_height, safe_width, 3), dtype=np.uint8)


def fit_image_to_size(
    image: Optional[np.ndarray],
    width: Optional[int],
    height: Optional[int],
    fallback_size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    if width is None and height is None:
        return image

    target_width = int(width) if width is not None else int(fallback_size[0])
    target_height = int(height) if height is not None else int(fallback_size[1])

    if image is None or not hasattr(image, "shape"):
        return _fallback_frame(target_width, target_height)

    if image.size == 0 or len(image.shape) < 2:
        return _fallback_frame(target_width, target_height)

    if target_width <= 0 or target_height <= 0:
        return image

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return _fallback_frame(target_width, target_height)

    ratio_h = 0.0
    ratio_w = 0.0
    if target_width > target_height:
        ratio_h = target_height / h
    else:
        ratio_w = target_width / w
    ratio = max(ratio_w, ratio_h)
    if ratio <= 0:
        return image

    new_size = (max(1, int(ratio * w)), max(1, int(ratio * h)))
    return gpu_resize(image, dsize=new_size)
