from __future__ import annotations

import sys
import types
import unittest

fake_numpy = types.ModuleType("numpy")
setattr(fake_numpy, "uint8", int)


class FakeArray:
    def __init__(self, shape, dtype=int) -> None:
        self.shape = shape
        self.dtype = dtype
        size = 1
        for dim in shape:
            size *= dim
        self.size = size


def fake_zeros(shape, dtype=int):
    return FakeArray(shape, dtype=dtype)


def fake_empty(shape, dtype=int):
    return FakeArray(shape, dtype=dtype)


setattr(fake_numpy, "zeros", fake_zeros)
setattr(fake_numpy, "empty", fake_empty)
setattr(fake_numpy, "ndarray", FakeArray)
sys.modules.setdefault("numpy", fake_numpy)

fake_gpu_processing = types.ModuleType("modules.gpu_processing")


def fake_gpu_resize(src, dsize, fx=0.0, fy=0.0, interpolation=None):
    if dsize and dsize != (0, 0):
        width, height = dsize
    else:
        width = max(1, int(src.shape[1] * fx))
        height = max(1, int(src.shape[0] * fy))
    channels = src.shape[2] if len(src.shape) > 2 else 1
    return FakeArray((height, width, channels), dtype=src.dtype)


setattr(fake_gpu_processing, "gpu_resize", fake_gpu_resize)
sys.modules.setdefault("modules.gpu_processing", fake_gpu_processing)

fake_cv2 = types.ModuleType("cv2")
setattr(fake_cv2, "IMREAD_COLOR", 1)
setattr(fake_cv2, "imdecode", lambda *args, **kwargs: None)
setattr(fake_cv2, "imencode", lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None)))
sys.modules.setdefault("cv2", fake_cv2)

from modules._image_sizing import fit_image_to_size  # noqa: E402


class FitImageToSizeTests(unittest.TestCase):
    def test_resizes_valid_frame_with_non_zero_result(self) -> None:
        image = fake_zeros((100, 200, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 50, 50)

        self.assertEqual(resized.shape[:2], (25, 50))

    def test_returns_fallback_frame_for_none_input(self) -> None:
        resized = fit_image_to_size(None, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)

    def test_returns_fallback_frame_for_empty_input(self) -> None:
        image = fake_empty((0, 0, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)

    def test_preserves_original_frame_when_target_size_is_invalid(self) -> None:
        image = fake_zeros((12, 18, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 0, 240)

        self.assertIs(resized, image)


if __name__ == "__main__":
    unittest.main()
