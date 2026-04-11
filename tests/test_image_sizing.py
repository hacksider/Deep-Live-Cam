from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock

fake_numpy = types.ModuleType("numpy")
setattr(fake_numpy, "uint8", int)


class FakeArray:
    def __init__(self, shape, dtype=int, fill_value=0) -> None:
        self.shape = shape
        self.dtype = dtype
        self.fill_value = fill_value
        size = 1
        for dim in shape:
            size *= dim
        self.size = size


def fake_zeros(shape, dtype=int):
    return FakeArray(shape, dtype=dtype, fill_value=0)


def fake_empty(shape, dtype=int):
    return FakeArray(shape, dtype=dtype, fill_value=0)


setattr(fake_numpy, "zeros", fake_zeros)
setattr(fake_numpy, "empty", fake_empty)
setattr(fake_numpy, "ndarray", FakeArray)

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

fake_cv2 = types.ModuleType("cv2")
setattr(fake_cv2, "IMREAD_COLOR", 1)
setattr(fake_cv2, "imdecode", lambda *args, **kwargs: None)
setattr(fake_cv2, "imencode", lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None)))
with mock.patch.dict(
    sys.modules,
    {
        "numpy": fake_numpy,
        "modules.gpu_processing": fake_gpu_processing,
        "cv2": fake_cv2,
    },
    clear=False,
):
    fit_image_to_size = importlib.import_module("modules._image_sizing").fit_image_to_size


class FitImageToSizeTests(unittest.TestCase):
    def test_resizes_valid_frame_with_non_zero_result(self) -> None:
        image = fake_zeros((100, 200, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 50, 50)

        self.assertEqual(resized.shape[:2], (25, 50))

    def test_returns_fallback_frame_for_none_input(self) -> None:
        resized = fit_image_to_size(None, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)
        self.assertEqual(resized.fill_value, 0)

    def test_returns_fallback_frame_for_empty_input(self) -> None:
        image = fake_empty((0, 0, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)
        self.assertEqual(resized.fill_value, 0)

    def test_returns_fallback_frame_for_1d_input(self) -> None:
        image = fake_empty((10,), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)
        self.assertEqual(resized.fill_value, 0)

    def test_returns_fallback_frame_for_scalar_input(self) -> None:
        image = fake_empty((), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 320, 240)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)
        self.assertEqual(resized.fill_value, 0)

    def test_returns_fallback_frame_when_size_is_not_provided_and_image_is_none(self) -> None:
        resized = fit_image_to_size(None, None, None)

        self.assertEqual(resized.shape, (360, 640, 3))
        self.assertEqual(resized.dtype, fake_numpy.uint8)
        self.assertEqual(resized.fill_value, 0)

    def test_preserves_original_frame_when_target_size_is_invalid(self) -> None:
        image = fake_zeros((12, 18, 3), dtype=fake_numpy.uint8)

        resized = fit_image_to_size(image, 0, 240)

        self.assertIs(resized, image)


if __name__ == "__main__":
    unittest.main()
