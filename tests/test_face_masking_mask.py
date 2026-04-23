from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_face_masking_module():
    fake_numpy = types.ModuleType("numpy")

    class FakeArray:
        def __init__(self, shape, dtype=int) -> None:
            self.shape = shape
            self.dtype = dtype

    def fake_zeros(shape, dtype=int):
        return FakeArray(shape, dtype=dtype)

    fake_numpy.ndarray = FakeArray
    fake_numpy.zeros = fake_zeros
    fake_numpy.uint8 = int

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (
        True,
        types.SimpleNamespace(tofile=lambda *_args: None),
    )

    fake_globals = types.ModuleType("modules.globals")

    fake_gpu = types.ModuleType("modules.gpu_processing")
    fake_gpu.gpu_gaussian_blur = lambda img, *args, **kwargs: img
    fake_gpu.gpu_resize = lambda img, *args, **kwargs: img
    fake_gpu.gpu_cvt_color = lambda img, *args, **kwargs: img

    fake_typing = types.ModuleType("modules.typing")
    fake_typing.Face = object
    fake_typing.Frame = object

    sys.modules.pop("modules.processors.frame.face_masking", None)
    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "modules.globals": fake_globals,
            "modules.gpu_processing": fake_gpu,
            "modules.typing": fake_typing,
        },
        clear=False,
    ):
        return importlib.import_module("modules.processors.frame.face_masking")


class CreateFaceMaskTests(unittest.TestCase):
    @staticmethod
    def _dummy_face():
        class DummyFace:
            landmark_2d_106 = []

        return DummyFace()

    def assert_empty_mask(self, face_masking, frame) -> None:
        result = face_masking.create_face_mask(self._dummy_face(), frame)
        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(result.dtype, face_masking.np.uint8)

    def test_create_face_mask_returns_empty_mask_when_frame_is_none(self) -> None:
        face_masking = _load_face_masking_module()
        self.assert_empty_mask(face_masking, None)

    def test_create_face_mask_returns_empty_mask_when_frame_has_no_shape(self) -> None:
        face_masking = _load_face_masking_module()
        self.assert_empty_mask(face_masking, 123)

    def test_create_face_mask_returns_empty_mask_when_frame_is_1d(self) -> None:
        face_masking = _load_face_masking_module()
        one_dimensional_frame = types.SimpleNamespace(shape=(10,))
        self.assert_empty_mask(face_masking, one_dimensional_frame)


if __name__ == "__main__":
    unittest.main()
