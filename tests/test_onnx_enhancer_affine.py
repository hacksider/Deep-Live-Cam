from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_onnx_enhancer_module():
    fake_numpy = types.ModuleType("numpy")

    class FakeArray:
        def __init__(self, value):
            self.value = value
            self.ndim, self.shape = self._shape(value)

        @staticmethod
        def _shape(value):
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return 2, (len(value), len(value[0]))
                return 1, (len(value),)
            return 0, ()

        def __mul__(self, _other):
            return self

    def fake_array(value, dtype=None):
        return FakeArray(value)

    def fake_asarray(value, dtype=None):
        if not isinstance(value, list):
            raise TypeError("only lists are supported by this test fake")
        return FakeArray(value)

    fake_numpy.ndarray = FakeArray
    fake_numpy.float32 = "float32"
    fake_numpy.uint8 = int
    fake_numpy.array = fake_array
    fake_numpy.asarray = fake_asarray
    fake_numpy.fromfile = lambda *args, **kwargs: b""

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.LMEDS = 4
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (
        True,
        types.SimpleNamespace(tofile=lambda *_args, **_kwargs: None),
    )
    fake_cv2.estimateAffinePartial2D = mock.Mock(
        side_effect=AssertionError("malformed landmarks should not reach cv2"),
    )
    fake_cv2.invertAffineTransform = mock.Mock()

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.InferenceSession = object
    fake_onnxruntime.SessionOptions = lambda: types.SimpleNamespace(
        graph_optimization_level=None,
    )
    fake_onnxruntime.GraphOptimizationLevel = types.SimpleNamespace(
        ORT_ENABLE_ALL="ORT_ENABLE_ALL",
    )
    fake_onnxruntime.OrtValue = types.SimpleNamespace(
        ortvalue_from_numpy=lambda *args, **kwargs: object(),
    )

    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_providers = []

    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "onnxruntime": fake_onnxruntime,
            "modules.globals": fake_globals,
        },
        clear=False,
    ):
        sys.modules.pop("modules.processors.frame._onnx_enhancer", None)
        return importlib.import_module("modules.processors.frame._onnx_enhancer")


class GetFaceAffineLandmarkTests(unittest.TestCase):
    @staticmethod
    def _face_with_landmarks(landmarks):
        return types.SimpleNamespace(kps=None, landmark_2d_106=landmarks)

    def test_short_landmark_106_returns_empty_transform(self) -> None:
        onnx_enhancer = _load_onnx_enhancer_module()

        transform, inverse = onnx_enhancer._get_face_affine(
            self._face_with_landmarks([[0, 0]] * 88),
            512,
        )

        self.assertIsNone(transform)
        self.assertIsNone(inverse)

    def test_malformed_landmark_106_returns_empty_transform(self) -> None:
        onnx_enhancer = _load_onnx_enhancer_module()

        transform, inverse = onnx_enhancer._get_face_affine(
            self._face_with_landmarks([0] * 106),
            512,
        )

        self.assertIsNone(transform)
        self.assertIsNone(inverse)

    def test_unconvertible_landmark_106_returns_empty_transform(self) -> None:
        onnx_enhancer = _load_onnx_enhancer_module()

        transform, inverse = onnx_enhancer._get_face_affine(
            self._face_with_landmarks(object()),
            512,
        )

        self.assertIsNone(transform)
        self.assertIsNone(inverse)


if __name__ == "__main__":
    unittest.main()
