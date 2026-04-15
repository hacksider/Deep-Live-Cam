from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock

def _load_face_swapper_module():
    fake_numpy = types.ModuleType("numpy")

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

    setattr(fake_numpy, "ndarray", FakeArray)
    setattr(fake_numpy, "zeros", fake_zeros)
    setattr(fake_numpy, "uint8", int)

    fake_cv2 = types.ModuleType("cv2")
    setattr(fake_cv2, "IMREAD_COLOR", 1)
    setattr(fake_cv2, "imdecode", lambda *args, **kwargs: None)
    setattr(
        fake_cv2,
        "imencode",
        lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None)),
    )
    fake_insightface = types.ModuleType("insightface")

    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_providers = []
    fake_globals.face_mask_blur = 31

    fake_core = types.ModuleType("modules.processors.frame.core")
    fake_core.process_video = lambda *args, **kwargs: None

    fake_core_module = types.ModuleType("modules.core")
    fake_core_module.update_status = lambda *args, **kwargs: None

    fake_face_analyser = types.ModuleType("modules.face_analyser")
    fake_face_analyser.get_one_face = lambda *args, **kwargs: None
    fake_face_analyser.get_many_faces = lambda *args, **kwargs: []
    fake_face_analyser.default_source_face = lambda *args, **kwargs: None

    fake_typing = types.ModuleType("modules.typing")
    fake_typing.Face = object
    fake_typing.Frame = object

    fake_utilities = types.ModuleType("modules.utilities")
    fake_utilities.conditional_download = lambda *args, **kwargs: None
    fake_utilities.is_image = lambda *args, **kwargs: False
    fake_utilities.is_video = lambda *args, **kwargs: False

    fake_cluster = types.ModuleType("modules.cluster_analysis")
    fake_cluster.find_closest_centroid = lambda *args, **kwargs: None

    fake_gpu = types.ModuleType("modules.gpu_processing")
    fake_gpu.gpu_gaussian_blur = lambda img, *args, **kwargs: img
    fake_gpu.gpu_sharpen = lambda img, *args, **kwargs: img
    fake_gpu.gpu_add_weighted = lambda src1, *args, **kwargs: src1
    fake_gpu.gpu_resize = lambda img, *args, **kwargs: img
    fake_gpu.gpu_cvt_color = lambda img, *args, **kwargs: img

    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "insightface": fake_insightface,
            "numpy": fake_numpy,
            "modules.globals": fake_globals,
            "modules.processors.frame.core": fake_core,
            "modules.core": fake_core_module,
            "modules.face_analyser": fake_face_analyser,
            "modules.typing": fake_typing,
            "modules.utilities": fake_utilities,
            "modules.cluster_analysis": fake_cluster,
            "modules.gpu_processing": fake_gpu,
        },
        clear=False,
    ):
        return importlib.import_module("modules.processors.frame.face_swapper")


class CreateFaceMaskTests(unittest.TestCase):
    @staticmethod
    def _dummy_face():
        class DummyFace:
            landmark_2d_106 = []

        return DummyFace()

    def assert_empty_mask(self, face_swapper, frame) -> None:
        result = face_swapper.create_face_mask(self._dummy_face(), frame)
        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(result.dtype, face_swapper.np.uint8)

    def test_create_face_mask_returns_empty_mask_when_frame_is_none(self) -> None:
        face_swapper = _load_face_swapper_module()
        self.assert_empty_mask(face_swapper, None)

    def test_create_face_mask_returns_empty_mask_when_frame_has_no_shape(self) -> None:
        face_swapper = _load_face_swapper_module()
        self.assert_empty_mask(face_swapper, 123)

    def test_create_face_mask_returns_empty_mask_when_frame_is_1d(self) -> None:
        face_swapper = _load_face_swapper_module()
        one_dimensional_frame = types.SimpleNamespace(shape=(10,))
        self.assert_empty_mask(face_swapper, one_dimensional_frame)


if __name__ == "__main__":
    unittest.main()
