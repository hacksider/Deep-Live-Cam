import tempfile
import types
import unittest
from pathlib import Path
import sys


fake_cv2 = types.ModuleType("cv2")
fake_cv2.__dict__["IMREAD_COLOR"] = 1
_ = sys.modules.setdefault("cv2", fake_cv2)

fake_numpy = types.ModuleType("numpy")
fake_numpy.__dict__["uint8"] = object()
_ = sys.modules.setdefault("numpy", fake_numpy)

from modules.processors.frame._face_swapper_model import resolve_face_swapper_model_path


class FaceSwapperModelPathTests(unittest.TestCase):
    def test_prefers_fp32_model_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "inswapper_128.onnx").write_text("fp32")
            (model_dir / "inswapper_128_fp16.onnx").write_text("fp16")

            result = resolve_face_swapper_model_path(model_dir)

            self.assertEqual(result, str(model_dir / "inswapper_128.onnx"))

    def test_falls_back_to_fp16_model_when_fp32_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "inswapper_128_fp16.onnx").write_text("fp16")

            result = resolve_face_swapper_model_path(str(model_dir))

            self.assertEqual(result, str(model_dir / "inswapper_128_fp16.onnx"))

    def test_returns_default_fp32_target_when_no_model_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            result = resolve_face_swapper_model_path(str(model_dir))

            self.assertEqual(result, str(model_dir / "inswapper_128.onnx"))


if __name__ == "__main__":
    unittest.main()
