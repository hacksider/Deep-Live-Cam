import tempfile
import unittest
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "processors"
    / "frame"
    / "_face_swapper_model.py"
)
SPEC = importlib.util.spec_from_file_location("_face_swapper_model", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
resolve_face_swapper_model_path = MODULE.resolve_face_swapper_model_path


class FaceSwapperModelPathTests(unittest.TestCase):
    def test_prefers_fp32_model_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "inswapper_128.onnx").write_text("fp32")
            (model_dir / "inswapper_128_fp16.onnx").write_text("fp16")

            result = resolve_face_swapper_model_path(str(model_dir))

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
