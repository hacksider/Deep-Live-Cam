"""Model-loading regressions; no weights, camera, or GPU are needed.

The real ONNX optimizer parses a deliberately corrupt file for the failure case.
InsightFace inference and unrelated application services are stubbed at import.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import onnx

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _loading_environment():
    analyser = types.ModuleType("modules.face_analyser")
    for name in (
        "get_face_analyser", "get_one_face", "get_many_faces", "default_source_face",
        "add_blank_map", "detect_many_faces_fast", "detect_one_face_fast",
        "ensure_landmarks", "get_unique_faces_from_target_image",
        "get_unique_faces_from_target_video", "has_valid_map", "reset_face_analyser",
        "simplify_maps",
    ):
        setattr(analyser, name, Mock())
    get_model = Mock()
    status = Mock()
    stubs = {
        "insightface": types.SimpleNamespace(
            model_zoo=types.SimpleNamespace(get_model=get_model)
        ),
        "torch": None,
        "modules.core": types.SimpleNamespace(update_status=status),
        "modules.face_analyser": analyser,
        "modules.typing": types.SimpleNamespace(Frame=np.ndarray, Face=object),
        "modules.cluster_analysis": types.SimpleNamespace(find_closest_centroid=Mock()),
    }
    with patch.dict(sys.modules, stubs):
        import modules.globals
        swapper = _module_from_file(
            "_test_swapper_loading", ROOT / "modules/processors/frame/face_swapper.py"
        )
        with patch.dict(sys.modules, {"modules.processors.frame.face_swapper": swapper}), \
             patch.object(modules.globals, "execution_providers", ["CoreMLExecutionProvider"]):
            swapper.IS_APPLE_SILICON = True
            yield swapper, get_model, status


class FaceSwapperLoadingTests(unittest.TestCase):
    def test_corrupt_onnx_is_reported_without_escaping_the_loader(self):
        import modules.onnx_optimize as optimizer
        with _loading_environment() as (swapper, get_model, status), \
             tempfile.TemporaryDirectory() as directory, \
             patch.object(optimizer, "IS_APPLE_SILICON", True):
            swapper.models_dir = directory
            model_path = Path(directory) / "inswapper_128.onnx"
            model_path.write_bytes(b"not an ONNX model")

            self.assertIsNone(swapper.get_face_swapper())
            self.assertIsNone(swapper.FACE_SWAPPER)
            get_model.assert_not_called()
            self.assertIn("Error loading face swapper model", status.call_args.args[0])

    def test_replacing_corrupt_model_allows_a_later_retry(self):
        import modules.onnx_optimize as optimizer
        with _loading_environment() as (swapper, get_model, _status), \
             tempfile.TemporaryDirectory() as directory, \
             patch.object(optimizer, "IS_APPLE_SILICON", True):
            swapper.models_dir = directory
            model_path = Path(directory) / "inswapper_128.onnx"
            model_path.write_bytes(b"not an ONNX model")
            self.assertIsNone(swapper.get_face_swapper())

            graph = onnx.helper.make_graph([], "empty", [], [])
            onnx.save(onnx.helper.make_model(graph), model_path)
            model = get_model.return_value
            self.assertIs(swapper.get_face_swapper(), model)
            self.assertIs(swapper.get_face_swapper(), model)
            get_model.assert_called_once()

    def test_successful_coreml_load_uses_optimized_path_and_caches_model(self):
        with _loading_environment() as (swapper, get_model, _status), \
             patch.object(swapper.os.path, "exists", return_value=True), \
             patch("modules.onnx_optimize.optimize_for_coreml", return_value="optimized.onnx") as optimize:
            self.assertIs(swapper.get_face_swapper(), get_model.return_value)
            self.assertIs(swapper.get_face_swapper(), get_model.return_value)
            optimize.assert_called_once()
            self.assertEqual(get_model.call_args.args, ("optimized.onnx",))
            get_model.assert_called_once()

    def test_cpu_load_skips_coreml_preparation(self):
        with _loading_environment() as (swapper, get_model, _status), \
             patch.object(swapper.os.path, "exists", return_value=True), \
             patch("modules.onnx_optimize.optimize_for_coreml") as optimize:
            swapper.IS_APPLE_SILICON = False
            with patch.object(swapper.modules.globals, "execution_providers", ["CPUExecutionProvider"]):
                self.assertIs(swapper.get_face_swapper(), get_model.return_value)
            optimize.assert_not_called()
            self.assertEqual(get_model.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_inference_loader_failure_keeps_model_unset(self):
        with _loading_environment() as (swapper, get_model, status), \
             patch.object(swapper.os.path, "exists", return_value=True), \
             patch("modules.onnx_optimize.optimize_for_coreml", return_value="optimized.onnx"):
            get_model.side_effect = RuntimeError("session creation failed")
            self.assertIsNone(swapper.get_face_swapper())
            self.assertIsNone(swapper.FACE_SWAPPER)
            self.assertIn("session creation failed", status.call_args.args[0])


class LiveModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _check_live_start(self, model):
        with _loading_environment() as (swapper, _get_model, _status):
            ui = _module_from_file("_test_ui_model_loading", ROOT / "modules/ui.py")
            window = types.SimpleNamespace(
                cb_camera=types.SimpleNamespace(currentIndex=lambda: 0),
                _camera_indices=[3],
            )
            with patch.object(swapper, "get_face_swapper", return_value=model), \
                 patch.object(ui.modules.globals, "map_faces", False), \
                 patch.object(ui.modules.globals, "source_path", "source.jpg"), \
                 patch.object(ui, "_open_webcam_preview") as open_preview:
                ui.MainWindow._on_live(window)
                return open_preview.call_args_list

    def test_failed_model_load_does_not_open_preview(self):
        self.assertEqual(self._check_live_start(None), [])

    def test_successful_model_load_opens_selected_camera(self):
        calls = self._check_live_start(object())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, (3,))


if __name__ == "__main__":
    unittest.main()
