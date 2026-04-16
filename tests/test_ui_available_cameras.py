from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


def _call_get_available_cameras(devices):
    video_capture_calls = []

    fake_ctk = DummyModule("customtkinter")
    fake_ctk.__path__ = []
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (
        True,
        types.SimpleNamespace(tofile=lambda *_args: None),
    )

    def fake_video_capture(index):
        video_capture_calls.append(index)
        raise AssertionError("VideoCapture should not be probed when no Windows cameras are found")

    fake_cv2.VideoCapture = fake_video_capture

    fake_platform = types.ModuleType("platform")
    fake_platform.system = lambda: "Windows"

    fake_graph_module = types.ModuleType("pygrabber.dshow_graph")

    class FakeFilterGraph:
        def get_input_devices(self):
            return devices

    fake_graph_module.FilterGraph = FakeFilterGraph

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.ModuleType("PIL.Image")
    fake_pil.ImageOps = types.ModuleType("PIL.ImageOps")

    fake_globals = types.ModuleType("modules.globals")
    fake_globals.file_types = (("Images", "*.png"), ("Videos", "*.mp4"))
    fake_metadata = types.ModuleType("modules.metadata")

    fake_gpu = types.ModuleType("modules.gpu_processing")
    fake_gpu.gpu_cvt_color = lambda *args, **kwargs: None
    fake_gpu.gpu_resize = lambda *args, **kwargs: None
    fake_gpu.gpu_flip = lambda *args, **kwargs: None

    fake_face_analyser = types.ModuleType("modules.face_analyser")
    fake_face_analyser.get_one_face = lambda *args, **kwargs: None
    fake_face_analyser.get_many_faces = lambda *args, **kwargs: []
    fake_face_analyser.get_unique_faces_from_target_image = lambda *args, **kwargs: None
    fake_face_analyser.get_unique_faces_from_target_video = lambda *args, **kwargs: None
    fake_face_analyser.add_blank_map = lambda *args, **kwargs: None
    fake_face_analyser.has_valid_map = lambda *args, **kwargs: False
    fake_face_analyser.simplify_maps = lambda *args, **kwargs: None

    fake_capturer = types.ModuleType("modules.capturer")
    fake_capturer.get_video_frame = lambda *args, **kwargs: None
    fake_capturer.get_video_frame_total = lambda *args, **kwargs: 0

    fake_core = types.ModuleType("modules.processors.frame.core")
    fake_core.get_frame_processors_modules = lambda *args, **kwargs: []

    fake_utilities = types.ModuleType("modules.utilities")
    fake_utilities.is_image = lambda *args, **kwargs: False
    fake_utilities.is_video = lambda *args, **kwargs: False
    fake_utilities.resolve_relative_path = lambda path: path
    fake_utilities.has_image_extension = lambda *args, **kwargs: False

    fake_video_capture_module = types.ModuleType("modules.video_capture")
    fake_video_capture_module.VideoCapturer = object

    fake_gettext = types.ModuleType("modules.gettext")
    fake_gettext.LanguageManager = object

    fake_tooltip = types.ModuleType("modules.ui_tooltip")
    fake_tooltip.ToolTip = object

    sys.modules.pop("modules.ui", None)
    with mock.patch.dict(
        sys.modules,
        {
            "customtkinter": fake_ctk,
            "cv2": fake_cv2,
            "numpy": DummyModule("numpy"),
            "requests": types.ModuleType("requests"),
            "PIL": fake_pil,
            "PIL.Image": fake_pil.Image,
            "PIL.ImageOps": fake_pil.ImageOps,
            "platform": fake_platform,
            "pygrabber.dshow_graph": fake_graph_module,
            "modules.globals": fake_globals,
            "modules.metadata": fake_metadata,
            "modules.gpu_processing": fake_gpu,
            "modules.face_analyser": fake_face_analyser,
            "modules.capturer": fake_capturer,
            "modules.processors.frame.core": fake_core,
            "modules.utilities": fake_utilities,
            "modules.video_capture": fake_video_capture_module,
            "modules.gettext": fake_gettext,
            "modules.ui_tooltip": fake_tooltip,
        },
        clear=False,
    ):
        import modules

        modules.globals = fake_globals
        ui = importlib.import_module("modules.ui")
        return ui.get_available_cameras(), video_capture_calls


class GetAvailableCamerasTests(unittest.TestCase):
    def test_windows_no_directshow_devices_does_not_probe_opencv_fallback(self) -> None:
        result, video_capture_calls = _call_get_available_cameras([])

        self.assertEqual(result, ([], ["No cameras found"]))
        self.assertEqual(video_capture_calls, [])

    def test_windows_directshow_devices_are_returned_without_opencv_probe(self) -> None:
        result, video_capture_calls = _call_get_available_cameras(["Integrated Camera", "USB Camera"])

        self.assertEqual(result, ([0, 1], ["Integrated Camera", "USB Camera"]))
        self.assertEqual(video_capture_calls, [])


if __name__ == "__main__":
    unittest.main()
