from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


class StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


def _module(name, **attrs):
    module = StubModule(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _call_get_available_cameras(devices):
    video_capture_calls = []

    fake_ctk = _module("customtkinter")
    fake_ctk.__path__ = []

    def fake_video_capture(index):
        video_capture_calls.append(index)
        return types.SimpleNamespace(isOpened=lambda: False, release=lambda: None)

    fake_cv2 = _module("cv2", VideoCapture=fake_video_capture)
    fake_platform = _module("platform", system=lambda: "Windows")

    fake_graph_module = types.ModuleType("pygrabber.dshow_graph")

    class FakeFilterGraph:
        def get_input_devices(self):
            return devices

    fake_graph_module.FilterGraph = FakeFilterGraph

    fake_pil = _module(
        "PIL",
        Image=types.ModuleType("PIL.Image"),
        ImageOps=types.ModuleType("PIL.ImageOps"),
    )
    fake_globals = _module(
        "modules.globals",
        file_types=(("Images", "*.png"), ("Videos", "*.mp4")),
    )

    sys.modules.pop("modules.ui", None)
    with mock.patch.dict(
        sys.modules,
        {
            "customtkinter": fake_ctk,
            "cv2": fake_cv2,
            "numpy": StubModule("numpy"),
            "requests": types.ModuleType("requests"),
            "PIL": fake_pil,
            "PIL.Image": fake_pil.Image,
            "PIL.ImageOps": fake_pil.ImageOps,
            "platform": fake_platform,
            "pygrabber.dshow_graph": fake_graph_module,
            "modules.globals": fake_globals,
            "modules.metadata": StubModule("modules.metadata"),
            "modules.gpu_processing": StubModule("modules.gpu_processing"),
            "modules.face_analyser": StubModule("modules.face_analyser"),
            "modules.capturer": StubModule("modules.capturer"),
            "modules.processors.frame.core": StubModule("modules.processors.frame.core"),
            "modules.utilities": _module(
                "modules.utilities",
                resolve_relative_path=lambda path: path,
            ),
            "modules.video_capture": _module("modules.video_capture", VideoCapturer=object),
            "modules.gettext": _module("modules.gettext", LanguageManager=object),
            "modules.ui_tooltip": _module("modules.ui_tooltip", ToolTip=object),
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
