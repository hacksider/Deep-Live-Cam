from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_utilities_module():
    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_threads = 0
    fake_globals.log_level = "error"
    fake_globals.execution_providers = []
    fake_globals.video_encoder = "libx264"
    fake_globals.video_quality = 23

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None))

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.uint8 = int
    fake_numpy.fromfile = lambda *args, **kwargs: b""

    with mock.patch.dict(
        sys.modules,
        {
            "modules.globals": fake_globals,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
        },
        clear=False,
    ):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.utilities", None)
        return importlib.import_module("modules.utilities")


class DetectFpsTests(unittest.TestCase):
    def test_detect_fps_returns_fraction_result(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess, "check_output", return_value="30000/1001\n"
        ):
            self.assertAlmostEqual(utilities.detect_fps("demo.mp4"), 30000 / 1001)

    def test_detect_fps_falls_back_on_command_failure(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=utilities.subprocess.CalledProcessError(1, ["ffprobe"]),
        ):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_malformed_output(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess, "check_output", return_value="not-a-fraction"
        ):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_zero_denominator(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="30/0"):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)


if __name__ == "__main__":
    unittest.main()
