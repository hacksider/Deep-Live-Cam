import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class FakeCapture:
    def __init__(self):
        self.read_count = 0
        self.set_calls = []
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        if prop == 3:
            return 960
        if prop == 4:
            return 540
        if prop == 5:
            return 30
        return 0

    def read(self):
        self.read_count += 1
        return True, f"frame-{self.read_count}"

    def release(self):
        self.opened = False


def import_video_capture(fake_capture):
    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_FOURCC=6,
        CAP_DSHOW=700,
        CAP_MSMF=1400,
        CAP_ANY=0,
        VideoWriter_fourcc=lambda *_args: 1196444237,
        VideoCapture=lambda *_args, **_kwargs: fake_capture,
    )
    module_path = Path(__file__).resolve().parents[1] / "modules" / "video_capture.py"
    module_name = "video_capture_under_test"
    fake_numpy = types.SimpleNamespace(ndarray=object)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy, module_name: module}):
        spec.loader.exec_module(module)
    return module


class VideoCaptureFpsProbeTests(unittest.TestCase):
    def test_start_does_not_consume_frames_for_fps_probe(self):
        fake_capture = FakeCapture()
        video_capture = import_video_capture(fake_capture)

        with patch.object(video_capture.platform, "system", return_value="Linux"):
            capturer = video_capture.VideoCapturer(0)
            self.assertTrue(capturer.start())

        self.assertEqual(0, fake_capture.read_count)
        self.assertEqual(30, capturer.actual_fps)
        self.assertTrue(capturer.is_running)

    def test_actual_fps_updates_from_delivered_frames(self):
        fake_capture = FakeCapture()
        video_capture = import_video_capture(fake_capture)
        ticks = iter(i * 0.1 for i in range(30))

        with patch.object(video_capture.platform, "system", return_value="Linux"):
            with patch.object(video_capture.time, "perf_counter", side_effect=lambda: next(ticks)):
                capturer = video_capture.VideoCapturer(0)
                self.assertTrue(capturer.start())
                frames = [capturer.read()[1] for _ in range(30)]

        self.assertEqual("frame-1", frames[0])
        self.assertEqual("frame-30", frames[-1])
        self.assertEqual(30, fake_capture.read_count)
        self.assertAlmostEqual(10.0, capturer.actual_fps)


if __name__ == "__main__":
    unittest.main()
