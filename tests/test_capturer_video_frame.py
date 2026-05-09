import importlib
import sys
import types
import unittest


class FakeCapture:
    def __init__(self, frame_total):
        self.frame_total = frame_total
        self.set_calls = []
        self.released = False

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return self.frame_total

    def read(self):
        return True, "frame"

    def release(self):
        self.released = True


def _load_capturer(captures):
    cv2 = types.SimpleNamespace(
        CAP_PROP_FOURCC=1,
        CAP_PROP_CONVERT_RGB=2,
        CAP_PROP_FRAME_COUNT=3,
        CAP_PROP_POS_FRAMES=4,
        COLOR_BGR2RGB=5,
        IMREAD_COLOR=6,
        imread=lambda *_args, **_kwargs: None,
        imdecode=lambda *_args, **_kwargs: None,
        imencode=lambda *_args, **_kwargs: (True, types.SimpleNamespace(tofile=lambda *_a, **_k: None)),
        VideoWriter_fourcc=lambda *_args: 0,
        VideoCapture=lambda _path: captures.pop(0),
    )
    sys.modules["cv2"] = cv2
    sys.modules["numpy"] = types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b"")
    globals_stub = types.SimpleNamespace(color_correction=False)
    sys.modules["modules.globals"] = globals_stub
    sys.modules["modules.gpu_processing"] = types.SimpleNamespace(gpu_cvt_color=lambda frame, _code: frame)
    sys.modules.pop("modules.capturer", None)
    modules_pkg = importlib.import_module("modules")
    modules_pkg.globals = globals_stub
    return importlib.import_module("modules.capturer"), cv2


class CapturerVideoFrameTests(unittest.TestCase):
    def test_uses_zero_based_frame_number_without_subtracting_one(self):
        first = FakeCapture(frame_total=10)
        second = FakeCapture(frame_total=10)
        capturer, cv2 = _load_capturer([first, second])

        self.assertEqual(capturer.get_video_frame("video.mp4", 0), "frame")
        self.assertEqual(capturer.get_video_frame("video.mp4", 1), "frame")

        self.assertIn((cv2.CAP_PROP_POS_FRAMES, 0), first.set_calls)
        self.assertIn((cv2.CAP_PROP_POS_FRAMES, 1), second.set_calls)

    def test_clamps_invalid_negative_and_beyond_last_frame_numbers(self):
        first = FakeCapture(frame_total=10)
        second = FakeCapture(frame_total=10)
        third = FakeCapture(frame_total="not-a-number")
        capturer, cv2 = _load_capturer([first, second, third])

        capturer.get_video_frame("video.mp4", -3)
        capturer.get_video_frame("video.mp4", 99)
        capturer.get_video_frame("video.mp4", 5)

        self.assertIn((cv2.CAP_PROP_POS_FRAMES, 0), first.set_calls)
        self.assertIn((cv2.CAP_PROP_POS_FRAMES, 9), second.set_calls)
        self.assertIn((cv2.CAP_PROP_POS_FRAMES, 0), third.set_calls)


if __name__ == "__main__":
    unittest.main()
