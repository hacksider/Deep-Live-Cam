import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.virtual_camera import VirtualCameraSink, find_virtual_camera_device


class VirtualCameraTests(unittest.TestCase):
    def test_finds_device_by_label(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(directory)
            sys_class = root / "sys"
            device_dir = sys_class / "video10"
            device_dir.mkdir(parents=True)
            (device_dir / "name").write_text("Deep Live Cam\n", encoding="utf-8")
            dev_root = root / "dev"
            dev_root.mkdir()
            (dev_root / "video10").touch()

            self.assertEqual(
                find_virtual_camera_device(sys_class, dev_root),
                str(dev_root / "video10"),
            )

    def test_resizes_and_converts_frames(self):
        sent = []

        class FakeCamera:
            backend = "test"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def send(self, frame):
                sent.append(frame)

            def close(self):
                pass

        fake_module = types.SimpleNamespace(
            Camera=FakeCamera,
            PixelFormat=types.SimpleNamespace(YUYV="YUYV"),
        )
        with patch.dict(sys.modules, {"pyvirtualcam": fake_module}):
            sink = VirtualCameraSink(640, 360, 30, device="/dev/video10")
            sink.send(np.ones((180, 320, 3), dtype=np.float32) * 300)
            sink.close()

        self.assertEqual(sent[0].shape, (360, 640, 2))
        self.assertEqual(sent[0].dtype, np.uint8)
        self.assertTrue(sent[0].flags.c_contiguous)
        self.assertEqual(tuple(sent[0][0, 0]), (235, 128))


if __name__ == "__main__":
    unittest.main()
