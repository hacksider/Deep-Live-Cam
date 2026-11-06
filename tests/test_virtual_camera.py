import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from modules.virtual_camera import VirtualCameraManager


class TestVirtualCameraManager(unittest.TestCase):
    def setUp(self):
        self.manager = VirtualCameraManager()

    def tearDown(self):
        self.manager.stop()

    def test_get_available_devices(self):
        devices = VirtualCameraManager.get_available_devices()
        self.assertIsInstance(devices, list)
        self.assertIn("Default / Auto", devices)

    @patch("modules.virtual_camera.pyvirtualcam")
    def test_start_and_send_frame(self, mock_pyvirtualcam):
        mock_cam = MagicMock()
        mock_cam.device = "Test Device"
        mock_cam.fps = 30.0
        mock_pyvirtualcam.Camera.return_value = mock_cam
        mock_pyvirtualcam.PixelFormat.RGB = "RGB"

        # Start virtual camera
        success = self.manager.start(640, 480, fps=30.0)
        self.assertTrue(success)
        self.assertTrue(self.manager.is_active)

        # Send frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (255, 0, 0)  # BGR blue
        sent = self.manager.send_bgr_frame(frame)
        self.assertTrue(sent)
        self.assertTrue(mock_cam.send.called)

        # Stop camera
        self.manager.stop()
        self.assertFalse(self.manager.is_active)
        self.assertTrue(mock_cam.close.called)

    @patch("modules.virtual_camera.pyvirtualcam")
    def test_start_failure_handles_gracefully(self, mock_pyvirtualcam):
        mock_pyvirtualcam.Camera.side_effect = RuntimeError("No virtual camera device found")
        success = self.manager.start(640, 480, fps=30.0)
        self.assertFalse(success)
        self.assertFalse(self.manager.is_active)
        self.assertIn("No virtual camera device found", self.manager.last_error)


if __name__ == "__main__":
    unittest.main()
