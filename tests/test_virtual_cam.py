"""Tests for the virtual camera module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modules.globals


class TestVirtualCamGlobalFlag:
    """Verify the virtual_cam global flag exists and defaults correctly."""

    def test_default_is_false(self):
        assert modules.globals.virtual_cam is False


class TestVirtualCamModule:
    """Test the virtual_cam module interface."""

    def test_import(self):
        from modules import virtual_cam

        assert hasattr(virtual_cam, "start")
        assert hasattr(virtual_cam, "stop")
        assert hasattr(virtual_cam, "send")
        assert hasattr(virtual_cam, "is_active")
        assert hasattr(virtual_cam, "is_available")

    def test_is_active_default(self):
        from modules import virtual_cam

        # Ensure clean state
        virtual_cam._camera = None
        assert virtual_cam.is_active() is False

    def test_stop_when_not_started(self):
        from modules import virtual_cam

        virtual_cam._camera = None
        # Should not raise
        virtual_cam.stop()
        assert virtual_cam.is_active() is False

    def test_send_when_not_started(self):
        from modules import virtual_cam

        virtual_cam._camera = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Should not raise
        virtual_cam.send(frame)


class TestVirtualCamWithMock:
    """Test start/stop/send lifecycle with mocked pyvirtualcam."""

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_start_success(self, mock_pyvirtualcam):
        from modules import virtual_cam

        mock_camera = MagicMock()
        mock_camera.device = "TestCam"
        mock_pyvirtualcam.Camera.return_value = mock_camera

        result = virtual_cam.start(640, 480, 30.0)

        assert result is True
        assert virtual_cam.is_active() is True
        mock_pyvirtualcam.Camera.assert_called_once()

        # Cleanup
        virtual_cam._camera = None

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_start_runtime_error(self, mock_pyvirtualcam):
        from modules import virtual_cam

        mock_pyvirtualcam.Camera.side_effect = RuntimeError("No backend")

        result = virtual_cam.start(640, 480)

        assert result is False
        assert virtual_cam.is_active() is False

    @patch("modules.virtual_cam._AVAILABLE", False)
    def test_start_not_installed(self):
        from modules import virtual_cam

        result = virtual_cam.start(640, 480)

        assert result is False
        assert virtual_cam.is_active() is False

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_send_frame(self, mock_pyvirtualcam):
        from modules import virtual_cam

        mock_camera = MagicMock()
        mock_camera.device = "TestCam"
        mock_camera.width = 640
        mock_camera.height = 480
        mock_pyvirtualcam.Camera.return_value = mock_camera

        virtual_cam.start(640, 480)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        virtual_cam.send(frame)

        mock_camera.send.assert_called_once()

        # Cleanup
        virtual_cam._camera = None

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_send_resizes_mismatched_frame(self, mock_pyvirtualcam):
        from modules import virtual_cam

        mock_camera = MagicMock()
        mock_camera.device = "TestCam"
        mock_camera.width = 640
        mock_camera.height = 480
        mock_pyvirtualcam.Camera.return_value = mock_camera

        virtual_cam.start(640, 480)

        # Send a frame with different dimensions
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        virtual_cam.send(frame)

        mock_camera.send.assert_called_once()
        sent_frame = mock_camera.send.call_args[0][0]
        assert sent_frame.shape == (480, 640, 3)

        # Cleanup
        virtual_cam._camera = None

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_stop(self, mock_pyvirtualcam):
        from modules import virtual_cam

        mock_camera = MagicMock()
        mock_camera.device = "TestCam"
        mock_pyvirtualcam.Camera.return_value = mock_camera

        virtual_cam.start(640, 480)
        assert virtual_cam.is_active() is True

        virtual_cam.stop()
        assert virtual_cam.is_active() is False
        mock_camera.close.assert_called_once()

    @patch("modules.virtual_cam._AVAILABLE", True)
    @patch("modules.virtual_cam.pyvirtualcam")
    def test_restart(self, mock_pyvirtualcam):
        """Starting while already active should stop the old camera first."""
        from modules import virtual_cam

        mock_camera_1 = MagicMock()
        mock_camera_1.device = "TestCam1"
        mock_camera_2 = MagicMock()
        mock_camera_2.device = "TestCam2"
        mock_pyvirtualcam.Camera.side_effect = [mock_camera_1, mock_camera_2]

        virtual_cam.start(640, 480)
        virtual_cam.start(640, 480)

        mock_camera_1.close.assert_called_once()
        assert virtual_cam.is_active() is True

        # Cleanup
        virtual_cam._camera = None
