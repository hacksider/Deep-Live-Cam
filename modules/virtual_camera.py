"""Virtual Camera output manager for Deep-Live-Cam.

Feeds processed live webcam frames into a virtual webcam device using pyvirtualcam,
enabling direct video routing to apps like Zoom, Discord, Google Meet, and OBS.
"""

from __future__ import annotations

import glob
import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except ImportError:
    HAS_PYVIRTUALCAM = False


class VirtualCameraManager:
    """Manages the pyvirtualcam Camera instance."""

    def __init__(self):
        self._cam: Optional[pyvirtualcam.Camera] = None
        self._current_size: Optional[Tuple[int, int]] = None
        self._device: Optional[str] = None
        self._last_error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self._cam is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @staticmethod
    def get_available_devices() -> List[str]:
        """Detect available virtual camera devices or loopbacks."""
        devices: List[str] = ["Default / Auto"]
        system = platform.system()
        if system == "Linux":
            # Search for v4l2 video loopback devices
            for dev in sorted(glob.glob("/dev/video*")):
                devices.append(dev)
        elif system == "Windows":
            # Common Windows virtual camera driver names
            devices.extend(["OBS Virtual Camera", "Unity Video Capture"])
        elif system == "Darwin":
            devices.extend(["OBS Virtual Camera"])
        return devices

    def start(self, width: int, height: int, fps: float = 30.0, device: Optional[str] = None) -> bool:
        """Start the virtual camera stream."""
        if not HAS_PYVIRTUALCAM:
            self._last_error = "pyvirtualcam is not installed"
            print(f"[VirtualCamera] {self._last_error}")
            return False

        self.stop()
        actual_device = None if (not device or device == "Default / Auto") else device

        try:
            # pyvirtualcam expects RGB format
            self._cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
                device=actual_device,
            )
            self._current_size = (width, height)
            self._device = actual_device
            self._last_error = None
            print(
                f"[VirtualCamera] Virtual camera started: {self._cam.device} "
                f"({width}x{height} @ {fps:.1f}fps)"
            )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[VirtualCamera] Failed to initialize virtual camera: {exc}")
            self._cam = None
            return False

    def send_bgr_frame(self, bgr_frame: np.ndarray) -> bool:
        """Send a BGR frame from OpenCV to the virtual camera."""
        if self._cam is None or bgr_frame is None or bgr_frame.size == 0:
            return False

        h, w = bgr_frame.shape[:2]
        # Re-initialize camera if resolution changed dynamically
        if self._current_size != (w, h):
            if not self.start(w, h, fps=self._cam.fps, device=self._device):
                return False

        try:
            # Convert OpenCV BGR to RGB for pyvirtualcam
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            self._cam.send(rgb_frame)
            return True
        except Exception as exc:
            print(f"[VirtualCamera] Error sending frame: {exc}")
            return False

    def stop(self) -> None:
        """Release the virtual camera."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception as exc:
                print(f"[VirtualCamera] Error closing virtual camera: {exc}")
            finally:
                self._cam = None
                self._current_size = None


# Global singleton instance
VIRTUAL_CAM = VirtualCameraManager()
