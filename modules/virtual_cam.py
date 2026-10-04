"""Optional live output to a virtual camera (v4l2loopback on Linux, OBS
Virtual Camera on Windows/macOS) via the pyvirtualcam package."""

from typing import Optional, Tuple

import numpy as np


class VirtualCamOutput:
    """Wraps a pyvirtualcam.Camera, opening it lazily from the first frame.

    Re-opens automatically if the frame size changes (e.g. camera
    resolution switched mid-session). Disables itself after a failure so
    callers can keep calling send() every frame without extra checks.
    """

    def __init__(self, fps: float, device: Optional[str] = None):
        self._fps = max(1, round(fps))
        self._device = device or None
        self._cam = None
        self._shape: Optional[Tuple[int, int]] = None
        self._disabled = False

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._disabled:
            return
        shape = frame_bgr.shape[:2]
        if self._cam is None or self._shape != shape:
            self._open(frame_bgr.shape[1], frame_bgr.shape[0])
            self._shape = shape
        if self._cam is None:
            return
        try:
            self._cam.send(frame_bgr)
        except Exception as exc:
            print(f"[virtual_cam] send failed, disabling: {exc}")
            self.close()
            self._disabled = True

    def _open(self, width: int, height: int) -> None:
        self.close()
        try:
            import pyvirtualcam
        except ImportError:
            print(
                "[virtual_cam] pyvirtualcam not installed — "
                "run: pip install pyvirtualcam"
            )
            self._disabled = True
            return
        try:
            self._cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=self._fps,
                device=self._device,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            print(
                f"[virtual_cam] streaming {width}x{height}@{self._fps}fps "
                f"to {self._cam.device}"
            )
        except Exception as exc:
            print(f"[virtual_cam] failed to open virtual camera: {exc}")
            self._cam = None
            self._disabled = True

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
