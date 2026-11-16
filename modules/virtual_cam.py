"""Optional live output to a virtual camera (v4l2loopback on Linux, OBS
Virtual Camera on Windows/macOS) via the pyvirtualcam package."""

import threading
from typing import Optional, Tuple

import numpy as np


class VirtualCamOutput:
    """Wraps a pyvirtualcam.Camera, opening it lazily from the first frame.

    Re-opens automatically if the frame size changes (e.g. camera
    resolution switched mid-session). Disables itself after a failure so
    callers can keep calling send() every frame without extra checks.

    send() runs on the processing thread while close() can be called from
    the UI thread at any time (e.g. toggling "Virtual Camera" off mid
    session) — both take the same lock so a close() can never run
    concurrently with a send()/_open() and tear down self._cam out from
    under an in-flight pyvirtualcam call.
    """

    def __init__(self, fps: float, device: Optional[str] = None):
        self._fps = max(1, round(fps))
        self._device = device or None
        self._cam = None
        self._shape: Optional[Tuple[int, int]] = None
        self._disabled = False
        self._lock = threading.Lock()

    def send(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if self._disabled:
                return
            shape = frame_bgr.shape[:2]
            if self._cam is None or self._shape != shape:
                self._open_locked(frame_bgr.shape[1], frame_bgr.shape[0])
                self._shape = shape
            if self._cam is None:
                return
            try:
                self._cam.send(frame_bgr)
            except Exception as exc:
                print(f"[virtual_cam] send failed, disabling: {exc}")
                self._close_locked()
                self._disabled = True

    def _open_locked(self, width: int, height: int) -> None:
        self._close_locked()
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

    def _close_locked(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()
            # Closed explicitly (e.g. switch toggled off) — never silently
            # reopen on a subsequent send() from a thread that hasn't
            # noticed yet.
            self._disabled = True
