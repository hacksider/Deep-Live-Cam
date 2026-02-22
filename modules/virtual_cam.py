"""Virtual camera output — sends processed frames to a system virtual camera device.

Requires pyvirtualcam (optional dependency). Platform backends:
  - macOS: OBS Virtual Camera (OBS 30+ must be installed and started once)
  - Linux: v4l2loopback kernel module
  - Windows: OBS Virtual Camera (OBS 26+)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvirtualcam

    _AVAILABLE = True
except ImportError:
    pyvirtualcam = None  # type: ignore[assignment]
    _AVAILABLE = False

_camera = None


def is_available() -> bool:
    """Return True if pyvirtualcam is installed."""
    return _AVAILABLE


def is_active() -> bool:
    """Return True if the virtual camera is currently sending frames."""
    return _camera is not None


def start(width: int, height: int, fps: float = 30.0) -> bool:
    """Open the virtual camera device.

    Returns True on success, False on failure (missing backend, etc.).
    """
    global _camera

    if not _AVAILABLE:
        logger.warning(
            "pyvirtualcam is not installed. "
            "Install with: pip install pyvirtualcam"
        )
        return False

    if _camera is not None:
        logger.debug("Virtual camera already active, stopping first")
        stop()

    try:
        _camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        logger.info("Virtual camera started: %s (%dx%d @ %.0f fps)",
                     _camera.device, width, height, fps)
        return True
    except RuntimeError as exc:
        import sys

        if sys.platform == "darwin":
            hint = "Install OBS 30+, launch it, start Virtual Camera once, then close OBS."
        elif sys.platform == "linux":
            hint = ("Install v4l2loopback: "
                    "sudo apt install v4l2loopback-dkms && "
                    "sudo modprobe v4l2loopback devices=1")
        else:
            hint = "Install OBS 26+ to provide the virtual camera backend."

        logger.error("Failed to start virtual camera: %s\nHint: %s", exc, hint)
        _camera = None
        return False


def send(frame: np.ndarray) -> None:
    """Send a BGR frame to the virtual camera.

    The frame is resized to match the camera dimensions if needed.
    Must be called from a single thread (pyvirtualcam is not thread-safe).
    """
    if _camera is None:
        return

    h, w = frame.shape[:2]
    cam_h, cam_w = _camera.height, _camera.width

    if (h, w) != (cam_h, cam_w):
        import cv2

        frame = cv2.resize(frame, (cam_w, cam_h))

    _camera.send(frame)


def stop() -> None:
    """Close the virtual camera device."""
    global _camera

    if _camera is not None:
        try:
            _camera.close()
        except Exception:
            pass
        logger.info("Virtual camera stopped")
        _camera = None
