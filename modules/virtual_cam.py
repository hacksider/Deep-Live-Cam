"""Virtual camera output manager for DeepLiveCam using pyvirtualcam."""

from typing import Optional, Tuple
import numpy as np

try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except ImportError:
    HAS_PYVIRTUALCAM = False

_CAM: Optional["pyvirtualcam.Camera"] = None
_CURRENT_SIZE: Optional[Tuple[int, int]] = None


def is_available() -> bool:
    """Check if pyvirtualcam is installed and a virtual camera device exists."""
    if not HAS_PYVIRTUALCAM:
        return False
    try:
        with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
            return True
    except Exception:
        return False


def get_device_info() -> Optional[str]:
    """Return virtual camera device name and backend, or None."""
    if not HAS_PYVIRTUALCAM:
        return None
    try:
        with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
            return f"{cam.device} ({cam.backend})"
    except Exception:
        return None


def start(width: int, height: int, fps: float) -> bool:
    """Start or restart the virtual camera device."""
    global _CAM, _CURRENT_SIZE
    if not HAS_PYVIRTUALCAM:
        return False

    stop()
    try:
        fps_target = max(1.0, min(60.0, float(fps)))
        _CAM = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps_target,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        _CURRENT_SIZE = (width, height)
        print(
            f"[DLC.VCAM] Started virtual camera: {_CAM.device} ({_CAM.backend}) "
            f"at {width}x{height}@{fps_target:.1f}fps"
        )
        return True
    except Exception as exc:
        print(f"[DLC.VCAM] Virtual camera not started: {exc}")
        print("[DLC.VCAM] TIP: Right-click 'install_virtual_cam.bat' and select 'Run as Administrator' to register the virtual camera device, or use OBS Studio.")
        _CAM = None
        _CURRENT_SIZE = None
        return False


def send(frame: np.ndarray) -> None:
    """Send a BGR frame to the virtual camera if active."""
    global _CAM, _CURRENT_SIZE
    if _CAM is None:
        return

    h, w = frame.shape[:2]
    if (w, h) != _CURRENT_SIZE:
        import cv2
        frame = cv2.resize(frame, _CURRENT_SIZE, interpolation=cv2.INTER_LINEAR)

    try:
        _CAM.send(frame)
    except Exception as exc:
        print(f"[DLC.VCAM] Frame send error: {exc}")


def stop() -> None:
    """Close the virtual camera."""
    global _CAM, _CURRENT_SIZE
    if _CAM is not None:
        try:
            _CAM.close()
        except Exception:
            pass
        _CAM = None
        _CURRENT_SIZE = None
