import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        self.last_error = None

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        try:
            if platform.system() == "Windows":
                # Windows-specific capture methods
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),  # Try DirectShow first
                    (self.device_index, cv2.CAP_ANY),  # Then try default backend
                    (-1, cv2.CAP_ANY),  # Try -1 as fallback
                    (0, cv2.CAP_ANY),  # Finally try 0 without specific backend
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                # Unix-like systems (Linux/Mac) capture method.
                # On macOS, OpenCV is more reliable when AVFoundation is tried
                # explicitly before falling back to the default backend.
                capture_methods = [
                    (self.device_index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)),
                    (self.device_index, cv2.CAP_ANY),
                    (self.device_index, -1),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue

            if not self.cap or not self.cap.isOpened():
                if platform.system() == "Darwin":
                    raise RuntimeError(
                        "Failed to open camera on macOS. Check Camera permission in System Settings, "
                        "then try a different camera index."
                    )
                raise RuntimeError("Failed to open camera")

            # Configure format
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            self.is_running = True
            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
