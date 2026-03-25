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
        self.target_width: Optional[int] = None
        self.target_height: Optional[int] = None

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 640, height: int = 480, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        try:
            if platform.system() == "Windows":
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_ANY),
                    (-1, cv2.CAP_ANY),
                    (0, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            elif platform.system() == "Darwin":
                # On macOS, never silently fall back to index 0 because AVFoundation
                # index 0 is often "Capture screen", which produces recursive/gray
                # preview artifacts instead of webcam frames.
                capture_methods = [
                    (self.device_index, cv2.CAP_AVFOUNDATION),
                ]
                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            self.device_index = dev_id
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except Exception:
                pass
            self.target_width = int(width)
            self.target_height = int(height)

            # MJPG tends to be unstable on AVFoundation (macOS) and may produce
            # grayscale/flickering frames. Keep it for non-macOS paths only.
            if platform.system() != "Darwin":
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass

            if platform.system() == "Darwin":
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass

            self.is_running = True
            return True

        except Exception as e:
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
            # Normalize to 3-channel BGR to keep downstream detectors/processors
            # consistent across camera backends.
            try:
                if frame is not None and frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif (
                    frame is not None
                    and frame.ndim == 3
                    and frame.shape[2] == 4
                ):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

            if (
                self.target_width
                and self.target_height
                and frame is not None
                and (
                    frame.shape[1] != self.target_width
                    or frame.shape[0] != self.target_height
                )
            ):
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA
                    if frame.shape[1] > self.target_width
                    else cv2.INTER_LINEAR,
                )
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
