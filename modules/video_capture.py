import cv2
import numpy as np
import sys
import time
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
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0
        self._fps_sample_target: int = 30
        self._fps_sample_count: int = 0
        self._fps_sample_started_at: Optional[float] = None
        self._fps_sample_done: bool = False

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
                # device_index comes from pygrabber.FilterGraph (DirectShow
                # enumeration), so open with DSHOW first to preserve mapping.
                # MSMF and DirectShow enumerate cameras in different orders, so
                # opening MSMF with a DSHOW index silently selects the wrong
                # camera. MSMF/ANY remain as fallbacks for cameras DSHOW can't
                # open.
                #
                # Pass codec + resolution + fps as construction params (OpenCV
                # 4.6+). DSHOW locks the pixel format at open time and ignores
                # later cap.set(CAP_PROP_FOURCC, ...) — without this, DSHOW
                # falls back to uncompressed YUYV at 1080p, which is USB-
                # bandwidth-limited to ~5 fps. Setting MJPG at construction
                # negotiates compressed frames from the first read.
                mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                open_params = [
                    cv2.CAP_PROP_FOURCC, mjpg,
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, fps,
                ]
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),
                    (self.device_index, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend, open_params)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                # Unix-like systems (Linux/Mac) capture method
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Belt-and-braces: also set via cap.set() for backends that honor
            # post-open changes (MSMF, V4L2). DSHOW ignores these, but the
            # construction params above already handled it.
            if platform.system() != "Windows":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Read back resolution (usually reliable)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # CAP_PROP_FPS is unreliable on DirectShow — often reports 30
            # even when the camera delivers 60.  Use it as an immediate
            # startup value, then refine actual_fps from frames that callers
            # already read.  Avoid doing a warmup+sample read burst here: that
            # discards the first camera frames before live preview can consume
            # them.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = reported_fps or fps
            self._reset_fps_measurement(sample=30)

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

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
            self._record_fps_sample()
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

    def _reset_fps_measurement(self, sample: int = 30) -> None:
        """Prepare opportunistic FPS measurement without consuming frames."""
        self._fps_sample_target = max(2, sample)
        self._fps_sample_count = 0
        self._fps_sample_started_at = None
        self._fps_sample_done = False

    def _record_fps_sample(self) -> None:
        """Update actual_fps from frames already delivered to callers.

        Measuring in read() preserves the first live-preview frames while still
        correcting backends whose CAP_PROP_FPS value is inaccurate.
        """
        if self._fps_sample_done:
            return
        try:
            now = time.perf_counter()
            if self._fps_sample_started_at is None:
                self._fps_sample_started_at = now
                self._fps_sample_count = 1
                return

            self._fps_sample_count += 1
            if self._fps_sample_count < self._fps_sample_target:
                return

            elapsed = now - self._fps_sample_started_at
            if elapsed <= 0:
                return

            # N frames contain N-1 frame intervals from the first timestamp
            # to the current one.
            self.actual_fps = (self._fps_sample_count - 1) / elapsed
            self._fps_sample_done = True
        except (OSError, RuntimeError, ValueError):
            self._fps_sample_done = True

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
