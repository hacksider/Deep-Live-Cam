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
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            try:
                self.graph = FilterGraph()
                # Verify device exists
                devices = self.graph.get_input_devices()
                if self.device_index >= len(devices):
                    # Fallback or logging, rather than immediate raise for flexibility
                    print(f"Warning: Device index {device_index} might be out of range. Available: {len(devices)}. Will attempt to open anyway.")
            except Exception as e:
                print(f"Warning: Could not initialize FilterGraph for device enumeration: {e}")
                self.graph = None


    def _capture_loop(self) -> None:
        while self.is_running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self._frame_lock:
                        self._latest_frame = frame
                else:
                    # Handle camera read failure, e.g., camera disconnected
                    print("Warning: Failed to read frame from camera in capture loop.")
                    # Small sleep to prevent tight loop on continuous read errors
                    threading.Event().wait(0.1)
            except Exception as e:
                print(f"Error in capture loop: {e}")
                self.is_running = False # Stop loop on critical error
                break
            # Small sleep to yield execution and not busy-wait if camera FPS is low
            # Adjust sleep time as needed; too high adds latency, too low uses more CPU.
            threading.Event().wait(0.001) # 1 ms sleep

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture in a separate thread."""
        if self.is_running:
            print("Capture already running.")
            return True
        try:
            if platform.system() == "Windows":
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),
                    (self.device_index, cv2.CAP_ANY),
                    (-1, cv2.CAP_ANY),
                    (0, cv2.CAP_ANY)
                ]
                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap and self.cap.isOpened():
                            print(f"Successfully opened camera {dev_id} with backend {backend}")
                            break
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                    except Exception:
                        continue
            else: # Unix-like
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with device index {self.device_index} using available methods.")

            # Configure format
            # Note: Setting properties might not always work or might reset after opening.
            # It's often better to request a format the camera natively supports if known.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Verify settings if possible (actual values might differ)
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Requested: {width}x{height}@{fps}fps. Actual: {actual_width}x{actual_height}@{actual_fps}fps")


            self.is_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            # Wait briefly for the first frame to be captured, makes initial read() more likely to succeed.
            # This is optional and can be adjusted or removed.
            threading.Event().wait(0.5) # Wait up to 0.5 seconds

            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_running = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame from the camera (non-blocking)."""
        if not self.is_running:
            return False, None

        frame_copy = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame_copy = self._latest_frame.copy()

        if frame_copy is not None:
            return True, frame_copy
        else:
            # No frame available yet, or thread stopped
            return False, None

    def release(self) -> None:
        """Stop capture thread and release resources."""
        if self.is_running:
            self.is_running = False # Signal the thread to stop
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=1.0) # Wait for thread to finish
                if self._capture_thread.is_alive():
                    print("Warning: Capture thread did not terminate cleanly.")
            self._capture_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self._frame_lock: # Clear last frame
            self._latest_frame = None
        print("Video capture released.")

    # frame_callback is removed as direct polling via read() is now non-blocking and preferred with threaded capture.
    # If a callback mechanism is still desired, it would need to be integrated carefully with the thread.
