import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform
import threading
import time
from collections import deque

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
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.current_fps = 0
        self.target_fps = 30
        self.frame_skip = 1
        self.frame_counter = 0
        
        # Buffer management
        self.frame_buffer = deque(maxlen=3)
        self.buffer_lock = threading.Lock()

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
        """Initialize and start video capture with performance optimizations"""
        try:
            self.target_fps = fps
            
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
                # Unix-like systems (Linux/Mac) capture method
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Configure format with performance optimizations
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Additional performance settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera with performance optimizations"""
        if not self.is_running or self.cap is None:
            return False, None

        start_time = time.time()
        
        # Implement frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            # Skip this frame but still read to clear buffer
            ret, _ = self.cap.read()
            return ret, self._current_frame if ret else None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            
            # Update performance metrics
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            self._update_performance_metrics()
            
            # Add to buffer for processing
            with self.buffer_lock:
                self.frame_buffer.append(frame.copy())
            
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None
    
    def _update_performance_metrics(self):
        """Update FPS and adjust frame skipping based on performance"""
        if len(self.frame_times) >= 10:
            avg_frame_time = sum(list(self.frame_times)[-10:]) / 10
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Adaptive frame skipping
            if self.current_fps < self.target_fps * 0.8:
                self.frame_skip = min(3, self.frame_skip + 1)
            elif self.current_fps > self.target_fps * 0.95:
                self.frame_skip = max(1, self.frame_skip - 1)
    
    def get_buffered_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from buffer"""
        with self.buffer_lock:
            return self.frame_buffer[-1] if self.frame_buffer else None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
