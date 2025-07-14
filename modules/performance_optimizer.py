"""
Performance optimization module for Deep-Live-Cam
Provides frame caching, adaptive quality, and FPS optimization
"""
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import threading
from collections import deque
import modules.globals

class PerformanceOptimizer:
    def __init__(self):
        self.frame_cache = {}
        self.face_cache = {}
        self.last_detection_time = 0
        self.detection_interval = 0.1  # Detect faces every 100ms
        self.adaptive_quality = True
        self.target_fps = 30
        self.frame_times = deque(maxlen=10)
        self.current_fps = 0
        self.quality_level = 1.0
        self.min_quality = 0.5
        self.max_quality = 1.0
        
    def should_detect_faces(self) -> bool:
        """Determine if we should run face detection based on timing"""
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_interval:
            self.last_detection_time = current_time
            return True
        return False
    
    def update_fps_stats(self, frame_time: float):
        """Update FPS statistics and adjust quality accordingly"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) >= 5:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            if self.adaptive_quality:
                self._adjust_quality()
    
    def _adjust_quality(self):
        """Dynamically adjust processing quality based on FPS"""
        if self.current_fps < self.target_fps * 0.8:  # Below 80% of target
            self.quality_level = max(self.min_quality, self.quality_level - 0.1)
            self.detection_interval = min(0.2, self.detection_interval + 0.02)
        elif self.current_fps > self.target_fps * 0.95:  # Above 95% of target
            self.quality_level = min(self.max_quality, self.quality_level + 0.05)
            self.detection_interval = max(0.05, self.detection_interval - 0.01)
    
    def get_optimal_resolution(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get optimal processing resolution based on current quality level"""
        width, height = original_size
        scale = self.quality_level
        return (int(width * scale), int(height * scale))
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal performance"""
        if self.quality_level < 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.quality_level)
            new_width = int(width * self.quality_level)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return frame
    
    def postprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess frame to target resolution"""
        if frame.shape[:2][::-1] != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        return frame

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()