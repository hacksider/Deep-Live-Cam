"""
Enhanced Live Face Swapper with optimized performance and quality
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Any
from collections import deque
import modules.globals
from modules.face_analyser import get_one_face, get_many_faces
from modules.processors.frame.face_swapper import get_face_swapper
# Removed performance_optimizer import to maximize FPS
from modules.video_capture import VideoCapturer


class LiveFaceSwapper:
    def __init__(self):
        self.is_running = False
        self.source_face = None
        self.video_capturer = None
        self.processing_thread = None
        self.display_callback = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processed_frames = 0
        
        # Frame processing
        self.input_queue = deque(maxlen=2)  # Small queue to reduce latency
        self.output_queue = deque(maxlen=2)
        self.queue_lock = threading.Lock()
        
        # Quality settings
        self.quality_mode = "balanced"  # "fast", "balanced", "quality"
        self.adaptive_quality = True
        
    def set_source_face(self, source_image_path: str) -> bool:
        """Set the source face for swapping"""
        try:
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                return False
                
            face = get_one_face(source_image)
            if face is None:
                return False
                
            self.source_face = face
            return True
        except Exception as e:
            print(f"Error setting source face: {e}")
            return False
    
    def start_live_swap(self, camera_index: int, display_callback: Callable[[np.ndarray, float], None]) -> bool:
        """Start live face swapping"""
        try:
            if self.source_face is None:
                print("No source face set")
                return False
                
            self.display_callback = display_callback
            self.video_capturer = VideoCapturer(camera_index)
            
            # Start video capture with optimized settings
            if not self.video_capturer.start(width=960, height=540, fps=30):
                return False
            
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start capture loop
            self._capture_loop()
            return True
            
        except Exception as e:
            print(f"Error starting live swap: {e}")
            return False
    
    def stop_live_swap(self):
        """Stop live face swapping"""
        self.is_running = False
        if self.video_capturer:
            self.video_capturer.release()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _capture_loop(self):
        """Main capture loop"""
        while self.is_running:
            try:
                ret, frame = self.video_capturer.read()
                if ret and frame is not None:
                    # Add frame to processing queue
                    with self.queue_lock:
                        if len(self.input_queue) < self.input_queue.maxlen:
                            self.input_queue.append(frame.copy())
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                break
    
    def _processing_loop(self):
        """Background processing loop for face swapping"""
        while self.is_running:
            try:
                frame_to_process = None
                
                # Get frame from input queue
                with self.queue_lock:
                    if self.input_queue:
                        frame_to_process = self.input_queue.popleft()
                
                if frame_to_process is not None:
                    # Process the frame
                    processed_frame = self._process_frame(frame_to_process)
                    
                    # Add to output queue
                    with self.queue_lock:
                        if len(self.output_queue) < self.output_queue.maxlen:
                            self.output_queue.append(processed_frame)
                    
                    # Update FPS and call display callback
                    self._update_fps()
                    if self.display_callback:
                        self.display_callback(processed_frame, self.current_fps)
                
                else:
                    # No frame to process, small delay
                    time.sleep(0.005)
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.01)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-fast frame processing - maximum FPS priority"""
        try:
            # Use the fastest face swapping method for maximum FPS
            if modules.globals.many_faces:
                many_faces = get_many_faces(frame)
                if many_faces:
                    for target_face in many_faces:
                        if self.source_face and target_face:
                            from modules.processors.frame.face_swapper import swap_face
                            frame = swap_face(self.source_face, target_face, frame)
            else:
                target_face = get_one_face(frame)
                if target_face and self.source_face:
                    from modules.processors.frame.face_swapper import swap_face
                    frame = swap_face(self.source_face, target_face, frame)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def set_quality_mode(self, mode: str):
        """Set quality mode: 'fast', 'balanced', or 'quality'"""
        self.quality_mode = mode
        # Removed performance_optimizer references for maximum FPS
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            'fps': self.current_fps,
            'quality_level': 1.0,  # Fixed value for maximum FPS
            'detection_interval': 0.1,  # Fixed value for maximum FPS
            'processed_frames': self.processed_frames
        }


# Global instance
live_face_swapper = LiveFaceSwapper()