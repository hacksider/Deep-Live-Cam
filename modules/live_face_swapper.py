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
from modules.processors.frame.face_swapper import swap_face_enhanced, get_face_swapper
from modules.performance_optimizer import performance_optimizer
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
        """Process a single frame with face swapping, tracking, and occlusion handling"""
        try:
            start_time = time.time()
            
            # Apply performance optimizations
            original_size = frame.shape[:2][::-1]
            processed_frame = performance_optimizer.preprocess_frame(frame)
            
            # Import face tracker
            from modules.face_tracker import face_tracker
            
            # Detect and track faces based on performance settings
            if modules.globals.many_faces:
                if performance_optimizer.should_detect_faces():
                    detected_faces = get_many_faces(processed_frame)
                    # Apply tracking to each face
                    tracked_faces = []
                    for face in (detected_faces or []):
                        tracked_face = face_tracker.track_face(face, processed_frame)
                        if tracked_face:
                            tracked_faces.append(tracked_face)
                    performance_optimizer.face_cache['many_faces'] = tracked_faces
                else:
                    tracked_faces = performance_optimizer.face_cache.get('many_faces', [])
                
                if tracked_faces:
                    for target_face in tracked_faces:
                        if self.source_face and target_face:
                            # Use enhanced swap with occlusion handling
                            from modules.processors.frame.face_swapper import swap_face_enhanced_with_occlusion
                            processed_frame = swap_face_enhanced_with_occlusion(
                                self.source_face, target_face, processed_frame, frame
                            )
            else:
                if performance_optimizer.should_detect_faces():
                    detected_face = get_one_face(processed_frame)
                    tracked_face = face_tracker.track_face(detected_face, processed_frame)
                    performance_optimizer.face_cache['single_face'] = tracked_face
                else:
                    tracked_face = performance_optimizer.face_cache.get('single_face')
                
                if tracked_face and self.source_face:
                    # Use enhanced swap with occlusion handling
                    from modules.processors.frame.face_swapper import swap_face_enhanced_with_occlusion
                    processed_frame = swap_face_enhanced_with_occlusion(
                        self.source_face, tracked_face, processed_frame, frame
                    )
                else:
                    # Try to use tracking even without detection (for occlusion handling)
                    tracked_face = face_tracker.track_face(None, processed_frame)
                    if tracked_face and self.source_face:
                        from modules.processors.frame.face_swapper import swap_face_enhanced_with_occlusion
                        processed_frame = swap_face_enhanced_with_occlusion(
                            self.source_face, tracked_face, processed_frame, frame
                        )
            
            # Post-process back to original size
            final_frame = performance_optimizer.postprocess_frame(processed_frame, original_size)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            performance_optimizer.update_fps_stats(processing_time)
            
            return final_frame
            
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
        
        if mode == "fast":
            performance_optimizer.quality_level = 0.7
            performance_optimizer.detection_interval = 0.15
        elif mode == "balanced":
            performance_optimizer.quality_level = 0.85
            performance_optimizer.detection_interval = 0.1
        elif mode == "quality":
            performance_optimizer.quality_level = 1.0
            performance_optimizer.detection_interval = 0.05
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            'fps': self.current_fps,
            'quality_level': performance_optimizer.quality_level,
            'detection_interval': performance_optimizer.detection_interval,
            'processed_frames': self.processed_frames
        }


# Global instance
live_face_swapper = LiveFaceSwapper()