"""
Advanced Face Tracking with Occlusion Handling and Stabilization
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import time
from modules.typing import Face, Frame


class FaceTracker:
    def __init__(self):
        # Face tracking history
        self.face_history = deque(maxlen=10)
        self.stable_face_position = None
        self.last_valid_face = None
        self.tracking_confidence = 0.0
        
        # Stabilization parameters
        self.position_smoothing = 0.7  # Higher = more stable, lower = more responsive
        self.size_smoothing = 0.8
        self.landmark_smoothing = 0.6
        
        # Occlusion detection
        self.occlusion_threshold = 0.3
        self.face_template = None
        self.template_update_interval = 30  # frames
        self.frame_count = 0
        
        # Kalman filter for position prediction
        self.kalman_filter = self._init_kalman_filter()
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter for face position prediction"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        return kalman
    
    def track_face(self, current_face: Optional[Face], frame: Frame) -> Optional[Face]:
        """
        Track face with stabilization and occlusion handling
        """
        self.frame_count += 1
        
        if current_face is not None:
            # We have a detected face
            stabilized_face = self._stabilize_face(current_face)
            self._update_face_history(stabilized_face)
            self._update_face_template(frame, stabilized_face)
            self.last_valid_face = stabilized_face
            self.tracking_confidence = min(1.0, self.tracking_confidence + 0.1)
            return stabilized_face
        
        else:
            # No face detected - handle occlusion
            if self.last_valid_face is not None and self.tracking_confidence > 0.3:
                # Try to predict face position using tracking
                predicted_face = self._predict_face_position(frame)
                if predicted_face is not None:
                    self.tracking_confidence = max(0.0, self.tracking_confidence - 0.05)
                    return predicted_face
            
            # Gradually reduce confidence
            self.tracking_confidence = max(0.0, self.tracking_confidence - 0.1)
            return None
    
    def _stabilize_face(self, face: Face) -> Face:
        """Apply stabilization to reduce jitter"""
        if len(self.face_history) == 0:
            return face
        
        # Get the last stable face
        last_face = self.face_history[-1]
        
        # Smooth the bounding box
        face.bbox = self._smooth_bbox(face.bbox, last_face.bbox)
        
        # Smooth landmarks if available
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            if hasattr(last_face, 'landmark_2d_106') and last_face.landmark_2d_106 is not None:
                face.landmark_2d_106 = self._smooth_landmarks(
                    face.landmark_2d_106, last_face.landmark_2d_106
                )
        
        # Update Kalman filter
        center_x = (face.bbox[0] + face.bbox[2]) / 2
        center_y = (face.bbox[1] + face.bbox[3]) / 2
        self.kalman_filter.correct(np.array([[center_x], [center_y]], dtype=np.float32))
        
        return face
    
    def _smooth_bbox(self, current_bbox: np.ndarray, last_bbox: np.ndarray) -> np.ndarray:
        """Smooth bounding box coordinates"""
        alpha = 1 - self.position_smoothing
        return alpha * current_bbox + (1 - alpha) * last_bbox
    
    def _smooth_landmarks(self, current_landmarks: np.ndarray, last_landmarks: np.ndarray) -> np.ndarray:
        """Smooth facial landmarks"""
        alpha = 1 - self.landmark_smoothing
        return alpha * current_landmarks + (1 - alpha) * last_landmarks
    
    def _update_face_history(self, face: Face):
        """Update face tracking history"""
        self.face_history.append(face)
    
    def _update_face_template(self, frame: Frame, face: Face):
        """Update face template for occlusion detection"""
        if self.frame_count % self.template_update_interval == 0:
            try:
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    face_region = frame[y1:y2, x1:x2]
                    self.face_template = cv2.resize(face_region, (64, 64))
            except Exception:
                pass
    
    def _predict_face_position(self, frame: Frame) -> Optional[Face]:
        """Predict face position during occlusion"""
        if self.last_valid_face is None:
            return None
        
        try:
            # Use Kalman filter prediction
            prediction = self.kalman_filter.predict()
            pred_x, pred_y = prediction[0, 0], prediction[1, 0]
            
            # Create predicted face based on last valid face
            predicted_face = self._create_predicted_face(pred_x, pred_y)
            
            # Verify prediction using template matching if available
            if self.face_template is not None:
                confidence = self._verify_prediction(frame, predicted_face)
                if confidence > self.occlusion_threshold:
                    return predicted_face
            else:
                return predicted_face
                
        except Exception:
            pass
        
        return None
    
    def _create_predicted_face(self, center_x: float, center_y: float) -> Face:
        """Create a predicted face object"""
        # Use the last valid face as template
        predicted_face = type(self.last_valid_face)()
        
        # Copy attributes from last valid face
        for attr in dir(self.last_valid_face):
            if not attr.startswith('_'):
                try:
                    setattr(predicted_face, attr, getattr(self.last_valid_face, attr))
                except:
                    pass
        
        # Update position
        last_center_x = (self.last_valid_face.bbox[0] + self.last_valid_face.bbox[2]) / 2
        last_center_y = (self.last_valid_face.bbox[1] + self.last_valid_face.bbox[3]) / 2
        
        offset_x = center_x - last_center_x
        offset_y = center_y - last_center_y
        
        # Update bbox
        predicted_face.bbox = self.last_valid_face.bbox + [offset_x, offset_y, offset_x, offset_y]
        
        # Update landmarks if available
        if hasattr(predicted_face, 'landmark_2d_106') and predicted_face.landmark_2d_106 is not None:
            predicted_face.landmark_2d_106 = self.last_valid_face.landmark_2d_106 + [offset_x, offset_y]
        
        return predicted_face
    
    def _verify_prediction(self, frame: Frame, predicted_face: Face) -> float:
        """Verify predicted face position using template matching"""
        try:
            x1, y1, x2, y2 = predicted_face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            current_region = frame[y1:y2, x1:x2]
            current_region = cv2.resize(current_region, (64, 64))
            
            # Template matching
            result = cv2.matchTemplate(current_region, self.face_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            return max_val
            
        except Exception:
            return 0.0
    
    def is_face_stable(self) -> bool:
        """Check if face tracking is stable"""
        return len(self.face_history) >= 5 and self.tracking_confidence > 0.7
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.face_history.clear()
        self.stable_face_position = None
        self.last_valid_face = None
        self.tracking_confidence = 0.0
        self.face_template = None
        self.kalman_filter = self._init_kalman_filter()


# Global face tracker instance
face_tracker = FaceTracker()