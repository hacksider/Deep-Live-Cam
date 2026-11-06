import cv2
import numpy as np
import os
import threading
from typing import Optional

_landmarker = None
_landmarker_lock = threading.Lock()
_INITIALIZED = False

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task",
)


def _get_hands_detector():
    global _landmarker, _INITIALIZED
    if _INITIALIZED:
        return _landmarker
    with _landmarker_lock:
        if _INITIALIZED:
            return _landmarker
        try:
            import mediapipe as mp
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=_MODEL_PATH,
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
            _landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception:
            _landmarker = None
        _INITIALIZED = True
    return _landmarker


def detect_hand_mask(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect hands and return a feathered uint8 mask (255 = hand)."""
    detector = _get_hands_detector()
    if detector is None:
        return None

    try:
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)

        if not result.hand_landmarks:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        for hand in result.hand_landmarks:
            pts = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in hand],
                dtype=np.int32,
            )
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

        k = max(5, min(h, w) // 80) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask
    except Exception:
        return None


def apply_hand_occlusion(
    swapped_frame: np.ndarray,
    original_frame: np.ndarray,
    hand_mask: np.ndarray,
) -> np.ndarray:
    """Composite original hand pixels back over the swapped face."""
    alpha = hand_mask.astype(np.float32) * (1.0 / 255.0)
    alpha_3c = alpha[:, :, np.newaxis]
    blended = (
        original_frame.astype(np.float32) * alpha_3c
        + swapped_frame.astype(np.float32) * (1.0 - alpha_3c)
    )
    np.clip(blended, 0, 255, out=blended)
    return blended.astype(np.uint8)
