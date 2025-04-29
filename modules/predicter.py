import numpy as np
import opennsfw2
from PIL import Image
import cv2
import modules.globals
import logging
from functools import lru_cache
from typing import Union, Any

from modules.typing import Frame

logger = logging.getLogger(__name__)

# Global model instance for reuse
_model = None

@lru_cache(maxsize=1)
def load_nsfw_model():
    """
    Load the NSFW prediction model with caching
    
    Returns:
        Loaded NSFW model
    """
    try:
        logger.info("Loading NSFW detection model")
        return opennsfw2.make_open_nsfw_model()
    except Exception as e:
        logger.error(f"Failed to load NSFW model: {str(e)}")
        return None

def get_nsfw_model():
    """
    Get or initialize the NSFW model
    
    Returns:
        NSFW model instance
    """
    global _model
    if _model is None:
        _model = load_nsfw_model()
    return _model

def predict_frame(target_frame: Frame) -> bool:
    """
    Predict if a frame contains NSFW content
    
    Args:
        target_frame: Frame to analyze as numpy array
        
    Returns:
        True if NSFW content detected, False otherwise
    """
    try:
        if target_frame is None:
            logger.warning("Cannot predict on None frame")
            return False
            
        # Get threshold from globals
        threshold = getattr(modules.globals, 'nsfw_threshold', 0.85)
        
        # Convert the frame to RGB if needed
        expected_format = 'RGB' if modules.globals.color_correction else 'BGR'
        if expected_format == 'RGB' and target_frame.shape[2] == 3:
            processed_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        else:
            processed_frame = target_frame
            
        # Convert to PIL image and preprocess
        image = Image.fromarray(processed_frame)
        image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
        
        # Get model and predict
        model = get_nsfw_model()
        if model is None:
            logger.error("NSFW model not available")
            return False
            
        views = np.expand_dims(image, axis=0)
        _, probability = model.predict(views)[0]
        
        logger.debug(f"NSFW probability: {probability:.4f}")
        return probability > threshold
        
    except Exception as e:
        logger.error(f"Error during NSFW prediction: {str(e)}")
        return False

def predict_image(target_path: str) -> bool:
    """
    Predict if an image file contains NSFW content
    
    Args:
        target_path: Path to image file
        
    Returns:
        True if NSFW content detected, False otherwise
    """
    try:
        threshold = getattr(modules.globals, 'nsfw_threshold', 0.85)
        return opennsfw2.predict_image(target_path) > threshold
    except Exception as e:
        logger.error(f"Error predicting NSFW for image {target_path}: {str(e)}")
        return False

def predict_video(target_path: str) -> bool:
    """
    Predict if a video file contains NSFW content
    
    Args:
        target_path: Path to video file
        
    Returns:
        True if NSFW content detected, False otherwise
    """
    try:
        threshold = getattr(modules.globals, 'nsfw_threshold', 0.85)
        _, probabilities = opennsfw2.predict_video_frames(
            video_path=target_path, 
            frame_interval=100
        )
        return any(probability > threshold for probability in probabilities)
    except Exception as e:
        logger.error(f"Error predicting NSFW for video {target_path}: {str(e)}")
        return False