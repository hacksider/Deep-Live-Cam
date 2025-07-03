import numpy
import opennsfw2
from PIL import Image
import cv2
import modules.globals
from modules.typing import Frame

MAX_PROBABILITY = 0.85

# Preload the model once for efficiency
model = None

def predict_frame(target_frame: numpy.ndarray) -> bool:
    """Predict if a frame is NSFW using OpenNSFW2."""
    try:
        # Convert the frame to RGB before processing if color correction is enabled
        if modules.globals.color_correction:
            target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
            
        image = Image.fromarray(target_frame)
        image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
        global model
        if model is None: 
            model = opennsfw2.make_open_nsfw_model()
            
        views = numpy.expand_dims(image, axis=0)
        _, probability = model.predict(views)[0]
        return probability > MAX_PROBABILITY
    except Exception as e:
        print(f"Error in predict_frame: {e}")
        return False


def predict_image(target_path: str) -> bool:
    """Predict if an image file is NSFW."""
    try:
        return opennsfw2.predict_image(target_path) > MAX_PROBABILITY
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return False


def predict_video(target_path: str) -> bool:
    """Predict if any frame in a video is NSFW."""
    try:
        _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
        return any(probability > MAX_PROBABILITY for probability in probabilities)
    except Exception as e:
        print(f"Error in predict_video: {e}")
        return False
