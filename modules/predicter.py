import numpy as np
import opennsfw2
from PIL import Image
from modules.typing import Frame

MAX_PROBABILITY = 0.85

# Preload the model once for efficiency
model = opennsfw2.make_open_nsfw_model()

def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    views = np.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY

def predict_image(target_path: str) -> bool:
    probability = opennsfw2.predict_image(target_path, model=model)
    return probability > MAX_PROBABILITY

def predict_video(target_path: str) -> bool:
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100, model=model)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
